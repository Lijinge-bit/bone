"""Evaluate one explicit candidate without training or activating a model.

The existing RSNA test partition has already been inspected in this project.
Its results are exploratory, even when evaluation is run in a separate process.
New independently collected data are required for a final G01 claim.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.metrics import metrics
from app.model_registry import (CSV_FIELDS, RegistryError, atomic_write, canonical_sha,
                                code_hashes, json_bytes, load_candidate, require, sha256)
from app.network import build_model, denormalize_target, normalize_for_model
from app.training_data import BoneAgeDataset, load_data_reference


def reference_summary(reference):
    result = {key: reference[key] for key in ("manifest_sha256", "split_sha256", "audit_sha256", "source_sha256")}
    result.update({"split_counts": reference["audit"]["splits"],
                   "patient_isolation": reference["audit"].get("patient_isolation", "UNVERIFIED"),
                   "grouping": reference["audit"].get("grouping", "unverified")})
    if reference["audit"].get("cache_sha256"):
        result["cache_sha256"] = reference["audit"]["cache_sha256"]
    return result


def freeze_protocol(meta, metadata_path, reference, split, bootstrap_samples, bootstrap_seed):
    rows = reference["records"]
    role = "exploratory_previously_seen_test" if split == "test" else "model_selection_validation"
    return {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": meta["model_id"], "architecture": meta["architecture"],
        "checkpoint_sha256": meta["checkpoint_sha256"], "metadata_sha256": sha256(metadata_path),
        "preprocess_version": meta["preprocess_version"], "image_size": meta["image_size"],
        "code_sha256": code_hashes(), "data_reference": reference_summary(reference),
        "split": split, "evaluation_role": role,
        "independent_final_test": False, "patient_independence_verified": False,
        "test_previously_seen": split == "test", "metrics_use_unclipped_predictions": True,
        "target_mae_months": 6,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed,
                      "confidence": 0.95, "method": "percentile", "unit": "image"},
        "split_membership": [[row["source_id"], row["split"]] for row in rows],
        "expected_samples": [{"source_image_id": row["source_id"], "reference_months": row["boneage"],
                              "sex": row["sex"], "image_sha256": row["sha256"]}
                             for row in rows if row["split"] == split],
        "limitations": ["Previously inspected public research data; no new independent final cohort",
                        "Patient identifiers are absent; patient independence remains unverified",
                        "Bootstrap intervals resample images, not verified independent patients",
                        "Labels are public RSNA research labels, not an adjudicated clinical standard"],
    }


def evaluate_candidate(candidate, data, split, output, *, acknowledge_exploratory_test=False,
                       bootstrap_samples=1000, bootstrap_seed=20261023, batch_size=32,
                       workers=0, threads=2, device="cpu"):
    require(split in ("validation", "test"), "Only validation/test evaluation is supported")
    require(split != "test" or acknowledge_exploratory_test,
            "Previously inspected test data require --acknowledge-exploratory-test")
    require(type(bootstrap_samples) is int and 1 <= bootstrap_samples <= 100000, "Invalid bootstrap sample count")
    require(type(bootstrap_seed) is int, "Invalid bootstrap seed")
    require(batch_size > 0 and workers >= 0 and threads > 0, "Invalid evaluation resource settings")
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    require(output != candidate and candidate not in output.parents,
            "Evaluation output must be outside the candidate directory")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Evaluation output must be a new or empty directory; previous evidence is never overwritten")
    meta, metadata_path, checkpoint = load_candidate(candidate)
    reference = load_data_reference(data)
    summary = reference_summary(reference)
    for key in ("manifest_sha256", "split_sha256", "audit_sha256", "source_sha256", "split_counts"):
        require(summary[key] == meta["data_reference"].get(key), f"Candidate and evaluation data differ: {key}")
    dataset = BoneAgeDataset(data, split, meta["image_size"], reference)
    require(len(dataset) == summary["split_counts"][split] > 0, "Incomplete evaluation dataset")
    protocol = freeze_protocol(meta, metadata_path, reference, split, bootstrap_samples, bootstrap_seed)
    require(canonical_sha(protocol["split_membership"]) == summary["split_sha256"], "Fixed split hash mismatch")
    require(dataset.ids == [row["source_image_id"] for row in protocol["expected_samples"]], "Dataset cohort order changed")
    output.mkdir(parents=True, exist_ok=True)
    # Freeze the complete protocol before opening weights or computing outputs.
    protocol_path = output / "frozen-protocol.json"
    atomic_write(protocol_path, json_bytes(protocol))
    protocol_sha = sha256(protocol_path)
    torch.set_num_threads(threads)
    torch.manual_seed(bootstrap_seed)
    if device == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    else:
        require(device == "cpu", "Device must be cpu or cuda")
    model = build_model(meta["architecture"], weights_path=checkpoint).to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    predictions, labels, sexes, identifiers = [], [], [], []
    started = time.monotonic()
    with torch.inference_mode():
        for images, batch_sex, targets, ids in loader:
            inputs = normalize_for_model(images.to(device), meta["preprocess_version"])
            values = denormalize_target(model(inputs, batch_sex.to(device))).detach().cpu().reshape(-1)
            require(len(values) == len(ids), "Model output sample count mismatch")
            predictions.extend(float(value) for value in values.tolist())
            labels.extend(float(value) for value in targets.reshape(-1).tolist())
            sexes.extend("M" if float(value) == 1 else "F" for value in batch_sex.reshape(-1).tolist())
            identifiers.extend(str(value) for value in ids)
    require(identifiers == dataset.ids, "Evaluation omitted/reordered samples")
    expected = protocol["expected_samples"]
    require(labels == [float(row["reference_months"]) for row in expected] and
            sexes == [row["sex"] for row in expected], "Cached labels or sex differ from the frozen cohort")
    results = metrics(predictions, labels, sexes, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
    prediction_path = output / "predictions.csv"
    with prediction_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for row, prediction, truth, sex in zip(expected, predictions, labels, sexes):
            writer.writerow([row["source_image_id"], truth, prediction, sex, split, row["image_sha256"]])
    require(sha256(metadata_path) == protocol["metadata_sha256"] and
            sha256(checkpoint) == protocol["checkpoint_sha256"] and
            code_hashes() == protocol["code_sha256"] and sha256(protocol_path) == protocol_sha,
            "Candidate, protocol or evaluation code changed during evaluation")
    final_reference = reference_summary(load_data_reference(data))
    require(final_reference == summary, "Audited dataset changed during evaluation")
    complete = results["complete"] and results["unknown_sex_n"] == 0 and len(predictions) == len(expected)
    report = {key: protocol[key] for key in ("schema_version", "model_id", "architecture", "checkpoint_sha256",
              "metadata_sha256", "preprocess_version", "image_size", "code_sha256", "data_reference", "split",
              "evaluation_role", "independent_final_test", "patient_independence_verified", "test_previously_seen",
              "metrics_use_unclipped_predictions", "target_mae_months", "limitations")}
    report.update({
        "created_at": datetime.now(timezone.utc).isoformat(), "kind": "candidate_evaluation",
        "status": "evaluated" if complete else "invalid", "n_expected": len(expected),
        "n_evaluated": len(predictions), "finite_predictions": results["finite_prediction_n"],
        "metrics": results,
        "mae_target_met": bool(complete and results["mae_months"] < 6), "target_passed": False,
        "gates": {"G01": {"passed": False, "reason": "No new independent final data; public test was previously inspected and patient isolation is unverified"}},
        "patient_isolation": summary["patient_isolation"],
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "device": device,
                    "threads": threads, "batch_size": batch_size, "workers": workers},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "artifacts": {"predictions.csv": {"sha256": sha256(prediction_path), "rows": len(predictions)},
                      "frozen-protocol.json": {"sha256": protocol_sha}},
    })
    atomic_write(output / "evaluation.json", json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-exploratory-test", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261023)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        report = evaluate_candidate(args.candidate, args.data, args.split, args.output,
                                    acknowledge_exploratory_test=args.acknowledge_exploratory_test,
                                    bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed,
                                    batch_size=args.batch_size, workers=args.workers, threads=args.threads, device=args.device)
    except (RegistryError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation refused: {exc}\n")
    print(json.dumps({"model_id": report["model_id"], "status": report["status"], "split": report["split"],
                      "evaluation_role": report["evaluation_role"], "n": report["n_evaluated"],
                      "metrics": report["metrics"], "mae_target_met": report["mae_target_met"],
                      "target_passed": False, "evaluation_sha256": sha256(args.output / "evaluation.json"),
                      "artifacts": report["artifacts"]}, ensure_ascii=False, indent=2, allow_nan=False))
    if report["status"] != "evaluated":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
