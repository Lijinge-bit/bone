"""Exercise a candidate on six checksum-verified public images, on CPU.

This is an integration check, not an accuracy evaluation.  New candidates are
not required to reproduce the old CNN's values.  ``--reference`` optionally
enables the immutable historical runtime comparison for that same checkpoint.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import Settings
from app.imaging import ImageValidationError, letterbox, validate_png
from app.network import denormalize_target, normalize_for_model
from app.predictor import ModelUnavailable, Predictor


PUBLIC_IDS = frozenset(("1377", "1378", "3806", "6197", "11929", "14466"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_samples(samples_dir, manifest_path):
    """Resolve the fixed public samples without trusting arbitrary filenames."""
    rows = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    selected = []
    for row in rows:
        name = row.get("file", "")
        if not name.startswith("train") or not name.endswith(".png"):
            continue
        source_id = name[5:-4]
        if source_id not in PUBLIC_IDS:
            continue
        digest = row.get("sha256", "")
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                "Invalid public sample digest")
        choices = (Path(samples_dir) / (digest + ".png"), Path(samples_dir) / name,
                   Path(samples_dir) / (source_id + ".png"))
        path = next((p for p in choices if p.is_file()), None)
        require(path is not None, f"Public sample missing: {name}; supply --samples")
        require(sha256(path) == digest, f"Public sample checksum mismatch: {name}")
        selected.append((row, path))
    require(len(selected) == 6 and {r["file"][5:-4] for r, _ in selected} == PUBLIC_IDS,
            "Manifest must contain exactly the six fixed public samples")
    require(len({r["sha256"] for r, _ in selected}) == 6,
            "Public sample images must be distinct")
    return sorted(selected, key=lambda item: item[0]["file"])


def expected_alpha(model, image, sex, meta):
    """Reconstruct spatial CAM mapping separately from Predictor.assess.

    Checking dimensions alone would miss resizing the letterbox padding into
    the original image.  Recompute the feature CAM and explicitly invert the
    letterbox transform, then compare every alpha pixel.
    """
    array, transform = letterbox(image, meta["image_size"])
    tensor = normalize_for_model(torch.from_numpy(array)[None, None], meta["preprocess_version"])
    sex_tensor = torch.tensor([[float(sex == "M")]], dtype=torch.float32)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        features = model.features(tensor)
        features.retain_grad()
        output = denormalize_target(model.predict_from_features(features, sex_tensor))
        output.backward()
        weights = features.grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1).relu().detach().cpu().numpy()[0]
    require(np.isfinite(cam).all() and float(np.ptp(cam)) >= 1e-8,
            "Feature CAM has no finite spatial variation")
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    heat = Image.fromarray(np.uint8(cam * 255)).resize(
        (meta["image_size"], meta["image_size"]), Image.Resampling.BILINEAR)
    left, top = transform["left"], transform["top"]
    heat = heat.crop((left, top, left + transform["width"], top + transform["height"]))
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    return np.uint8(np.asarray(heat, dtype=np.float32) / 255 * 160), transform


def verify_heatmap(heatmap, image, expected):
    require(heatmap is not None and heatmap.mode == "RGBA" and heatmap.size == image.size,
            "Heatmap must be RGBA in original image coordinates")
    rgba = np.asarray(heatmap)
    require(rgba.dtype == np.uint8 and rgba.shape == (image.height, image.width, 4),
            "Invalid RGBA heatmap array")
    alpha = rgba[:, :, 3]
    require(0 < int(alpha.max()) <= 160 and np.ptp(alpha) > 0,
            "Heatmap alpha is empty, constant, or outside its documented range")
    # A constant colour block with variable alpha is not the documented CAM.
    require(any(np.ptp(rgba[:, :, channel]) > 0 for channel in range(3)),
            "Heatmap RGB has no spatial variation")
    difference = int(np.abs(alpha.astype(np.int16) - expected.astype(np.int16)).max())
    require(difference <= 1, "Heatmap does not map the valid letterbox area onto the original image")
    grid = np.asarray(heatmap.getchannel("A").resize((16, 16), Image.Resampling.BILINEAR))
    require(np.ptp(grid) > 0, "Heatmap has no coarse spatial variation")
    return {"heatmap_size": list(heatmap.size), "heatmap_mode": heatmap.mode,
            "heatmap_sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
            "alpha_min": int(alpha.min()), "alpha_max": int(alpha.max()),
            "alpha_mean": float(alpha.mean()), "alpha_std": float(alpha.std()),
            "alpha_grid_16x16": grid.ravel().tolist(),
            "alpha_grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
            "spatial_mapping_max_alpha_difference": difference}


def check_corruption(candidate, meta, image):
    """A fresh predictor must reject a modified checkpoint before using it."""
    with tempfile.TemporaryDirectory(prefix="bone-candidate-corruption-") as temporary:
        directory = Path(temporary)
        shutil.copyfile(candidate / "candidate.json", directory / "candidate.json")
        shutil.copyfile(candidate / meta["checkpoint"], directory / meta["checkpoint"])
        with (directory / meta["checkpoint"]).open("ab") as stream:
            stream.write(b"\nintentional integration-check corruption\n")
        try:
            Predictor(directory, metadata_filename="candidate.json").assess(
                image, "F", meta["model_id"], meta["checkpoint_sha256"])
        except ModelUnavailable:
            return True
        raise RuntimeError("Predictor accepted a corrupt checkpoint")


def check_image_corruption(path):
    try:
        validate_png(Path(path).read_bytes()[:40])
    except ImageValidationError:
        return True
    raise RuntimeError("Image validation accepted a corrupt public sample")


def check_legacy_reference(samples, meta, reference_path):
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    require(reference["model_id"] == meta["model_id"] and
            reference["checkpoint_sha256"] == meta["checkpoint_sha256"],
            "Historical reference belongs to a different checkpoint")
    expected = {(row["sample"], row["sex"]): row for row in reference["samples"]}
    require(set(expected) == {(row["sample"], row["sex"]) for row in samples},
            "Historical reference sample/sex pairs do not match")
    differences = []
    for row in samples:
        baseline = expected[(row["sample"], row["sex"])]
        require(row["image_sha256"] == baseline["image_sha256"] and
                row["heatmap_size"] == baseline["image_size"],
                "Historical reference image identity or geometry differs")
        difference = abs(row["prediction_months"] - baseline["prediction_months"])
        require(difference <= reference["prediction_tolerance_months"],
                "Historical prediction changed beyond its frozen tolerance")
        grid_delta = np.abs(np.asarray(row["alpha_grid_16x16"], dtype=np.int16) -
                            np.asarray(baseline["alpha_grid_16x16"], dtype=np.int16))
        require(int(grid_delta.max()) <= reference["heatmap_grid_tolerance"],
                "Historical spatial attention changed beyond its frozen tolerance")
        require(abs(row["alpha_mean"] - baseline["alpha_mean"]) <= reference["heatmap_alpha_mean_tolerance"],
                "Historical heatmap opacity changed beyond its frozen tolerance")
        differences.append(difference)
    return {"status": "passed", "reference_sha256": sha256(reference_path),
            "max_prediction_difference_months": max(differences)}


def check_candidate(candidate, samples_dir, manifest_path, reference_path=None):
    candidate = Path(candidate).resolve()
    metadata_path = candidate / "candidate.json"
    metadata_sha = sha256(metadata_path)
    predictor = Predictor(candidate, metadata_filename="candidate.json")
    meta = predictor.metadata()
    require(meta is not None, "Candidate metadata or checkpoint is invalid")
    require(sha256(candidate / meta["checkpoint"]) == meta["checkpoint_sha256"],
            "Candidate checkpoint checksum mismatch")
    selected = public_samples(samples_dir, manifest_path)
    samples = []
    first_image = None
    for row, path in selected:
        image = validate_png(path.read_bytes())
        require(image.size == (row["width"], row["height"]),
                "Public sample geometry does not match its manifest")
        if first_image is None:
            first_image = image.copy()
        for sex in ("F", "M"):
            value, heatmap, error = predictor.assess(
                image, sex, meta["model_id"], meta["checkpoint_sha256"])
            repeated, repeated_heatmap, repeat_error = predictor.assess(
                image, sex, meta["model_id"], meta["checkpoint_sha256"])
            require(error is None and repeat_error is None, f"Heatmap unavailable: {row['file']}/{sex}")
            require(np.isfinite(value) and np.isfinite(repeated) and 0 <= value <= 360,
                    "Candidate produced an invalid integration sample prediction")
            require(abs(value - repeated) <= 1e-6, "Candidate prediction is not repeatable")
            require(predictor.model is not None and
                    all(p.device.type == "cpu" for p in predictor.model.parameters()) and
                    all(b.device.type == "cpu" for b in predictor.model.buffers()),
                    "Candidate inference did not run entirely on CPU")
            alpha, transform = expected_alpha(predictor.model, image, sex, meta)
            spatial = verify_heatmap(heatmap, image, alpha)
            require(repeated_heatmap is not None and np.array_equal(np.asarray(heatmap), np.asarray(repeated_heatmap)),
                    "Candidate spatial heatmap is not repeatable")
            samples.append({"sample": row["file"], "image_sha256": row["sha256"], "sex": sex,
                            "prediction_months": float(value),
                            "repeat_difference_months": float(abs(value - repeated)),
                            "letterbox_transform": transform, **spatial})
    corruption_rejected = check_corruption(candidate, meta, first_image)
    image_corruption_rejected = check_image_corruption(selected[0][1])
    legacy = check_legacy_reference(samples, meta, reference_path) if reference_path else None
    require(sha256(metadata_path) == metadata_sha and
            sha256(candidate / meta["checkpoint"]) == meta["checkpoint_sha256"],
            "Candidate files changed during the integration check")
    code_paths = ("scripts/check_candidate.py", "app/predictor.py", "app/network.py", "app/imaging.py")
    return {"schema_version": 1, "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "Public sample CPU integration check; not an independent accuracy evaluation",
            "model_id": meta["model_id"], "checkpoint_sha256": meta["checkpoint_sha256"],
            "metadata_sha256": metadata_sha, "architecture": meta["architecture"],
            "preprocess_version": meta["preprocess_version"], "image_size": meta["image_size"],
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "device": "cpu", "threads": torch.get_num_threads()},
            "public_sample_count": len(selected), "sample_image_sex_pairs": len(samples),
            "sample_manifest_sha256": sha256(manifest_path),
            "code_sha256": {name: sha256(ROOT / name) for name in code_paths},
            "checks": {"finite_outputs": True, "repeatable_predictions": True,
                       "repeatable_heatmaps": True, "rgba_heatmaps": True, "spatial_heatmaps": True,
                       "original_image_mapping": True, "corrupt_checkpoint_rejected": corruption_rejected,
                       "corrupt_image_rejected": image_corruption_rejected,
                       "cpu_inference": True},
            "samples": samples, "legacy_reference": legacy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--storage", type=Path, default=Settings.from_env().storage)
    parser.add_argument("--samples", type=Path, help="Directory containing verified public PNG files")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/reports/sample-manifest.json")
    parser.add_argument("--reference", type=Path, help="Optional frozen reference for the original checkpoint only")
    parser.add_argument("--output", type=Path, help="Report file or directory (default: candidate/integration-check.json)")
    args = parser.parse_args()
    report = check_candidate(args.candidate, args.samples or args.storage / "images", args.manifest, args.reference)
    output = args.output or args.candidate / "integration-check.json"
    if output.is_dir() or output.suffix.lower() != ".json":
        output = output / "integration-check.json"
    candidate_meta = json.loads((args.candidate / "candidate.json").read_text(encoding="utf-8"))
    require(output.resolve() not in {(args.candidate / "candidate.json").resolve(),
                                     (args.candidate / candidate_meta["checkpoint"]).resolve(),
                                     (args.candidate / "active.json").resolve()},
            "Report output cannot replace candidate metadata, active metadata, or weights")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=".integration-check-", suffix=".json",
                                     dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    print(json.dumps({"status": report["status"], "model_id": report["model_id"],
                      "public_sample_count": report["public_sample_count"],
                      "sample_image_sex_pairs": report["sample_image_sex_pairs"], "output": str(output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
