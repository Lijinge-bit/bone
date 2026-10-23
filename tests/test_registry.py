"""Integrity and publication tests use synthetic evidence, never real storage."""
import csv
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.metrics import metrics
from app.model_registry import (CSV_FIELDS, INTEGRATION_CODE, ROOT, RegistryError,
                                canonical_sha, code_hashes, json_bytes, load_json,
                                register_candidate, rollback_model, sha256)


def make_evidence(base, model_id="candidate-a", predictions=(100.0, 100.0, 100.0)):
    base = Path(base)
    candidate, evaluation = base / model_id / "export", base / model_id / "evaluation"
    candidate.mkdir(parents=True)
    evaluation.mkdir()
    membership = [["tr0", "train"], ["tr1", "train"], ["va0", "validation"],
                  ["va1", "validation"], ["va2", "validation"], ["te0", "test"], ["te1", "test"]]
    counts = {"train": 2, "validation": 3, "test": 2}
    source = {"manifest_sha256": "1" * 64, "split_sha256": canonical_sha(membership), "audit_sha256": "2" * 64}
    reference = {**source, "source_sha256": canonical_sha(source), "split_counts": counts,
                 "patient_isolation": "UNVERIFIED", "grouping": "image-only"}
    expected = [{"source_image_id": f"va{i}", "reference_months": i * 20,
                 "sex": "M" if i % 2 else "F", "image_sha256": str(i + 3) * 64} for i in range(3)]
    result = metrics(predictions, [0, 20, 40], ["F", "M", "F"], bootstrap_samples=30, bootstrap_seed=5)
    (candidate / "model.pt").write_bytes(b"synthetic checkpoint " + model_id.encode())
    meta = {"schema_version": 2, "model_id": model_id, "architecture": "boneage-small-cnn-v1", "checkpoint": "model.pt",
            "checkpoint_sha256": sha256(candidate / "model.pt"), "image_size": 96, "preprocess_version": "gray-letterbox-v1",
            "input_channels": 1, "sex_encoding": {"M": 1, "F": 0}, "target_normalization": {"offset": 120.0, "scale": 60.0},
            "explanation_method": "Grad-CAM", "gradcam_layer": "features", "created_at": "2026-10-23T00:00:00+00:00",
            "usage": "research_only", "data_reference": reference,
            "evaluation": {"validation": result, "test": None, "target_passed": False,
                           "training_samples": 2, "validation_samples": 3, "test_samples": None}}
    (candidate / "candidate.json").write_bytes(json_bytes(meta))
    protocol = {"schema_version": 2, "model_id": model_id, "architecture": meta["architecture"],
                "checkpoint_sha256": meta["checkpoint_sha256"], "metadata_sha256": sha256(candidate / "candidate.json"),
                "preprocess_version": meta["preprocess_version"], "image_size": 96, "split": "validation",
                "evaluation_role": "model_selection_validation", "independent_final_test": False,
                "patient_independence_verified": False, "data_reference": reference,
                "code_sha256": code_hashes(), "split_membership": membership, "expected_samples": expected,
                "bootstrap": {"samples": 30, "seed": 5, "confidence": 0.95, "unit": "image", "method": "percentile"}}
    (evaluation / "frozen-protocol.json").write_bytes(json_bytes(protocol))
    with (evaluation / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for row, prediction in zip(expected, predictions):
            writer.writerow([row["source_image_id"], row["reference_months"], prediction,
                             row["sex"], "validation", row["image_sha256"]])
    report = {key: protocol[key] for key in ("schema_version", "model_id", "architecture", "checkpoint_sha256", "metadata_sha256",
              "preprocess_version", "image_size", "split", "evaluation_role", "independent_final_test",
              "patient_independence_verified", "data_reference", "code_sha256")}
    report.update({"status": "evaluated", "metrics": result, "n_expected": 3, "n_evaluated": 3,
                   "target_mae_months": 6, "target_passed": False, "mae_target_met": result["mae_months"] < 6,
                   "gates": {"G01": {"passed": False, "reason": "No independent data"}},
                   "artifacts": {"predictions.csv": {"sha256": sha256(evaluation / "predictions.csv"), "rows": 3},
                                 "frozen-protocol.json": {"sha256": sha256(evaluation / "frozen-protocol.json")}}})
    (evaluation / "evaluation.json").write_bytes(json_bytes(report))
    public = [row for row in load_json(ROOT / "docs/reports/sample-manifest.json")
              if row["file"].startswith("train")]
    samples = []
    for row in public:
        for sex in ("F", "M"):
            samples.append({"sample": row["file"], "sex": sex, "image_sha256": row["sha256"],
                            "heatmap_size": [row["width"], row["height"]], "heatmap_mode": "RGBA",
                            "prediction_months": 100.0, "repeat_difference_months": 0.0,
                            "spatial_mapping_max_alpha_difference": 0, "alpha_max": 120, "alpha_min": 0,
                            "alpha_std": 4.0, "alpha_grid_sha256": "a" * 64, "heatmap_sha256": "b" * 64})
    integration = {"schema_version": 1, "status": "passed", "model_id": model_id,
                   "checkpoint_sha256": meta["checkpoint_sha256"], "metadata_sha256": protocol["metadata_sha256"],
                   "architecture": meta["architecture"], "preprocess_version": meta["preprocess_version"], "image_size": 96,
                   "code_sha256": code_hashes(INTEGRATION_CODE), "runtime": {"device": "cpu"},
                   "sample_manifest_sha256": sha256(ROOT / "docs/reports/sample-manifest.json"),
                   "public_sample_count": 6, "sample_image_sex_pairs": 12, "samples": samples,
                   "checks": {key: True for key in ("finite_outputs", "repeatable_predictions", "repeatable_heatmaps",
                              "rgba_heatmaps", "spatial_heatmaps", "original_image_mapping", "corrupt_checkpoint_rejected",
                              "corrupt_image_rejected", "cpu_inference")}}
    integration_path = base / model_id / "integration-check.json"
    integration_path.write_bytes(json_bytes(integration))
    return candidate, evaluation, integration_path


def publish(paths, storage):
    return register_candidate(*paths, storage, operator="unit tester", reason="Verify integrity", research_only=True)


def publish_process(paths, storage, result):
    try:
        active = publish(paths, storage)
        result.put(("ok", active["model_id"]))
    except Exception as exc:
        result.put(("error", str(exc)))


class RegistryTests(unittest.TestCase):
    def test_explicit_research_registration_preserves_failure_and_immutable_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base)
            original = (paths[0] / "candidate.json").read_bytes()
            active = publish(paths, base / "var")
            self.assertFalse(active["evaluation"]["target_passed"])
            self.assertFalse(active["evaluation"]["mae_target_met"])
            self.assertEqual(active["evaluation"]["registration_status"], "research_only_target_not_met")
            self.assertEqual(active["checkpoint"], "candidate-a.pt")
            self.assertEqual((paths[0] / "candidate.json").read_bytes(), original)
            version = base / "var/models/versions/candidate-a"
            for name in ("candidate.json", "model.pt", "evaluation.json", "predictions.csv", "frozen-protocol.json", "integration-check.json"):
                self.assertTrue((version / name).is_file())
            self.assertEqual(sha256(version / "model.pt"), active["checkpoint_sha256"])
            self.assertEqual(publish(paths, base / "var")["model_id"], "candidate-a")

    def test_legacy_model_is_snapshotted_and_can_be_rolled_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            models = base / "var/models"
            models.mkdir(parents=True)
            (models / "old.pt").write_bytes(b"old known checkpoint")
            old = {"model_id": "old", "checkpoint": "old.pt", "checkpoint_sha256": sha256(models / "old.pt"),
                   "evaluation": {"test": {"n": 10, "mae_months": 5.5}, "target_passed": True}}
            (models / "active.json").write_bytes(json_bytes(old))
            (models / "evaluation.json").write_bytes(json_bytes({"target_passed": True, "test": old["evaluation"]["test"]}))
            publish(make_evidence(base), base / "var")
            self.assertEqual(load_json(models / "versions/old/legacy-active.json"), old)
            restored = rollback_model("old", base / "var", operator="tester", reason="Restore previous research model")
            self.assertEqual(restored, old)
            self.assertEqual(load_json(models / "active.json"), old)
            events = [json.loads(line) for line in (models / "registry-audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["operation"] == "rollback" for event in events))

    def test_tampered_csv_and_missing_cases_refuse_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base)
            csv_path = paths[1] / "predictions.csv"
            csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("va1", "missing-id"), encoding="utf-8")
            with self.assertRaisesRegex(RegistryError, "checksum"):
                publish(paths, base / "var")
            report = load_json(paths[1] / "evaluation.json")
            report["artifacts"]["predictions.csv"]["sha256"] = sha256(csv_path)
            (paths[1] / "evaluation.json").write_bytes(json_bytes(report))
            with self.assertRaisesRegex(RegistryError, "identity"):
                publish(paths, base / "var")
            self.assertFalse((base / "var/models/active.json").exists())

    def test_nan_prediction_and_unverified_usage_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base, predictions=(0, float("nan"), 40))
            with self.assertRaisesRegex(RegistryError, "Non-finite"):
                publish(paths, base / "var")
            with self.assertRaisesRegex(RegistryError, "Explicit"):
                register_candidate(*paths, base / "var", operator="tester", reason="no explicit research usage")

    def test_mismatched_integration_and_false_g01_claim_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base)
            report = load_json(paths[2]); report["checkpoint_sha256"] = "0" * 64
            paths[2].write_bytes(json_bytes(report))
            with self.assertRaisesRegex(RegistryError, "Integration checkpoint"):
                publish(paths, base / "var")
            evaluation = load_json(paths[1] / "evaluation.json"); evaluation["target_passed"] = True
            (paths[1] / "evaluation.json").write_bytes(json_bytes(evaluation))
            with self.assertRaisesRegex(RegistryError, "G01"):
                publish(paths, base / "var")

    def test_concurrent_processes_publish_complete_versions_and_keep_lock_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first, second = make_evidence(base, "candidate-a"), make_evidence(base, "candidate-b")
            context = multiprocessing.get_context("spawn")
            result = context.Queue()
            processes = [context.Process(target=publish_process, args=(paths, base / "var", result)) for paths in (first, second)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            answers = [result.get(timeout=2) for _ in processes]
            self.assertEqual([answer[0] for answer in answers], ["ok", "ok"], answers)
            models = base / "var/models"
            active = load_json(models / "active.json")
            self.assertIn(active["model_id"], ("candidate-a", "candidate-b"))
            self.assertEqual(sha256(models / active["checkpoint"]), active["checkpoint_sha256"])
            self.assertTrue((models / ".registry.lock").is_file())
            self.assertTrue((models / "versions/candidate-a/version-manifest.json").is_file())
            self.assertTrue((models / "versions/candidate-b/version-manifest.json").is_file())

    def test_metadata_or_immutable_version_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base)
            active = publish(paths, base / "var")
            version = base / "var/models/versions" / active["model_id"]
            (version / "model.pt").chmod(0o644)
            (version / "model.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RegistryError, "corrupt"):
                rollback_model(active["model_id"], base / "var", operator="tester", reason="Refuse corrupt snapshot")

    def test_rollback_recovers_corrupt_flat_checkpoint_from_verified_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            active = publish(make_evidence(base), base / "var")
            models = base / "var/models"
            checkpoint = models / active["checkpoint"]
            checkpoint.write_bytes(b"bad live checkpoint")
            corrupt_sha = sha256(checkpoint)
            restored = rollback_model(active["model_id"], base / "var", operator="tester", reason="Recover corrupt live weight")
            self.assertEqual(sha256(checkpoint), restored["checkpoint_sha256"])
            events = [json.loads(line) for line in (models / "registry-audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event.get("phase") == "committed" and event.get("operation") == "rollback" and
                                event.get("previous_actual_checkpoint_sha256") == corrupt_sha for event in events))

    def test_missing_protocol_fields_or_forged_independence_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base)
            protocol_path = paths[1] / "frozen-protocol.json"
            protocol = load_json(protocol_path)
            protocol["independent_final_test"] = True
            protocol_path.write_bytes(json_bytes(protocol))
            evaluation = load_json(paths[1] / "evaluation.json")
            evaluation["artifacts"]["frozen-protocol.json"]["sha256"] = sha256(protocol_path)
            (paths[1] / "evaluation.json").write_bytes(json_bytes(evaluation))
            with self.assertRaisesRegex(RegistryError, "independent final"):
                publish(paths, base / "var")

    def test_test_evaluation_requires_acknowledgment_before_reading_data(self):
        from scripts.evaluate_model import evaluate_candidate
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            with self.assertRaisesRegex(RegistryError, "acknowledge-exploratory-test"):
                evaluate_candidate("missing-candidate", "missing-data", "test", output)
            self.assertFalse(output.exists())

    def test_evidence_changed_after_validation_cannot_be_published(self):
        import app.model_registry as registry
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base)
            write_version = registry._write_version

            def change_before_copy(*args, **kwargs):
                with (paths[1] / "predictions.csv").open("a", encoding="utf-8") as stream:
                    stream.write("tampered after validation\n")
                return write_version(*args, **kwargs)

            with patch.object(registry, "_write_version", side_effect=change_before_copy):
                with self.assertRaisesRegex(RegistryError, "changed while copying"):
                    publish(paths, base / "var")
            self.assertFalse((base / "var/models/active.json").exists())

    def test_version_manifest_cannot_drop_required_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            active = publish(make_evidence(base), base / "var")
            manifest_path = base / "var/models/versions" / active["model_id"] / "version-manifest.json"
            manifest = load_json(manifest_path)
            del manifest["files"]["integration-check.json"]
            manifest_path.chmod(0o644)
            manifest_path.write_bytes(json_bytes(manifest))
            with self.assertRaisesRegex(RegistryError, "complete evaluation"):
                rollback_model(active["model_id"], base / "var", operator="tester", reason="Reject incomplete snapshot")

    def test_audit_intent_failure_prevents_activation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths = make_evidence(base)
            with patch("app.model_registry._audit", side_effect=OSError("audit disk unavailable")):
                with self.assertRaisesRegex(OSError, "audit disk"):
                    publish(paths, base / "var")
            self.assertFalse((base / "var/models/active.json").exists())


if __name__ == "__main__":
    unittest.main()
