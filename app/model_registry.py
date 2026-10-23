"""Evidence-bound, local research-model publication and rollback.

Versions are immutable snapshots; active.json is the sole atomic activation
point. The cross-process lock works on Windows and Unix and is never unlinked.
"""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import uuid

from filelock import FileLock, Timeout

from app.metrics import metrics


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CODE = ("app/imaging.py", "app/network.py", "app/metrics.py",
                   "app/training_data.py", "scripts/evaluate_model.py")
INTEGRATION_CODE = ("scripts/check_candidate.py", "app/predictor.py",
                    "app/network.py", "app/imaging.py")
CSV_FIELDS = ("source_image_id", "reference_months", "predicted_months", "sex", "split", "image_sha256")


class RegistryError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise RegistryError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def load_json(path, *, expected_sha=None):
    def reject_constant(value):
        raise RegistryError(f"Non-finite JSON value: {value}")

    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        raw = Path(path).read_bytes()
        if expected_sha is not None:
            require(hashlib.sha256(raw).hexdigest() == expected_sha, f"JSON evidence changed: {path}")
        value = json.loads(raw.decode("utf-8"),
                           parse_constant=reject_constant, object_pairs_hook=pairs)
        # JSON numbers such as 1e999 also must not slip through as infinity.
        json.dumps(value, allow_nan=False)
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise RegistryError(f"Invalid JSON evidence {path}: {exc}") from exc


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sync_directory(path):
    # Windows does not expose a portable directory-fsync descriptor. File data
    # is flushed before os.replace; on Unix also persist the directory entry.
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def code_hashes(names=EVALUATION_CODE):
    return {name: sha256(ROOT / name) for name in names}


def safe_model_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value)
            and ".." not in value, "Invalid model_id")
    return value


def evidence_file(directory, name):
    directory, name = Path(directory).resolve(), str(name)
    require(Path(name).name == name and "/" not in name and "\\" not in name,
            "Evidence filename must be local to its directory")
    path = directory / name
    require(path.is_file() and not path.is_symlink() and path.resolve().parent == directory,
            f"Missing or unsafe evidence file: {path}")
    return path


def load_candidate(candidate, *, expected_metadata_sha=None):
    from app.network import preprocess_version_for

    candidate = Path(candidate).resolve()
    metadata_path = evidence_file(candidate, "candidate.json")
    meta = load_json(metadata_path, expected_sha=expected_metadata_sha)
    require(isinstance(meta, dict) and meta.get("schema_version") == 2, "Candidate requires schema_version 2")
    safe_model_id(meta.get("model_id"))
    required = {"architecture", "checkpoint", "checkpoint_sha256", "image_size", "preprocess_version",
                "input_channels", "sex_encoding", "target_normalization", "explanation_method",
                "gradcam_layer", "data_reference", "evaluation", "usage", "created_at"}
    require(required <= meta.keys(), "Candidate metadata is incomplete")
    require(meta["usage"] == "research_only", "Candidate usage must be research_only")
    require(meta["checkpoint"] == "model.pt", "Candidate checkpoint must be model.pt")
    require(type(meta["image_size"]) is int and 64 <= meta["image_size"] <= 512,
            "Invalid candidate image_size")
    try:
        expected_preprocess = preprocess_version_for(meta["architecture"])
    except (ValueError, KeyError) as exc:
        raise RegistryError("Unknown candidate architecture") from exc
    require(meta["preprocess_version"] == expected_preprocess, "Candidate preprocessing/code mismatch")
    expected_channels = 1 if meta["architecture"] == "boneage-small-cnn-v1" else 3
    require(meta["input_channels"] == expected_channels, "Candidate channel metadata mismatch")
    require(meta["sex_encoding"] == {"M": 1, "F": 0}, "Unsupported sex encoding")
    require(meta["target_normalization"] == {"offset": 120.0, "scale": 60.0}, "Unsupported target normalization")
    expected_layer = "features" if expected_channels == 1 else "features.layer4"
    require(meta["explanation_method"] == "Grad-CAM" and meta["gradcam_layer"] == expected_layer,
            "Candidate explanation metadata mismatch")
    reference = meta["data_reference"]
    require(isinstance(reference, dict) and all(is_digest(reference.get(key)) for key in
            ("manifest_sha256", "split_sha256", "audit_sha256", "source_sha256")),
            "Candidate requires complete audited data hashes")
    require(reference["source_sha256"] == canonical_sha({key: reference[key] for key in
            ("audit_sha256", "manifest_sha256", "split_sha256")}), "Candidate source data hash is inconsistent")
    counts = reference.get("split_counts")
    require(isinstance(counts, dict) and all(type(counts.get(key)) is int and counts[key] > 0
            for key in ("train", "validation", "test")), "Candidate split counts are missing")
    evaluation = meta["evaluation"]
    require(isinstance(evaluation, dict) and isinstance(evaluation.get("validation"), dict)
            and evaluation.get("test") is None and evaluation.get("target_passed") is False,
            "Candidate must retain validation and an unevaluated test; target_passed must be false")
    require(evaluation.get("validation_samples") == counts["validation"] and
            evaluation.get("training_samples") == counts["train"] and
            evaluation["validation"].get("n") == counts["validation"], "Candidate sample counts mismatch")
    checkpoint = evidence_file(candidate, meta["checkpoint"])
    require(is_digest(meta["checkpoint_sha256"]) and sha256(checkpoint) == meta["checkpoint_sha256"],
            "Candidate checkpoint checksum mismatch")
    return meta, metadata_path, checkpoint


def is_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def verify_integration(path, meta, metadata_sha, *, expected_report_sha=None):
    report = load_json(path, expected_sha=expected_report_sha)
    require(isinstance(report, dict) and report.get("schema_version") == 1 and report.get("status") == "passed",
            "Integration check did not pass")
    for key, expected in (("model_id", meta["model_id"]), ("checkpoint_sha256", meta["checkpoint_sha256"]),
                          ("metadata_sha256", metadata_sha), ("architecture", meta["architecture"]),
                          ("preprocess_version", meta["preprocess_version"]), ("image_size", meta["image_size"])):
        require(report.get(key) == expected, f"Integration {key} mismatch")
    require(report.get("code_sha256") == code_hashes(INTEGRATION_CODE), "Integration preprocessing/code hashes changed")
    require(report.get("runtime", {}).get("device") == "cpu", "CPU integration evidence is required")
    checks = ("finite_outputs", "repeatable_predictions", "repeatable_heatmaps", "rgba_heatmaps",
              "spatial_heatmaps", "original_image_mapping", "corrupt_checkpoint_rejected",
              "corrupt_image_rejected", "cpu_inference")
    require(all(report.get("checks", {}).get(key) is True for key in checks), "Incomplete integration checks")
    manifest_path = ROOT / "docs/reports/sample-manifest.json"
    require(report.get("sample_manifest_sha256") == sha256(manifest_path), "Public sample manifest mismatch")
    from scripts.check_candidate import PUBLIC_IDS
    expected = {row["file"]: row for row in load_json(manifest_path)
                if row.get("file", "").startswith("train") and row["file"][5:-4] in PUBLIC_IDS}
    samples = report.get("samples", [])
    require(isinstance(samples, list) and all(isinstance(row, dict) for row in samples),
            "Malformed integration sample evidence")
    require(report.get("public_sample_count") == len(expected) == 6 and
            report.get("sample_image_sex_pairs") == len(samples) == 12, "Six public images and both sexes are required")
    seen = set()
    for row in samples:
        name, sex = row.get("sample"), row.get("sex")
        require(name in expected and sex in ("F", "M") and (name, sex) not in seen,
                "Unexpected or duplicate integration sample")
        seen.add((name, sex))
        source = expected[name]
        require(row.get("image_sha256") == source["sha256"] and
                row.get("heatmap_size") == [source["width"], source["height"]] and
                row.get("heatmap_mode") == "RGBA", "Integration image/heatmap evidence mismatch")
        value = row.get("prediction_months")
        require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 360,
                "Non-finite or out-of-range integration prediction")
        require(0 <= row.get("repeat_difference_months", math.inf) <= 1e-6 and
                0 <= row.get("spatial_mapping_max_alpha_difference", math.inf) <= 1 and
                0 < row.get("alpha_max", 0) <= 160 and row.get("alpha_std", 0) > 0 and
                is_digest(row.get("alpha_grid_sha256")) and is_digest(row.get("heatmap_sha256")),
                "Invalid repeatability or spatial heatmap evidence")
    return report


def validate_evidence(candidate, evaluation_directory, integration_path):
    # Bind the bytes parsed below, rather than re-hashing live files only when
    # copying them after validation. The registry lock does not lock candidates.
    initial_metadata = evidence_file(Path(candidate).resolve(), "candidate.json")
    metadata_sha = sha256(initial_metadata)
    meta, metadata_path, checkpoint = load_candidate(candidate, expected_metadata_sha=metadata_sha)
    require(sha256(metadata_path) == metadata_sha, "Candidate metadata changed during validation")
    evaluation_directory = Path(evaluation_directory).resolve()
    evaluation_path = evidence_file(evaluation_directory, "evaluation.json")
    prediction_path = evidence_file(evaluation_directory, "predictions.csv")
    protocol_path = evidence_file(evaluation_directory, "frozen-protocol.json")
    fixed_hashes = {"candidate.json": metadata_sha, "model.pt": meta["checkpoint_sha256"],
                    "evaluation.json": sha256(evaluation_path), "predictions.csv": sha256(prediction_path),
                    "frozen-protocol.json": sha256(protocol_path)}
    report = load_json(evaluation_path, expected_sha=fixed_hashes["evaluation.json"])
    protocol = load_json(protocol_path, expected_sha=fixed_hashes["frozen-protocol.json"])
    require(isinstance(report, dict) and isinstance(protocol, dict) and
            report.get("schema_version") == 2 and report.get("status") == "evaluated", "Evaluation is incomplete or invalid")
    split = report.get("split")
    require(split in ("validation", "test"), "Invalid evaluation split")
    for document in (report, protocol):
        require(document.get("schema_version") == 2, "Invalid frozen evaluation protocol")
        for key, expected in (("model_id", meta["model_id"]), ("checkpoint_sha256", meta["checkpoint_sha256"]),
                              ("metadata_sha256", metadata_sha), ("architecture", meta["architecture"]),
                              ("preprocess_version", meta["preprocess_version"]), ("image_size", meta["image_size"]),
                              ("split", split)):
            require(document.get(key) == expected, f"Evaluation {key} mismatch")
        require(document.get("code_sha256") == code_hashes(), "Evaluation preprocessing/code hashes changed")
        reference = document.get("data_reference", {})
        for key in ("manifest_sha256", "split_sha256", "audit_sha256", "source_sha256", "split_counts"):
            require(reference.get(key) == meta["data_reference"].get(key), f"Evaluation data {key} mismatch")
        require(document.get("independent_final_test") is False and document.get("patient_independence_verified") is False,
                "Public historical data cannot establish independent final clinical validation")
        role = "exploratory_previously_seen_test" if split == "test" else "model_selection_validation"
        require(document.get("evaluation_role") == role, "Evaluation role must disclose reused data")
    require(report.get("target_passed") is False and report.get("gates", {}).get("G01", {}).get("passed") is False,
            "G01 cannot pass without new independent data")
    require(report.get("target_mae_months") == 6, "Unexpected MAE target")
    hashes = report.get("artifacts", {})
    for name, path in (("predictions.csv", prediction_path), ("frozen-protocol.json", protocol_path)):
        require(hashes.get(name, {}).get("sha256") == sha256(path), f"Evaluation artifact checksum mismatch: {name}")
    membership = protocol.get("split_membership")
    require(isinstance(membership, list) and canonical_sha(membership) == meta["data_reference"]["split_sha256"],
            "Frozen full split membership does not match training")
    require(all(isinstance(row, list) and len(row) == 2 and row[1] in ("train", "validation", "test") for row in membership),
            "Invalid full split membership")
    require(len({str(row[0]) for row in membership}) == len(membership), "Duplicate split source IDs")
    counts = {key: sum(row[1] == key for row in membership) for key in ("train", "validation", "test")}
    require(counts == meta["data_reference"]["split_counts"], "Frozen split counts mismatch")
    expected = protocol.get("expected_samples")
    require(isinstance(expected, list) and all(isinstance(row, dict) for row in expected)
            and len(expected) == counts[split], "Frozen cohort is incomplete")
    require([str(row.get("source_image_id")) for row in expected] == [str(row[0]) for row in membership if row[1] == split],
            "Frozen cohort identities/order mismatch")
    prediction_bytes = prediction_path.read_bytes()
    require(hashlib.sha256(prediction_bytes).hexdigest() == fixed_hashes["predictions.csv"],
            "Prediction CSV changed during validation")
    with io.StringIO(prediction_bytes.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(CSV_FIELDS), "Prediction CSV columns mismatch")
        rows = list(reader)
    require(len(rows) == len(expected) == report.get("n_expected") == report.get("n_evaluated") ==
            hashes["predictions.csv"].get("rows"), "Evaluation omitted cohort samples")
    predictions, labels, sexes = [], [], []
    for row, frozen in zip(rows, expected):
        require(set(row) == set(CSV_FIELDS) and None not in row.values(), "Malformed prediction CSV row")
        require(row["source_image_id"] == str(frozen.get("source_image_id")) and row["split"] == split and
                row["sex"] == frozen.get("sex") and row["sex"] in ("M", "F") and
                row["image_sha256"] == frozen.get("image_sha256") and is_digest(row["image_sha256"]),
                "Prediction cohort identity/sex/image mismatch")
        try:
            prediction, label = float(row["predicted_months"]), float(row["reference_months"])
        except (ValueError, TypeError) as exc:
            raise RegistryError("Invalid numeric prediction CSV field") from exc
        require(math.isfinite(prediction) and math.isfinite(label), "Non-finite prediction or reference refuses registration")
        require(label == frozen.get("reference_months"), "Prediction reference label mismatch")
        predictions.append(prediction); labels.append(label); sexes.append(row["sex"])
    bootstrap = protocol.get("bootstrap", {})
    require(type(bootstrap.get("samples")) is int and 1 <= bootstrap["samples"] <= 100000 and
            type(bootstrap.get("seed")) is int, "Invalid bootstrap protocol")
    require(bootstrap.get("confidence") == 0.95 and bootstrap.get("unit") == "image" and
            bootstrap.get("method") == "percentile", "Unsupported bootstrap confidence or resampling method")
    calculated = metrics(predictions, labels, sexes, bootstrap_samples=bootstrap["samples"],
                         bootstrap_seed=bootstrap["seed"])
    require(report.get("metrics") == calculated and calculated["complete"] and calculated["unknown_sex_n"] == 0,
            "Evaluation metrics do not match the complete raw predictions")
    require(report.get("mae_target_met") is (calculated["mae_months"] < 6), "Misstated numeric MAE target result")
    integration_path = Path(integration_path)
    require(integration_path.is_file() and not integration_path.is_symlink(), "Missing integration evidence")
    integration_path = integration_path.resolve()
    fixed_hashes["integration-check.json"] = sha256(integration_path)
    verify_integration(integration_path, meta, metadata_sha,
                       expected_report_sha=fixed_hashes["integration-check.json"])
    validated_paths = {"candidate.json": metadata_path, "model.pt": checkpoint,
                       "evaluation.json": evaluation_path, "predictions.csv": prediction_path,
                       "frozen-protocol.json": protocol_path, "integration-check.json": integration_path}
    require(all(sha256(path) == fixed_hashes[name] for name, path in validated_paths.items()),
            "Candidate or evaluation evidence changed during validation")
    return {"metadata": meta, "metadata_sha256": metadata_sha, "checkpoint": checkpoint,
            "candidate": metadata_path, "evaluation": evaluation_path, "predictions": prediction_path,
            "protocol": protocol_path, "integration": integration_path, "report": report,
            "file_sha256": fixed_hashes}


@contextmanager
def registry_lock(models):
    models.mkdir(parents=True, exist_ok=True)
    path = models / ".registry.lock"
    require(not path.is_symlink(), "Registry lock must not be a symbolic link")
    lock = FileLock(str(path), timeout=60)
    try:
        lock.acquire()
        require(path.is_file() and not path.is_symlink(), "Registry lock must be a regular file")
        yield
    except Timeout as exc:
        raise RegistryError("Another model operation holds the registry lock; retry later") from exc
    finally:
        lock.release()


def _read_active(models, *, verify_checkpoint=True):
    path = models / "active.json"
    if not path.exists():
        return None
    require(not path.is_symlink(), "Unsafe active metadata")
    meta = load_json(path)
    require(isinstance(meta, dict), "Current active metadata must be an object")
    safe_model_id(meta.get("model_id"))
    checkpoint_name = meta.get("checkpoint", "")
    require(isinstance(checkpoint_name, str) and checkpoint_name not in ("", ".", "..") and
            Path(checkpoint_name).name == checkpoint_name and "/" not in checkpoint_name and "\\" not in checkpoint_name,
            "Current active checkpoint must be a flat filename")
    require(is_digest(meta.get("checkpoint_sha256")), "Current active checkpoint digest is invalid")
    checkpoint = models / checkpoint_name
    require(not checkpoint.is_symlink(), "Unsafe active checkpoint")
    if verify_checkpoint:
        checkpoint = evidence_file(models, checkpoint_name)
        require(sha256(checkpoint) == meta["checkpoint_sha256"],
                "Current active checkpoint is corrupt; refusing to overwrite evidence")
    return meta


def _verify_version(version):
    version = Path(version)
    require(version.is_dir() and not version.is_symlink() and not version.parent.is_symlink(),
            "Unsafe immutable version directory")
    manifest = load_json(evidence_file(version, "version-manifest.json"))
    require(manifest.get("schema_version") == 1 and manifest.get("model_id") == version.name and
            isinstance(manifest.get("files"), dict), "Invalid immutable version manifest")
    safe_model_id(version.name)
    common = {"activation.json", "registration.json"}
    candidate_files = common | {"candidate.json", "model.pt", "evaluation.json", "predictions.csv",
                                "frozen-protocol.json", "integration-check.json"}
    kind = manifest.get("kind")
    files = set(manifest["files"])
    require(kind in ("candidate_registration", "legacy_snapshot") and common <= files,
            "Immutable version is missing its activation or registration evidence")
    if kind == "candidate_registration":
        require(files == candidate_files, "Immutable candidate version is missing complete evaluation evidence")
    else:
        require("legacy-active.json" in files, "Immutable legacy version is missing its original metadata")
    for name, expected in manifest["files"].items():
        require(is_digest(expected) and sha256(evidence_file(version, name)) == expected,
                f"Immutable version artifact is corrupt: {name}")
    active = load_json(version / "activation.json")
    require(isinstance(active, dict) and active.get("model_id") == version.name and
            is_digest(active.get("checkpoint_sha256")), "Immutable activation identity is invalid")
    checkpoint_name = "model.pt" if kind == "candidate_registration" else active.get("checkpoint", "")
    require(checkpoint_name in files and manifest["files"][checkpoint_name] == active["checkpoint_sha256"],
            "Immutable activation checkpoint is missing or mismatched")
    record = load_json(version / "registration.json")
    require(isinstance(record, dict) and isinstance(record.get("operator"), str) and record["operator"].strip()
            and isinstance(record.get("reason"), str) and record["reason"].strip(),
            "Immutable registration audit is incomplete")
    if kind == "legacy_snapshot":
        require(files <= common | {"legacy-active.json", checkpoint_name, "evaluation.json",
                                    "test-predictions.csv", "validation-predictions.csv"} and
                load_json(version / "legacy-active.json") == active and
                record.get("operation") == "legacy_snapshot", "Immutable legacy snapshot is inconsistent")
    else:
        candidate = load_json(version / "candidate.json")
        require(isinstance(candidate, dict) and candidate.get("model_id") == version.name and
                candidate.get("checkpoint") == "model.pt" and
                candidate.get("checkpoint_sha256") == active["checkpoint_sha256"] and
                active.get("candidate_metadata_sha256") == manifest["files"]["candidate.json"] and
                active.get("evaluation_report_sha256") == manifest["files"]["evaluation.json"] and
                record.get("operation") == "register" and record.get("current") == _identity(active),
                "Immutable candidate activation or registration evidence is inconsistent")
    return manifest


def _write_version(models, model_id, sources, activation, record, kind, expected_hashes=None):
    versions = models / "versions"
    versions.mkdir(exist_ok=True)
    require(versions.is_dir() and not versions.is_symlink(), "Unsafe immutable versions directory")
    version = versions / model_id
    expected = dict(expected_hashes) if expected_hashes is not None else {name: sha256(path) for name, path in sources.items()}
    require(set(expected) == set(sources) and all(is_digest(value) for value in expected.values()),
            "Incomplete fixed evidence hashes")
    if version.exists():
        manifest = _verify_version(version)
        require(manifest.get("kind") == kind and all(manifest["files"].get(name) == digest for name, digest in expected.items()),
                "model_id already exists with different immutable evidence")
        return version, load_json(version / "activation.json")
    staging = Path(tempfile.mkdtemp(prefix=".version-", dir=models))
    try:
        for name, source in sources.items():
            shutil.copyfile(source, staging / name)
            require(sha256(staging / name) == expected[name], "Evidence changed while copying the version")
        (staging / "activation.json").write_bytes(json_bytes(activation))
        (staging / "registration.json").write_bytes(json_bytes(record))
        files = {path.name: sha256(path) for path in staging.iterdir()}
        manifest = {"schema_version": 1, "kind": kind, "model_id": model_id, "files": files}
        (staging / "version-manifest.json").write_bytes(json_bytes(manifest))
        for path in staging.iterdir():
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
            path.chmod(0o444)
        sync_directory(staging)
        os.rename(staging, version)
        sync_directory(versions)
        return version, activation
    finally:
        if staging.exists():
            def writable_remove(function, path, error):
                os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
                function(path)
            shutil.rmtree(staging, onexc=writable_remove)


def _snapshot_active(models, active, operator, reason):
    if active is None:
        return
    version = models / "versions" / active["model_id"]
    if version.exists():
        _verify_version(version)
        require(load_json(version / "activation.json") == active,
                "Existing immutable version does not preserve the current active metadata")
        return
    sources = {"legacy-active.json": models / "active.json", active["checkpoint"]: models / active["checkpoint"]}
    fixed_hashes = {"legacy-active.json": sha256(sources["legacy-active.json"]),
                    active["checkpoint"]: active["checkpoint_sha256"]}
    require(load_json(sources["legacy-active.json"], expected_sha=fixed_hashes["legacy-active.json"]) == active and
            sha256(sources[active["checkpoint"]]) == active["checkpoint_sha256"],
            "Active model changed before its legacy snapshot")
    for name in ("evaluation.json", "test-predictions.csv", "validation-predictions.csv"):
        if (models / name).is_file():
            sources[name] = evidence_file(models, name)
            fixed_hashes[name] = sha256(sources[name])
    _write_version(models, active["model_id"], sources, active,
                   {"operation": "legacy_snapshot", "created_at": _now(), "operator": operator,
                    "reason": reason, "original_evaluation_retained": True}, "legacy_snapshot",
                   expected_hashes=fixed_hashes)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identity(meta):
    return None if meta is None else {"model_id": meta["model_id"], "checkpoint_sha256": meta["checkpoint_sha256"]}


def _audit(models, record):
    descriptor = os.open(models / "registry-audit.jsonl",
                         os.O_CREAT | os.O_WRONLY | os.O_APPEND |
                         getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0), 0o600)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Registry audit must be a regular file")
        data = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        while data:
            written = os.write(descriptor, data)
            require(written > 0, "Cannot persist registry audit")
            data = data[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sync_directory(models)


def _activate(models, version, active, *, repair_flat_checkpoint=False):
    manifest = _verify_version(version)
    source_name = "model.pt" if "model.pt" in manifest["files"] else active["checkpoint"]
    source = evidence_file(version, source_name)
    require(sha256(source) == active["checkpoint_sha256"], "Rollback/activation checkpoint mismatch")
    destination = models / active["checkpoint"]
    require(destination.parent.resolve() == models.resolve(), "Activation checkpoint must be a flat filename")
    copy_checkpoint = not destination.exists()
    if destination.exists():
        require(destination.is_file() and not destination.is_symlink(), "Unsafe flat checkpoint destination")
        if sha256(destination) != active["checkpoint_sha256"]:
            require(repair_flat_checkpoint, "Refusing to overwrite a different flat checkpoint")
            copy_checkpoint = True
    if copy_checkpoint:
        temporary = models / (".checkpoint-" + uuid.uuid4().hex)
        try:
            shutil.copyfile(source, temporary)
            require(sha256(temporary) == active["checkpoint_sha256"], "Checkpoint copy failed")
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    for name in ("evaluation.json", "test-predictions.csv", "validation-predictions.csv"):
        if name in manifest["files"]:
            atomic_write(models / name, (version / name).read_bytes())
    if "predictions.csv" in manifest["files"]:
        split = load_json(version / "evaluation.json")["split"]
        atomic_write(models / f"{split}-predictions.csv", (version / "predictions.csv").read_bytes())
    atomic_write(models / "active.json", json_bytes(active))


def register_candidate(candidate, evaluation, integration, storage, *, operator, reason,
                       research_only=False, usage=None):
    require(isinstance(operator, str) and operator.strip() and isinstance(reason, str) and reason.strip(),
            "operator and reason are required for audit")
    require(research_only or usage == "research_only", "Explicit --research-only or --usage research_only is required")
    require(usage in (None, "research_only"), "Only research_only usage is supported without independent validation")
    models = Path(storage).resolve() / "models"
    with registry_lock(models):
        proof = validate_evidence(candidate, evaluation, integration)
        meta, report = proof["metadata"], proof["report"]
        previous = _read_active(models)
        require(previous is None or previous["model_id"] != meta["model_id"] or
                previous["checkpoint_sha256"] == meta["checkpoint_sha256"], "Cannot reuse an active model_id for different weights")
        _snapshot_active(models, previous, operator, reason)
        active = dict(meta)
        active["checkpoint"] = meta["model_id"] + ".pt"
        active["usage"] = "research_only"
        active["candidate_metadata_sha256"] = proof["metadata_sha256"]
        active["evaluation_report_sha256"] = proof["file_sha256"]["evaluation.json"]
        active["registry_version"] = meta["model_id"]
        active["evaluation"] = dict(meta["evaluation"])
        active["evaluation"][report["split"]] = report["metrics"]
        active["evaluation"][report["split"] + "_samples"] = report["n_evaluated"]
        active["evaluation"].update({"target_passed": False, "mae_target_met": report["mae_target_met"],
                                     "evaluation_role": report["evaluation_role"], "independent_final_test": False,
                                     "protocol_sha256": proof["file_sha256"]["frozen-protocol.json"],
                                     "patient_independence_verified": False, "gates": report["gates"],
                                     "registration_status": "research_only_target_met" if report["mae_target_met"] else "research_only_target_not_met"})
        record = {"operation": "register", "created_at": _now(), "operator": operator.strip(), "reason": reason.strip(),
                  "previous": _identity(previous), "current": _identity(active), "usage": "research_only",
                  "mae_target_met": report["mae_target_met"], "target_passed": False,
                  "metadata_sha256": proof["metadata_sha256"],
                  "evaluation_sha256": proof["file_sha256"]["evaluation.json"],
                  "integration_sha256": proof["file_sha256"]["integration-check.json"]}
        sources = {"candidate.json": proof["candidate"], "model.pt": proof["checkpoint"],
                   "evaluation.json": proof["evaluation"], "predictions.csv": proof["predictions"],
                   "frozen-protocol.json": proof["protocol"], "integration-check.json": proof["integration"]}
        version, active = _write_version(models, meta["model_id"], sources, active, record, "candidate_registration",
                                         expected_hashes=proof["file_sha256"])
        operation_id = uuid.uuid4().hex
        _audit(models, {**record, "operation_id": operation_id, "phase": "intent"})
        _activate(models, version, active)
        _audit(models, {**record, "operation_id": operation_id, "phase": "committed"})
        return active


def rollback_model(model_id, storage, *, operator, reason):
    model_id = safe_model_id(model_id)
    require(isinstance(operator, str) and operator.strip() and isinstance(reason, str) and reason.strip(),
            "operator and reason are required for audit")
    models = Path(storage).resolve() / "models"
    with registry_lock(models):
        previous = _read_active(models, verify_checkpoint=False)
        previous_actual_sha = None
        if previous is not None:
            previous_checkpoint = models / previous["checkpoint"]
            if previous_checkpoint.is_file():
                previous_actual_sha = sha256(previous_checkpoint)
            if previous_actual_sha == previous["checkpoint_sha256"]:
                _snapshot_active(models, previous, operator, reason)
            else:
                preserved = models / "versions" / previous["model_id"]
                require(preserved.is_dir() and not preserved.is_symlink(),
                        "Current flat checkpoint is corrupt or missing and has no immutable snapshot; refusing rollback")
                _verify_version(preserved)
                require(load_json(preserved / "activation.json") == previous,
                        "Current corrupt checkpoint metadata does not match its preserved immutable version")
        version = models / "versions" / model_id
        require(version.is_dir() and not version.is_symlink(), "Unknown rollback model_id")
        _verify_version(version)
        active = load_json(version / "activation.json")
        require(active.get("model_id") == model_id, "Rollback version identity mismatch")
        record = {"operation": "rollback", "created_at": _now(), "operator": operator.strip(), "reason": reason.strip(),
                  "previous": _identity(previous), "current": _identity(active), "operation_id": uuid.uuid4().hex,
                  "previous_actual_checkpoint_sha256": previous_actual_sha}
        _audit(models, {**record, "phase": "intent"})
        _activate(models, version, active, repair_flat_checkpoint=True)
        _audit(models, {**record, "phase": "committed"})
        return active
