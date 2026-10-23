"""Audited, fixed-split training data and memory-mapped image caches.

High resolution caches are always made from the original PNG ZIP, never from
the legacy 96 pixel cache. No ZIP handle is held by a Dataset or its workers.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
import time
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.data import Dataset

from app.imaging import PREPROCESS_VERSION, letterbox, validate_png


SPLITS = ("train", "validation", "test")
IMAGE_SIZES = (96, 192, 224, 384, 512)
CACHE_SCHEMA = "boneage-training-cache-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_FILES: set[tuple] = set()


class DataIntegrityError(ValueError):
    """The audited source or a prepared cache no longer matches its hashes."""


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value, description):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise DataIntegrityError(f"Missing or invalid SHA256: {description}")
    return value


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"Cannot read JSON metadata: {path}") from exc


def _archive_member(value):
    # ZIP member names are portable POSIX paths. Never normalize traversal or
    # Windows drive/UNC names into apparently safe names, even without extract().
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DataIntegrityError(f"Unsafe archive_path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or
            any(part in ("", ".", "..") for part in value.split("/")) or
            path.suffix.lower() != ".png"):
        raise DataIntegrityError(f"Unsafe archive_path: {value!r}")
    return value


def load_data_reference(data_dir) -> dict:
    """Validate the source audit and preserve its existing manifest partitions.

    The returned dictionary is JSON serializable. ``split_sha256`` hashes the
    ordered ``[[source_id, split], ...]`` list using canonical UTF-8 JSON.
    Small synthetic audits are supported; counts must always match audit.splits.
    """
    directory = Path(data_dir).expanduser().resolve()
    audit_path, manifest_path = directory / "audit.json", directory / "manifest.jsonl"
    audit = _read_json(audit_path)
    if not isinstance(audit, dict):
        raise DataIntegrityError("audit.json must contain an object")
    expected = _require_sha(audit.get("manifest_sha256"), "audit.manifest_sha256")
    try:
        manifest_sha = _sha256(manifest_path)
    except OSError as exc:
        raise DataIntegrityError(f"Cannot read audited manifest: {manifest_path}") from exc
    if manifest_sha != expected:
        raise DataIntegrityError("Manifest SHA256 mismatch against audit.json")
    if audit.get("preprocess_version", PREPROCESS_VERSION) != PREPROCESS_VERSION:
        raise DataIntegrityError("Unsupported audited preprocessing version")

    records, ids, paths = [], set(), set()
    group_splits = {}
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise DataIntegrityError(f"Manifest line {line_number} is not an object")
                if item.get("kind") != "train" or item.get("status") != "valid":
                    continue
                source_id = item.get("source_id")
                if not isinstance(source_id, str) or not source_id or source_id in ids:
                    raise DataIntegrityError(f"Missing or duplicate source_id at manifest line {line_number}")
                path = _archive_member(item.get("archive_path"))
                if PurePosixPath(path).stem != source_id or path in paths:
                    raise DataIntegrityError(f"Archive path/ID mismatch or duplicate: {source_id}")
                if item.get("split") not in SPLITS:
                    raise DataIntegrityError(f"Missing fixed split for source_id {source_id}")
                _require_sha(item.get("sha256"), f"PNG {source_id}")
                target = item.get("boneage")
                if (isinstance(target, bool) or not isinstance(target, (int, float)) or
                        not math.isfinite(target) or target < 0 or item.get("sex") not in ("M", "F")):
                    raise DataIntegrityError(f"Invalid target/sex for source_id {source_id}")
                for field in ("duplicate_group", "raster_sha256", "dhash"):
                    if item.get(field) is not None:
                        key = (field, str(item[field]))
                        previous = group_splits.setdefault(key, item["split"])
                        if previous != item["split"]:
                            raise DataIntegrityError(f"Duplicate group crosses fixed splits: {source_id}")
                ids.add(source_id)
                paths.add(path)
                records.append(item)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataIntegrityError("Cannot parse audited manifest.jsonl") from exc
    if not records:
        raise DataIntegrityError("Audited manifest contains no valid labeled training records")
    counts = {name: sum(item["split"] == name for item in records) for name in SPLITS}
    if audit.get("splits") != counts:
        raise DataIntegrityError(f"Fixed split counts differ from audit.splits: {counts}")
    if "matched_training" in audit and audit["matched_training"] != len(records):
        raise DataIntegrityError("Valid training record count differs from audit.matched_training")
    # The actual RSNA v2 source has 12,611 labeled images and this fixed split.
    if audit.get("training_rows") == 12611 and counts != {"train": 10091, "validation": 1260, "test": 1260}:
        raise DataIntegrityError("RSNA v2 fixed partitions must remain 10091/1260/1260")
    partition_sha = hashlib.sha256(_json_bytes([[r["source_id"], r["split"]] for r in records])).hexdigest()
    source = {"audit_sha256": _sha256(audit_path), "manifest_sha256": manifest_sha,
              "split_sha256": partition_sha}
    return {"data_dir": str(directory), "audit": audit, "records": records, **source,
            "source_sha256": hashlib.sha256(_json_bytes(source)).hexdigest()}


def _reference(data_dir, supplied=None):
    verified = load_data_reference(data_dir)
    if supplied is not None:
        for key in ("audit", "records", "audit_sha256", "manifest_sha256", "split_sha256", "source_sha256"):
            if supplied.get(key) != verified[key]:
                raise DataIntegrityError(f"Supplied data reference differs from the audited source: {key}")
    return verified


def _image_size(value):
    if isinstance(value, bool) or value not in IMAGE_SIZES:
        raise ValueError(f"image_size must be one of {IMAGE_SIZES}")
    return int(value)


def _cache_directory(data_dir, size):
    return Path(data_dir) / f"training-cache-{size}"


def _verify_file(path, expected):
    expected = _require_sha(expected, str(path))
    try:
        stat = path.stat()
        key = (os.getpid(), str(path.resolve()), stat.st_size, stat.st_mtime_ns,
               stat.st_ctime_ns, expected)
        if key not in _VERIFIED_FILES:
            if _sha256(path) != expected:
                raise DataIntegrityError(f"Cache SHA256 mismatch: {path}")
            _VERIFIED_FILES.add(key)
    except OSError as exc:
        raise DataIntegrityError(f"Missing prepared cache file: {path}") from exc


def _labels(records):
    return np.asarray([[r["boneage"], 1.0 if r["sex"] == "M" else 0.0]
                       for r in records], dtype=np.float32)


def _load_cache_metadata(directory, size, reference):
    metadata = _read_json(directory / "metadata.json")
    n = len(reference["records"])
    if (metadata.get("schema") != CACHE_SCHEMA or metadata.get("image_size") != size or
            metadata.get("preprocess_version") != PREPROCESS_VERSION or
            metadata.get("dtype") != "uint8" or metadata.get("shape") != [n, size, size] or
            metadata.get("splits") != reference["audit"]["splits"]):
        raise DataIntegrityError("Prepared cache schema, shape, or fixed splits do not match source")
    for key in ("audit_sha256", "manifest_sha256", "split_sha256", "source_sha256"):
        if metadata.get(key) != reference[key]:
            raise DataIntegrityError(f"Prepared cache source mismatch: {key}")
    files = metadata.get("files", {})
    for name in ("images.npy", "labels.npy", "ids.json"):
        item = files.get(name, {})
        _verify_file(directory / name, item.get("sha256"))
        if item.get("bytes") != (directory / name).stat().st_size:
            raise DataIntegrityError(f"Prepared cache byte count mismatch: {name}")
    index = _read_json(directory / "ids.json")
    expected_ids = [r["source_id"] for r in reference["records"]]
    expected_splits = [r["split"] for r in reference["records"]]
    if index != {"ids": expected_ids, "splits": expected_splits}:
        raise DataIntegrityError("Prepared cache IDs or fixed split assignments differ from manifest")
    try:
        images = np.load(directory / "images.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(directory / "labels.npy", mmap_mode="r", allow_pickle=False)
        if images.dtype != np.uint8 or images.shape != (n, size, size):
            raise DataIntegrityError("Prepared image array must be uint8 [N,H,W]")
        if labels.dtype != np.float32 or labels.shape != (n, 2) or not np.array_equal(labels, _labels(reference["records"])):
            raise DataIntegrityError("Prepared labels/sex differ from audited manifest")
    except (OSError, ValueError) as exc:
        if isinstance(exc, DataIntegrityError):
            raise
        raise DataIntegrityError("Cannot memory-map prepared cache arrays") from exc
    finally:
        # Do not retain handles here: datasets open their own mapping per process.
        if "images" in locals():
            del images
        if "labels" in locals():
            del labels
    return metadata


class BoneAgeDataset(Dataset):
    """One fixed partition, returning uint8 CHW, sex, age in months, and ID.

    Augmentation and normalization belong to the trainer. High resolution images
    are read from a process-local read-only mmap and copied only per sample.
    """

    def __init__(self, data_dir, split, image_size, reference=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.data_dir = str(Path(data_dir).expanduser().resolve())
        self.split, self.image_size = split, _image_size(image_size)
        self.reference = _reference(self.data_dir, reference)
        all_records = self.reference["records"]
        self.indices = np.asarray([i for i, r in enumerate(all_records) if r["split"] == split], dtype=np.int64)
        self.records = [all_records[i] for i in self.indices]
        self.ids = [r["source_id"] for r in self.records]
        self._images, self._pid = None, None
        self._legacy = None
        directory = _cache_directory(self.data_dir, self.image_size)
        self.cache_dir = str(directory)
        if (directory / "metadata.json").is_file():
            self.cache_metadata = _load_cache_metadata(directory, self.image_size, self.reference)
        elif self.image_size == 96:
            self._load_legacy()
        else:
            raise FileNotFoundError(
                f"Prepared {self.image_size}px cache is missing: {directory}. "
                f"Run python scripts/prepare_training_data.py --data \"{self.data_dir}\" "
                f"--size {self.image_size} first; high resolution images are never upsampled from cache-96.npz.")

    def _load_legacy(self):
        audit = self.reference["audit"]
        if audit.get("image_size") != 96:
            raise DataIntegrityError("Legacy 96 cache is not the image_size declared by audit.json")
        path = Path(self.data_dir) / "cache-96.npz"
        _verify_file(path, audit.get("cache_sha256"))
        records = self.reference["records"]
        with np.load(path, allow_pickle=False) as cache:
            required = {"images", "targets", "sex", "ids", "split"}
            if not required.issubset(cache.files):
                raise DataIntegrityError("Legacy cache lacks audited IDs/labels/splits")
            expected_ids = np.asarray([r["source_id"] for r in records])
            expected_split = np.asarray([SPLITS.index(r["split"]) for r in records], dtype=np.uint8)
            labels = _labels(records)
            if (not np.array_equal(cache["ids"], expected_ids) or
                    not np.array_equal(cache["split"], expected_split) or
                    not np.array_equal(cache["targets"], labels[:, 0]) or
                    not np.array_equal(cache["sex"], labels[:, 1])):
                raise DataIntegrityError("Legacy cache IDs, fixed splits, or labels differ from manifest")
            images = cache["images"]
            if images.dtype != np.uint8 or images.shape != (len(records), 96, 96):
                raise DataIntegrityError("Legacy cache images must be uint8 [N,96,96]")
            # The retained dataset contains only its selected partition.
            self._legacy = images[self.indices].copy()
        self.cache_metadata = {"schema": "legacy-audited-npz", "image_size": 96,
                               "cache_sha256": audit["cache_sha256"],
                               **{k: self.reference[k] for k in
                                  ("audit_sha256", "manifest_sha256", "split_sha256", "source_sha256")}}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self._legacy is not None:
            pixels = self._legacy[index]
        else:
            if self._images is None or self._pid != os.getpid():
                self._images = np.load(Path(self.cache_dir) / "images.npy", mmap_mode="r", allow_pickle=False)
                self._pid = os.getpid()
            pixels = self._images[self.indices[index]]
        image = torch.from_numpy(np.array(pixels, dtype=np.uint8, copy=True)).unsqueeze(0)
        sex = torch.tensor([1.0 if record["sex"] == "M" else 0.0], dtype=torch.float32)
        target = torch.tensor(float(record["boneage"]), dtype=torch.float32)
        return image, sex, target, record["source_id"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"], state["_pid"] = None, None
        return state


def _archive_path(data_dir, audit):
    configured = audit.get("archive_path")
    if configured is not None:
        if not isinstance(configured, str) or not configured:
            raise DataIntegrityError("audit.archive_path must be a filesystem path")
        path = Path(configured).expanduser()
        path = path if path.is_absolute() else Path(data_dir) / path
    else:
        path = Path(data_dir) / "rsna-bone-age-v2.zip"
    if not path.is_file():
        raise FileNotFoundError(f"Original audited PNG archive is missing: {path}")
    return path.resolve()


def prepare_training_cache(data_dir, image_size, output_dir=None) -> dict:
    """Atomically build a uint8 letterbox mmap cache from original audited PNGs.

    ``output_dir`` is the complete destination cache directory, defaulting to
    ``data_dir/training-cache-{image_size}``. Existing valid caches are reused;
    invalid existing caches are never silently overwritten.
    """
    size = _image_size(image_size)
    reference = load_data_reference(data_dir)
    destination = (Path(output_dir).expanduser().resolve() if output_dir is not None else
                   _cache_directory(reference["data_dir"], size))
    if destination.exists():
        metadata = _load_cache_metadata(destination, size, reference)
        print(f"Reusing verified cache: {destination}", flush=True)
        return metadata
    archive_path = _archive_path(reference["data_dir"], reference["audit"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    records = reference["records"]
    started = time.monotonic()
    images = None
    try:
        images = np.lib.format.open_memmap(temporary / "images.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(records), size, size))
        print(f"Preparing {len(records)} original PNGs at {size}px; preserving {reference['audit']['splits']}", flush=True)
        with ZipFile(archive_path) as archive:
            counts = Counter(info.filename for info in archive.infolist())
            for index, record in enumerate(records):
                member = _archive_member(record["archive_path"])
                if counts[member] != 1:
                    raise DataIntegrityError(f"Missing or ambiguous original ZIP member: {member}")
                info = archive.getinfo(member)
                if info.file_size > 30_000_000:
                    raise DataIntegrityError(f"Oversize original PNG: {member}")
                png = archive.read(info)  # ZipFile also verifies the original CRC.
                if hashlib.sha256(png).hexdigest() != record["sha256"]:
                    raise DataIntegrityError(f"Original PNG SHA256 mismatch: {member}")
                image = validate_png(png)
                for field, actual in (("width", image.width), ("height", image.height), ("mode", image.mode)):
                    if field in record and record[field] != actual:
                        raise DataIntegrityError(f"Original PNG {field} mismatch: {member}")
                if record.get("raster_sha256"):
                    raster_sha = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    if raster_sha != record["raster_sha256"]:
                        raise DataIntegrityError(f"Original decoded PNG SHA256 mismatch: {member}")
                images[index], _ = letterbox(image, size)
                if (index + 1) % 1000 == 0 or index + 1 == len(records):
                    print(f"Prepared {index + 1}/{len(records)} PNGs; elapsed {time.monotonic() - started:.1f}s", flush=True)
        images.flush()
        del images
        images = None
        np.save(temporary / "labels.npy", _labels(records), allow_pickle=False)
        (temporary / "ids.json").write_text(json.dumps(
            {"ids": [r["source_id"] for r in records], "splits": [r["split"] for r in records]},
            ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        files = {name: {"sha256": _sha256(temporary / name), "bytes": (temporary / name).stat().st_size}
                 for name in ("images.npy", "labels.npy", "ids.json")}
        metadata = {"schema": CACHE_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                    "image_size": size, "dtype": "uint8", "shape": [len(records), size, size],
                    "label_columns": ["boneage_months", "male"], "preprocess_version": PREPROCESS_VERSION,
                    "splits": reference["audit"]["splits"], "source_archive": str(archive_path),
                    "source_archive_bytes": archive_path.stat().st_size, "files": files,
                    **{k: reference[k] for k in ("audit_sha256", "manifest_sha256", "split_sha256", "source_sha256")},
                    "source_pixels": "original PNG archive; per-image SHA256 and CRC verified",
                    "elapsed_seconds": round(time.monotonic() - started, 2)}
        (temporary / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        # Close every array before publication, including on Windows. The final
        # directory becomes visible only after all images, IDs and hashes exist.
        os.replace(temporary, destination)
        print(f"Prepared cache: {destination}", flush=True)
        return metadata
    finally:
        if images is not None:
            del images
        if temporary.exists():
            shutil.rmtree(temporary)
