#!/usr/bin/env python3
"""Losslessly repack manifest NPZ caches as ZIP_LZMA, never rebuild pixels.

Run: python3 scripts/lossless_cache.py
Optional: --assets-dir PATH --work-dir PATH --report PATH
All candidates are verified before any replacement. Keep render/preparation jobs
stopped during maintenance: multi-file replacement is not one atomic transaction.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMPRESSION = {"container": "npz", "zip_method": "ZIP_LZMA", "zip_method_id": 14,
               "lossless": True}
HASH_CONVENTION = "SHA256 of all uint8 RGBA frames in N,H,W,4 C order, including transparent RGB; no NPY header"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def frames_hash(frames: np.ndarray) -> str:
    h = hashlib.sha256()
    for frame in frames:
        h.update(frame.tobytes(order="C"))
    return h.hexdigest()


def cache_metadata(frames: np.ndarray) -> dict:
    return {"compression": dict(COMPRESSION), "decoded_frames_sha256": frames_hash(frames),
            "decoded_frames_hash_convention": HASH_CONVENTION}


def write_npz_lzma(path: Path, **arrays: np.ndarray) -> None:
    """Public writer for prepare_assets; numpy NPY format, pickle forbidden."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_LZMA, allowZip64=True) as z:
        for name, array in arrays.items():
            if not name or "/" in name or "\\" in name:
                raise ValueError("Invalid array key")
            data = io.BytesIO()
            np.lib.format.write_array(data, np.asanyarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_LZMA
            z.writestr(info, data.getvalue())
    with Path(path).open("rb") as f:
        os.fsync(f.fileno())


def inspect_npz(path: Path, dog: dict | None = None) -> tuple[dict, dict]:
    """Read/CRC check every member and load every NPY with allow_pickle=False."""
    started = time.perf_counter()
    members = []
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"Empty/duplicate ZIP members: {path}")
        for info in infos:
            payload = z.read(info)  # zipfile verifies CRC; directories/extra files kept too.
            members.append({"name": info.filename, "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "zip_method_id": info.compress_type})
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            # NumPy exposes non-NPY members as bytes; preserve them but only
            # actual .npy members are arrays subject to dtype/shape validation.
            if name + ".npy" in names:
                arrays[name] = archive[name]
    if "frames" not in arrays:
        raise ValueError(f"Missing frames.npy: {path}")
    frames = arrays["frames"]
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 4 or frames.shape[0] < 1:
        raise ValueError(f"Bad RGBA contract: {path}")
    if dog is not None:
        if frames.shape != (dog["frame_count"], dog["height"], dog["width"], 4):
            raise ValueError(f"Manifest shape mismatch: {path}")
        expected = dog.get("decoded_frames_sha256")
        if expected is not None and frames_hash(frames) != expected:
            raise ValueError(f"Decoded frames hash mismatch: {path}")
    return arrays, {"members": members, "decode_seconds": round(time.perf_counter()-started, 6)}


def repack_npz(source: Path, target: Path) -> None:
    """Change ZIP codec only; NEVER serialize/reconstruct existing NPY members."""
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_LZMA,
                                                       allowZip64=True) as new:
        new.comment = old.comment
        for info in old.infolist():
            cloned = copy.copy(info)
            cloned.compress_type = zipfile.ZIP_LZMA
            new.writestr(cloned, old.read(info))
    with Path(target).open("rb") as f:
        os.fsync(f.fileno())


def compare_npz(source: Path, target: Path, dog: dict | None = None) -> dict:
    old, before = inspect_npz(source, dog)
    new, after = inspect_npz(target, dog)
    started = time.perf_counter()
    a = [(m["name"], m["bytes"], m["sha256"]) for m in before["members"]]
    b = [(m["name"], m["bytes"], m["sha256"]) for m in after["members"]]
    if a != b or list(old) != list(new):
        raise ValueError("ZIP member content/order or array keys changed")
    summaries = []
    for name in old:
        left, right = old[name], new[name]
        equal = np.array_equal(left, right)
        if not equal and left.dtype.kind in "fc":
            equal = np.array_equal(left, right, equal_nan=True)
        if left.dtype != right.dtype or left.shape != right.shape or not equal:
            raise ValueError(f"Array changed: {name}")
        summaries.append({"name": name, "dtype": str(left.dtype), "shape": list(left.shape),
                          "dtype_shape_values_equal": True})
    if not all(m["zip_method_id"] == 14 for m in after["members"]):
        raise ValueError("Candidate is not entirely ZIP_LZMA")
    frames = old["frames"]
    return {"all_members_byte_identical": True, "all_rgba_pixels_equal": True,
            "arrays": summaries, "old_members": before["members"], "new_members": after["members"],
            "old_decode_seconds": before["decode_seconds"], "new_decode_seconds": after["decode_seconds"],
            "compare_seconds": round(time.perf_counter()-started, 6),
            **cache_metadata(frames),
            "frame_pixels_sha256": [hashlib.sha256(f.tobytes(order="C")).hexdigest() for f in frames]}


def atomic_json(path: Path, value: dict, work: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="json-", suffix=".tmp", dir=work)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def staging_directory(work: Path):
    """Retain candidates/backups on failure, including rollback I/O failure."""
    stage = Path(tempfile.mkdtemp(prefix="transaction-", dir=work))
    try:
        yield stage
    except BaseException:
        # Do not erase forensic evidence or the only remaining backup.
        raise
    else:
        shutil.rmtree(stage)


def compress_assets(assets_dir: Path = ROOT / "assets", work_dir: Path = ROOT / "temp_data/lossless",
                    report_path: Path | None = None) -> dict:
    """Preflight whole manifest, stage whole batch, then replace with rollback.

    work_dir must share a filesystem with assets (os.replace); backups and failure
    diagnostics stay under work_dir, never in the baseline or source directories.
    A lock rejects concurrent maintenance invocations; it does not lock renderers.
    """
    assets, work = Path(assets_dir).resolve(), Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve() if report_path else work / "compression_report.json"
    if not report_path.is_relative_to(work):
        raise ValueError("Report must be under work_dir")
    if os.stat(assets).st_dev != os.stat(work).st_dev:
        raise ValueError("work_dir must share filesystem with assets for atomic replacement")
    lock = work / "maintenance.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    start = time.perf_counter()
    report = {"ok": False, "dogs": [], "compression": COMPRESSION,
              "decoded_frames_hash_convention": HASH_CONVENTION}
    try:
        manifest_path = assets / "dog_manifest.json"
        original_bytes = manifest_path.read_bytes()
        manifest = json.loads(original_bytes)
        if manifest.get("version") != 1 or not manifest.get("dogs"):
            raise ValueError("Expected version=1, nonempty manifest")
        ids, paths = set(), set()
        # Verify ALL trusted container hashes and contents before staging any.
        for dog in manifest["dogs"]:
            path = (assets / dog["file"]).resolve()
            if dog["id"] in ids or path in paths or not path.is_relative_to(assets / "dogs") or path.suffix != ".npz":
                raise ValueError("Duplicate/unsafe manifest cache path or ID")
            ids.add(dog["id"]); paths.add(path)
            if sha256(path) != dog["sha256"] or path.stat().st_size != dog["npz_bytes"]:
                raise ValueError(f"Cache hash/size mismatch: {dog['id']}")
            inspect_npz(path, dog)
        updated = copy.deepcopy(manifest)
        with staging_directory(work) as stage:
            # Save original manifest for recovery after process/power failure.
            (stage / "manifest-original.json").write_bytes(original_bytes)
            pending = []
            for index, dog in enumerate(updated["dogs"]):
                dog_start = time.perf_counter()
                source = (assets / dog["file"]).resolve()
                with zipfile.ZipFile(source) as z:
                    already = all(i.compress_type == 14 for i in z.infolist())
                target = source if already else stage / f"candidate-{index}.npz"
                if not already:
                    repack_npz(source, target)
                result = compare_npz(source, target, dog)
                old_size, new_size = source.stat().st_size, target.stat().st_size
                old_hash, new_hash = sha256(source), sha256(target)
                if not already and new_size >= old_size:
                    raise ValueError(f"No size reduction: {dog['id']}; refuse replacement")
                row = {"id": dog["id"], "old_bytes": old_size, "new_bytes": new_size,
                       "old_sha256": old_hash, "new_sha256": new_hash,
                       "saved_bytes": old_size-new_size, "skipped_already_lzma": already,
                       **result, "elapsed_seconds": round(time.perf_counter()-dog_start, 6)}
                report["dogs"].append(row)
                dog.update({"sha256": new_hash, "npz_bytes": new_size,
                            **{k: result[k] for k in cache_metadata_keys()}})
                if not already:
                    backup = stage / f"backup-{index}.npz"
                    shutil.copy2(source, backup)
                    if sha256(backup) != old_hash:
                        raise ValueError("Backup verification failed")
                    pending.append((source, target, backup, old_hash, new_hash))
                print(f"{dog['id']}: {old_size:,} -> {new_size:,} bytes; all members/RGBA equal", flush=True)
            # Recheck whole source snapshot immediately before commit.
            if manifest_path.read_bytes() != original_bytes:
                raise ValueError("Manifest changed concurrently")
            for dog in manifest["dogs"]:
                if sha256(assets / dog["file"]) != dog["sha256"]:
                    raise ValueError("Cache changed concurrently")
            replaced = []
            try:
                for source, target, backup, old_hash, new_hash in pending:
                    if sha256(target) != new_hash:
                        raise ValueError("Candidate changed after verification")
                    os.replace(target, source)
                    replaced.append((source, backup))
                if updated != manifest:
                    atomic_json(manifest_path, updated, stage)
            except BaseException:
                for source, backup in reversed(replaced):
                    os.replace(backup, source)
                # atomic_json either leaves original or installs whole new JSON.
                if manifest_path.read_bytes() != original_bytes:
                    restore = stage / "manifest-restore.json"
                    restore.write_bytes(original_bytes)
                    os.replace(restore, manifest_path)
                raise
        report.update(ok=True, old_cache_total_bytes=sum(r["old_bytes"] for r in report["dogs"]),
                      new_cache_total_bytes=sum(r["new_bytes"] for r in report["dogs"]),
                      all_members_byte_identical=True, all_rgba_pixels_equal=True,
                      metadata_unchanged_except_cache_fields=True,
                      total_frames=sum(d["frame_count"] for d in manifest["dogs"]))
        report["saved_bytes"] = report["old_cache_total_bytes"] - report["new_cache_total_bytes"]
        report["saved_percent"] = round(100*report["saved_bytes"]/report["old_cache_total_bytes"], 4)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter()-start, 6)
        try:
            atomic_json(report_path, report, work)
        finally:
            lock.unlink(missing_ok=True)
    return report


def cache_metadata_keys() -> tuple[str, ...]:
    return ("compression", "decoded_frames_sha256", "decoded_frames_hash_convention")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "temp_data/lossless")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = compress_assets(args.assets_dir, args.work_dir, args.report)
    print(json.dumps({k: v for k, v in result.items() if k != "dogs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
