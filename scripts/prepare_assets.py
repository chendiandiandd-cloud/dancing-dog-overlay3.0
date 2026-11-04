#!/usr/bin/env python3
"""Offline BGR -> straight-alpha RGBA dog assets; no network or engine changes.

Example:
  python scripts/prepare_assets.py --source-dir /path/to/videos
Border-connected green is keyed. Strict screen-like enclosed holes are also keyed
for non-green characters; green-dominant characters and verified overrides are protected.
Verified scene-change tails are trimmed from runtime frames, never from bundled sources.
One padded union crop and one pixel-space foot anchor are used for the whole clip.
NPZ stores frames (uint8, N,H,W,4); SHA256 in the manifest hashes the NPZ bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

SKILL_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "native-pixels-strict-holes-clean-prefix-v3"
# Verified against the bundled originals; overrides do not apply to unrelated
# videos with the same filename/ID. Frame indices below are ZERO-BASED.
PREFIX_OVERRIDES = {
    "44dc4ac51819c0178c8ffba0f99ef3584486322fd5b53cb1475d808b87adc3e6": 58,
}
GREEN_DETAIL_OVERRIDES = {
    "6f7fe8f08df3d95c4a9c37f714bf5d1ddae65a0145bd761c2a6789693a275fc1",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def probe(path: Path) -> dict:
    # No shell interpolation, bounded execution, absolute path even for '-' names.
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
         "-of", "json", str(path.resolve())],
        check=True, capture_output=True, text=True, timeout=60,
    )
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No video stream: {path}")
    s = data["streams"][0]
    numerator, denominator = map(int, s["avg_frame_rate"].split("/"))
    fps = numerator / denominator if denominator else 0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid FPS: {path}")
    if s.get("r_frame_rate") != s["avg_frame_rate"]:
        raise ValueError(f"Potential variable-frame-rate input not supported: {path}")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps,
            "fps_rational": s["avg_frame_rate"],
            "declared_frames": int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else None,
            "source_duration": float(s.get("duration", data.get("format", {}).get("duration", 0)))}


def border_color(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    band = max(2, min(h, w) // 50)
    samples = np.concatenate((rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
                              rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3)))
    s = samples.astype(np.float32)
    green = (s[:, 1] - np.maximum(s[:, 0], s[:, 2]) > 45) & (s[:, 1] > 90)
    if green.mean() < 0.60:
        raise ValueError("Border does not contain sufficient green-screen samples")
    bg = np.median(s[green], axis=0)
    spread = float(np.percentile(np.linalg.norm(s[green] - bg, axis=1), 99))
    return bg, spread


def matte(bgr: np.ndarray, preserve_enclosed_green: bool = True) -> tuple[np.ndarray, dict]:
    rgb8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb8.astype(np.float32)
    bg, spread = border_color(rgb8)
    delta = np.linalg.norm(rgb - bg, axis=2)
    dominance = rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])
    # Lenient band includes mixed AA pixels, but excludes black and luminous cyan.
    candidate = (dominance > 22) & (delta < 145)
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    lut = np.zeros(count, dtype=bool)
    lut[border_labels[border_labels != 0]] = True
    connected = lut[labels]
    enclosed_before = candidate & ~connected
    removed_holes = np.zeros(candidate.shape, dtype=bool)
    strict_limit = 18.0  # Fixed: never widened in response to noisy backgrounds.
    if not preserve_enclosed_green:
        hsv = cv2.cvtColor(rgb8, cv2.COLOR_RGB2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(np.rint(bg).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
        hue_delta = np.abs(hsv[..., 0] - bg_hsv[0])
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        strict = (enclosed_before & (delta <= strict_limit)
                  & (np.max(np.abs(rgb-bg), axis=2) <= strict_limit)
                  & (hue_delta <= 3) & (np.abs(hsv[..., 1]-bg_hsv[1]) <= 24)
                  & (np.abs(hsv[..., 2]-bg_hsv[2]) <= 14))
        areas = np.bincount(labels.ravel(), minlength=count)
        seeds = np.bincount(labels[strict], minlength=count)
        # Thin holes (e.g. a mouth slit) have mostly mixed edge pixels. A small
        # strict seed fraction is sufficient; RGB/HSV tolerances stay unchanged.
        hole_lut = (seeds >= 3) & (seeds >= 0.015 * areas) & ~lut
        hole_lut[0] = False
        # Strict seeds identify a screen hole. Its existing green candidate band
        # (not a newly widened band) supplies AA pixels for projection below.
        removed_holes = hole_lut[labels]
        connected = connected | removed_holes
    protected = candidate & ~connected
    core = ~connected
    # Nearest clean foreground for subpixel color unmixing. Never blur the whole
    # matte: that would erase thin black lines or grow halos onto the green screen.
    clean = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not clean.any():
        clean = core
    if not clean.any():
        raise ValueError("No foreground found")
    distance, nearest = cv2.distanceTransformWithLabels(
        (~clean).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros((int(nearest.max()) + 1, 3), dtype=np.float32)
    lookup[nearest[clean]] = rgb[clean]
    edge = connected & (distance <= 4.2)
    alpha = core.astype(np.float32)
    if edge.any():
        fg = lookup[nearest[edge]]
        v = fg - bg
        a = np.sum((rgb[edge] - bg) * v, axis=1) / np.maximum(np.sum(v * v, axis=1), 1)
        a = np.clip(a, 0, 1)
        # Suppress codec variation on near-pure background without eating glow.
        noise = max(5.0, min(18.0, spread + 3.0))
        a[delta[edge] <= noise] = 0
        alpha[edge] = a
    # Recover straight RGB from compositing equation; edge-local residual despill
    # is limited to keyed pixels, preserving enclosed green/cyan and black lines.
    colors = rgb.copy()
    mixed = (alpha > 0.015) & (alpha < 0.999)
    colors[mixed] = (rgb[mixed] - (1 - alpha[mixed, None]) * bg) / alpha[mixed, None]
    colors = np.clip(colors, 0, 255)
    edge_green = mixed & (colors[..., 1] > np.maximum(colors[..., 0], colors[..., 2]) + 25)
    colors[edge_green, 1] = np.maximum(colors[edge_green, 0], colors[edge_green, 2]) + 25
    a8 = np.rint(alpha * 255).astype(np.uint8)
    a8[a8 <= 2] = 0
    colors[a8 == 0] = 0
    rgba = np.dstack((np.rint(colors).astype(np.uint8), a8))
    ys, xs = np.where(a8 > 0)
    if not len(xs):
        raise ValueError("Empty foreground")
    # Largest solid component avoids grounding the character on detached sparks.
    _, cc, stats, _ = cv2.connectedComponentsWithStats((a8 >= 192).astype(np.uint8), 8)
    if len(stats) <= 1:
        raise ValueError("No solid foreground")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fy, fx = np.where(cc == largest)
    bottom = int(fy.max())
    # Stance center across both feet/lower limbs, rather than only the lowest
    # toe (which would pull a two-foot character's anchor to one side).
    lower = fx[fy >= bottom - max(4, (bottom - int(fy.min())) // 5)]
    feet_x = float((np.percentile(lower, 5) + np.percentile(lower, 95)) / 2)
    corners = np.concatenate((a8[:8, :8].ravel(), a8[:8, -8:].ravel(),
                              a8[-8:, :8].ravel(), a8[-8:, -8:].ravel()))
    return rgba, {"bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                  "foot": [feet_x, bottom + 1], "background_rgb": bg.tolist(),
                  "background_spread_p99": spread,
                  "foreground_rgb_median": np.median(rgb[(cc == largest) & (rgb.max(axis=2) > 90)], axis=0).tolist(),
                  "enclosed_green_before": int(enclosed_before.sum()),
                  "enclosed_screen_keyed_pixels": int(removed_holes.sum()),
                  "enclosed_screen_transparent_pixels": int((removed_holes & (a8 == 0)).sum()),
                  "protected_green_rgb_unchanged": bool(np.array_equal(rgba[protected, :3], rgb8[protected])),
                  "protected_green_pixels": int(protected.sum()),
                  "protected_green_opaque": bool(np.all(a8[protected] == 255)),
                  "solid_core_opaque": bool(np.all(a8[clean] == 255)),
                  "black_core_pixels": int((clean & (rgb.max(axis=2) < 70)).sum()),
                  "source_corner_alpha_max": int(corners.max()),
                  "source_edge_nonzero": int(np.count_nonzero(np.concatenate(
                      (a8[0], a8[-1], a8[:, 0], a8[:, -1]))))}


def resize_rgba(rgba: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # 1::
    if (rgba.shape[1], rgba.shape[0]) == size:
        return rgba.copy()
    # Area filtering in premultiplied space prevents black fringes on soft edges.
    a = rgba[..., 3].astype(np.float32) / 255
    premul = rgba[..., :3].astype(np.float32) * a[..., None]
    interp = cv2.INTER_AREA if size[0] <= rgba.shape[1] and size[1] <= rgba.shape[0] else cv2.INTER_LANCZOS4
    raw_a = cv2.resize(a, size, interpolation=interp)
    out_a = np.clip(raw_a, 0, 1)
    out_p = cv2.resize(premul, size, interpolation=interp)
    # Normalize using UNCLAMPED filter alpha: clipping alpha first changes
    # straight color on overshoot pixels (especially saturated outlines).
    out_rgb = out_p / np.maximum(raw_a[..., None], 1e-8)
    out_rgb[raw_a <= 0] = 0
    result = np.dstack((np.clip(np.rint(out_rgb), 0, 255).astype(np.uint8),
                       np.clip(np.rint(out_a * 255), 0, 255).astype(np.uint8)))
    result[result[..., 3] == 0, :3] = 0
    return result


def dominant(frames: np.ndarray) -> list[int]:
    hist = np.zeros(32 ** 3, dtype=np.int64)
    for f in frames:
        colors = f[..., :3][f[..., 3] >= 240].astype(np.int32)
        if not len(colors):
            continue
        q = colors // 8
        hist += np.bincount(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2], minlength=len(hist))
    if not hist.any():
        raise ValueError("No opaque color samples")
    peak = int(hist.argmax())
    center = np.array([peak // 1024, (peak // 32) % 32, peak % 32]) * 8 + 4
    total = np.zeros(3, dtype=np.float64)
    n = 0
    for f in frames:
        c = f[..., :3][f[..., 3] >= 240].astype(np.int32)
        selected = c[np.linalg.norm(c - center, axis=1) <= 16]
        total += selected.sum(axis=0)
        n += len(selected)
    return np.rint(total / max(n, 1)).astype(int).tolist()


def check_archive(path: Path, dog: dict) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["frames"]:
            raise ValueError(f"Unexpected NPZ keys: {path}")
        frames = archive["frames"]
    expected = (dog["frame_count"], dog["height"], dog["width"], 4)
    if "source_frame_count" in dog:
        if (dog["source_frame_count"] - dog["trimmed_source_frames"] != dog["frame_count"]
                or len(dog["trimmed_source_frame_indices"]) != dog["trimmed_source_frames"]
                or dog["runtime_source_frame_range"] != [0, dog["frame_count"]]):
            raise ValueError(f"Invalid clean-prefix/source frame accounting: {path}")
    if frames.dtype != np.uint8 or frames.shape != expected:
        raise ValueError(f"Bad RGBA contract: {path}")
    a = frames[..., 3]
    corners = np.concatenate((a[:, :3, :3].ravel(), a[:, :3, -3:].ravel(),
                              a[:, -3:, :3].ravel(), a[:, -3:, -3:].ravel()))
    edges = np.concatenate((a[:, 0].ravel(), a[:, -1].ravel(),
                            a[:, :, 0].ravel(), a[:, :, -1].ravel()))
    ax, ay = dog["anchor"]
    if not (0 <= ax < dog["width"] and 0 <= ay <= dog["height"]):
        raise ValueError(f"Anchor outside asset: {path}")
    if not np.all(np.any(a > 0, axis=(1, 2))):
        raise ValueError(f"Height/empty-frame check failed: {path}")
    if corners.max() or edges.max():
        raise ValueError(f"Nontransparent output corner/boundary: {path}")
    return {"corner_alpha_max": int(corners.max()), "edge_alpha_max": int(edges.max()),
            "soft_alpha_pixels": int(((a > 0) & (a < 255)).sum()),
            "uncompressed_bytes": int(frames.nbytes)}


def prepare_one(source: Path, output: Path, dog_id: str, signature: str,
                max_height: int, padding: int, old: dict | None, force: bool) -> tuple[dict, bool]:
    source_hash = digest(source)
    npz_path = output / "dogs" / f"{dog_id}.npz"
    copy_path = output / "source" / f"{dog_id}.mp4"
    same_source = source.resolve() == copy_path.resolve()
    if not same_source and (not copy_path.exists() or digest(copy_path) != source_hash):
        tmp = copy_path.with_suffix(".mp4.tmp")
        try:
            shutil.copyfile(source, tmp)
            if digest(tmp) != source_hash:
                raise ValueError(f"Source changed during copy: {source}")
            os.replace(tmp, copy_path)
        finally:
            tmp.unlink(missing_ok=True)
    if (not force and old and old.get("source_sha256") == source_hash
            and old.get("pipeline_signature") == signature and npz_path.exists()
            and digest(npz_path) == old.get("sha256")):
        check_archive(npz_path, old)
        return old, True
    started = time.perf_counter()
    info = probe(copy_path)
    cap = cv2.VideoCapture(str(copy_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot decode: {copy_path}")
    decoded = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if bgr.shape[:2] != (info["height"], info["width"]):
                raise ValueError("Frame dimensions changed")
            decoded.append(bgr)
    finally:
        cap.release()
    source_n = len(decoded)
    if not source_n or (info["declared_frames"] is not None and source_n != info["declared_frames"]):
        raise ValueError(f"Incomplete decode: {source}: {source_n}/{info['declared_frames']}")
    prefix_stop = PREFIX_OVERRIDES.get(source_hash, source_n)
    if not 1 <= prefix_stop <= source_n:
        raise ValueError("Verified prefix cutoff does not fit source")
    trimmed_indices = list(range(prefix_stop, source_n))
    decoded = decoded[:prefix_stop]
    n = len(decoded)
    samples = [matte(decoded[i])[0] for i in sorted(set(np.linspace(0, n-1, min(n, 5), dtype=int).tolist()))]
    policy_rgb = dominant(samples)
    del samples
    policy_hsv = cv2.cvtColor(np.array(policy_rgb, dtype=np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    preserve_green = bool(source_hash in GREEN_DETAIL_OVERRIDES
                          or (35 <= int(policy_hsv[0]) <= 90 and int(policy_hsv[1]) >= 70))
    policy_reason = ("verified green-detail/glow source override" if source_hash in GREEN_DETAIL_OVERRIDES
                     else "foreground dominant green hue" if preserve_green
                     else "non-green dominant hue: strict screen-color seeded holes")
    sequence, metrics = [], []
    for bgr in decoded:
        rgba, quality = matte(bgr, preserve_enclosed_green=preserve_green)
        sequence.append(rgba)
        metrics.append(quality)
    del decoded
    boxes = np.array([m["bbox"] for m in metrics])
    union = [int(boxes[:, 0].min()), int(boxes[:, 1].min()),
             int(boxes[:, 2].max()), int(boxes[:, 3].max())]
    pad = max(padding, int(math.ceil((union[3] - union[1]) * 0.025)))
    # Virtual padding also works when a source touches its own frame boundary.
    x0, y0, x1, y1 = union[0] - pad, union[1] - pad, union[2] + pad, union[3] + pad
    cw, ch = x1 - x0, y1 - y0
    scale = min(1.0, max_height / ch) if max_height else 1.0
    width, height = max(1, round(cw * scale)), max(1, round(ch * scale))
    frames = np.empty((n, height, width, 4), dtype=np.uint8)
    for i, rgba in enumerate(sequence):
        crop = np.zeros((ch, cw, 4), dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(info["width"], x1), min(info["height"], y1)
        crop[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = rgba[sy0:sy1, sx0:sx1]
        frames[i] = resize_rgba(crop, (width, height))
    del sequence
    foot_x = float(np.median([m["foot"][0] for m in metrics]))
    foot_y = float(max(m["foot"][1] for m in metrics))
    warnings = []
    if any(m["source_edge_nonzero"] for m in metrics):
        warnings.append("Source foreground touches original boundary; missing source action cannot be recovered.")
    colors_over_time = np.array([m["foreground_rgb_median"] for m in metrics])
    color_jumps = np.linalg.norm(np.diff(colors_over_time, axis=0), axis=1)
    suspicious_frames = (np.where(color_jumps > 140)[0] + 1).tolist()
    if suspicious_frames:
        warnings.append(f"Abrupt foreground color change before zero-based frames {suspicious_frames}; full source retained, inspect possible scene changes.")
    protected_max = max(m["protected_green_pixels"] for m in metrics)
    if protected_max:
        warnings.append("Enclosed green pixels kept opaque; these may include real green details or enclosed screen holes.")
    tmp = npz_path.with_suffix(".npz.tmp")
    try:
        with tmp.open("wb") as f:
            np.savez_compressed(f, frames=frames)
        os.replace(tmp, npz_path)
    finally:
        tmp.unlink(missing_ok=True)
    dog = {"id": dog_id, "file": f"dogs/{dog_id}.npz", "fps": info["fps"],
           "duration": n / info["fps"], "frame_count": n, "width": width, "height": height,
           "source_frame_count": source_n, "trimmed_source_frames": len(trimmed_indices),
           "trimmed_source_frame_indices": trimmed_indices,
           "runtime_source_frame_range": [0, prefix_stop],
           "runtime_source_frame_range_convention": "zero-based, end-exclusive",
           "trim_reason": "verified trailing blue-character scene change; clean prefix retained" if trimmed_indices else None,
           "enclosed_green_policy": {"preserve": preserve_green, "reason": policy_reason,
                                     "foreground_sample_rgb": policy_rgb,
                                     "foreground_sample_hue_opencv": int(policy_hsv[0]),
                                     "strict_screen_rgb_distance_max": 18.0,
                                     "strict_screen_channel_difference_max": 18},
           "anchor": [round((foot_x-x0)*width/cw, 4), round((foot_y-y0)*height/ch, 4)],
           "dominant_color": dominant(frames), "sha256": digest(npz_path),
           "source": f"source/{dog_id}.mp4", "source_sha256": source_hash,
           "original_filename": (old.get("original_filename", source.name)
                                 if old and old.get("source_sha256") == source_hash else source.name),
           "source_width": info["width"],
           "source_height": info["height"], "fps_rational": info["fps_rational"],
           "source_duration": info["source_duration"], "union_bbox": union,
           "crop_bbox": [x0, y0, x1, y1], "padding_source_px": pad,
           "scale_xy": [width/cw, height/ch],
           "resolution_provenance": {"source": f"source/{dog_id}.mp4", "source_sha256": source_hash,
                                     "decoded_resolution": [info["width"], info["height"]],
                                     "native_crop_resolution": [cw, ch], "cache_resolution": [width, height],
                                     "native_pixels_preserved": width == cw and height == ch,
                                     "max_height": max_height, "sharpen_amount": 0,
                                     "method": "original decoded frames, offline key + union crop; no generated pixels or super-resolution"},
           "anchor_method": "fixed median lower-limb stance-center x; maximum solid-largest-component floor y; pixel coordinates",
           "pipeline_signature": signature, "npz_bytes": npz_path.stat().st_size,
           "source_bytes": copy_path.stat().st_size,
           "quality": {"all_frames_inside_union": bool(np.all(boxes[:, :2] >= np.array(union[:2]))
                          and np.all(boxes[:, 2:] <= np.array(union[2:]))),
                       "suspicious_color_jump_frames": suspicious_frames,
                       "enclosed_screen_keyed_pixels_total": sum(m["enclosed_screen_keyed_pixels"] for m in metrics),
                       "enclosed_screen_transparent_pixels_total": sum(m["enclosed_screen_transparent_pixels"] for m in metrics),
                       "protected_green_rgb_all_unchanged": all(m["protected_green_rgb_unchanged"] for m in metrics),
                       "protected_green_max_per_frame": protected_max,
                       "protected_green_all_opaque": all(m["protected_green_opaque"] for m in metrics),
                       "solid_core_all_opaque": all(m["solid_core_opaque"] for m in metrics),
                       "black_core_pixels_total": sum(m["black_core_pixels"] for m in metrics),
                       "source_corner_alpha_max": max(m["source_corner_alpha_max"] for m in metrics),
                       "source_boundary_nonzero_total": sum(m["source_edge_nonzero"] for m in metrics),
                       "background_rgb_median": np.median([m["background_rgb"] for m in metrics], axis=0).tolist(),
                       "cycle_verified": False,
                       "warnings": warnings}}
    dog["quality"].update(check_archive(npz_path, dog))
    dog["processing_seconds"] = round(time.perf_counter() - started, 3)
    return dog, False


def contact_sheet(output: Path, dogs: list[dict]) -> Path:
    # Diagnostic only, deliberately outside assets/final delivery.
    target = SKILL_ROOT / "temp_data" / "dogs_checker_contact.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, columns = 600, 240, 2
    sheet = Image.new("RGB", (columns*cell_w, math.ceil(len(dogs)/columns)*cell_h), "#222222")
    draw = ImageDraw.Draw(sheet)
    yy, xx = np.indices((200, 196))
    checker = np.where(((xx//12 + yy//12) % 2)[..., None], 190, 240)
    checker = np.repeat(checker, 3, axis=2).astype(np.uint8)
    for i, dog in enumerate(dogs):
        ox, oy = (i % columns)*cell_w, (i//columns)*cell_h
        draw.text((ox+5, oy+5), f"{dog['id']}  {dog['frame_count']}f / {dog['fps']:g}fps  {dog['width']}x{dog['height']}", fill="white")
        with np.load(output / dog["file"], allow_pickle=False) as archive:
            frames = archive["frames"]
        for j, index in enumerate([0, len(frames)//2, len(frames)-1]):
            tile = Image.fromarray(checker, "RGB").convert("RGBA")
            im = Image.fromarray(frames[index], "RGBA")
            s = min(180/im.height, 180/im.width)
            tw, th = max(1, round(im.width*s)), max(1, round(im.height*s))
            im = im.resize((tw, th), Image.Resampling.LANCZOS)
            tx, ty = (196-tw)//2, (200-th)//2
            tile.alpha_composite(im, (tx, ty))
            td = ImageDraw.Draw(tile)
            ax = tx + dog["anchor"][0]*tw/dog["width"]
            ay = ty + dog["anchor"][1]*th/dog["height"]
            td.line((ax-5, ay, ax+5, ay), fill="#ff3366", width=1)
            td.line((ax, ay-5, ax, ay+5), fill="#ff3366", width=1)
            sheet.paste(tile.convert("RGB"), (ox+j*200, oy+30))
    sheet.save(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=SKILL_ROOT / "assets")
    parser.add_argument("--max-height", type=int, default=0,
                        help="0 preserves native cropped pixels (default); positive value only downsamples")
    parser.add_argument("--padding", type=int, default=8, help="minimum source-space padding")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if (args.max_height != 0 and not 32 <= args.max_height <= 8192) or args.padding < 4:
        parser.error("max-height must be 0 (native) or 32..8192 and padding >=4")
    source_dir, output = args.source_dir.resolve(), args.output_dir.resolve()
    # Two explicit naming schemes only; music and other mp4 files are ignored.
    indexed = {}
    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(r"(\d+)_.*动态素材\.mp4", path.name)
        bundled = re.fullmatch(r"dog_(\d+)\.mp4", path.name)
        found = match or bundled
        if not found:
            continue
        number = int(found.group(1))
        if number < 1 or number in indexed:
            parser.error(f"Invalid or duplicate dog ID in input: {path.name}")
        indexed[number] = path
    sources = sorted(indexed.items())
    if not sources:
        parser.error(f"No numbered *动态素材.mp4 or dog_NN.mp4 in {source_dir}")
    for number, path in sources:
        bundled_target = output / "source" / f"dog_{number:02d}.mp4"
        if path.resolve().is_relative_to(output) and path.resolve() != bundled_target.resolve():
            parser.error("Input within output must be its exact bundled source/dog_NN.mp4")
    for subdir in ("dogs", "source"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dog_manifest.json"
    old = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("version") == 1:
                old = {dog["id"]: dog for dog in previous.get("dogs", [])}
        except (ValueError, KeyError, TypeError):
            pass
    signature = hashlib.sha256((PIPELINE_VERSION + digest(Path(__file__)) +
                                f"opencv={cv2.__version__};height={args.max_height};padding={args.padding}").encode()).hexdigest()
    cv2.setNumThreads(1)
    start = time.perf_counter()
    dogs, cache_hits = [], 0
    for number, source in sources:
        dog_id = f"dog_{number:02d}"
        dog, cached = prepare_one(source, output, dog_id, signature, args.max_height,
                                  args.padding, old.get(dog_id), args.force)
        dogs.append(dog)
        cache_hits += int(cached)
        print(f"{dog_id}: {'CACHE' if cached else 'BUILT'} {dog['frame_count']}f "
              f"{dog['width']}x{dog['height']} anchor={dog['anchor']} "
              f"{dog['npz_bytes']/1048576:.2f} MiB warnings={len(dog['quality']['warnings'])}", flush=True)
    # No timestamps/run metrics in the manifest: identical cached reruns leave it untouched.
    manifest = {"version": 1, "dogs": dogs}
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != encoded:
        atomic_json(manifest_path, manifest)
    preview = None if args.no_preview else str(contact_sheet(output, dogs))
    print(json.dumps({"dogs": len(dogs), "cache_hits": cache_hits,
                      "total_frames": sum(d["frame_count"] for d in dogs),
                      "total_source_frames": sum(d["source_frame_count"] for d in dogs),
                      "total_trimmed_source_frames": sum(d["trimmed_source_frames"] for d in dogs),
                      "npz_bytes": sum(d["npz_bytes"] for d in dogs),
                      "source_bytes": sum(d["source_bytes"] for d in dogs),
                      "elapsed_seconds": round(time.perf_counter()-start, 3),
                      "manifest": str(manifest_path), "diagnostic_preview": preview}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
