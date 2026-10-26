"""DEV-ONLY helper: generate a tiny synthetic frame slice matching the real frame
format (PNG + manifest.csv) with known degradation segments, so the full local
pipeline can be validated for correctness before the real TartanDrive bag arrives.
This is NOT data for any reportable result.

Usage:
    python src/_make_synthetic.py <out_dir> <n_frames>
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def main(out_dir: str, n: int = 80):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    H, W = 544, 1024
    rng = np.random.default_rng(0)
    # static-ish textured background so optical flow finds trackable corners
    base = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    # known fault segments
    blur_seg = set(range(20, 33))     # blurry
    dark_seg = set(range(45, 60))     # washed-out / dark
    with open(out / "manifest.csv", "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["frame_idx", "filename", "timestamp_ns", "encoding", "width", "height"])
        for i in range(n):
            # pan the background to create real motion for optical flow
            shift = (i * 3) % 50
            img = np.roll(base, shift, axis=1).copy()
            # add small per-frame noise so corners move
            img = np.clip(img.astype(np.int16) + rng.integers(-5, 6, img.shape), 0, 255).astype(np.uint8)
            if i in blur_seg:
                img = cv2.GaussianBlur(img, (31, 31), 12)
            if i in dark_seg:
                img = np.clip(img.astype(np.float32) * 0.25 + 20, 0, 255).astype(np.uint8)
            fn = f"frame_{i:06d}.png"
            cv2.imwrite(str(out / fn), img)
            w.writerow([i, fn, 100_000_000 * i, "bgr8", W, H])
    print(f"wrote {n} synthetic frames to {out}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data/frames/_synthetic"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    main(d, n)
