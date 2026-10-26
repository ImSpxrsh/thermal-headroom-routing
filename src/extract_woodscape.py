"""Track C extractor (WoodScape Soiling, fisheye). LOCAL/UNCOMMITTED.

Reads WoodScape soiling RGB images + soiling masks, and produces:
  1. frames_dir: UNDISTORTED (rectilinear/cylindrical) PNG frames + manifest.csv, for the
     YOLO pseudo-GT pass (phase3), exactly like extract_frames.py / extract_radiate.py.
  2. trackC_observations_raw.csv: per-frame fisheye-correct degradation observations
     (polar blur + illumination on the raw fisheye, occlusion from the soiling mask),
     consumed by phase1_woodscape.py.

WoodScape soiling layout is flexible across releases; we glob for rgb images and match
masks by filename stem. Calibration defaults to the vendored example (front.json); pass
--calib for the actual soiling-camera calibration if you have it.

    python src/extract_woodscape.py --soiling-dir data/woodscape_soiling/train
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import abspath, REPO_ROOT
import woodscape_track as wt


def _find_pairs(root: Path):
    """Return [(rgb_path, mask_path), ...] matched by filename stem."""
    rgb_dirs = [d for d in root.rglob("*") if d.is_dir() and d.name.lower() in
                ("rgbimages", "rgb_images", "images", "rgb")]
    mask_dirs = [d for d in root.rglob("*") if d.is_dir() and d.name.lower() in
                 ("gtlabels", "gt_labels", "masks", "gtlabel", "labels")]
    if not rgb_dirs:
        # fallback: any dir with images, masks alongside
        rgb_dirs = [root]
    rgb = {}
    for d in rgb_dirs:
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            rgb[p.stem] = p
    masks = {}
    for d in mask_dirs:
        for p in sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")):
            masks[p.stem] = p
    pairs = []
    for stem in sorted(rgb):
        m = masks.get(stem)
        if m is None:  # try common mask suffixes
            for suf in ("", "_gtLabels", "_mask", "_labels"):
                if (stem + suf) in masks:
                    m = masks[stem + suf]; break
        pairs.append((rgb[stem], m))
    return pairs


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--soiling-dir", required=True, help="WoodScape soiling dir (rgbImages/ + gtLabels/)")
    ap.add_argument("--calib", default=None, help="WoodScape calibration json (default: vendored example)")
    ap.add_argument("--out", default="data/frames/woodscape_soiling")
    ap.add_argument("--obs-out", default="data/cache/trackC_observations_raw.csv")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    root = abspath(args.soiling_dir)
    pairs = _find_pairs(root)
    if args.max_frames:
        pairs = pairs[: args.max_frames]
    if not pairs:
        print(f"ERROR: no rgb/mask pairs under {root}", file=sys.stderr); return 2
    n_with_mask = sum(1 for _, m in pairs if m is not None)
    print(f"Found {len(pairs)} rgb images ({n_with_mask} with masks) under {root}")

    out = abspath(args.out); out.mkdir(parents=True, exist_ok=True)
    obs_out = abspath(args.obs_out); obs_out.parent.mkdir(parents=True, exist_ok=True)
    m1, m2, _ = wt.build_undistort_maps(args.calib)

    with open(out / "manifest.csv", "w", newline="") as mf, open(obs_out, "w", newline="") as of:
        mw = csv.writer(mf); mw.writerow(["frame_idx", "filename", "timestamp_ns", "encoding", "width", "height"])
        ow = csv.writer(of); ow.writerow(["frame_idx", "timestamp_ns", "blur", "illumination", "occlusion", "soiled_frac"])
        for i, (rgb_p, mask_p) in enumerate(pairs):
            fish = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
            if fish is None:
                print(f"  skip unreadable {rgb_p.name}"); continue
            mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED) if mask_p else np.zeros(fish.shape[:2], np.uint8)
            if mask is None:
                mask = np.zeros(fish.shape[:2], np.uint8)
            # undistort for the detector (rectilinear frames)
            rect = wt.undistort(fish, m1, m2)
            fn = f"frame_{i:06d}.png"
            cv2.imwrite(str(out / fn), rect)
            ts = int(i * (1e9 / 10.0))  # ordered series; synth 10 "fps" timeline (see caveat)
            mw.writerow([i, fn, ts, "bgr8", rect.shape[1], rect.shape[0]])
            # fisheye-correct observations on the RAW fisheye + mask
            obs = wt.observations_for_frame(fish, mask)
            ow.writerow([i, ts, obs["blur"], obs["illumination"], obs["occlusion"],
                         round(1.0 - obs["occlusion"], 4)])
            if (i + 1) % 100 == 0:
                print(f"  {i+1} frames...")

    print(f"DONE: {len(pairs)} frames -> {out}; observations -> {obs_out}")
    print("NOTE: WoodScape soiling is a frame collection; the temporal axis is frame ORDER,"
          " so persistence/onset structure is the dataset ordering (documented limitation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
