"""Track D: turn a RADIATE sequence into the pipeline's frame format.

RADIATE (Heriot-Watt, CC BY-NC-SA 4.0) is real adverse-weather driving video. We use
only the LEFT ZED camera, rectified to `camera_left_rect` exactly as the radiate_sdk
does (stereoRectify + initUndistortRectifyMap + remap, calibration vendored in
config/radiate-calib.yaml). Radar, lidar, and the provided annotations are NOT used:
pseudo-GT is YOLO11x on the camera frames, the same as every other track.

Output is the SAME PNG + manifest.csv format as extract_frames.py, so the existing
Phases 1-6 run on Track D unchanged (point frames_dir / FRAMES_DIR at the output).

    python src/extract_radiate.py --seq-dir data/radiate/tiny_foggy
    python src/extract_radiate.py --seq-dir data/radiate/rain_4_0 --max-frames 0
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import abspath, REPO_ROOT


def _load_calib(path: Path):
    with open(path) as f:
        c = yaml.safe_load(f)
    L, R, S = c["left_cam_calib"], c["right_cam_calib"], c["stereo_calib"]
    left_mat = np.array([[L["fx"], 0, L["cx"]], [0, L["fy"], L["cy"]], [0, 0, 1]])
    left_dist = np.array([L["k1"], L["k2"], L["p1"], L["p2"]], dtype=np.float64)
    right_mat = np.array([[R["fx"], 0, R["cx"]], [0, R["fy"], R["cy"]], [0, 0, 1]])
    right_dist = np.array([R["k1"], R["k2"], R["p1"], R["p2"]], dtype=np.float64)
    res = tuple(L["res"])
    stereoR = np.array(S["R"], dtype=np.float64)
    stereoT = np.array([S["TX"], S["TY"], S["TZ"]], dtype=np.float64)
    return left_mat, left_dist, right_mat, right_dist, res, stereoR, stereoT


def _left_rect_maps(calib_path: Path):
    import cv2
    lm, ld, rm, rd, res, sR, sT = _load_calib(calib_path)
    leftRect, _, leftProj, _, _, _, _ = cv2.stereoRectify(
        cameraMatrix1=lm, distCoeffs1=ld, cameraMatrix2=rm, distCoeffs2=rd,
        imageSize=res, R=sR, T=sT, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    mapx, mapy = cv2.initUndistortRectifyMap(lm, ld, leftRect, leftProj, res, cv2.CV_32FC1)
    return mapx, mapy


def _timestamps(seq_dir: Path, n: int) -> list:
    """Parse zed_left.txt ('Frame: NNNNNN Time: <epoch.sec>') -> ns ints. Fallback 15 fps."""
    tf = seq_dir / "zed_left.txt"
    ts = []
    if tf.exists():
        for line in tf.read_text().splitlines():
            if "Time:" in line:
                try:
                    ts.append(int(float(line.split("Time:")[1].strip()) * 1e9))
                except Exception:
                    pass
    if len(ts) < n:  # fallback / pad at 15 fps
        ts = [int(i * (1e9 / 15.0)) for i in range(n)]
    return ts[:n]


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="RADIATE sequence dir (contains zed_left/)")
    ap.add_argument("--out", default=None, help="frames output dir (default data/frames/radiate_<seq>)")
    ap.add_argument("--calib", default="config/radiate-calib.yaml")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--no-rectify", action="store_true", help="use raw left frames (not recommended)")
    args = ap.parse_args()

    seq = abspath(args.seq_dir)
    left = seq / "zed_left"
    if not left.is_dir():
        # allow pointing directly at a folder of camera PNGs (e.g. a downloaded
        # zed_left/ on its own, without the parent sequence structure)
        if sorted(seq.glob("*.png")):
            left = seq
        else:
            print(f"ERROR: no zed_left/ and no *.png under {seq}", file=sys.stderr); return 2
    frames = sorted(left.glob("*.png"))
    if args.max_frames:
        frames = frames[: args.max_frames]
    if not frames:
        print(f"ERROR: no PNGs in {left}", file=sys.stderr); return 3

    out = abspath(args.out) if args.out else (REPO_ROOT / "data" / "frames" / f"radiate_{seq.name}")
    out.mkdir(parents=True, exist_ok=True)
    ts = _timestamps(seq, len(frames))

    mapx = mapy = None
    if not args.no_rectify:
        mapx, mapy = _left_rect_maps(abspath(args.calib))
        print(f"Rectifying left frames to camera_left_rect ({seq.name})")
    else:
        print("Using RAW left frames (no rectification)")

    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["frame_idx", "filename", "timestamp_ns", "encoding", "width", "height"])
        for i, fp in enumerate(frames):
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  skip unreadable {fp.name}"); continue
            if mapx is not None:
                img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            fn = f"frame_{i:06d}.png"
            cv2.imwrite(str(out / fn), img)
            w.writerow([i, fn, ts[i], "bgr8", img.shape[1], img.shape[0]])
    print(f"DONE: {len(frames)} frames -> {out}")
    print(f"Run the pipeline on Track D with:  FRAMES_DIR={out.relative_to(REPO_ROOT)} python src/run_pipeline.py --skip-extract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
