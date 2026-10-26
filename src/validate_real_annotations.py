"""RQ-Coupling (d): validate the pseudo-GT accuracy axis against REAL RADIATE
annotations on one dataset (Track D).

Every accuracy number elsewhere in this paper is "agreement with C1" (pseudo-GT):
a cheaper config's F1 against C1's own output on the same frame, not against
independent ground truth. This script computes REAL detection F1 (class-agnostic,
IoU>=0.5, greedy-matched) against RADIATE's real radar-derived object annotations
(projected onto the camera image via radiate_annotations.RadiateProjector), for all
four frontier configs, and checks:
  (a) does the real-GT accuracy ranking across configs match the pseudo-GT ranking
      (does the heavier/lighter config ordering the paper assumes actually hold)?
  (b) how well does per-frame pseudo-GT F1 correlate with per-frame real-GT F1?
This validates -- or flags problems with -- the accuracy axis used throughout the
paper, on real ground truth, without requiring a full re-profiling of the RQ-H
substrate (which would need per-config latency/accuracy matrices rebuilt on
real-GT and is left as future work).

    python src/validate_real_annotations.py --track outputs/trackD_rain_4_0 \
        --seq-dir data/radiate/rain_4_0 --frames data/frames/radiate_rain_4_0 \
        --max-frames 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


# RADIATE annotates ONLY vehicles (car/van/bus/truck seen in this sequence); scoring
# ALL YOLO/COCO classes against a vehicle-only GT would count every detected
# pedestrian/cyclist/etc as a false positive it was never asked to find. We restrict
# predictions to COCO's vehicle-like classes before matching (car=2, motorcycle=3,
# bus=5, truck=7; "van" has no COCO class and is left to fall under car/truck as YOLO
# itself would predict it), then match class-agnostically within that restricted set
# (RADIATE's van/car/bus/truck boundary doesn't map cleanly onto COCO's, so we do not
# require an exact class match, only "is this GT vehicle" vs "is this a predicted
# vehicle").
VEHICLE_COCO_CLASSES = {2, 3, 5, 7}


def class_agnostic_f1(pred_xyxy, pred_conf, gt_xyxy, iou_thr=0.5):
    """Greedy IoU matching, ignoring class labels (RADIATE's {car,van,bus,truck}
    taxonomy doesn't map 1:1 onto COCO, so we validate localization/detection
    agreement rather than attempting a class mapping)."""
    n_pred, n_gt = len(pred_xyxy), len(gt_xyxy)
    if n_pred == 0 and n_gt == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "n_pred": 0, "n_gt": 0}
    iou = iou_matrix(pred_xyxy, gt_xyxy)
    order = np.argsort(-pred_conf) if n_pred else np.array([], int)
    matched = set()
    tp = 0
    for pi in order:
        best_j, best_iou = -1, iou_thr
        for gj in range(n_gt):
            if gj in matched:
                continue
            if iou[pi, gj] >= best_iou:
                best_iou = iou[pi, gj]; best_j = gj
        if best_j >= 0:
            tp += 1; matched.add(best_j)
    prec = tp / max(n_pred, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "n_pred": n_pred, "n_gt": n_gt}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--seq-dir", required=True, help="raw RADIATE sequence dir (has annotations/)")
    ap.add_argument("--frames", required=True, help="extracted frames dir for this track")
    ap.add_argument("--max-frames", type=int, default=1500)
    args = ap.parse_args()
    os.environ["OUTPUTS_DIR"] = args.track
    os.environ["FRAMES_DIR"] = args.frames

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath, resolve_device
    from data_harness import FrameSource
    from detector import build_detector
    from radiate_annotations import RadiateProjector

    cfg = load_config()
    device = resolve_device(cfg)
    out_dir = abspath(args.track)
    proj = RadiateProjector(args.seq_dir)

    src = FrameSource(cfg=cfg)
    n = min(args.max_frames, len(src))
    frames = [fr.image for fr in src.iter(stop=n)]
    print(f"[real-annotations] {args.track}: {n} frames, projecting real RADIATE annotations")

    gt_boxes = [proj.boxes_for_camera_frame(i) for i in range(n)]
    n_with_gt = sum(1 for b in gt_boxes if b)
    print(f"  {n_with_gt}/{n} frames have >=1 projected real annotation")

    keys = list(cfg["configs"].keys())
    per_config_f1 = {c: [] for c in keys}
    pseudo_gt_df = pd.read_csv(out_dir / "phase3" / "per_frame_accuracy.csv")

    for c in keys:
        det = build_detector(cfg, c, device).load()
        det.warmup(frames[0])
        for i, img in enumerate(frames):
            d = det.predict(img, conf=float(cfg["profiling"]["conf_threshold"]))
            veh_mask = np.isin(d.cls, list(VEHICLE_COCO_CLASSES))
            pred_xyxy, pred_conf = d.xyxy[veh_mask], d.conf[veh_mask]
            gt = np.array([[b[0], b[1], b[2], b[3]] for b in gt_boxes[i]]) if gt_boxes[i] else np.zeros((0, 4))
            res = class_agnostic_f1(pred_xyxy, pred_conf, gt)
            per_config_f1[c].append(res["f1"])
        print(f"  {c}: real-GT F1 mean={np.mean(per_config_f1[c]):.3f}")

    # --- (a) ranking check: does real-GT accuracy order the configs the same way
    #     as pseudo-GT (agreement-with-C1)? ---
    pseudo_mean = {c: float(pseudo_gt_df[pseudo_gt_df["config"] == c]["f1"].iloc[:n].mean())
                   for c in keys}
    real_mean = {c: float(np.mean(per_config_f1[c])) for c in keys}
    rank_pseudo = sorted(keys, key=lambda c: -pseudo_mean[c])
    rank_real = sorted(keys, key=lambda c: -real_mean[c])

    # --- (b) per-frame correlation between pseudo-GT F1 and real-GT F1, per config ---
    corr = {}
    for c in keys:
        pf = pseudo_gt_df[pseudo_gt_df["config"] == c].set_index("frame_idx")["f1"].reindex(range(n)).to_numpy()
        rf = np.array(per_config_f1[c])
        mask = ~np.isnan(pf) & ~np.isnan(rf)
        corr[c] = float(np.corrcoef(pf[mask], rf[mask])[0, 1]) if mask.sum() > 2 else float("nan")

    summary = {
        "track": args.track, "n_frames": n, "n_frames_with_real_gt": n_with_gt,
        "real_gt_f1_mean": real_mean, "pseudo_gt_f1_mean": pseudo_mean,
        "ranking_real_gt": rank_real, "ranking_pseudo_gt": rank_pseudo,
        "ranking_matches": rank_real == rank_pseudo,
        "per_frame_corr_pseudo_vs_real": corr,
    }
    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(extras / "real_annotations_validation.json", "w"), indent=2, default=str)

    print(f"\n{'='*70}\n[real-annotations validation] {args.track}\n{'='*70}")
    print(f"real-GT F1:    " + "  ".join(f"{c}={real_mean[c]:.3f}" for c in keys))
    print(f"pseudo-GT F1:  " + "  ".join(f"{c}={pseudo_mean[c]:.3f}" for c in keys))
    print(f"ranking real-GT:   {rank_real}")
    print(f"ranking pseudo-GT: {rank_pseudo}  (match={rank_real == rank_pseudo})")
    print(f"per-config pseudo-vs-real correlation: " + "  ".join(f"{c}={corr[c]:.3f}" for c in keys))
    print(f"  -> {extras / 'real_annotations_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
