"""Real-label validation of the pseudo-GT accuracy metric on the annotated RADIATE
subset (addresses reviewer note #2: accuracy is agreement-with-C1, not external GT).

This is a SCAFFOLD: it must run where the RADIATE annotation files exist (the friend's
machine), since the public sample on this repo has no labels committed. It does NOT
change any headline result; it produces a single validation correlation showing whether
agreement-with-reference (pseudo-GT) tracks real detection quality (mAP/F1 vs GT).

What it does:
  1. For each annotated RADIATE frame, run every config Ck and the reference C1.
  2. Compute (a) pseudo-GT: IoU-matched F1 of Ck vs C1 (the metric used in the paper),
     and (b) real F1/AP: IoU-matched F1 of Ck vs the RADIATE ground-truth boxes.
  3. Report Pearson/Spearman correlation between per-frame pseudo-GT and real F1, plus a
     scatter. High correlation => agreement-with-reference is a valid proxy for quality.

RADIATE annotations: each sequence has annotations.json (or per-frame label files) with
object boxes in the left-camera frame. Map them through the same rectification used in
src/extract_radiate.py (camera_left_rect) before matching to detections.

    python src/validate_pseudo_gt.py --seq-dir data/radiate/rain_4_0 --max-frames 500

Outputs: outputs/validation/pseudo_gt_vs_real.{json,png}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def iou_matched_f1(pred_boxes, gt_boxes, iou_thr=0.5):
    """IoU-matched per-frame F1 (greedy 1-1 match), classes ignored for simplicity here;
    extend to per-class to mirror agreement.py exactly."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return np.nan
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    matched_gt = set()
    tp = 0
    for p in pred_boxes:
        best, bj = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            v = iou(p, g)
            if v > best:
                best, bj = v, j
        if best >= iou_thr and bj >= 0:
            matched_gt.add(bj); tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def load_radiate_gt(seq_dir: Path, frame_idx: int):
    """Return ground-truth boxes [[x1,y1,x2,y2],...] for a frame, rectified to
    camera_left_rect. STUB: implement using the sequence's annotation file. RADIATE
    annotations are in annotations/annotations.json or per-frame .txt; map object
    polygons/boxes through the same K,D,R used in extract_radiate.py."""
    raise NotImplementedError(
        "Wire this to the RADIATE annotation files on the machine that has them. "
        "Parse annotations.json -> boxes for frame_idx -> rectify with the calib in "
        "src/extract_radiate.py (camera_left_rect).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="RADIATE sequence dir with annotations")
    ap.add_argument("--max-frames", type=int, default=500)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config
    from detector import load_detector  # whatever the repo uses to run YOLO configs

    cfg = load_config()
    ref = cfg["reference_config"]
    keys = [k for k in cfg["configs"] if k != ref]

    seq = Path(args.seq_dir)
    print(f"Validating pseudo-GT vs real labels on {seq} (ref={ref}, configs={keys})")
    print("NOTE: requires RADIATE annotation files; load_radiate_gt() is a stub to wire up.")

    pseudo, real = {k: [] for k in keys}, {k: [] for k in keys}
    # frame loop (pseudo-code; fill in detector calls mirroring profile_frontier.py):
    #   for idx, img in enumerate(frames[:max_frames]):
    #       det_ref = detect(ref, img)
    #       gt = load_radiate_gt(seq, idx)
    #       for k in keys:
    #           det_k = detect(k, img)
    #           pseudo[k].append(iou_matched_f1(det_k, det_ref))   # vs reference (paper metric)
    #           real[k].append(iou_matched_f1(det_k, gt))          # vs real GT
    raise SystemExit("Scaffold only: implement detect()/load_radiate_gt() on the "
                     "annotated-data machine, then this writes outputs/validation/.")


if __name__ == "__main__":
    raise SystemExit(main())
