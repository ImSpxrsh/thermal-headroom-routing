"""Pseudo-ground-truth agreement metric.

Task accuracy of a cheaper config is the AGREEMENT of its detections with the
reference config (C1) on the same frame, matched by IoU per class. This is
agreement-with-reference, NOT agreement with external truth: any systematic error
in C1 is invisible here. Stated wherever used (see config.yaml meta.pseudo_gt_note).
"""
from __future__ import annotations

import numpy as np

from detector import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between box sets a (Na,4) and b (Nb,4), xyxy."""
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


def agreement_f1(pred: Detection, ref: Detection, iou_thr: float = 0.5) -> dict:
    """Greedy per-class IoU matching of pred against ref (ref treated as GT)."""
    tp = 0
    matched_ref = set()
    order = np.argsort(-pred.conf) if len(pred.conf) else np.array([], int)
    iou = iou_matrix(pred.xyxy, ref.xyxy)
    for pi in order:
        best_j, best_iou = -1, iou_thr
        for rj in range(len(ref.cls)):
            if rj in matched_ref or ref.cls[rj] != pred.cls[pi]:
                continue
            if iou[pi, rj] >= best_iou:
                best_iou = iou[pi, rj]; best_j = rj
        if best_j >= 0:
            tp += 1; matched_ref.add(best_j)
    n_pred, n_ref = len(pred.cls), len(ref.cls)
    fp = n_pred - tp
    fn = n_ref - tp
    prec = tp / max(n_pred, 1)
    rec = tp / max(n_ref, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # define agreement as 1.0 when both empty (cheap config correctly agrees "nothing")
    if n_pred == 0 and n_ref == 0:
        prec = rec = f1 = 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "n_pred": n_pred, "n_ref": n_ref}


def mean_agreement(rows: list) -> dict:
    """Aggregate per-frame agreement dicts into mean precision/recall/f1."""
    if not rows:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    return {
        "precision": float(np.mean([r["precision"] for r in rows])),
        "recall": float(np.mean([r["recall"] for r in rows])),
        "f1": float(np.mean([r["f1"] for r in rows])),
    }
