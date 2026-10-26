"""Oracle-infeasibility labeling and abstention scoring (Phase 4).

Post hoc, with the full profiled frontier and the realized latencies, a frame is
oracle-infeasible if NO config would have met both constraints that frame:
  realized latency <= deadline  AND  agreement-with-C1 >= accuracy_floor.
Abstention is scored as precision/recall of the policy's ABSTAIN decisions against
this oracle label. (Latency is agreement-with-reference, so the accuracy floor is on
pseudo-GT; stated as a limitation.)
"""
from __future__ import annotations

import numpy as np

import policies as pol


def oracle_infeasible(sub, accuracy_floor: float) -> np.ndarray:
    T = sub.T
    infeasible = np.ones(T, dtype=bool)
    for c in pol.CONFIG_KEYS:
        meets = (sub.L[c] <= sub.deadline_s) & (np.nan_to_num(sub.acc[c], nan=0.0) >= accuracy_floor)
        infeasible &= ~meets
    return infeasible


def score_abstention(choices, infeasible: np.ndarray) -> dict:
    abstain = np.array([c == pol.ABSTAIN for c in choices], dtype=bool)
    tp = int((abstain & infeasible).sum())
    fp = int((abstain & ~infeasible).sum())
    fn = int((~abstain & infeasible).sum())
    precision = tp / max(tp + fp, 1) if (tp + fp) else float("nan")
    recall = tp / max(tp + fn, 1) if (tp + fn) else float("nan")
    return {"abstain_precision": precision, "abstain_recall": recall,
            "n_infeasible": int(infeasible.sum()), "n_abstain": int(abstain.sum()),
            "tp": tp, "fp": fp, "fn": fn}
