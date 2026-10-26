"""Track B: controlled persistent fault injection with a known ground-truth timeline.

VALIDATION AID ONLY. Track B is never a headline result. It exists so the belief
estimator (Phase 2) and the coupling/persistence measurements (Phases 5-6) can be
checked against a known fault onset/duration timeline. Track A (real degradation)
drives all headline numbers.

The plan (which faults are active on which frames) is generated deterministically
from a seed so runs are reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np


@dataclass
class InjEvent:
    channel: str
    onset: int
    duration: int
    intensity: float

    @property
    def end(self) -> int:
        return self.onset + self.duration - 1


def make_injection_plan(n_frames: int, cfg: dict, seed: int) -> Dict[str, object]:
    """Return {'events': [InjEvent...], 'active': {channel: bool array of len n_frames},
    'ground_truth': bool array (any fault)}. Non-overlapping per channel."""
    rng = np.random.default_rng(seed)
    b = cfg["tracks"]["B_injection"]
    chans = b["fault_types"]
    onset_lo, onset_hi = b["onset_frame_range"]
    dur_lo, dur_hi = b["duration_frame_range"]
    n_events = b["n_events_per_run"]
    intensity = float(b["intensity"])

    active = {c: np.zeros(n_frames, dtype=bool) for c in chans}
    events: List[InjEvent] = []
    for _ in range(n_events):
        ch = chans[rng.integers(0, len(chans))]
        onset = int(rng.integers(min(onset_lo, n_frames - 1),
                                 min(onset_hi, n_frames)))
        dur = int(rng.integers(dur_lo, dur_hi + 1))
        end = min(onset + dur, n_frames)
        if active[ch][onset:end].any():
            continue  # skip overlap to keep the timeline clean
        active[ch][onset:end] = True
        events.append(InjEvent(ch, onset, end - onset, intensity))

    gt = np.zeros(n_frames, dtype=bool)
    for c in chans:
        gt |= active[c]
    return {"events": events, "active": active, "ground_truth": gt}


def apply_injection(image: np.ndarray, active: Dict[str, bool], intensity: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Overlay the active synthetic faults on a single BGR frame."""
    import cv2
    img = image.copy()
    if active.get("blur"):
        k = max(3, int(round(2 * (intensity * 15) + 1)) | 1)  # odd kernel
        img = cv2.GaussianBlur(img, (k, k), sigmaX=intensity * 8)
    if active.get("illumination"):
        # wash out toward a flat low-entropy image: contrast crush + darken
        gain = 1.0 - 0.7 * intensity
        img = np.clip(img.astype(np.float32) * gain + 30 * intensity, 0, 255).astype(np.uint8)
    if active.get("occlusion"):
        # drop trackable texture: paste a soft dark blob over a random region
        h, w = img.shape[:2]
        bw, bh = int(w * 0.45 * intensity + 1), int(h * 0.45 * intensity + 1)
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        patch = img[y0:y0 + bh, x0:x0 + bw]
        blurred = cv2.GaussianBlur(patch, (31, 31), 0)
        img[y0:y0 + bh, x0:x0 + bw] = (0.3 * patch + 0.7 * blurred * 0.4).astype(np.uint8)
    return img


def plan_to_dict(plan: Dict[str, object]) -> dict:
    """JSON-serializable view of a plan for saving alongside outputs."""
    return {
        "events": [asdict(e) for e in plan["events"]],
        "ground_truth_count": int(np.sum(plan["ground_truth"])),
    }
