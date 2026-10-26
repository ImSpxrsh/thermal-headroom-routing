"""Per-frame degradation observations and the deterministic fault-labeling rule.

Three raw observations per frame:
  blur         = variance of the Laplacian (high = sharp, low = blurry)
  illumination = Shannon entropy of the grayscale histogram (low = washed out/dark)
  occlusion    = sparse optical-flow feature-track survival rate between
                 consecutive frames (low = occluded / dust / lost texture)

Fault-segment labels come from a single documented threshold-and-smooth rule, the
same for every channel, reported as part of the method (see config.yaml ->
fault_labeling). No hand-labeling anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Raw observations
# ---------------------------------------------------------------------------
def blur_vol(gray: np.ndarray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def illumination_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def occlusion_track_survival(prev_gray: np.ndarray, gray: np.ndarray, cfg: dict) -> float:
    """Fraction of sparse features detected in prev frame that survive forward-backward
    LK tracking into the current frame. 1.0 = all survive (clean), 0.0 = none."""
    import cv2
    d = cfg["degradation"]["occlusion"]
    pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=d["max_corners"],
        qualityLevel=d["quality_level"],
        minDistance=d["min_distance"],
    )
    if pts is None or len(pts) == 0:
        return 0.0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None)
    fb_err = np.linalg.norm((pts - back).reshape(-1, 2), axis=1)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb_err < d["survival_radius_px"])
    return float(good.sum() / len(pts))


def compute_observations(frame_source, start: int = 0, stop=None) -> pd.DataFrame:
    """Run the three observations over a frame range -> tidy DataFrame."""
    import cv2
    rows = []
    prev_gray = None
    for fr in frame_source.iter(start=start, stop=stop):
        gray = cv2.cvtColor(fr.image, cv2.COLOR_BGR2GRAY)
        blur = blur_vol(gray)
        illum = illumination_entropy(gray)
        if prev_gray is None:
            occ = np.nan
        else:
            occ = occlusion_track_survival(prev_gray, gray, frame_source.cfg)
        rows.append(
            dict(frame_idx=fr.idx, timestamp_ns=fr.timestamp_ns,
                 blur=blur, illumination=illum, occlusion=occ)
        )
        prev_gray = gray
    df = pd.DataFrame(rows)
    # first-frame occlusion is undefined; back-fill with the first valid value
    if df["occlusion"].isna().any():
        df["occlusion"] = df["occlusion"].bfill()
    return df


# ---------------------------------------------------------------------------
# Deterministic threshold-and-smooth fault labeling (part of the method)
# ---------------------------------------------------------------------------
def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else (np.std(x) + 1e-9)
    return (x - med) / scale


def _apply_min_and_hysteresis(raw_bad: np.ndarray, min_len: int, recov_hold: int) -> np.ndarray:
    """Turn a noisy boolean 'bad' signal into clean fault segments:
    drop fault runs shorter than min_len, and require recov_hold consecutive
    good frames before declaring recovery (asymmetric onset/recovery)."""
    n = len(raw_bad)
    state = np.zeros(n, dtype=bool)
    i = 0
    cur = False
    good_run = 0
    for i in range(n):
        if not cur:
            if raw_bad[i]:
                # candidate onset: confirm min_len ahead
                if raw_bad[i:i + min_len].sum() >= min_len:
                    cur = True
                    good_run = 0
        else:
            if raw_bad[i]:
                good_run = 0
            else:
                good_run += 1
                if good_run >= recov_hold:
                    cur = False
        state[i] = cur
    return state


@dataclass
class Segment:
    channel: str
    onset: int       # frame index of fault onset
    end: int         # last faulted frame index
    recovery: int    # first nominal frame after the fault (== end+1 if any)


def label_faults(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, List[Segment]]:
    fl = cfg["fault_labeling"]
    out = df.copy()
    segments: List[Segment] = []
    for ch in ("blur", "illumination", "occlusion"):
        x = out[ch].to_numpy(dtype=np.float64)
        sm = _ema(x, fl["ema_alpha"])
        z = _robust_z(sm)
        direction = fl["direction"][ch]
        raw_bad = (z < -fl["robust_z_threshold"]) if direction == "low" else (z > fl["robust_z_threshold"])
        state = _apply_min_and_hysteresis(
            raw_bad, fl["min_segment_frames"], fl["recovery_hysteresis_frames"]
        )
        out[f"{ch}_z"] = z
        out[f"{ch}_fault"] = state
        # extract segments
        idxs = out["frame_idx"].to_numpy()
        in_seg = False
        s0 = 0
        for k in range(len(state)):
            if state[k] and not in_seg:
                in_seg = True
                s0 = k
            elif not state[k] and in_seg:
                in_seg = False
                segments.append(Segment(ch, int(idxs[s0]), int(idxs[k - 1]),
                                        int(idxs[k])))
        if in_seg:
            segments.append(Segment(ch, int(idxs[s0]), int(idxs[-1]), int(idxs[-1])))
    out["any_fault"] = out[["blur_fault", "illumination_fault", "occlusion_fault"]].any(axis=1)
    return out, segments
