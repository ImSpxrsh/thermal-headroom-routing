"""Phase 2 driver: fit the per-channel sensor-fault belief estimator and validate
belief tracking on both tracks. Produces one belief-vs-fault plot per channel per
track in outputs/phase2/ and reports detection lag and frame-level agreement.

Track A validation: belief vs the deterministic threshold-and-smooth labels.
Track B validation: belief vs the KNOWN injected timeline (clean lag measurement).

Run after Phase 1:
    python src/phase2_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath
import sensor_belief as sb

CHANS = ["blur", "illumination", "occlusion"]


def _segments(b: np.ndarray):
    segs = []
    in_seg = False; s0 = 0
    for k in range(len(b)):
        if b[k] and not in_seg:
            in_seg = True; s0 = k
        elif not b[k] and in_seg:
            in_seg = False; segs.append((s0, k - 1))
    if in_seg:
        segs.append((s0, len(b) - 1))
    return segs


def detection_lag(p_fault: np.ndarray, gt: np.ndarray, thr: float):
    """Median frames from each ground-truth onset to first belief>=thr within the
    segment. Returns (median_lag or nan, n_detected, n_segments)."""
    lags = []
    detected = 0
    segs = _segments(gt)
    for s0, s1 in segs:
        cross = np.where(p_fault[s0:s1 + 1] >= thr)[0]
        if len(cross):
            lags.append(int(cross[0])); detected += 1
    med = float(np.median(lags)) if lags else float("nan")
    return med, detected, len(segs)


def frame_agreement(p_fault: np.ndarray, gt: np.ndarray, thr: float) -> dict:
    pred = p_fault >= thr
    tp = int((pred & gt).sum()); tn = int((~pred & ~gt).sum())
    fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    return {"balanced_acc": 0.5 * (tpr + tnr), "tpr": tpr, "tnr": tnr,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _plot(track, ch, fidx, signal, p_fault, gt, out):
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(fidx, signal, color="tab:gray", lw=0.8)
    for s0, s1 in _segments(gt):
        ax[0].axvspan(fidx[s0], fidx[s1], color="red", alpha=0.15)
    ax[0].set_ylabel(f"{ch} signal")
    ax[1].plot(fidx, p_fault, color="tab:purple", lw=1.1, label="P(faulted+degrading)")
    for s0, s1 in _segments(gt):
        ax[1].axvspan(fidx[s0], fidx[s1], color="red", alpha=0.15,
                      label="ground-truth fault" if s0 == _segments(gt)[0][0] else None)
    ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_ylim(-0.02, 1.02); ax[1].set_ylabel("belief"); ax[1].set_xlabel("frame")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"{track}: {ch} belief tracking")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main() -> int:
    cfg = load_config()
    thr = cfg["sensor_hmm"]["decision_threshold"]
    p1 = abspath(cfg["paths"]["outputs_dir"]) / "phase1"
    out = abspath(cfg["paths"]["outputs_dir"]) / "phase2"
    out.mkdir(parents=True, exist_ok=True)

    dfa = pd.read_csv(p1 / "trackA_observations.csv")
    hmms = sb.fit_all_channels(dfa, cfg)

    results = {"trackA": {}, "trackB": {}}
    fidx_a = dfa["frame_idx"].to_numpy()
    for ch in CHANS:
        belief = sb.belief_for_channel(dfa[ch].to_numpy(float), hmms[ch], cfg)
        pf = sb.p_faulted(belief)
        gt = dfa[f"{ch}_fault"].to_numpy(bool)
        lag, det, nseg = detection_lag(pf, gt, thr)
        agree = frame_agreement(pf, gt, thr)
        results["trackA"][ch] = {"median_lag_frames": lag, "segments": nseg,
                                 "detected": det, **agree}
        _plot("Track A (real)", ch, fidx_a, dfa[ch].to_numpy(), pf, gt,
              out / f"trackA_belief_{ch}.png")

    # Track B: fit per-channel from the KNOWN injected timeline, measure clean lag
    if (p1 / "trackB_observations.csv").exists():
        dfb = pd.read_csv(p1 / "trackB_observations.csv")
        fidx_b = dfb["frame_idx"].to_numpy()
        for ch in CHANS:
            gtcol = f"gt_{ch}"
            if gtcol not in dfb:
                continue
            gt = dfb[gtcol].to_numpy(bool)
            direction = cfg["fault_labeling"]["direction"][ch]
            hmm = sb.fit_channel_hmm(dfb[ch].to_numpy(float), gt, cfg, direction)
            pf = sb.p_faulted(sb.belief_for_channel(dfb[ch].to_numpy(float), hmm, cfg))
            lag, det, nseg = detection_lag(pf, gt, thr)
            agree = frame_agreement(pf, gt, thr)
            results["trackB"][ch] = {"median_lag_frames": lag, "segments": nseg,
                                     "detected": det, **agree}
            _plot("Track B (validation, known timeline)", ch, fidx_b,
                  dfb[ch].to_numpy(), pf, gt, out / f"trackB_belief_{ch}.png")

    with open(out / "phase2_tracking.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
