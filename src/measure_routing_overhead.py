"""Measure the wall-clock cost of the ROUTING ITSELF (belief update + noisy-OR
fusion + config selection), separate from the underlying YOLO inference, and
report it as an absolute time and a percentage of the reference config's
median latency. Answers "how much does running the router cost, on top of
whatever inference config it picks?"

Two numbers, both on CPU (the router never touches the GPU):
  1. Belief-fit + full-trace filtering (sensor HMMs x3 channels + compute HMM),
     amortized per frame -- an UPPER BOUND, since HMM parameter fitting is a
     one-time calibration cost in a real deployment, not paid every frame.
  2. Policy decision loop only (JointPolicy.decide() over the trace, given
     already-computed beliefs) -- the representative STEADY-STATE per-frame
     routing cost once beliefs are already being tracked online.

    python src/measure_routing_overhead.py --track outputs/trackD_fog_6_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="outputs")
    ap.add_argument("--reps", type=int, default=5, help="repeats for stable timing")
    args = ap.parse_args()
    os.environ["OUTPUTS_DIR"] = args.track

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath
    import simulate as sim
    import sensor_belief as sb
    import policies as pol

    cfg = load_config()
    out_dir = abspath(args.track)
    if not (out_dir / "phase3" / "phase3_summary.json").exists():
        print(f"ERROR: {args.track} has no profiled frontier (phase3)."); return 2
    seeds = cfg["seeds"]
    rt = cfg["routing"]["reliability_target"]
    dwell = cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]

    lab, acc_df, _ = sim.load_phase_outputs(cfg)
    T = len(lab)

    # --- (1) belief fit + full-trace filter, amortized per frame (upper bound) ---
    fit_filter_times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        hmms = sb.fit_all_channels(lab, cfg)
        alpha = cfg["fault_labeling"]["ema_alpha"]
        for ch in ("blur", "illumination", "occlusion"):
            obs = sb._robust_z(sb._ema(lab[ch].to_numpy(float), alpha))
            hmms[ch].filter(obs)
        fit_filter_times.append(time.perf_counter() - t0)
    fit_filter_s = float(np.median(fit_filter_times))

    # --- (2) policy decision loop only, given already-computed beliefs ---
    sub = sim.build_substrate(cfg, regime="coupled", seed=seeds[0])
    decide_times = []
    for _ in range(args.reps):
        joint = pol.JointPolicy(sub.fm, sub.kappa, rt, pol.Hysteresis(dwell))
        t0 = time.perf_counter()
        for t in range(sub.T):
            joint.decide(sub.s_belief[t], sub.c_belief[t])
        decide_times.append(time.perf_counter() - t0)
    decide_s = float(np.median(decide_times))

    ref_config = list(cfg["configs"].keys())[0]  # C1 by convention
    latnpz = np.load(out_dir / "phase3" / "latency_distributions.npz")
    ref_median_ms = float(np.median(latnpz[f"{ref_config}_nominal"])) * 1e3

    fit_filter_per_frame_us = fit_filter_s / T * 1e6
    decide_per_frame_us = decide_s / T * 1e6

    summary = {
        "track": args.track, "T": T, "reps": args.reps,
        "reference_config": ref_config, "reference_median_latency_ms": ref_median_ms,
        "belief_fit_and_filter_total_s": fit_filter_s,
        "belief_fit_and_filter_per_frame_us": fit_filter_per_frame_us,
        "belief_fit_and_filter_pct_of_reference_latency": 100.0 * (fit_filter_per_frame_us / 1e3) / ref_median_ms,
        "policy_decide_total_s": decide_s,
        "policy_decide_per_frame_us": decide_per_frame_us,
        "policy_decide_pct_of_reference_latency": 100.0 * (decide_per_frame_us / 1e3) / ref_median_ms,
        "note": ("belief_fit_and_filter is an UPPER BOUND (includes one-time HMM "
                 "parameter fitting, amortized over the trace, not just incremental "
                 "filtering); policy_decide is the representative steady-state "
                 "per-frame routing cost given beliefs already being tracked. Both "
                 "measured on CPU (Apple M5); the router never touches the GPU."),
    }
    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(extras / "routing_overhead.json", "w"), indent=2, default=str)

    print(f"\n{'='*70}\n[routing overhead] {args.track}  (T={T} frames, reference {ref_config} "
          f"median {ref_median_ms:.1f} ms)\n{'='*70}")
    print(f"belief fit+filter (upper bound): {fit_filter_per_frame_us:.1f} us/frame "
          f"({summary['belief_fit_and_filter_pct_of_reference_latency']:.4f}% of reference latency)")
    print(f"policy decide only (steady-state): {decide_per_frame_us:.1f} us/frame "
          f"({summary['policy_decide_pct_of_reference_latency']:.4f}% of reference latency)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
