"""Systematic injection sweep -> RQ-H (coupling x onset-rate) phase diagram (Part 2).

Reuses an ALREADY-PROFILED track's frontier (outputs/<track>/phase3 latency distributions
+ per-frame accuracy; no re-profiling). For each cell of a (coupling-strength x onset-rate)
grid it builds a CONTROLLED substrate: a synthetic fault timeline at the given onset rate,
a contention schedule coupled at the given strength, belief signals from the synthetic
timeline, and per-(frame,config) realized latency drawn from the real profiled
distributions. It then runs the joint vs decoupled policies and records the deadline-miss
reduction (with 95% CIs over seeds) and the ACHIEVED fault-onset count and coupling
correlation. Output: a phase-diagram heatmap + json.

This is labeled controlled/synthetic (like Track B): it sweeps the regime parameters to
show WHERE the coupling helps, anchored to a real latency/accuracy profile.

    python src/run_injection_sweep.py --track outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="outputs")
    ap.add_argument("--n-frames", type=int, default=1500, help="synthetic timeline length")
    args = ap.parse_args()
    os.environ["OUTPUTS_DIR"] = args.track

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath
    import simulate as sim
    import policies as pol
    import sensor_belief as sb
    import compute_belief as cb
    import contention as cont
    import ci as cistats

    cfg = load_config()
    keys = list(cfg["configs"].keys())
    pol.CONFIG_KEYS = keys
    out_dir = abspath(args.track)
    p3sum = out_dir / "phase3" / "phase3_summary.json"
    if not p3sum.exists():
        print(f"ERROR: {args.track} has no profiled frontier (phase3)."); return 2
    import json as _json
    deadline_s = float(_json.load(open(p3sum))["deadline_ms"]) / 1e3  # THIS track's deadline
    latnpz = np.load(out_dir / "phase3" / "latency_distributions.npz")
    accdf = pd.read_csv(out_dir / "phase3" / "per_frame_accuracy.csv")
    lab = pd.read_csv(out_dir / "phase1" / "trackA_observations.csv")
    Tprof = min(len(lab), int(accdf["frame_idx"].max()) + 1)
    fa_real = lab["any_fault"].to_numpy(bool)[:Tprof]
    # accuracy buckets per config (nominal vs faulted) from the real profile
    acc_nom, acc_flt = {}, {}
    for c in keys:
        a = accdf[accdf["config"] == c].set_index("frame_idx")["f1"].reindex(range(Tprof)).to_numpy()
        acc_nom[c] = float(np.nanmean(a[~fa_real])) if (~fa_real).any() else float(np.nanmean(a))
        acc_flt[c] = float(np.nanmean(a[fa_real])) if fa_real.any() else float(np.nanmean(a))
    lat_dists = {c: {"nominal": latnpz[f"{c}_nominal"], "contended": latnpz[f"{c}_contended"]} for c in keys}
    fm = pol.build_frontier_model(deadline_s, lat_dists, acc_nom, acc_flt)
    rt = cfg["routing"]["reliability_target"]
    dwell = cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]
    lag = cfg["regimes"]["coupled"]["onset_to_contention_lag_frames"]
    fps = cfg["dataset"]["nominal_fps"]; n = args.n_frames; seeds = cfg["seeds"]

    # grid: coupling strength (P(cont|fault)-P(cont|nominal)) x onset rate (events/min)
    COUPLINGS = [0.0, 0.2, 0.4, 0.6, 0.8]
    ONSET_RATES = [2.0, 6.0, 12.0, 24.0]   # events per minute

    def gen_fault(lam_per_min, seed):
        rng = np.random.default_rng(4000 + seed)
        p_on = lam_per_min / (60.0 * fps)          # onset prob per frame
        p_off = 0.08
        f = np.zeros(n, bool); st = False
        for t in range(n):
            st = (not (rng.random() < p_off)) if st else (rng.random() < p_on)
            f[t] = st
        return f

    def gen_contention(fault, strength, seed):
        rng = np.random.default_rng(5000 + seed)
        p_nom = 0.05; p_flt = min(1.0, p_nom + strength)
        lead = np.zeros(n, bool); lead[lag:] = fault[:n - lag]
        return rng.random(n) < np.where(lead, p_flt, p_nom)

    def belief_signals(fault, state, seed):
        rng = np.random.default_rng(6000 + seed)
        obs = rng.normal(0, 1, n) - 3.0 * fault
        hmm = sb.fit_channel_hmm(obs, fault, cfg, "low")
        sbel = sb.p_faulted(sb.belief_for_channel(obs, hmm, cfg))
        # compute belief from a probe-config latency monitor
        rng2 = np.random.default_rng(7000 + seed)
        probe = keys[len(keys) // 2]
        plat = np.where(state, rng2.choice(lat_dists[probe]["contended"], n),
                        rng2.choice(lat_dists[probe]["nominal"], n))
        st = cont.sliding_window_stats(plat, cfg["contention"]["latency_window_frames"], fps)
        feats = np.column_stack([st["p95_s"], st["p99_s"], st["queue_depth"]])
        fs, _ = cb.standardize(feats)
        chmm = cb.fit_compute_hmm(fs, state, cfg)
        return sbel, cb.p_contended(chmm.filter(fs)), np.repeat(sbel[:, None], 3, 1)

    grid = np.full((len(COUPLINGS), len(ONSET_RATES)), np.nan)
    onsets_grid = np.zeros_like(grid)
    detail = []
    for i, strength in enumerate(COUPLINGS):
        for j, lam in enumerate(ONSET_RATES):
            diffs, onset_counts = [], []
            for seed in seeds:
                fault = gen_fault(lam, seed)
                state = gen_contention(fault, strength, seed)
                sbel, cbel, sbch = belief_signals(fault, state, seed)
                rng = np.random.default_rng(8000 + seed)
                L = {c: np.where(state, rng.choice(lat_dists[c]["contended"], n),
                                 rng.choice(lat_dists[c]["nominal"], n)) for c in keys}
                acc = {c: np.where(fault, acc_flt[c], acc_nom[c]) for c in keys}
                kap = pol.fit_coupling(fault, state, lag)
                sub = sim.Substrate(T=n, s_belief=sbel, s_instant=sbel, c_belief=cbel,
                                    c_instant=cbel, state=state, fault_active=fault, L=L, acc=acc,
                                    fm=fm, kappa=kap, deadline_s=deadline_s, p_self_transition=0.95,
                                    s_belief_channels=sbch)
                jm = sim.run_policy(pol.JointPolicy(fm, kap, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"]
                dm = sim.run_policy(pol.DecoupledPolicy(fm, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"]
                diffs.append(dm - jm)
                onset_counts.append(int(np.sum(fault[1:] & ~fault[:-1])))
            c = cistats.mean_ci(diffs)
            grid[i, j] = c["mean"]; onsets_grid[i, j] = np.mean(onset_counts)
            detail.append({"coupling": strength, "onset_per_min": lam,
                           "miss_reduction": c["mean"], "ci_lo": c["lo"], "ci_hi": c["hi"],
                           "significant": bool(c["lo"] > 0 or c["hi"] < 0),
                           "avg_onsets": float(np.mean(onset_counts))})

    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump({"track": args.track, "couplings": COUPLINGS, "onset_rates": ONSET_RATES,
               "grid_miss_reduction_pp": (grid * 100).tolist(), "cells": detail},
              open(extras / "phase_diagram.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(grid * 100, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ONSET_RATES))); ax.set_xticklabels(ONSET_RATES)
    ax.set_yticks(range(len(COUPLINGS))); ax.set_yticklabels(COUPLINGS)
    ax.set_xlabel("fault onset rate (events/min)"); ax.set_ylabel("coupling strength")
    ax.set_title(f"RQ-H phase diagram: joint-vs-decoupled deadline-miss reduction (pp)\n{args.track}")
    for i in range(len(COUPLINGS)):
        for j in range(len(ONSET_RATES)):
            ax.text(j, i, f"{grid[i,j]*100:.1f}", ha="center", va="center",
                    color="white", fontsize=8)
    fig.colorbar(im, label="miss reduction (pp)")
    fig.tight_layout(); fig.savefig(extras / "phase_diagram.png", dpi=140); plt.close(fig)
    print(f"phase diagram -> {extras}/phase_diagram.png")
    print("row=coupling, col=onset/min, value=joint-vs-decoupled miss reduction (pp):")
    print(np.round(grid * 100, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
