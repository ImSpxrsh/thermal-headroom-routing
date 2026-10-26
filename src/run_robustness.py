"""Robustness add-ons for an ALREADY-PROFILED track, reusing the committed outputs
(phase1 labels + phase3 latency distributions / per-frame accuracy). CPU only, no GPU,
no datasets. Reproduces the exact RQ-H substrate (sim.build_substrate + calibration
kappa, identical seeds/draws as phase5), then computes, on the same draws:

  - Threshold-rule baseline (hard union p_cont=max(b_c,1[b_f>0.5])) vs joint vs decoupled
    deadline-miss, with paired 95% CIs                    -> paper "noisy-OR necessary?" table
  - On-time utility U = mean over served frames of a_t * 1[lat_t <= D] (a late frame
    scores 0), joint vs decoupled with paired CI          -> paper utility table
  - Moving-block bootstrap (block=50, B=2000) of the per-frame miss reduction, pooled
    across seeds, over the temporal axis                  -> paper bootstrap robustness
  - (fog / --sweeps) deadline-strictness sweep (0.8-1.2x) and kappa sweep (0,0.5,0.75,1)

Writes outputs/<track>/extras/robustness.json and prints a summary block.

    python src/run_robustness.py --track outputs/trackD_fog_6_0 --sweeps
    python src/run_robustness.py --track outputs/trackD_rain_4_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


def slice_sub(sub, lo, hi):
    """A Substrate restricted to frames [lo:hi) (for held-out temporal evaluation)."""
    return replace(
        sub, T=hi - lo,
        s_belief=sub.s_belief[lo:hi], s_instant=sub.s_instant[lo:hi],
        c_belief=sub.c_belief[lo:hi], c_instant=sub.c_instant[lo:hi],
        state=sub.state[lo:hi], fault_active=sub.fault_active[lo:hi],
        L={c: sub.L[c][lo:hi] for c in sub.L}, acc={c: sub.acc[c][lo:hi] for c in sub.acc},
        s_belief_channels=(sub.s_belief_channels[lo:hi]
                           if sub.s_belief_channels is not None else None))


def on_time_utility(choices, sub, pol) -> float:
    """U = mean over served, non-nan-accuracy frames of (a_t if on time else 0)."""
    num, den = 0.0, 0
    for t, ch in enumerate(choices):
        if ch == pol.ABSTAIN:
            continue
        a = sub.acc[ch][t]
        if np.isnan(a):
            continue
        den += 1
        if sub.L[ch][t] <= sub.deadline_s:
            num += float(a)
    return num / den if den else float("nan")


def miss_indicator(choices, sub) -> np.ndarray:
    return np.array([1.0 if sub.L[ch][t] > sub.deadline_s else 0.0
                     for t, ch in enumerate(choices)], dtype=float)


def block_bootstrap_ci(d: np.ndarray, block: int = 50, B: int = 2000, seed: int = 12345):
    """Moving-block bootstrap CI of the mean of a (temporally autocorrelated) series."""
    rng = np.random.default_rng(seed)
    T = len(d)
    if T <= block:
        m = float(d.mean())
        return m, m
    n_blocks = int(np.ceil(T / block))
    starts_max = T - block
    offs = np.arange(block)
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:T]
        means[b] = d[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="outputs")
    ap.add_argument("--sweeps", action="store_true",
                    help="also run deadline + kappa sweeps (auto-on for fog tracks)")
    ap.add_argument("--noise-sweep", action="store_true",
                    help="also sweep observation noise std (auto-on for fog/rain tracks)")
    args = ap.parse_args()
    os.environ["OUTPUTS_DIR"] = args.track

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath
    import simulate as sim
    import policies as pol
    import ci as cistats

    cfg = load_config()
    out_dir = abspath(args.track)
    if not (out_dir / "phase3" / "phase3_summary.json").exists():
        print(f"ERROR: {args.track} has no profiled frontier (phase3)."); return 2
    seeds = cfg["seeds"]
    rt = cfg["routing"]["reliability_target"]
    dwell = cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]
    do_sweeps = args.sweeps or ("fog" in args.track)
    do_noise = args.noise_sweep or ("fog" in args.track) or ("rain" in args.track)

    # calibration kappa, exactly as phase5 (fit on offset calibration draws, not the test trace)
    kappa_cal = sim.calibration_kappa(cfg, "coupled", [s + 100 for s in seeds])

    # build the coupled-regime substrate once per seed (deterministic; reused everywhere)
    subs = [sim.build_substrate(cfg, regime="coupled", seed=s) for s in seeds]
    base_deadline_s = subs[0].deadline_s
    latnpz = np.load(out_dir / "phase3" / "latency_distributions.npz")
    keys = list(cfg["configs"].keys())
    lat_dists = {c: {"nominal": latnpz[f"{c}_nominal"], "contended": latnpz[f"{c}_contended"]}
                 for c in keys}

    def run(kind, sub, fm, kappa=kappa_cal):
        if kind == "joint":
            p = pol.JointPolicy(fm, kappa, rt, pol.Hysteresis(dwell))
        elif kind == "decoupled":
            p = pol.DecoupledPolicy(fm, rt, pol.Hysteresis(dwell))
        elif kind == "threshold":
            p = pol.ThresholdPolicy(fm, rt, pol.Hysteresis(dwell))
        return sim.run_policy(p, sub)

    # --- 1) threshold vs joint vs decoupled + utility, base deadline, coupled regime ---
    miss = {k: [] for k in ("joint", "decoupled", "threshold")}
    util = {k: [] for k in ("joint", "decoupled", "threshold")}
    d_per_frame = []  # per-seed (decoupled - joint) per-frame miss reduction
    for sub in subs:
        res = {k: run(k, sub, sub.fm) for k in ("joint", "decoupled", "threshold")}
        for k in res:
            miss[k].append(res[k]["deadline_miss_rate"])
            util[k].append(on_time_utility(res[k]["choices"], sub, pol))
        mj = miss_indicator(res["joint"]["choices"], sub)
        md = miss_indicator(res["decoupled"]["choices"], sub)
        d_per_frame.append(md - mj)

    summary = {
        "track": args.track, "kappa_cal": kappa_cal, "seeds": seeds,
        "deadline_ms": base_deadline_s * 1e3,
        "miss": {k: cistats.mean_ci(miss[k]) for k in miss},
        "utility": {k: cistats.mean_ci(util[k]) for k in util},
        "joint_minus_threshold_miss": cistats.paired_diff_ci(miss["joint"], miss["threshold"]),
        "threshold_vs_decoupled_miss": cistats.paired_diff_ci(miss["decoupled"], miss["threshold"]),
        "utility_joint_minus_decoupled": cistats.paired_diff_ci(util["joint"], util["decoupled"]),
        "rqh_miss_reduction_dec_minus_joint": cistats.paired_diff_ci(miss["decoupled"], miss["joint"]),
    }

    # --- 2) moving-block bootstrap of the per-frame miss reduction, pooled across seeds ---
    # Pool = concatenate the per-seed per-frame (decoupled - joint) series, then resample
    # contiguous 50-frame blocks. This probes robustness to temporal structure WITHIN a
    # sequence (not cross-sequence generalization), pooling the seeds' draws.
    d_pool = np.concatenate(d_per_frame)
    blo, bhi = block_bootstrap_ci(d_pool)
    summary["bootstrap_miss_reduction"] = {
        "block": 50, "B": 2000, "mean_pp": float(d_pool.mean() * 100),
        "lo_pp": blo * 100, "hi_pp": bhi * 100}

    lag = cfg["regimes"]["coupled"]["onset_to_contention_lag_frames"]

    def rqh_at(dl):
        """Joint-vs-decoupled miss reduction at an arbitrary absolute deadline (s)."""
        jl, dlst = [], []
        fm = pol.build_frontier_model(dl, lat_dists, subs[0].fm.acc_nominal, subs[0].fm.acc_faulted)
        for sub in subs:
            saved = sub.deadline_s
            sub.deadline_s = dl
            jl.append(run("joint", sub, fm)["deadline_miss_rate"])
            dlst.append(run("decoupled", sub, fm)["deadline_miss_rate"])
            sub.deadline_s = saved
        return cistats.paired_diff_ci(dlst, jl)

    # --- 3) deadline-strictness sweep (EVERY condition, not just fog) ---
    dl_sweep = []
    for mult in (0.8, 0.9, 1.0, 1.1, 1.2):
        dl = base_deadline_s * mult
        diff = rqh_at(dl)
        dl_sweep.append({"mult": mult, "deadline_ms": dl * 1e3, "reduction_pp": diff["mean"] * 100,
                         "lo_pp": diff["lo"] * 100, "hi_pp": diff["hi"] * 100,
                         "significant": diff["significant"]})
    summary["deadline_sweep"] = dl_sweep

    # --- 3b) externally-motivated FIXED deadlines (perception budget, not self-calibrated) ---
    fixed = []
    for hz, dl_ms in ((10.0, 100.0), (5.0, 200.0)):
        diff = rqh_at(dl_ms / 1e3)
        fixed.append({"hz": hz, "deadline_ms": dl_ms, "reduction_pp": diff["mean"] * 100,
                      "lo_pp": diff["lo"] * 100, "hi_pp": diff["hi"] * 100,
                      "significant": diff["significant"]})
    summary["fixed_deadline"] = fixed

    # --- 3c) temporally-separated calibration (leakage / circularity rebuttal) ---
    # Fit kappa ONLY on the first half of the trace (its realized contention draw), then
    # evaluate RQ-H on the held-out SECOND half. kappa is a structural regime constant, so
    # a temporal split should leave it (and the result) essentially unchanged. This makes
    # calibration/test temporal separation explicit and answers the leakage concern.
    half = subs[0].T // 2
    fa = subs[0].fault_active
    calib_ff = float(fa[:half].mean()); test_ff = float(fa[half:].mean())
    k_split, jm_h, dm_h = [], [], []
    for sub in subs:
        ks = pol.fit_coupling(sub.fault_active[:half], sub.state[:half], lag)
        k_split.append(ks)
        test = slice_sub(sub, half, sub.T)
        jm_h.append(run("joint", test, test.fm, kappa=ks)["deadline_miss_rate"])
        dm_h.append(run("decoupled", test, test.fm)["deadline_miss_rate"])
    csplit = cistats.paired_diff_ci(dm_h, jm_h)
    summary["calibration_split"] = {
        "kappa_full": kappa_cal, "kappa_split_firsthalf_mean": float(np.mean(k_split)),
        "calib_window_fault_frac": calib_ff, "test_window_fault_frac": test_ff,
        "heldout_test_frames": int(subs[0].T - half),
        "heldout_reduction_pp": csplit["mean"] * 100, "lo_pp": csplit["lo"] * 100,
        "hi_pp": csplit["hi"] * 100, "significant": csplit["significant"],
        "note": ("held-out test window is fault-sparse; split underpowered"
                 if test_ff < 0.03 else "held-out test window has faults")}

    # --- 3d) kappa sweep (fog / --sweeps only) ---
    if do_sweeps:
        kap_sweep = []
        for kap in (0.0, 0.5, 0.75, 1.0):
            jl, dlst = [], []
            for sub in subs:
                jl.append(run("joint", sub, sub.fm, kappa=kap)["deadline_miss_rate"])
                dlst.append(run("decoupled", sub, sub.fm)["deadline_miss_rate"])
            diff = cistats.paired_diff_ci(dlst, jl)
            kap_sweep.append({"kappa": kap, "reduction_pp": diff["mean"] * 100,
                              "lo_pp": diff["lo"] * 100, "hi_pp": diff["hi"] * 100,
                              "significant": diff["significant"]})
        summary["kappa_sweep"] = kap_sweep

    # --- 4) observation-noise sweep: joint-vs-threshold gap stability (fog/rain) ---
    if do_noise:
        noise_sweep = []
        for sigma in (0.15, 0.30, 0.50, 0.75):
            cfg.setdefault("experiments", {})["observation_noise_std"] = sigma
            gaps = []
            for s in seeds:
                sub = sim.build_substrate(cfg, regime="coupled", seed=s)
                jm = run("joint", sub, sub.fm)["deadline_miss_rate"]
                tm = run("threshold", sub, sub.fm)["deadline_miss_rate"]
                gaps.append(jm - tm)
            g = cistats.mean_ci(gaps)
            noise_sweep.append({"sigma": sigma, "joint_minus_threshold_pp": g["mean"] * 100,
                                "lo_pp": g["lo"] * 100, "hi_pp": g["hi"] * 100,
                                "significant": bool(g["lo"] > 0 or g["hi"] < 0)})
        summary["noise_sweep_joint_minus_threshold"] = noise_sweep

    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(extras / "robustness.json", "w"), indent=2, default=str)

    # --- print summary ---
    print(f"\n{'='*70}\n[robustness] {args.track}  (deadline {base_deadline_s*1e3:.0f} ms, kappa {kappa_cal:.2f})\n{'='*70}")
    m = summary["miss"]; u = summary["utility"]
    print(f"miss:  decoupled {m['decoupled']['mean']:.3f}  threshold {m['threshold']['mean']:.3f}  joint {m['joint']['mean']:.3f}")
    jt = summary["joint_minus_threshold_miss"]
    print(f"  joint-threshold = {jt['mean']*100:+.2f}pp [{jt['lo']*100:.2f},{jt['hi']*100:.2f}]  (sig={jt['significant']})")
    print(f"utility: U_joint {u['joint']['mean']:.3f}  U_decoupled {u['decoupled']['mean']:.3f}")
    du = summary["utility_joint_minus_decoupled"]
    print(f"  dU(joint-decoupled) = {du['mean']*100:+.2f}pp [{du['lo']*100:.2f},{du['hi']*100:.2f}]  (sig={du['significant']})")
    b = summary["bootstrap_miss_reduction"]
    print(f"bootstrap miss-reduction: {b['mean_pp']:+.2f}pp [{b['lo_pp']:.2f},{b['hi_pp']:.2f}]")
    rqh = summary["rqh_miss_reduction_dec_minus_joint"]
    print(f"(sanity) RQ-H reduction (dec-joint): {rqh['mean']*100:+.2f}pp [{rqh['lo']*100:.2f},{rqh['hi']*100:.2f}]")
    cs = summary["calibration_split"]
    print(f"calib-split (kappa on 1st half {cs['kappa_split_firsthalf_mean']:.2f} vs full {cs['kappa_full']:.2f}); "
          f"held-out 2nd-half ({cs['heldout_test_frames']} fr) reduction "
          f"{cs['heldout_reduction_pp']:+.2f}pp [{cs['lo_pp']:.2f},{cs['hi_pp']:.2f}] (sig={cs['significant']})")
    print("deadline sweep:  " + "  ".join(f"{r['mult']}x:{r['reduction_pp']:+.2f}" for r in summary["deadline_sweep"]))
    print("fixed deadline:  " + "  ".join(f"{r['hz']:.0f}Hz/{r['deadline_ms']:.0f}ms:{r['reduction_pp']:+.2f}"
                                          f"[{r['lo_pp']:.1f},{r['hi_pp']:.1f}]" for r in summary["fixed_deadline"]))
    if do_sweeps:
        print("kappa sweep:     " + "  ".join(f"k={r['kappa']}:{r['reduction_pp']:+.2f}" for r in summary["kappa_sweep"]))
    if do_noise:
        print("noise sweep (joint-threshold pp):  " + "  ".join(
            f"s={r['sigma']}:{r['joint_minus_threshold_pp']:+.2f}[{r['lo_pp']:.1f},{r['hi_pp']:.1f}]"
            for r in summary["noise_sweep_joint_minus_threshold"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
