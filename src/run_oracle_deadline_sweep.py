"""Reviewer #3: is the oracle/belief result an artifact of a coarse (median-thresholded)
feasibility rule, and does the benefit only exist at the tight self-calibrated deadline?

This sweeps the deadline across a wide tight->loose band and runs ALL five policies
(joint, decoupled, oracle-contention, reactive-latency) on the identical RQ-H substrate at
each deadline. It shows (a) where the coupling benefit (decoupled-joint) actually lives,
(b) the joint<->oracle gap across the band, and (c) reactive vs joint across the band --
so the claim can be scoped explicitly to the deadline band where it holds, instead of
implying general utility.

    python src/run_oracle_deadline_sweep.py --track outputs/trackD_fog_6_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

MULTS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="outputs")
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

    kappa_cal = sim.calibration_kappa(cfg, "coupled", [s + 100 for s in seeds])
    subs = [sim.build_substrate(cfg, regime="coupled", seed=s) for s in seeds]
    base = subs[0].deadline_s
    latnpz = np.load(out_dir / "phase3" / "latency_distributions.npz")
    keys = list(cfg["configs"].keys())
    lat_dists = {c: {"nominal": latnpz[f"{c}_nominal"], "contended": latnpz[f"{c}_contended"]} for c in keys}
    acc_nom, acc_flt = subs[0].fm.acc_nominal, subs[0].fm.acc_faulted

    def miss_at(dl):
        fm = pol.build_frontier_model(dl, lat_dists, acc_nom, acc_flt)
        out = {k: [] for k in ("joint", "decoupled", "oracle", "reactive")}
        for sub in subs:
            saved = sub.deadline_s
            sub.deadline_s = dl
            out["joint"].append(sim.run_policy(pol.JointPolicy(fm, kappa_cal, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"])
            out["decoupled"].append(sim.run_policy(pol.DecoupledPolicy(fm, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"])
            out["oracle"].append(sim.run_oracle_policy(pol.OracleContentionPolicy(fm, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"])
            out["reactive"].append(sim.run_reactive_policy(pol.ReactiveLatencyPolicy(fm, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"])
            sub.deadline_s = saved
        return out

    rows = []
    for mult in MULTS:
        dl = base * mult
        m = miss_at(dl)
        red = cistats.paired_diff_ci(m["decoupled"], m["joint"])      # coupling benefit
        j_or = cistats.paired_diff_ci(m["joint"], m["oracle"])        # belief headroom to oracle
        r_j = cistats.paired_diff_ci(m["reactive"], m["joint"])       # reactive vs joint
        rows.append({
            "mult": mult, "deadline_ms": dl * 1e3,
            "joint_miss": float(np.mean(m["joint"])), "decoupled_miss": float(np.mean(m["decoupled"])),
            "oracle_miss": float(np.mean(m["oracle"])), "reactive_miss": float(np.mean(m["reactive"])),
            "coupling_reduction_pp": red["mean"] * 100, "coupling_sig": red["significant"],
            "joint_vs_oracle_pp": j_or["mean"] * 100, "joint_vs_oracle_sig": j_or["significant"],
            "reactive_vs_joint_pp": r_j["mean"] * 100, "reactive_vs_joint_sig": r_j["significant"],
        })

    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump({"track": args.track, "base_deadline_ms": base * 1e3, "kappa_cal": kappa_cal, "rows": rows},
              open(extras / "oracle_deadline_sweep.json", "w"), indent=2, default=str)

    print(f"\n{'='*88}\n[oracle deadline sweep] {args.track}  (base deadline {base*1e3:.0f} ms, kappa {kappa_cal:.2f})\n{'='*88}")
    print(f"{'mult':>5} {'dl_ms':>7} {'joint':>7} {'decoup':>7} {'oracle':>7} {'react':>7} "
          f"{'coupl_d':>11} {'joint-oracle':>13} {'react-joint':>12}")
    for r in rows:
        print(f"{r['mult']:>5} {r['deadline_ms']:>7.0f} {r['joint_miss']:>7.3f} {r['decoupled_miss']:>7.3f} "
              f"{r['oracle_miss']:>7.3f} {r['reactive_miss']:>7.3f} "
              f"{r['coupling_reduction_pp']:>+9.2f}{'*' if r['coupling_sig'] else ' '}  "
              f"{r['joint_vs_oracle_pp']:>+11.2f}{'*' if r['joint_vs_oracle_sig'] else ' '}  "
              f"{r['reactive_vs_joint_pp']:>+10.2f}{'*' if r['reactive_vs_joint_sig'] else ' '}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
