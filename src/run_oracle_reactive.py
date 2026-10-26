"""RQ-Baselines: oracle contention router and a naive reactive-latency-only
scheduler, alongside Joint/Decoupled/Threshold, on an already-profiled track.

  OracleContentionPolicy   knows the TRUE contention state exactly each frame
                           (not a belief). Upper bound on what perfect compute-
                           state knowledge could buy the joint mechanism.
  ReactiveLatencyPolicy    reacts only to the observed latency trend; carries
                           NO sensor-fault information at all (not even for
                           accuracy weighting). Tests whether naive latency
                           reactivity alone already captures the benefit.

Reuses the exact RQ-H substrate (sim.build_substrate + calibration kappa,
identical seeds/draws as phase5), so all five policies are compared on
identical draws.

    python src/run_oracle_reactive.py --track outputs/trackD_fog_6_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


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

    from run_robustness import on_time_utility
    keys = ("joint", "decoupled", "threshold", "oracle", "reactive")
    miss = {k: [] for k in keys}
    acc = {k: [] for k in keys}
    util = {k: [] for k in keys}
    for sub in subs:
        fm = sub.fm
        joint = pol.JointPolicy(fm, kappa_cal, rt, pol.Hysteresis(dwell))
        decoup = pol.DecoupledPolicy(fm, rt, pol.Hysteresis(dwell))
        thresh = pol.ThresholdPolicy(fm, rt, pol.Hysteresis(dwell))
        oracle = pol.OracleContentionPolicy(fm, rt, pol.Hysteresis(dwell))
        reactive = pol.ReactiveLatencyPolicy(fm, rt, pol.Hysteresis(dwell))

        res = {
            "joint": sim.run_policy(joint, sub),
            "decoupled": sim.run_policy(decoup, sub),
            "threshold": sim.run_policy(thresh, sub),
            "oracle": sim.run_oracle_policy(oracle, sub),
            "reactive": sim.run_reactive_policy(reactive, sub),
        }
        for k in keys:
            miss[k].append(res[k]["deadline_miss_rate"])
            acc[k].append(res[k]["mean_accuracy"])
            util[k].append(on_time_utility(res[k]["choices"], sub, pol))

    summary = {
        "track": args.track, "kappa_cal": kappa_cal, "seeds": seeds,
        "miss": {k: cistats.mean_ci(miss[k]) for k in keys},
        "accuracy": {k: cistats.mean_ci(acc[k]) for k in keys},
        "utility": {k: cistats.mean_ci(util[k]) for k in keys},
        "joint_minus_oracle_miss": cistats.paired_diff_ci(miss["joint"], miss["oracle"]),
        "reactive_minus_joint_miss": cistats.paired_diff_ci(miss["reactive"], miss["joint"]),
        "reactive_minus_decoupled_miss": cistats.paired_diff_ci(miss["reactive"], miss["decoupled"]),
        "decoupled_minus_joint_miss": cistats.paired_diff_ci(miss["decoupled"], miss["joint"]),
        "joint_minus_reactive_utility": cistats.paired_diff_ci(util["joint"], util["reactive"]),
        "joint_minus_reactive_accuracy": cistats.paired_diff_ci(acc["joint"], acc["reactive"]),
    }

    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(extras / "oracle_reactive.json", "w"), indent=2, default=str)

    m, a, u = summary["miss"], summary["accuracy"], summary["utility"]
    print(f"\n{'='*70}\n[oracle/reactive] {args.track}  (kappa {kappa_cal:.2f})\n{'='*70}")
    print(f"miss:      oracle {m['oracle']['mean']:.3f}  joint {m['joint']['mean']:.3f}  "
          f"threshold {m['threshold']['mean']:.3f}  decoupled {m['decoupled']['mean']:.3f}  "
          f"reactive {m['reactive']['mean']:.3f}")
    print(f"accuracy:  oracle {a['oracle']['mean']:.3f}  joint {a['joint']['mean']:.3f}  "
          f"threshold {a['threshold']['mean']:.3f}  decoupled {a['decoupled']['mean']:.3f}  "
          f"reactive {a['reactive']['mean']:.3f}")
    print(f"utility U: oracle {u['oracle']['mean']:.3f}  joint {u['joint']['mean']:.3f}  "
          f"threshold {u['threshold']['mean']:.3f}  decoupled {u['decoupled']['mean']:.3f}  "
          f"reactive {u['reactive']['mean']:.3f}")
    jo = summary["joint_minus_oracle_miss"]
    print(f"  joint-oracle miss headroom = {jo['mean']*100:+.2f}pp [{jo['lo']*100:.2f},{jo['hi']*100:.2f}] "
          f"(sig={jo['significant']}) -- gap closable only by better contention estimation, not coupling")
    rj = summary["reactive_minus_joint_miss"]
    print(f"  reactive-joint miss = {rj['mean']*100:+.2f}pp [{rj['lo']*100:.2f},{rj['hi']*100:.2f}] "
          f"(sig={rj['significant']})")
    ju_acc = summary["joint_minus_reactive_accuracy"]
    print(f"  joint-reactive ACCURACY = {ju_acc['mean']*100:+.2f}pp [{ju_acc['lo']*100:.2f},{ju_acc['hi']*100:.2f}] "
          f"(sig={ju_acc['significant']}) -- the accuracy cost of reactive's lower miss rate")
    ju_u = summary["joint_minus_reactive_utility"]
    print(f"  joint-reactive UTILITY = {ju_u['mean']*100:+.2f}pp [{ju_u['lo']*100:.2f},{ju_u['hi']*100:.2f}] "
          f"(sig={ju_u['significant']}) -- combined timeliness+accuracy metric")
    dj = summary["decoupled_minus_joint_miss"]
    print(f"  (sanity) decoupled-joint (RQ-H) = {dj['mean']*100:+.2f}pp [{dj['lo']*100:.2f},{dj['hi']*100:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
