"""Phase 4 driver: run the full routing pipeline end-to-end and confirm every
joint/decoupled x hysteresis combination plus abstention and the oracle labeler all
work. Writes per-frame decisions and a summary to outputs/phase4/.

Run after Phase 3 (needs the deadline and the profiled frontier):
    python src/phase4_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath
import simulate as sim
import policies as pol
import oracle


def make_hysteresis(cfg, sub, variant):
    if variant == "fixed":
        return pol.Hysteresis(cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"])
    dwell = pol.model_derived_dwell(
        cfg["routing"]["hysteresis"]["model_derived"]["target_false_transition_rate"],
        sub.p_self_transition)
    return pol.Hysteresis(dwell)


def build_policy(kind, cfg, sub, variant, kappa_cal):
    rt = cfg["routing"]["reliability_target"]
    hy = make_hysteresis(cfg, sub, variant)
    if kind == "joint":
        # calibration kappa (not the in-sample sub.kappa), abstention ON for Phase 4
        return pol.JointPolicy(sub.fm, kappa_cal, rt, hy, allow_abstain=True)
    if kind == "decoupled":
        return pol.DecoupledPolicy(sub.fm, rt, hy, allow_abstain=True)
    raise ValueError(kind)


def main() -> int:
    cfg = load_config()
    if cfg["deadline"]["value_ms"] is None:
        print("ERROR: deadline not set. Run src/profile_frontier.py (Phase 3) first.")
        return 2
    out = abspath(cfg["paths"]["outputs_dir"]) / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    afloor = cfg["routing"]["accuracy_floor"]

    rows = []
    sub = sim.build_substrate(cfg, regime="coupled", seed=0)
    kappa_cal = sim.calibration_kappa(cfg, "coupled", [s + 100 for s in cfg["seeds"]])
    infeasible = oracle.oracle_infeasible(sub, afloor)
    print(f"T={sub.T} kappa_cal={kappa_cal:.3f} deadline={sub.deadline_s*1e3:.1f}ms "
          f"oracle_infeasible={int(infeasible.sum())}")

    decisions = {"frame_idx": list(range(sub.T))}
    for kind in ("joint", "decoupled"):
        for variant in ("fixed", "model_derived"):
            policy = build_policy(kind, cfg, sub, variant, kappa_cal)
            res = sim.run_policy(policy, sub)
            absc = oracle.score_abstention(res["choices"], infeasible)
            decisions[f"{kind}_{variant}"] = res["choices"]
            rows.append({"policy": kind, "hysteresis": variant,
                         **{k: v for k, v in res.items() if k != "choices"}, **absc})

    # memoryless reference (RQ-A1 mechanism check; full ablation in Phase 6)
    mp = pol.MemorylessPolicy(sub.fm, cfg["routing"]["reliability_target"])
    res_m = sim.run_policy(mp, sub, memoryless=True)
    decisions["memoryless"] = res_m["choices"]
    rows.append({"policy": "memoryless", "hysteresis": "none",
                 **{k: v for k, v in res_m.items() if k != "choices"},
                 **oracle.score_abstention(res_m["choices"], infeasible)})

    pd.DataFrame(decisions).to_csv(out / "per_frame_decisions.csv", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "phase4_summary.csv", index=False)
    print(summ.to_string(index=False))
    with open(out / "phase4_summary.json", "w") as f:
        json.dump({"kappa": sub.kappa, "deadline_ms": sub.deadline_s * 1e3,
                   "oracle_infeasible": int(infeasible.sum()),
                   "rows": rows}, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
