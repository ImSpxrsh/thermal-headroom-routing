"""Phase 5 driver: RQ-H centerpiece. Joint vs decoupled policy on identical data, in
both coupling regimes, deadline-miss rate (primary) + accuracy at matched miss rate
(secondary), with 95% CIs over >=5 seeds. Produces the centerpiece figure and an
explicit verdict.

Headline hypothesis: the joint benefit appears in the COUPLED regime and shrinks or
vanishes in the UNCOUPLED regime. A negligible delta (< ~1-2% absolute) is reported
honestly as a negative coupling result with a diagnosis. No tuning to force a sign.

Run after Phase 3:
    python src/phase5_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath
import simulate as sim
import policies as pol
import ci as cistats

REGIMES = ["coupled", "uncoupled"]


def _hyst(cfg):
    # Phase 5 holds hysteresis fixed and identical for both policies, so the ONLY
    # difference is the coupling. The dwell ablation is RQ-A2 (Phase 6).
    return cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]


def main() -> int:
    cfg = load_config()
    if cfg["deadline"]["value_ms"] is None:
        print("ERROR: deadline not set; run Phase 3 first."); return 2
    out = abspath(cfg["paths"]["outputs_dir"]) / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    seeds = cfg["seeds"]
    rt = cfg["routing"]["reliability_target"]
    dwell = _hyst(cfg)

    raw = {r: {"joint": {"miss": [], "acc": []}, "decoupled": {"miss": [], "acc": []},
               "kappa": []} for r in REGIMES}
    for regime in REGIMES:
        # coupling coefficient fit on separate calibration draws (seeds offset),
        # never on the test trace, to avoid leakage into the joint policy.
        kappa_cal = sim.calibration_kappa(cfg, regime, [s + 100 for s in seeds])
        for seed in seeds:
            sub = sim.build_substrate(cfg, regime=regime, seed=seed)
            raw[regime]["kappa"].append(kappa_cal)
            for kind in ("joint", "decoupled"):
                if kind == "joint":
                    p = pol.JointPolicy(sub.fm, kappa_cal, rt, pol.Hysteresis(dwell))
                else:
                    p = pol.DecoupledPolicy(sub.fm, rt, pol.Hysteresis(dwell))
                res = sim.run_policy(p, sub)
                raw[regime][kind]["miss"].append(res["deadline_miss_rate"])
                raw[regime][kind]["acc"].append(res["mean_accuracy"])

    # aggregate
    summary = {}
    for regime in REGIMES:
        jm = cistats.mean_ci(raw[regime]["joint"]["miss"])
        dm = cistats.mean_ci(raw[regime]["decoupled"]["miss"])
        diff = cistats.paired_diff_ci(raw[regime]["decoupled"]["miss"], raw[regime]["joint"]["miss"])
        summary[regime] = {
            "kappa_mean": float(np.mean(raw[regime]["kappa"])),
            "joint_miss": jm, "decoupled_miss": dm,
            "joint_acc": cistats.mean_ci(raw[regime]["joint"]["acc"]),
            "decoupled_acc": cistats.mean_ci(raw[regime]["decoupled"]["acc"]),
            "miss_reduction_decoupled_minus_joint": diff,
        }

    # verdict
    coup = summary["coupled"]["miss_reduction_decoupled_minus_joint"]
    unc = summary["uncoupled"]["miss_reduction_decoupled_minus_joint"]
    benefit_coupled = coup["mean"]
    benefit_uncoupled = unc["mean"]
    material = 0.01  # ~1% absolute threshold for "material"
    if coup["significant"] and benefit_coupled >= material and benefit_coupled > benefit_uncoupled:
        verdict = ("POSITIVE: joint reduces deadline-miss vs decoupled in the coupled "
                   f"regime by {benefit_coupled*100:.1f}pp (95% CI excludes 0), and the "
                   f"benefit is smaller/absent in the uncoupled regime "
                   f"({benefit_uncoupled*100:.1f}pp). The coupling drives the benefit.")
    else:
        verdict = ("NEGATIVE/INCONCLUSIVE: the joint-vs-decoupled deadline-miss delta in "
                   f"the coupled regime is {benefit_coupled*100:.1f}pp "
                   f"(95% CI [{coup['lo']*100:.1f},{coup['hi']*100:.1f}]pp), "
                   f"uncoupled {benefit_uncoupled*100:.1f}pp. Diagnosis required (coupling "
                   "too weak in this data, contention not actually fault-correlated, or "
                   "deadline not binding). Reported as a valid negative coupling result.")
    summary["verdict"] = verdict
    print(verdict)

    # centerpiece figure: grouped bars deadline-miss by regime, joint vs decoupled, CIs
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(REGIMES)); w = 0.35
    for i, kind in enumerate(["joint", "decoupled"]):
        means = [summary[r][f"{kind}_miss"]["mean"] for r in REGIMES]
        errs = [summary[r][f"{kind}_miss"]["half_width"] for r in REGIMES]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=5,
               label=kind, color=["tab:blue", "tab:orange"][i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n(kappa={summary[r]['kappa_mean']:.2f})" for r in REGIMES])
    ax.set_ylabel("deadline-miss rate")
    ax.set_title("RQ-H centerpiece: joint vs decoupled deadline-miss by regime\n"
                 f"(95% CI over {len(seeds)} seeds)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "rqh_centerpiece.png", dpi=140); plt.close(fig)

    with open(out / "phase5_rqh.json", "w") as f:
        json.dump({"summary": summary, "raw": raw, "seeds": seeds}, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
