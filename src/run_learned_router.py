"""Run the learned-router baseline vs the joint belief policy on an ALREADY-PROFILED
track (reuses its outputs/<track>/phase1 + phase3; no re-profiling).

    python src/run_learned_router.py --track outputs                       # Track A
    python src/run_learned_router.py --track outputs/trackD_rain_4_0        # RADIATE rain
    python src/run_learned_router.py --track outputs/trackC                 # WoodScape fisheye

Comparison: deadline-miss rate (primary) and accuracy on a held-out TEST split, with the
MLP trained on the TRAIN split only, split by sequence (contiguous) not frame, over >=5
seeds with 95% CIs. Both outcomes are valid: joint wins -> the belief-state structure
does real work; learned matches -> the value is in the features.
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


def _metrics(choices, sub, idx):
    miss = np.mean([sub.L[choices[t]][t] > sub.deadline_s for t in idx])
    accs = [sub.acc[choices[t]][t] for t in idx if not np.isnan(sub.acc[choices[t]][t])]
    return float(miss), float(np.mean(accs)) if accs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="outputs", help="track outputs dir (has phase1/ + phase3/)")
    ap.add_argument("--regime", default="coupled")
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()
    os.environ["OUTPUTS_DIR"] = args.track  # build_substrate reads from here

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath
    import simulate as sim
    import policies as pol
    import learned_router as lr
    import ci as cistats

    cfg = load_config()
    if cfg["deadline"]["value_ms"] is None:
        print("ERROR: deadline not set; this track has no profiled frontier."); return 2
    seeds = cfg["seeds"]; rt = cfg["routing"]["reliability_target"]
    dwell = cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]
    afloor = cfg["routing"]["accuracy_floor"]
    kappa_cal = sim.calibration_kappa(cfg, args.regime, [s + 100 for s in seeds])

    raw = {"joint": {"miss": [], "acc": []}, "learned": {"miss": [], "acc": []}, "train_acc": []}
    for seed in seeds:
        sub = sim.build_substrate(cfg, regime=args.regime, seed=seed)
        T = sub.T; cut = int(T * args.train_frac)
        train_idx = np.arange(0, cut); test_idx = np.arange(cut, T)
        X = lr.features(sub); y = lr.oracle_config_labels(sub, afloor)

        model = lr.LearnedRouter(in_dim=X.shape[1], n_configs=len(pol.CONFIG_KEYS), seed=seed)
        model.fit(X[train_idx], y[train_idx])
        # apply the SAME fixed hysteresis to the learned router's per-frame predictions, so
        # the comparison isolates the decision rule (belief structure vs learned function),
        # not smoothing (the joint also uses this hysteresis).
        hy_l = pol.Hysteresis(dwell)
        learned_choices = [hy_l.step(c) for c in lr.choices_from_indices(model.predict(X))]
        raw["train_acc"].append(float(np.mean(model.predict(X[train_idx]) == y[train_idx])))

        joint = pol.JointPolicy(sub.fm, kappa_cal, rt, pol.Hysteresis(dwell))
        joint_choices = sim.run_policy(joint, sub)["choices"]

        jm, ja = _metrics(joint_choices, sub, test_idx)
        lm, la = _metrics(learned_choices, sub, test_idx)
        raw["joint"]["miss"].append(jm); raw["joint"]["acc"].append(ja)
        raw["learned"]["miss"].append(lm); raw["learned"]["acc"].append(la)

    jm = cistats.mean_ci(raw["joint"]["miss"]); lm = cistats.mean_ci(raw["learned"]["miss"])
    diff = cistats.paired_diff_ci(raw["learned"]["miss"], raw["joint"]["miss"])  # learned - joint
    if diff["significant"] and diff["mean"] > 0:
        verdict = (f"JOINT WINS: joint deadline-miss {jm['mean']:.3f} < learned {lm['mean']:.3f} "
                   f"(learned-minus-joint {diff['mean']*100:.2f}pp, 95% CI "
                   f"[{diff['lo']*100:.2f},{diff['hi']*100:.2f}]). The belief-state structure helps "
                   "beyond the features.")
    elif diff["significant"] and diff["mean"] < 0:
        verdict = (f"LEARNED WINS: learned {lm['mean']:.3f} < joint {jm['mean']:.3f}. The features "
                   "carry the signal; the joint structure is not necessary on this track.")
    else:
        verdict = (f"MATCH (no significant difference): joint {jm['mean']:.3f} vs learned "
                   f"{lm['mean']:.3f} (CI [{diff['lo']*100:.2f},{diff['hi']*100:.2f}]pp). On the same "
                   "features a learned function matches the belief-space policy; an honest finding.")

    out = abspath(args.track) / "extras"; out.mkdir(parents=True, exist_ok=True)
    res = {"track": args.track, "regime": args.regime, "train_frac": args.train_frac,
           "n_configs": len(pol.CONFIG_KEYS), "mlp_train_acc": float(np.mean(raw["train_acc"])),
           "joint_miss": jm, "learned_miss": lm,
           "joint_acc": cistats.mean_ci(raw["joint"]["acc"]), "learned_acc": cistats.mean_ci(raw["learned"]["acc"]),
           "learned_minus_joint_miss": diff, "verdict": verdict, "raw": raw}
    with open(out / "learned_router.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar([0, 1], [jm["mean"], lm["mean"]], yerr=[jm["half_width"], lm["half_width"]],
           capsize=6, color=["tab:blue", "tab:gray"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["joint (belief)", "learned MLP"])
    ax.set_ylabel("deadline-miss rate (test split)")
    ax.set_title(f"Learned router vs joint belief policy\n{args.track} (95% CI, {len(seeds)} seeds)")
    fig.tight_layout(); fig.savefig(out / "learned_router.png", dpi=140); plt.close(fig)

    print(verdict)
    print(json.dumps({k: res[k] for k in ("joint_miss", "learned_miss", "mlp_train_acc")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
