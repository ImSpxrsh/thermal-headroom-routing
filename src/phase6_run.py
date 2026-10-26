"""Phase 6 driver: RQ-A1 (persistence kill-switch) and RQ-A2 (hysteresis ablation).

RQ-A1: belief (HMM-filtered, persistent) vs memoryless (same features, no filtering)
       detector, at MATCHED fault-detection accuracy, measuring config-switching
       frequency / control effort over >=5 seeds with CIs. Self-decided go/no-go.

RQ-A2: fixed vs model-derived hysteresis dwell under stationary AND non-stationary
       fault arrival (controlled arrival, labeled like Track B), CIs. The claim is
       that model-derived dwell dominates fixed under non-stationary arrival.

Run after Phase 3:
    python src/phase6_run.py
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
import sensor_belief as sb
import policies as pol
import ci as cistats


# ---------------------------------------------------------------------------
# RQ-A1 helpers
# ---------------------------------------------------------------------------
def balanced_acc(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = (pred & gt).sum(); tn = (~pred & ~gt).sum()
    fp = (pred & ~gt).sum(); fn = (~pred & gt).sum()
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    return 0.5 * (tpr + tnr)


def match_threshold(signal: np.ndarray, gt: np.ndarray, target_bacc: float):
    """Find threshold whose balanced accuracy is closest to target_bacc."""
    best_thr, best_gap = 0.5, 1e9
    for thr in np.linspace(0.05, 0.95, 91):
        ba = balanced_acc(signal >= thr, gt)
        gap = abs(ba - target_bacc)
        if gap < best_gap:
            best_gap, best_thr = gap, thr
    return best_thr, balanced_acc(signal >= best_thr, gt)


def cheapest_feasible(fm, c_belief_t, rt):
    feas = [c for c in pol.CONFIG_KEYS if fm.p_meet_given(c, c_belief_t) >= rt]
    pool = feas if feas else pol.CONFIG_KEYS
    return min(pool, key=lambda c: fm.cost_rank[c])


def controller_choices(detected, c_belief, fm, rt):
    """Isolates the detector's effect: faulted -> most accurate config, else cheapest
    deadline-feasible config. Same compute side for both detectors."""
    most_acc = max(pol.CONFIG_KEYS, key=lambda c: fm.acc_faulted[c])
    return [most_acc if detected[t] else cheapest_feasible(fm, c_belief[t], rt)
            for t in range(len(detected))]


def switch_rate(choices):
    return sum(1 for t in range(1, len(choices)) if choices[t] != choices[t - 1]) / max(len(choices) - 1, 1)


def rq_a1(cfg, out) -> dict:
    seeds = cfg["seeds"]; rt = cfg["routing"]["reliability_target"]
    rows = {"belief": [], "memoryless": [], "bacc_belief": [], "bacc_mem": []}
    for seed in seeds:
        sub = sim.build_substrate(cfg, regime="coupled", seed=seed)
        gt = sub.fault_active
        # belief operating point at 0.5, then match memoryless to it
        bacc_b = balanced_acc(sub.s_belief >= 0.5, gt)
        thr_m, bacc_m = match_threshold(sub.s_instant, gt, bacc_b)
        det_b = sub.s_belief >= 0.5
        det_m = sub.s_instant >= thr_m
        ch_b = controller_choices(det_b, sub.c_belief, sub.fm, rt)
        ch_m = controller_choices(det_m, sub.c_belief, sub.fm, rt)
        rows["belief"].append(switch_rate(ch_b))
        rows["memoryless"].append(switch_rate(ch_m))
        rows["bacc_belief"].append(bacc_b); rows["bacc_mem"].append(bacc_m)

    diff = cistats.paired_diff_ci(rows["memoryless"], rows["belief"])  # mem - belief > 0 => belief fewer switches
    belief_ci = cistats.mean_ci(rows["belief"]); mem_ci = cistats.mean_ci(rows["memoryless"])
    matched = abs(np.mean(rows["bacc_belief"]) - np.mean(rows["bacc_mem"])) < 0.05
    if diff["significant"] and diff["mean"] > 0 and matched:
        verdict = (f"PERSISTENCE JUSTIFIED: at matched detection accuracy "
                   f"(belief {np.mean(rows['bacc_belief']):.2f} vs memoryless "
                   f"{np.mean(rows['bacc_mem']):.2f}), belief reduces switch rate from "
                   f"{mem_ci['mean']:.3f} to {belief_ci['mean']:.3f} "
                   f"(diff {diff['mean']:.3f}, 95% CI [{diff['lo']:.3f},{diff['hi']:.3f}]).")
        go = True
    else:
        verdict = (f"NOT SUPPORTED: belief switch rate {belief_ci['mean']:.3f} vs memoryless "
                   f"{mem_ci['mean']:.3f} (diff {diff['mean']:.3f}, 95% CI "
                   f"[{diff['lo']:.3f},{diff['hi']:.3f}]); matched={matched}. Diagnosis: "
                   "detector under-confident or dataset rate-of-change too low. The "
                   "belief-state design is unsupported on this data (reported honestly).")
        go = False

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar([0, 1], [mem_ci["mean"], belief_ci["mean"]],
           yerr=[mem_ci["half_width"], belief_ci["half_width"]], capsize=6,
           color=["tab:gray", "tab:green"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["memoryless", "belief (persistent)"])
    ax.set_ylabel("config switch rate (per frame)")
    ax.set_title("RQ-A1: switching at matched detection accuracy")
    fig.tight_layout(); fig.savefig(out / "rqa1_switching.png", dpi=140); plt.close(fig)
    return {"verdict": verdict, "go": go, "belief": belief_ci, "memoryless": mem_ci,
            "paired_diff": diff, "bacc_belief": float(np.mean(rows["bacc_belief"])),
            "bacc_mem": float(np.mean(rows["bacc_mem"])), "raw": rows}


# ---------------------------------------------------------------------------
# RQ-A2 helpers (controlled fault arrival; labeled like Track B validation)
# ---------------------------------------------------------------------------
def gen_fault_timeline(n, arrival, seed):
    """2-state fault chain. stationary: constant onset prob. non-stationary: onset
    prob modulated in bursts over time."""
    rng = np.random.default_rng(7000 + seed)
    fault = np.zeros(n, dtype=bool)
    p_off_to_on_base = 0.02
    p_on_to_off = 0.08
    state = False
    for t in range(n):
        if arrival == "stationary":
            p_on = p_off_to_on_base
        else:
            # bursty: high onset in windows, near-zero between (non-stationary rate)
            p_on = 0.12 if (t // 80) % 2 == 0 else 0.002
        if state:
            if rng.random() < p_on_to_off:
                state = False
        else:
            if rng.random() < p_on:
                state = True
        fault[t] = state
    return fault


def belief_from_timeline(fault, cfg, seed):
    """Synthetic single-channel observation from a known fault timeline -> sensor HMM
    belief (so RQ-A2 exercises the real estimator + hysteresis)."""
    rng = np.random.default_rng(8000 + seed)
    base = rng.normal(0.0, 1.0, len(fault))
    obs = base - 3.0 * fault  # faulted -> low (bad) direction
    import pandas as pd
    df = pd.DataFrame({"blur": obs, "illumination": obs, "occlusion": obs})
    for ch in ("blur", "illumination", "occlusion"):
        df[f"{ch}_fault"] = fault
    df["any_fault"] = fault
    hmm = sb.fit_channel_hmm(obs, fault, cfg, direction="low")
    return sb.p_faulted(sb.belief_for_channel(obs, hmm, cfg)), hmm


def rq_a2(cfg, out) -> dict:
    seeds = cfg["seeds"]; n = 500
    target_ftr = cfg["routing"]["hysteresis"]["model_derived"]["target_false_transition_rate"]
    fixed_dwell = cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]
    results = {}
    for arrival in ("stationary", "nonstationary"):
        res = {"fixed": {"switch": [], "lag": []}, "model_derived": {"switch": [], "lag": []}}
        for seed in seeds:
            fault = gen_fault_timeline(n, arrival, seed)
            belief, hmm = belief_from_timeline(fault, cfg, seed)
            p_self = float(hmm.A[sb.NOMINAL, sb.NOMINAL])
            md_dwell = pol.model_derived_dwell(target_ftr, p_self)
            target = ["C1" if belief[t] >= 0.5 else "C4" for t in range(n)]
            for variant, dwell in (("fixed", fixed_dwell), ("model_derived", md_dwell)):
                hy = pol.Hysteresis(dwell)
                choices = [hy.step(target[t]) for t in range(n)]
                res[variant]["switch"].append(switch_rate(choices))
                res[variant]["lag"].append(sim.reconfig_latency(
                    choices, [i for i in range(n) if fault[i] and (i == 0 or not fault[i - 1])]))
        results[arrival] = {
            "fixed_switch": cistats.mean_ci(res["fixed"]["switch"]),
            "model_derived_switch": cistats.mean_ci(res["model_derived"]["switch"]),
            "fixed_lag": cistats.mean_ci(res["fixed"]["lag"]),
            "model_derived_lag": cistats.mean_ci(res["model_derived"]["lag"]),
            "switch_diff_fixed_minus_md": cistats.paired_diff_ci(
                res["fixed"]["switch"], res["model_derived"]["switch"]),
            "raw": res,
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, arrival in zip(axes, ("stationary", "nonstationary")):
        r = results[arrival]
        ax.bar([0, 1], [r["fixed_switch"]["mean"], r["model_derived_switch"]["mean"]],
               yerr=[r["fixed_switch"]["half_width"], r["model_derived_switch"]["half_width"]],
               capsize=6, color=["tab:gray", "tab:purple"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["fixed", "model-derived"])
        ax.set_title(f"{arrival} arrival"); ax.set_ylabel("switch rate")
    fig.suptitle("RQ-A2: fixed vs model-derived hysteresis (controlled arrival)")
    fig.tight_layout(); fig.savefig(out / "rqa2_hysteresis.png", dpi=140); plt.close(fig)
    return results


def main() -> int:
    cfg = load_config()
    if cfg["deadline"]["value_ms"] is None:
        print("ERROR: deadline not set; run Phase 3 first."); return 2
    out = abspath(cfg["paths"]["outputs_dir"]) / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    a1 = rq_a1(cfg, out)
    a2 = rq_a2(cfg, out)
    print("RQ-A1:", a1["verdict"])
    with open(out / "phase6_results.json", "w") as f:
        json.dump({"rq_a1": a1, "rq_a2": a2}, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
