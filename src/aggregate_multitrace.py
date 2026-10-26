"""Experiment B: aggregate several sequences of the SAME condition into a cross-sequence
(trajectory-level) result. Turns "5 seeds on one frozen trace" into "generalizes across
independent traces" -- the CI is now over TRAJECTORIES, which is the honest
generalization statistic (reviewer #7 / #4).

For each sequence output dir it reads the committed RQ-H (phase5_rqh.json) -> one
per-sequence point estimate of the coupled miss reduction. Across the N sequences it
reports the mean, a t-interval over sequences, and a cluster (resample-sequences)
bootstrap CI. If measure_real_coupling.py was run per sequence, it also pools the measured
Pc(fault)/Pc(nominal) and corr(fault, load).

Works on ANY set of sequence outputs (RADIATE, or any dataset run through the pipeline).

    python src/aggregate_multitrace.py --condition rain \
        --tracks outputs/trackD_rain_1_0 outputs/trackD_rain_2_0 outputs/trackD_rain_3_0

POOLED mode (--pool): treat every listed sequence as one draw regardless of condition and
run EXACT nonparametric tests across all of them -- a one-sided exact sign test (k of n
positive) and a Wilcoxon signed-rank test. With small per-condition n (e.g. 3 traces) the
parametric t-interval is underpowered; "positive in n/n independent sequences, sign test
p=1/2^n" is the defensible cross-trace claim (p=0.016 at 6/6, p=0.004 at 8/8).

    python src/aggregate_multitrace.py --condition all --pool \
        --tracks outputs/trackD_rain_* outputs/trackD_fog_* outputs/trackD_snow_1_0 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def cluster_bootstrap_ci(vals, B=2000, seed=23):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def t_ci(vals, ci=95.0):
    v = np.asarray(vals, float)
    n = len(v)
    if n < 2:
        return float(v.mean()) if n else float("nan"), float("nan"), float("nan")
    from scipy import stats
    m = float(v.mean()); se = float(v.std(ddof=1)) / np.sqrt(n)
    t = stats.t.ppf(0.5 + ci / 200.0, df=n - 1)
    return m, m - t * se, m + t * se


def sign_test_onesided(vals):
    """Exact one-sided sign test: P(>=k positives out of n | p=0.5). Ties (exact zeros)
    are dropped per convention."""
    v = [x for x in vals if x != 0.0]
    n = len(v)
    k = sum(1 for x in v if x > 0)
    if n == 0:
        return float("nan"), 0, 0
    from math import comb
    p = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return float(p), k, n


def wilcoxon_onesided(vals):
    v = np.asarray([x for x in vals if x != 0.0], float)
    if len(v) < 4:            # scipy needs a few nonzero values; sign test covers small n
        return None
    from scipy import stats
    try:
        res = stats.wilcoxon(v, alternative="greater")
        return float(res.pvalue)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, help="label, e.g. rain (or 'all' with --pool)")
    ap.add_argument("--tracks", nargs="+", required=True, help="sequence output dirs for this condition")
    ap.add_argument("--pool", action="store_true",
                    help="pool ALL listed sequences (any condition) and run exact sign + Wilcoxon tests")
    args = ap.parse_args()

    reductions, per_seq = [], []
    pcf, pcn, corr = [], [], []
    for t in args.tracks:
        d = Path(t)
        if not d.is_absolute() and not (d / "phase5").exists():
            d = ROOT / t                      # resolve relative dirs against the repo root
        rqh = jload(d / "phase5" / "phase5_rqh.json")
        if not rqh:
            print(f"  SKIP {t}: no phase5_rqh.json (run the pipeline on it first)"); continue
        c = rqh["summary"]["coupled"]
        red = c["miss_reduction_decoupled_minus_joint"]["mean"] * 100
        reductions.append(red)
        row = {"track": t, "reduction_pp": red,
               "joint_miss": c["joint_miss"]["mean"], "decoupled_miss": c["decoupled_miss"]["mean"],
               "seed_ci_pp": [c["miss_reduction_decoupled_minus_joint"]["lo"] * 100,
                              c["miss_reduction_decoupled_minus_joint"]["hi"] * 100]}
        rc = jload(d / "extras" / "real_coupling.json")
        if rc:
            row["measured_Pc_fault"] = rc.get("measured_Pc_fault")
            row["measured_Pc_nominal"] = rc.get("measured_Pc_nominal")
            row["corr_fault_load"] = rc.get("corr_fault_load")
            row["decirc_reduction_pp"] = rc.get("decirc_reduction_pp")
            if rc.get("measured_Pc_fault") is not None:
                pcf.append(rc["measured_Pc_fault"]); pcn.append(rc["measured_Pc_nominal"])
            if rc.get("corr_fault_load") is not None:
                corr.append(rc["corr_fault_load"])
        per_seq.append(row)

    if not reductions:
        print("No sequences with results found."); return 2

    m_t, lo_t, hi_t = t_ci(reductions)
    m_b, lo_b, hi_b = cluster_bootstrap_ci(reductions)
    summary = {
        "condition": args.condition, "n_sequences": len(reductions),
        "per_sequence_reduction_pp": reductions,
        "cross_sequence_mean_pp": m_t,
        "t_interval_pp": [lo_t, hi_t],
        "cluster_bootstrap_ci_pp": [lo_b, hi_b],
        "generalizes": bool((lo_t > 0) or (hi_t < 0)) if len(reductions) >= 2 else None,
        "per_sequence": per_seq,
    }
    if pcf:
        summary["measured_Pc_fault_mean"] = float(np.mean(pcf))
        summary["measured_Pc_nominal_mean"] = float(np.mean(pcn))
    if corr:
        summary["corr_fault_load_mean"] = float(np.mean(corr))

    # pooled exact tests across all listed sequences (the small-n honest statistic)
    if args.pool:
        p_sign, k, n_nz = sign_test_onesided(reductions)
        p_wilx = wilcoxon_onesided(reductions)
        summary["pooled_sign_test"] = {"positives": k, "n_nonzero": n_nz, "p_one_sided": p_sign}
        summary["pooled_wilcoxon_p_one_sided"] = p_wilx
        # does-no-harm on the de-circularized (real-load) reductions, if present
        decirc = [r["decirc_reduction_pp"] for r in per_seq if r.get("decirc_reduction_pp") is not None]
        if decirc:
            summary["decirc_pooled"] = {
                "n": len(decirc), "min_pp": float(np.min(decirc)), "mean_pp": float(np.mean(decirc)),
                "n_negative": int(sum(1 for x in decirc if x < 0))}

    outdir = ROOT / "outputs" / "multitrace"; outdir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(outdir / f"{args.condition}.json", "w"), indent=2, default=str)

    print(f"\n=== {args.condition}: {len(reductions)} sequences ===")
    for r in per_seq:
        print(f"  {r['track']:32s} reduction {r['reduction_pp']:+.2f}pp")
    print(f"  CROSS-SEQUENCE mean {m_t:+.2f}pp  t-CI [{lo_t:.2f},{hi_t:.2f}]  "
          f"cluster-bootstrap [{lo_b:.2f},{hi_b:.2f}]")
    if summary.get("generalizes") is not None:
        print(f"  generalizes across traces (CI excludes 0): {summary['generalizes']}")
    if args.pool:
        st = summary["pooled_sign_test"]
        print(f"  POOLED sign test: {st['positives']}/{st['n_nonzero']} sequences positive, "
              f"one-sided p={st['p_one_sided']:.4f}")
        if summary.get("pooled_wilcoxon_p_one_sided") is not None:
            print(f"  POOLED Wilcoxon signed-rank: one-sided p={summary['pooled_wilcoxon_p_one_sided']:.4f}")
        if "decirc_pooled" in summary:
            dp = summary["decirc_pooled"]
            print(f"  de-circularized (real load): n={dp['n']}, mean {dp['mean_pp']:+.2f}pp, "
                  f"min {dp['min_pp']:+.2f}pp, negative on {dp['n_negative']} sequences")
    if pcf:
        print(f"  measured Pc(fault)={np.mean(pcf):.3f} vs Pc(nominal)={np.mean(pcn):.3f} "
              f"(paper hard-codes 0.85 / 0.05); corr(fault,load)={np.mean(corr):+.3f}"
              if corr else "")
    print(f"  -> {outdir / (args.condition + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
