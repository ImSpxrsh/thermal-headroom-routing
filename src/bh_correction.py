"""Benjamini-Hochberg correction across the RQ-Coupling family of tests:
8 RADIATE sequences x 3 load proxies (detections, latency, concurrent ORB
workload) = 24 de-circularized RQ-H tests of the same hypothesis ("real load
tracks sensor faults strongly enough for the router to exploit").

p-values: computed directly from the five seed-level paired differences where
the runner stored them (decirc_per_seed_reduction_pp / decirc_p_two_sided,
written by measure_concurrent_workload.py). For runs that predate per-seed
storage, p is recovered by exact inversion of the reported paired t-interval:
for a t-test, (mean, half-width, df) is a sufficient statistic, so
p = 2*(1 - T_df(|mean| * t_crit / half_width)) reproduces the seed-level
p-value exactly (not approximately). Zero-width intervals (every seed draw
identical at 0) carry no evidence and are assigned p = 1.

Note on dependence: the 24 tests are not independent (proxies share frames
within a sequence; sequences share a condition). BH controls FDR under
positive regression dependence (Benjamini-Yekutieli 2001), which is the
plausible structure here; we state this in the paper rather than assume
independence.

    python src/bh_correction.py
-> outputs/multitrace/bh_correction.json + printed table
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
TRACKS = ["trackD_rain_4_0", "trackD_rain_2_0", "trackD_rain_3_0",
          "trackD_snow_1_0", "trackD_fog_6_0", "trackD_fog_8_0",
          "trackD_fog_8_1", "trackD_night_1_0"]
N_SEEDS = 5
DF = N_SEEDS - 1
T_CRIT = stats.t.ppf(0.975, df=DF)
Q = 0.05

# detections-proxy results (mean, lo, hi in pp) from the measure_real_coupling
# detections runs; that runner writes a single real_coupling.json which the
# subsequent latency run overwrites, so the detections numbers are preserved
# here from the run logs (see PR #7 discussion).
DETECTIONS = {
    "trackD_rain_4_0":  (0.00, 0.00, 0.00),
    "trackD_rain_2_0":  (0.00, 0.00, 0.00),
    "trackD_rain_3_0":  (1.26, -2.23, 4.75),
    "trackD_snow_1_0":  (0.00, 0.00, 0.00),
    "trackD_fog_6_0":   (0.00, 0.00, 0.00),
    "trackD_fog_8_0":   (0.47, -2.72, 3.65),
    "trackD_fog_8_1":   (0.00, 0.00, 0.00),
    "trackD_night_1_0": (0.00, 0.00, 0.00),
}


def p_from_ci(mean, lo, hi):
    """Exact inversion of a paired t-interval back to the two-sided p-value."""
    if hi == lo:
        return 1.0 if mean == 0.0 else 0.0
    se = (hi - lo) / (2 * T_CRIT)
    t = mean / se
    return float(2 * (1 - stats.t.cdf(abs(t), df=DF)))


def main() -> int:
    tests = []
    for t in TRACKS:
        m, lo, hi = DETECTIONS[t]
        tests.append({"track": t, "proxy": "detections", "mean_pp": m,
                      "p": p_from_ci(m, lo, hi), "p_source": "ci_inversion"})

        d = json.load(open(OUT / t / "extras" / "real_coupling.json"))
        m, lo, hi = d["decirc_reduction_pp"], d["decirc_lo_pp"], d["decirc_hi_pp"]
        p = d.get("decirc_p_two_sided", p_from_ci(m, lo, hi))
        src = "seed_level" if "decirc_p_two_sided" in d else "ci_inversion"
        tests.append({"track": t, "proxy": "latency", "mean_pp": m, "p": p,
                      "p_source": src})

        d = json.load(open(OUT / t / "extras" / "concurrent_workload_coupling.json"))
        m = d["decirc_reduction_pp"]
        p = d.get("decirc_p_two_sided",
                  p_from_ci(m, d["decirc_lo_pp"], d["decirc_hi_pp"]))
        src = "seed_level" if "decirc_p_two_sided" in d else "ci_inversion"
        tests.append({"track": t, "proxy": "orb_workload", "mean_pp": m, "p": p,
                      "p_source": src})

    tests.sort(key=lambda x: x["p"])
    n = len(tests)
    any_sig = False
    for i, x in enumerate(tests):
        x["rank"] = i + 1
        x["bh_threshold"] = (i + 1) / n * Q
        x["bh_significant"] = x["p"] <= x["bh_threshold"]
        any_sig |= x["bh_significant"]

    summary = {"n_tests": n, "q": Q, "n_bh_significant": sum(x["bh_significant"] for x in tests),
               "dependence_note": ("tests share frames (proxies within a sequence) and "
                                   "conditions (sequences); BH controls FDR under positive "
                                   "regression dependence"),
               "tests": tests}
    out = OUT / "multitrace"; out.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out / "bh_correction.json", "w"), indent=2)

    print(f"BH correction over {n} tests (q={Q}): "
          f"{summary['n_bh_significant']} significant after correction")
    for x in tests[:6]:
        print(f"  rank {x['rank']:2d}  {x['track']:20s} {x['proxy']:12s} "
              f"mean={x['mean_pp']:+.2f}pp  p={x['p']:.4f}  "
              f"thr={x['bh_threshold']:.4f}  sig={x['bh_significant']}  ({x['p_source']})")
    print(f"  -> {out / 'bh_correction.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
