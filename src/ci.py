"""Confidence intervals over seeds/draws. Every cross-method comparison reports a
95% CI (autonomy rule 5). Uses the t-interval for small sample sizes (>=5 seeds)."""
from __future__ import annotations

import numpy as np


def mean_ci(values, ci: float = 95.0):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    n = len(v)
    mean = float(np.mean(v)) if n else float("nan")
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "n": n, "sd": 0.0, "half_width": 0.0}
    sd = float(np.std(v, ddof=1))
    se = sd / np.sqrt(n)
    from scipy import stats
    t = stats.t.ppf(0.5 + ci / 200.0, df=n - 1)
    hw = t * se
    return {"mean": mean, "lo": mean - hw, "hi": mean + hw, "n": n, "sd": sd, "half_width": hw}


def paired_diff_ci(a, b, ci: float = 95.0):
    """CI on the paired difference a-b (same seeds), whether it excludes 0, and the
    two-sided paired t-test p-value computed directly from the seed-level differences."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    res = mean_ci(d, ci)
    res["significant"] = bool(res["lo"] > 0 or res["hi"] < 0)
    res["p_two_sided"] = paired_p_two_sided(d)
    return res


def paired_p_two_sided(d) -> float:
    """Two-sided paired t-test p-value from the per-seed differences d."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return 1.0
    sd = float(np.std(d, ddof=1))
    if sd == 0.0:
        # all differences identical: zero difference -> no evidence; nonzero constant
        # difference -> degenerate (report smallest representable evidence honestly as 0)
        return 1.0 if float(np.mean(d)) == 0.0 else 0.0
    from scipy import stats
    t = float(np.mean(d)) / (sd / np.sqrt(n))
    return float(2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1)))
