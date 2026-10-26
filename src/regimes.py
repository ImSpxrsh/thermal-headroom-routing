"""Two coupling regimes that drive when compute contention is active.

uncoupled: contention follows a fixed periodic schedule, INDEPENDENT of sensor
           faults. RQ-H benefit should shrink or vanish here.
coupled:   contention correlates with detected sensor-fault onset (with a lag and
           per-state probabilities). This EXPERIMENTALLY IMPOSES the sensor-compute
           coupling so RQ-H is testable; whether such coupling occurs naturally is
           tested separately (measure_real_coupling.py, measure_concurrent_workload.py).

Both schedules are deterministic given a seed. The alignment scheme for the
coupled regime: shift the fault-active signal forward by onset_to_contention_lag
frames, then sample contention Bernoulli with p = contention_given_fault_prob when
the lagged fault is active and contention_given_nominal_prob otherwise.
"""
from __future__ import annotations

import numpy as np


def uncoupled_schedule(n_frames: int, cfg: dict, seed: int = 0) -> np.ndarray:
    r = cfg["regimes"]["uncoupled"]
    period = int(r["period_frames"])
    duty = float(r["duty_cycle"])
    on_len = int(round(period * duty))
    # phase offset varies with seed so draws differ but stay periodic+fault-independent
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, period))
    # periodic schedule with a seed-dependent phase offset, fault-independent
    phase = (np.arange(n_frames) + offset) % period
    return phase < on_len


def coupled_schedule(n_frames: int, fault_active: np.ndarray, cfg: dict,
                     seed: int = 0) -> np.ndarray:
    r = cfg["regimes"]["coupled"]
    lag = int(r["onset_to_contention_lag_frames"])
    p_fault = float(r["contention_given_fault_prob"])
    p_nom = float(r["contention_given_nominal_prob"])
    rng = np.random.default_rng(1000 + seed)
    lagged = np.zeros(n_frames, dtype=bool)
    if lag < n_frames:
        lagged[lag:] = fault_active[: n_frames - lag]
    probs = np.where(lagged, p_fault, p_nom)
    return rng.random(n_frames) < probs


def make_schedule(regime: str, n_frames: int, fault_active: np.ndarray, cfg: dict,
                  seed: int = 0) -> np.ndarray:
    if regime == "uncoupled":
        return uncoupled_schedule(n_frames, cfg, seed)
    if regime == "coupled":
        return coupled_schedule(n_frames, fault_active, cfg, seed)
    raise ValueError(f"unknown regime {regime!r}")


def empirical_coupling(fault_active: np.ndarray, contention: np.ndarray) -> dict:
    """Report the realized correlation between faults and contention, so the regime
    is documented rather than assumed."""
    fa = fault_active.astype(float)
    co = contention.astype(float)
    if fa.std() < 1e-9 or co.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(fa, co)[0, 1])
    p_c_given_f = float(co[fault_active].mean()) if fault_active.any() else float("nan")
    p_c_given_n = float(co[~fault_active].mean()) if (~fault_active).any() else float("nan")
    return {"pearson_r": corr, "p_contention_given_fault": p_c_given_f,
            "p_contention_given_nominal": p_c_given_n}
