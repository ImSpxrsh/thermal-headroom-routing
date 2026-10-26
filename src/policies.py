"""Phase 4: routing policies.

The centerpiece comparison (RQ-H) is JointPolicy vs DecoupledPolicy. Both consume
the SAME two belief signals (sensor-fault belief and compute-contention belief). The
only difference is whether the policy models the COUPLING between them:

  JointPolicy    fuses compute belief with a data-fit prediction of contention from
                 sensor-fault belief (the coupling term), so it can anticipate
                 contention at fault onset before the latency monitor reacts.
  DecoupledPolicy runs a sensor-driven choice and a compute-driven choice through
                 separate thresholds and combines them by a fixed "more conservative"
                 rule. It never uses one signal to predict the other.

The novel claim lives entirely in the delta between these two on deadline-miss rate,
and should appear in the coupled regime and shrink in the uncoupled regime. The
coupling coefficient is fit from data (~0 when faults do not predict contention), so
the decoupled baseline is a fair union, not a strawman.

Config cost order (most to least expensive): C1 > C3 > C2 > C4 by compute. We use a
"safer for the deadline" = cheaper ordering for the conservative combine rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

ABSTAIN = "ABSTAIN"
CONFIG_KEYS = ["C1", "C2", "C3", "C4"]


# ---------------------------------------------------------------------------
# Models shared by the policies (fit from data / Phase 3 profile, not hand-set)
# ---------------------------------------------------------------------------
def fit_coupling(fault_active: np.ndarray, contended: np.ndarray, lag: int) -> float:
    """Coupling coefficient kappa = P(contended_{t} | fault_{t-lag}) -
    P(contended_{t} | not fault_{t-lag}), clamped to [0,1]. ~0 when faults do not
    predict contention (uncoupled regime), so the joint gains nothing there."""
    n = len(fault_active)
    if lag >= n:
        return 0.0
    lead = np.zeros(n, dtype=bool)
    lead[lag:] = fault_active[: n - lag]
    if lead.any() and (~lead).any():
        p_f = contended[lead].mean()
        p_n = contended[~lead].mean()
        return float(np.clip(p_f - p_n, 0.0, 1.0))
    return 0.0


def latency_cdf_leq(samples: np.ndarray, deadline_s: float) -> float:
    """Empirical P(latency <= deadline) from a sample of realized latencies."""
    if len(samples) == 0:
        return 1.0
    return float(np.mean(samples <= deadline_s))


@dataclass
class FrontierModel:
    """Everything the policies need to reason about configs, from Phase 3."""
    deadline_s: float
    # P(meet deadline | config, compute_state) from empirical latency dists
    p_meet: Dict[str, Dict[str, float]]              # p_meet[config][state]
    # expected accuracy | config, sensor condition (nominal vs faulted bucket)
    acc_nominal: Dict[str, float]
    acc_faulted: Dict[str, float]
    cost_rank: Dict[str, int] = field(default_factory=lambda: {"C4": 0, "C2": 1, "C3": 2, "C1": 3})

    def exp_accuracy(self, c: str, s_belief: float) -> float:
        return (1 - s_belief) * self.acc_nominal[c] + s_belief * self.acc_faulted[c]

    def p_meet_given(self, c: str, p_contended: float) -> float:
        return (1 - p_contended) * self.p_meet[c]["nominal"] + p_contended * self.p_meet[c]["contended"]


def build_frontier_model(deadline_s: float, lat_dists: Dict[str, Dict[str, np.ndarray]],
                         acc_nominal: Dict[str, float], acc_faulted: Dict[str, float]) -> FrontierModel:
    p_meet = {}
    for c in lat_dists:
        p_meet[c] = {st: latency_cdf_leq(lat_dists[c][st], deadline_s)
                     for st in ("nominal", "contended")}
    # cost rank by nominal median latency (cheapest = lowest latency); works for any
    # config-set size. For the default 4 configs this reproduces the original ordering.
    med = {c: float(np.median(lat_dists[c]["nominal"])) if len(lat_dists[c]["nominal"]) else 0.0
           for c in lat_dists}
    cost_rank = {c: i for i, c in enumerate(sorted(lat_dists, key=lambda c: med[c]))}
    return FrontierModel(deadline_s=deadline_s, p_meet=p_meet,
                         acc_nominal=acc_nominal, acc_faulted=acc_faulted, cost_rank=cost_rank)


# ---------------------------------------------------------------------------
# Hysteresis / dwell
# ---------------------------------------------------------------------------
class Hysteresis:
    """Holds a chosen config for at least dwell frames unless the alternative is
    decisively better, to suppress chattering. Two variants:
      fixed:         dwell_frames from config (hand-tuned).
      model_derived: dwell set from the belief model's expected false-transition rate.
    """
    def __init__(self, dwell_frames: int):
        self.dwell = max(1, int(dwell_frames))
        self.current: Optional[str] = None
        self._held = 0

    def step(self, proposed: str) -> str:
        if self.current is None:
            self.current = proposed; self._held = 0
            return self.current
        if proposed == self.current:
            self._held += 1
            return self.current
        # different proposal: only switch if dwell satisfied
        if self._held >= self.dwell:
            self.current = proposed; self._held = 0
        else:
            self._held += 1
        return self.current


def model_derived_dwell(false_transition_rate_target: float, p_self_transition: float) -> int:
    """Dwell time so the expected number of spurious switches per fault-free run is
    bounded by the target. With per-frame false-transition probability q = 1 -
    p_self_transition, holding d frames cuts spurious switches by ~q^? We set
    dwell = ceil(log(target)/log(q)) bounded to a sane range."""
    q = max(1e-3, 1.0 - p_self_transition)
    if q >= 1.0:
        return 1
    import math
    d = math.ceil(math.log(max(false_transition_rate_target, 1e-3)) / math.log(q))
    return int(np.clip(d, 1, 40))


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
# Shared soft objective: maximize expected reliability = expected accuracy under the
# current sensor belief x probability of meeting the deadline under the contention
# probability p_cont. Both policies use BOTH belief signals through THIS objective.
# The ONLY difference between joint and decoupled is how p_cont is formed:
#   joint:     p_cont = noisy-OR(compute belief, coupling prediction from sensor belief)
#   decoupled: p_cont = compute belief alone (no cross-term)
# This makes the decoupled a fair union (it still uses the sensor signal for accuracy
# weighting), not a strawman, and isolates the coupling as the sole difference (RQ-H).
def _choose(fm: FrontierModel, s_belief: float, p_cont: float, rt: float,
            hysteresis: "Hysteresis", allow_abstain: bool) -> str:
    """Predicted-compute-state feasibility, then maximum accuracy among feasible.

    The deadline is the median C1-nominal latency (locked rule), so feasibility in a
    state is "the config's median latency in that state meets the deadline", i.e.
    p_meet[c][state] >= 0.5. The predicted state is the MAP of p_cont. This realizes
    the spec's intent: C1 is usable when compute is predicted nominal and is dropped
    when contention is predicted, while cheaper configs remain feasible. The joint and
    decoupled differ ONLY in how p_cont (hence the predicted state) is formed, so any
    deadline-miss gap is attributable to the coupling.
    """
    state = "contended" if p_cont >= 0.5 else "nominal"
    pmeet = {c: fm.p_meet[c][state] for c in CONFIG_KEYS}
    # abstain only if no config even reaches the reliability target in the predicted state
    if allow_abstain and max(pmeet.values()) < rt:
        return hysteresis.step(ABSTAIN)
    feasible = [c for c in CONFIG_KEYS if pmeet[c] >= 0.5]
    if not feasible:
        feasible = [max(CONFIG_KEYS, key=lambda c: pmeet[c])]  # safest available
    best = max(feasible, key=lambda c: fm.exp_accuracy(c, s_belief))
    return hysteresis.step(best)


@dataclass
class JointPolicy:
    fm: FrontierModel
    kappa: float                  # coupling coefficient (fit from a calibration draw)
    reliability_target: float
    hysteresis: Hysteresis
    allow_abstain: bool = False   # off for the RQ-H headline (no abstention confound)

    def decide(self, s_belief: float, c_belief: float) -> str:
        # fuse compute belief with the coupling prediction from sensor belief (noisy-OR)
        p_cont = 1.0 - (1.0 - c_belief) * (1.0 - self.kappa * s_belief)
        return _choose(self.fm, s_belief, p_cont, self.reliability_target,
                       self.hysteresis, self.allow_abstain)


@dataclass
class DecoupledPolicy:
    fm: FrontierModel
    reliability_target: float
    hysteresis: Hysteresis
    allow_abstain: bool = False

    def decide(self, s_belief: float, c_belief: float) -> str:
        # compute belief ALONE drives p_cont (no coupling cross-term); the sensor
        # belief still enters via expected accuracy in the shared objective.
        return _choose(self.fm, s_belief, c_belief, self.reliability_target,
                       self.hysteresis, self.allow_abstain)


@dataclass
class ThresholdPolicy:
    """Rule-based coupling ablation: instead of the soft noisy-OR, fuse the two beliefs
    by a hard union p_cont = max(c_belief, 1[s_belief > 0.5]). Same feasible-set / max-
    accuracy objective and same hysteresis as the joint policy, so the ONLY difference
    from JointPolicy is the fusion FUNCTION (hard threshold vs soft noisy-OR). Tests
    whether the specific noisy-OR form matters or whether any sensor-triggered
    anticipation suffices (paper Table: 'Is the noisy-OR necessary?')."""
    fm: FrontierModel
    reliability_target: float
    hysteresis: Hysteresis
    s_threshold: float = 0.5
    allow_abstain: bool = False

    def decide(self, s_belief: float, c_belief: float) -> str:
        p_cont = max(c_belief, 1.0 if s_belief > self.s_threshold else 0.0)
        return _choose(self.fm, s_belief, p_cont, self.reliability_target,
                       self.hysteresis, self.allow_abstain)


@dataclass
class OracleContentionPolicy:
    """Upper-bound baseline: knows the TRUE contention state (ground-truth regime
    label, not a belief) exactly, every frame. Still uses the sensor belief for
    accuracy weighting (same shared objective as Joint/Decoupled), so the ONLY
    thing this ablates is compute-state uncertainty -- it bounds how much the
    joint policy could gain from a perfect (rather than HMM-estimated) contention
    read. Not a deployable policy: no real system has oracle access to a
    competing process's future load."""
    fm: FrontierModel
    reliability_target: float
    hysteresis: Hysteresis
    allow_abstain: bool = False

    def decide(self, s_belief: float, true_contended: bool) -> str:
        p_cont = 1.0 if true_contended else 0.0
        return _choose(self.fm, s_belief, p_cont, self.reliability_target,
                       self.hysteresis, self.allow_abstain)


@dataclass
class ReactiveLatencyPolicy:
    """Strong naive baseline: reacts ONLY to the observed latency trend (the same
    memoryless compute signal c_instant used elsewhere), with NO sensor-fault
    information at all -- not even for accuracy weighting (unlike Decoupled, which
    still uses s_belief to weight expected accuracy). Feasibility is decided from
    recent latency alone, and among feasible configs it picks the one with the
    best NOMINAL-condition accuracy, since it has no way to know a fault is
    present. Tests whether plain latency-reactive scheduling -- no belief about
    sensor degradation at all -- already captures the routing benefit."""
    fm: FrontierModel
    reliability_target: float
    hysteresis: Hysteresis
    allow_abstain: bool = False

    def decide(self, c_instant: float) -> str:
        state = "contended" if c_instant >= 0.5 else "nominal"
        pmeet = {c: self.fm.p_meet[c][state] for c in CONFIG_KEYS}
        if self.allow_abstain and max(pmeet.values()) < self.reliability_target:
            return self.hysteresis.step(ABSTAIN)
        feasible = [c for c in CONFIG_KEYS if pmeet[c] >= 0.5]
        if not feasible:
            feasible = [max(CONFIG_KEYS, key=lambda c: pmeet[c])]
        best = max(feasible, key=lambda c: self.fm.acc_nominal[c])
        return self.hysteresis.step(best)


@dataclass
class MemorylessPolicy:
    """RQ-A1 comparison: same features and same objective, belief update bypassed. It
    consumes the instantaneous (unfiltered) sensor and compute signals with no
    persistence and no dwell, so it is free to chatter."""
    fm: FrontierModel
    reliability_target: float
    allow_abstain: bool = False

    def decide(self, s_instant: float, c_instant: float) -> str:
        return _choose(self.fm, s_instant, c_instant, self.reliability_target,
                       Hysteresis(1), self.allow_abstain)
