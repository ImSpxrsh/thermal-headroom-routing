"""Extreme-environment extension: a lumped thermal model of the perception SoC and
thermal-throttling contention that is ENDOGENOUS to the router's own choices.

In the original study contention is exogenous (an imposed schedule) and two-state,
non-thermal. Robots in extreme heat (desert field work, sun-facing surfaces in
space) hit a different failure: the perception stack heats its own SoC, DVFS
throttles it, and every config becomes slower -- a contention state the router
CAUSES. This module models that with a first-order RC thermal node:

    T[t+1] = T[t] + (dt / tau) * (T_amb + R_th * P(t) - T[t])

with power P(t) set by the config served at t (duty = nominal median latency /
deadline, clipped to 1) and a hysteretic throttle (enter >= t_hi, exit <= t_lo),
like a real DVFS governor. While throttled, latencies are drawn from the profiled
CONTENDED distribution (a stand-in for a down-clocked SoC) and power is scaled by
throttle_power_scale.

The parameters are illustrative, Jetson-Orin-class values, NOT measured: they are a
modeling assumption stated in the paper. jetson_harness.py can measure the real
ones (tegrastats exposes SoC temperature and power) and drop them into config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

import policies as pol


@dataclass
class ThermalModel:
    dt_s: float = 0.1                 # one frame at 10 fps
    tau_s: float = 30.0               # thermal time constant
    r_th: float = 2.0                 # degC per W, junction-to-ambient
    p_idle_w: float = 5.0
    p_max_w: float = 25.0
    t_hi: float = 90.0                # throttle entry
    t_lo: float = 85.0                # throttle exit
    throttle_power_scale: float = 0.6

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ThermalModel":
        th = cfg.get("thermal", {})
        return cls(**{k: float(v) for k, v in th.get("model", {}).items()})

    def power(self, duty: float, throttled: bool) -> float:
        p = self.p_idle_w + (self.p_max_w - self.p_idle_w) * float(np.clip(duty, 0.0, 1.0))
        return p * (self.throttle_power_scale if throttled else 1.0)

    def step(self, temp: float, t_amb: float, p_w: float) -> float:
        return temp + (self.dt_s / self.tau_s) * (t_amb + self.r_th * p_w - temp)

    def forecast(self, temp: float, t_amb: float, p_w: float, horizon_frames: int) -> float:
        """Closed-form temperature after holding power p_w for horizon_frames."""
        t_ss = t_amb + self.r_th * p_w
        decay = (1.0 - self.dt_s / self.tau_s) ** horizon_frames
        return t_ss + (temp - t_ss) * decay


def config_duty(fm: "pol.FrontierModel", lat_dists: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
    return {c: float(np.median(lat_dists[c]["nominal"])) / fm.deadline_s for c in lat_dists}


# ---------------------------------------------------------------------------
# Online compute belief (closed loop: the router only sees the latency of the
# config it actually served, one frame late)
# ---------------------------------------------------------------------------
class OnlineComputeBelief:
    """Two-state forward filter on a binary 'slow' observation: the last served
    config's latency exceeded slow_ratio x its own nominal median."""
    def __init__(self, stay_nom=0.95, stay_con=0.90, p_slow_nom=0.10, p_slow_con=0.90):
        self.A = np.array([[stay_nom, 1 - stay_nom], [1 - stay_con, stay_con]])
        self.E = np.array([p_slow_nom, p_slow_con])
        self.b = np.array([0.95, 0.05])

    def update(self, slow: Optional[bool]) -> float:
        prior = self.b @ self.A
        if slow is None:
            self.b = prior
        else:
            lik = self.E if slow else 1.0 - self.E
            post = prior * lik
            self.b = post / post.sum()
        return float(self.b[1])


# ---------------------------------------------------------------------------
# Policies for the thermal / compound regimes. All share the original frontier
# objective (predicted-state feasibility, then max expected accuracy).
# ---------------------------------------------------------------------------
@dataclass
class ThermalAwarePolicy:
    """Headroom-aware router: before committing a config, forecast the SoC
    temperature after holding it for horizon_frames and reject configs that would
    cross the throttle threshold (minus a margin). Uses the on-die temperature
    reading (tegrastats-style, with noise) and the thermal model; optionally also
    the fault-contention coupling (kappa > 0 gives the compound Joint+Thermal policy).
    The cheapest config is always allowed, so it never deadlocks."""
    fm: "pol.FrontierModel"
    model: ThermalModel
    duty: Dict[str, float]
    hysteresis: "pol.Hysteresis"
    kappa: float = 0.0
    horizon_frames: int = 50
    margin_c: float = 2.0

    def decide(self, s_belief: float, c_belief: float, temp_meas: float, t_amb: float,
               throttled_belief: bool) -> str:
        p_cont = 1.0 - (1.0 - c_belief) * (1.0 - self.kappa * s_belief)
        state = "contended" if (p_cont >= 0.5 or throttled_belief) else "nominal"
        cheapest = min(pol.CONFIG_KEYS, key=lambda c: self.fm.cost_rank[c])
        allowed = []
        for c in pol.CONFIG_KEYS:
            if self.fm.p_meet[c][state] < 0.5:
                continue
            t_h = self.model.forecast(temp_meas, t_amb, self.model.power(self.duty[c], False),
                                      self.horizon_frames)
            if t_h < self.model.t_hi - self.margin_c or c == cheapest:
                allowed.append(c)
        if not allowed:
            allowed = [cheapest]
        best = max(allowed, key=lambda c: self.fm.exp_accuracy(c, s_belief))
        return self.hysteresis.step(best)


@dataclass
class ThermostatPolicy:
    """Reviewer baseline with no forecast: the belief router, except that the most
    expensive configuration is barred whenever the measured temperature is at or above
    t_set. Same feasibility/accuracy objective and dwell as every other policy. The
    a-priori setting t_set = t_hi - margin (88 degC) uses exactly the threshold THR
    compares its forecast against, so the two differ only in forecasting."""
    fm: "pol.FrontierModel"
    hysteresis: "pol.Hysteresis"
    t_set: float = 88.0

    def decide(self, s_belief: float, c_belief: float, temp_meas: float) -> str:
        state = "contended" if c_belief >= 0.5 else "nominal"
        priciest = max(pol.CONFIG_KEYS, key=lambda c: self.fm.cost_rank[c])
        cands = [c for c in pol.CONFIG_KEYS if not (temp_meas >= self.t_set and c == priciest)]
        feasible = [c for c in cands if self.fm.p_meet[c][state] >= 0.5]
        if not feasible:
            feasible = [max(cands, key=lambda c: self.fm.p_meet[c][state])]
        best = max(feasible, key=lambda c: self.fm.exp_accuracy(c, s_belief))
        return self.hysteresis.step(best)


def run_closed_loop(kind: str, sub, lat_dists, model: ThermalModel, t_amb: float,
                    rng: np.random.Generator, n_frames: int, kappa: float = 0.0,
                    dwell: int = 10, exo_contention: Optional[np.ndarray] = None,
                    policy_model: Optional[ThermalModel] = None,
                    temp_noise_c: float = 1.0, c1_budget: float = 1.0,
                    return_trace: bool = False, duty: Optional[Dict[str, float]] = None,
                    thr_params: Optional[dict] = None, t_set: float = 88.0) -> dict:
    """Run one policy in closed loop with the thermal plant over n_frames (the real
    trace is tiled to reach a mission-length run). exo_contention (bool per frame of
    the tiled run) adds an exogenous contention source on top of throttling (the
    compound regime). policy_model lets the policy hold a MISSPECIFIED thermal model
    (robustness check); the plant always uses `model`.

    kind in {static_C1, static_C4, reactive, decoupled, joint, thermal, joint_thermal,
    duty_matched, fixed_cap, thermostat}. duty_matched / fixed_cap are the decoupled policy with a running cap on the
    fraction of frames served by C1 (c1_budget, set to what the thermal policy used):
    the control for "the thermal policy only wins because it runs C1 less", since it
    spends the same C1 budget with no knowledge of temperature. fixed_cap is the same
    mechanism with a budget tuned offline at one ambient. thermostat bars C1 while the
    measured temperature is >= t_set. `duty` overrides the per-config duty (power)
    model so a deadline sweep does not change power; thr_params overrides THR's
    horizon_frames / margin_c (sensitivity analysis).
    """
    T0 = sub.T
    idx = np.arange(n_frames) % T0
    s_bel = sub.s_belief[idx]
    acc = {c: sub.acc[c][idx] for c in pol.CONFIG_KEYS}
    duty = duty if duty is not None else config_duty(sub.fm, lat_dists)
    med_nom = {c: float(np.median(lat_dists[c]["nominal"])) for c in pol.CONFIG_KEYS}
    # pre-draw both latency states per (frame, config) so every policy sees identical draws
    Ln = {c: rng.choice(lat_dists[c]["nominal"], size=n_frames) for c in pol.CONFIG_KEYS}
    Lc = {c: rng.choice(lat_dists[c]["contended"], size=n_frames) for c in pol.CONFIG_KEYS}
    tnoise = rng.normal(0.0, temp_noise_c, n_frames)

    hyst = pol.Hysteresis(dwell)
    pm = policy_model or model
    if kind in ("thermal", "joint_thermal"):
        tp = ThermalAwarePolicy(sub.fm, pm, duty, hyst, kappa=kappa if kind == "joint_thermal" else 0.0,
                                **(thr_params or {}))
    elif kind == "thermostat":
        tp = ThermostatPolicy(sub.fm, hyst, t_set=t_set)
    elif kind == "joint":
        tp = pol.JointPolicy(sub.fm, kappa, 0.85, hyst)
    elif kind in ("decoupled", "duty_matched", "fixed_cap"):
        tp = pol.DecoupledPolicy(sub.fm, 0.85, hyst)
    elif kind == "reactive":
        tp = pol.ReactiveLatencyPolicy(sub.fm, 0.85, hyst)
    else:
        tp = None

    cbel = OnlineComputeBelief()
    temp = t_amb + model.r_th * model.p_idle_w
    temp = min(temp, model.t_lo - 1.0)   # start below throttle so the run is not pre-throttled
    throttled = False
    choices: List[str] = []
    miss = np.zeros(n_frames, bool)
    served_acc = np.zeros(n_frames)
    temps = np.zeros(n_frames)
    thr = np.zeros(n_frames, bool)
    energy_j = 0.0
    last_slow: Optional[bool] = None
    n_c1 = 0
    for t in range(n_frames):
        cb = cbel.update(last_slow)
        if kind == "static_C1":
            ch = "C1"
        elif kind == "static_C4":
            ch = "C4"
        elif kind in ("thermal", "joint_thermal"):
            ch = tp.decide(s_bel[t], cb, temp + tnoise[t], t_amb, cb >= 0.5)
        elif kind == "thermostat":
            ch = tp.decide(s_bel[t], cb, temp + tnoise[t])
        elif kind == "reactive":
            ch = tp.decide(cb)
        else:
            ch = tp.decide(s_bel[t], cb)
            if kind in ("duty_matched", "fixed_cap") and ch == "C1" and (n_c1 + 1) / (t + 1) > c1_budget:
                ch = "C2"
                tp.hysteresis.current = ch
        n_c1 += ch == "C1"
        contended = throttled or (exo_contention is not None and bool(exo_contention[t]))
        lat = Lc[ch][t] if contended else Ln[ch][t]
        miss[t] = lat > sub.deadline_s
        a = acc[ch][t]
        served_acc[t] = 0.0 if np.isnan(a) else a
        p = model.power(duty[ch], throttled)
        energy_j += p * model.dt_s
        temps[t] = temp; thr[t] = throttled
        temp = model.step(temp, t_amb, p)
        if not throttled and temp >= model.t_hi:
            throttled = True
        elif throttled and temp <= model.t_lo:
            throttled = False
        last_slow = bool(lat > 1.5 * med_nom[ch])
        choices.append(ch)

    out = {
        "deadline_miss_rate": float(miss.mean()),
        "mean_accuracy": float(np.nanmean([acc[c][t] for t, c in enumerate(choices)])),
        "on_time_utility": float(np.mean(np.where(miss, 0.0, served_acc))),
        "throttled_frac": float(thr.mean()),
        "peak_temp_c": float(temps.max()),
        "mean_power_w": float(energy_j / (n_frames * model.dt_s)),
        "frac_C1": float(np.mean([c == "C1" for c in choices])),
        **burst_stats(miss),
    }
    if return_trace:
        out["trace"] = {"temp_c": temps.tolist(), "throttled": thr.tolist(),
                        "miss": miss.tolist(), "choices": choices}
    return out


# ---------------------------------------------------------------------------
# Burst (perception blackout) metrics
# ---------------------------------------------------------------------------
def miss_runs(miss: np.ndarray) -> List[int]:
    runs, n = [], 0
    for m in miss:
        if m:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return runs


def burst_stats(miss: np.ndarray, k: int = 3) -> dict:
    """Longest run of consecutive deadline misses (a perception blackout), and the
    fraction of frames that fall inside a run of >= k misses (k=3 is 0.3 s at 10 fps)."""
    runs = miss_runs(np.asarray(miss, bool))
    in_burst = sum(r for r in runs if r >= k)
    return {"max_miss_run": int(max(runs) if runs else 0),
            f"burst_frac_k{k}": float(in_burst / max(len(miss), 1))}
