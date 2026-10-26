"""Trace-driven offline simulation substrate and policy runner (Phases 4-6).

Everything is profiled once (Phase 1 signals/labels, Phase 3 accuracy + latency
distributions), then a policy is run over the trace. For a given (regime, seed) the
substrate fixes the contention state timeline, the per-(frame,config) realized
latency draws, and the per-(frame,config) accuracy, so every policy is compared on
identical draws. Stochasticity across seeds (contention schedule + latency draws)
gives the confidence intervals required for every comparison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config_util import abspath
import sensor_belief as sb
import compute_belief as cb
import regimes as reg
import contention as cont
import policies as pol


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Substrate:
    T: int
    s_belief: np.ndarray            # P(fault) from sensor HMM (max over channels)
    s_instant: np.ndarray           # memoryless sensor signal in [0,1]
    c_belief: np.ndarray            # P(contended) from compute HMM
    c_instant: np.ndarray           # memoryless compute signal in [0,1]
    state: np.ndarray               # bool contended (regime ground truth)
    fault_active: np.ndarray        # bool any-fault (Track A labels)
    L: Dict[str, np.ndarray]        # realized latency per config per frame (s)
    acc: Dict[str, np.ndarray]      # agreement-with-C1 per config per frame
    fm: "pol.FrontierModel"
    kappa: float
    deadline_s: float
    p_self_transition: float        # sensor HMM nominal self-transition (for dwell)
    s_belief_channels: np.ndarray = None  # T x 3 per-channel P(fault) (blur,illum,occ); for the learned router


def calibration_kappa(cfg: dict, regime: str, cal_seeds) -> float:
    """Fit the coupling coefficient on separate calibration contention draws (NOT the
    test trace), so the joint policy never sees the realized correlation of the trace
    it is evaluated on. kappa is a structural regime parameter; this just keeps the
    fit honest. Uses the same real fault timeline (Track A) with different contention
    schedule seeds."""
    out = abspath(cfg["paths"]["outputs_dir"])
    lab = pd.read_csv(out / "phase1" / "trackA_observations.csv")
    acc_df = pd.read_csv(out / "phase3" / "per_frame_accuracy.csv")
    T = min(len(lab), int(acc_df["frame_idx"].max()) + 1)
    fault = lab["any_fault"].to_numpy(bool)[:T]
    lag = cfg["regimes"]["coupled"]["onset_to_contention_lag_frames"]
    ks = []
    for s in cal_seeds:
        state = reg.make_schedule(regime, T, fault, cfg, s)
        ks.append(pol.fit_coupling(fault, state, lag))
    return float(np.mean(ks)) if ks else 0.0


def load_phase_outputs(cfg: dict):
    out = abspath(cfg["paths"]["outputs_dir"])
    lab = pd.read_csv(out / "phase1" / "trackA_observations.csv")
    acc_df = pd.read_csv(out / "phase3" / "per_frame_accuracy.csv")
    lat = np.load(out / "phase3" / "latency_distributions.npz")
    return lab, acc_df, lat


def build_substrate(cfg: dict, regime: str, seed: int, state_override=None,
                    sensor_hmms=None) -> Substrate:
    """If state_override (a bool contention timeline) is given, it REPLACES the synthetic
    regime schedule. Use it to drive contention from a real, independently-measured load
    signal (e.g. per-frame detection count / real latency), so the fault->contention
    coupling is EMPIRICAL, not imposed (de-circularization; see measure_real_coupling.py).

    If sensor_hmms (a dict channel->ChannelHMM) is given, it REPLACES the per-track HMM fit
    with a fault-belief estimator trained on a DIFFERENT sequence. This tests whether the
    whole belief model -- not just the scalar kappa -- generalizes across sequences
    (cross-sequence HMM transfer; see measure_cross_sequence_calibration.py)."""
    lab, acc_df, latnpz = load_phase_outputs(cfg)
    # config set is dynamic: derived from config.yaml (supports the extended frontier).
    # Set the module-level CONFIG_KEYS so policies/oracle use the same set this run.
    keys = list(cfg["configs"].keys())
    pol.CONFIG_KEYS = keys
    T = min(len(lab), int(acc_df["frame_idx"].max()) + 1)
    lab = lab.iloc[:T].reset_index(drop=True)

    # --- accuracy matrices ---
    acc = {}
    for c in keys:
        sub = acc_df[acc_df["config"] == c].set_index("frame_idx")["f1"]
        acc[c] = sub.reindex(range(T)).to_numpy()
    fault_active = lab["any_fault"].to_numpy(bool)
    nominal_mask = ~fault_active
    acc_nominal = {c: float(np.nanmean(acc[c][nominal_mask])) if nominal_mask.any() else float(np.nanmean(acc[c])) for c in keys}
    acc_faulted = {c: float(np.nanmean(acc[c][fault_active])) if fault_active.any() else float(np.nanmean(acc[c])) for c in keys}

    # --- sensor belief (HMM) + memoryless instant signal, max over channels ---
    # The real frames are identical across seeds, so per-seed measurement noise is the
    # source of sensor-side variance for CIs. Both the belief (HMM-filtered) and the
    # memoryless (unfiltered) detectors see the SAME noisy observation, so the only
    # difference between them is persistence. This is exactly the condition RQ-A1 tests.
    noise_std = float(cfg.get("experiments", {}).get("observation_noise_std", 0.0))
    rng_n = np.random.default_rng(30_000 + seed)
    alpha = cfg["fault_labeling"]["ema_alpha"]
    hmms = sensor_hmms if sensor_hmms is not None else sb.fit_all_channels(lab, cfg)
    pf_chan = []
    inst_chan = []
    for ch in ("blur", "illumination", "occlusion"):
        z_clean = sb._robust_z(sb._ema(lab[ch].to_numpy(float), alpha))
        z_noisy = z_clean + rng_n.normal(0.0, noise_std, len(z_clean))
        belief = hmms[ch].filter(z_noisy)          # HMM filters the noisy obs (persistence)
        pf_chan.append(sb.p_faulted(belief))
        direction = cfg["fault_labeling"]["direction"][ch]
        inst = _sigmoid(-z_noisy) if direction == "low" else _sigmoid(z_noisy)  # no filtering
        inst_chan.append(inst)
    s_belief_channels = np.vstack(pf_chan).T   # T x 3 (blur, illumination, occlusion)
    s_belief = np.max(np.vstack(pf_chan), axis=0)
    s_instant = np.max(np.vstack(inst_chan), axis=0)
    p_self_transition = float(hmms["blur"].A[sb.NOMINAL, sb.NOMINAL])

    # --- contention state: real-load override, else synthetic regime schedule ---
    if state_override is not None:
        state = np.asarray(state_override, bool)
        if len(state) < T:              # pad short overrides with nominal so shapes match
            state = np.pad(state, (0, T - len(state)))
        state = state[:T]
    else:
        state = reg.make_schedule(regime, T, fault_active, cfg, seed)

    # --- realized latency draws per (frame, config) using state ---
    rng = np.random.default_rng(10_000 + seed)
    L = {}
    for c in keys:
        ln = latnpz[f"{c}_nominal"]; lc = latnpz[f"{c}_contended"]
        draws = np.where(state,
                         rng.choice(lc, size=T) if len(lc) else np.zeros(T),
                         rng.choice(ln, size=T) if len(ln) else np.zeros(T))
        L[c] = draws

    # --- compute belief from a probe-config latency monitor (shared by all policies) ---
    probe = "C2"
    rng2 = np.random.default_rng(20_000 + seed)
    ln = latnpz[f"{probe}_nominal"]; lc = latnpz[f"{probe}_contended"]
    probe_lat = np.where(state, rng2.choice(lc, size=T), rng2.choice(ln, size=T))
    win = cfg["contention"]["latency_window_frames"]
    fps = cfg["dataset"]["nominal_fps"]
    stats = cont.sliding_window_stats(probe_lat, win, fps)
    feats = np.column_stack([stats["p95_s"].to_numpy(), stats["p99_s"].to_numpy(),
                             stats["queue_depth"].to_numpy()])
    feats_std, _ = cb.standardize(feats)
    chmm = cb.fit_compute_hmm(feats_std, state, cfg)
    c_belief = cb.p_contended(chmm.filter(feats_std))
    # memoryless compute: instantaneous p95 -> logistic around its median
    p95 = stats["p95_s"].to_numpy()
    c_instant = _sigmoid((p95 - np.median(p95)) / (np.std(p95) + 1e-9))

    # --- frontier model + coupling ---
    # Prefer THIS track's own profiled deadline (outputs/<track>/phase3), not config.yaml's
    # value_ms (which holds whichever track was profiled last). Critical when analyzing a
    # track other than the one config.yaml currently points at (e.g. the add-on sections).
    p3sum = abspath(cfg["paths"]["outputs_dir"]) / "phase3" / "phase3_summary.json"
    if p3sum.exists():
        deadline_s = float(json.load(open(p3sum))["deadline_ms"]) / 1e3
    else:
        deadline_s = float(cfg["deadline"]["value_ms"]) / 1e3
    lat_dists = {c: {"nominal": latnpz[f"{c}_nominal"], "contended": latnpz[f"{c}_contended"]}
                 for c in keys}
    fm = pol.build_frontier_model(deadline_s, lat_dists, acc_nominal, acc_faulted)
    lag = cfg["regimes"]["coupled"]["onset_to_contention_lag_frames"]
    kappa = pol.fit_coupling(fault_active, state, lag)

    return Substrate(T=T, s_belief=s_belief, s_instant=s_instant, c_belief=c_belief,
                     c_instant=c_instant, state=state, fault_active=fault_active,
                     L=L, acc=acc, fm=fm, kappa=kappa, deadline_s=deadline_s,
                     p_self_transition=p_self_transition, s_belief_channels=s_belief_channels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _onset_indices(b: np.ndarray) -> List[int]:
    return [i for i in range(len(b)) if b[i] and (i == 0 or not b[i - 1])]


def _clearance_indices(b: np.ndarray) -> List[int]:
    return [i for i in range(len(b)) if (not b[i]) and i > 0 and b[i - 1]]


def reconfig_latency(choices: List[str], events: List[int]) -> float:
    """Median frames from each event to the next config change at/after the event."""
    lags = []
    for e in events:
        for t in range(e, len(choices)):
            if t > 0 and choices[t] != choices[t - 1]:
                lags.append(t - e); break
    return float(np.median(lags)) if lags else float("nan")


def _score_choices(choices: List[str], sub: "Substrate") -> dict:
    miss = 0
    served = 0
    acc_vals = []
    abstain = 0
    for t, ch in enumerate(choices):
        if ch == pol.ABSTAIN:
            abstain += 1
            continue
        served += 1
        if sub.L[ch][t] > sub.deadline_s:
            miss += 1
        a = sub.acc[ch][t]
        if not np.isnan(a):
            acc_vals.append(a)

    switches = sum(1 for t in range(1, len(choices)) if choices[t] != choices[t - 1])
    return {
        "deadline_miss_rate": miss / max(served, 1),
        "served_frames": served,
        "abstain_frames": abstain,
        "mean_accuracy": float(np.mean(acc_vals)) if acc_vals else float("nan"),
        "switch_count": switches,
        "switch_rate": switches / max(sub.T - 1, 1),
        "onset_to_reconfig": reconfig_latency(choices, _onset_indices(sub.fault_active)),
        "clearance_to_reconfig": reconfig_latency(choices, _clearance_indices(sub.fault_active)),
        "choices": choices,
    }


def run_policy(policy, sub: Substrate, memoryless: bool = False) -> dict:
    choices = []
    for t in range(sub.T):
        if memoryless:
            ch = policy.decide(sub.s_instant[t], sub.c_instant[t])
        else:
            ch = policy.decide(sub.s_belief[t], sub.c_belief[t])
        choices.append(ch)
    return _score_choices(choices, sub)


def run_oracle_policy(policy, sub: Substrate) -> dict:
    """Runner for OracleContentionPolicy: decide(s_belief, true_contended)."""
    choices = [policy.decide(sub.s_belief[t], bool(sub.state[t])) for t in range(sub.T)]
    return _score_choices(choices, sub)


def run_reactive_policy(policy, sub: Substrate) -> dict:
    """Runner for ReactiveLatencyPolicy: decide(c_instant) only, no sensor info."""
    choices = [policy.decide(sub.c_instant[t]) for t in range(sub.T)]
    return _score_choices(choices, sub)
