"""Phase 2: per-channel sensor-fault belief estimator.

A switching HMM over the continuous (robust-z, EMA-smoothed) observation, with a
discrete latent state z in {nominal, degrading, faulted, recovering}. Emissions are
Gaussian per state. Transitions are fit from the labeled segments, allowing
asymmetric onset and recovery dynamics. The output is a belief vector
b_t = P(z_t | o_{1:t}) per channel, a distribution, never a hard label.

The estimator is intentionally a recursive Bayesian filter (forward algorithm), so
it carries persistence: this is the design choice RQ-A1 puts on trial against a
memoryless detector with the same features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

NOMINAL, DEGRADING, FAULTED, RECOVERING = 0, 1, 2, 3
STATE_NAMES = ["nominal", "degrading", "faulted", "recovering"]


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad if mad > 1e-9 else (np.std(x) + 1e-9)
    return (x - med) / scale


def derive_state_sequence(fault_bool: np.ndarray, onset_k: int, recov_k: int) -> np.ndarray:
    """Expand a binary fault signal into a 4-state label sequence for fitting."""
    n = len(fault_bool)
    states = np.full(n, NOMINAL, dtype=int)
    in_seg = False
    s0 = 0
    segs = []
    for k in range(n):
        if fault_bool[k] and not in_seg:
            in_seg = True; s0 = k
        elif not fault_bool[k] and in_seg:
            in_seg = False; segs.append((s0, k - 1))
    if in_seg:
        segs.append((s0, n - 1))
    for s0, s1 in segs:
        states[s0:s1 + 1] = FAULTED
        deg_end = min(s1, s0 + onset_k - 1)
        states[s0:deg_end + 1] = DEGRADING
        rec_start = s1 + 1
        rec_end = min(n - 1, s1 + recov_k)
        if rec_start <= rec_end:
            states[rec_start:rec_end + 1] = RECOVERING
    return states


@dataclass
class ChannelHMM:
    A: np.ndarray            # 4x4 transition
    means: np.ndarray        # per-state emission mean (on robust-z obs)
    stds: np.ndarray         # per-state emission std
    pi: np.ndarray           # initial dist
    direction: str           # 'low' or 'high' (which tail is bad)

    def emission_loglik(self, o: float) -> np.ndarray:
        v = self.stds ** 2
        return -0.5 * (np.log(2 * np.pi * v) + (o - self.means) ** 2 / v)

    def filter(self, obs: np.ndarray) -> np.ndarray:
        """Forward filtering -> belief[t, state] = P(z_t | o_{1:t})."""
        n = len(obs)
        belief = np.zeros((n, 4))
        logA = np.log(self.A + 1e-12)
        # t=0
        lp = np.log(self.pi + 1e-12) + self.emission_loglik(obs[0])
        lp -= lp.max(); p = np.exp(lp); p /= p.sum()
        belief[0] = p
        for t in range(1, n):
            # predict: log-sum-exp over previous states
            prev = np.log(belief[t - 1] + 1e-12)
            pred = np.array([np.logaddexp.reduce(prev + logA[:, j]) for j in range(4)])
            lp = pred + self.emission_loglik(obs[t])
            lp -= lp.max(); p = np.exp(lp); p /= p.sum()
            belief[t] = p
        return belief


def fit_channel_hmm(raw_signal: np.ndarray, fault_bool: np.ndarray, cfg: dict,
                    direction: str) -> ChannelHMM:
    sh = cfg["sensor_hmm"]
    alpha = cfg["fault_labeling"]["ema_alpha"]
    obs = _robust_z(_ema(raw_signal, alpha))
    states = derive_state_sequence(fault_bool, sh["label_onset_frames"],
                                   sh["label_recovery_frames"])

    # emission fit (fallback priors when a state is unobserved)
    thr = cfg["fault_labeling"]["robust_z_threshold"]
    sign = -1.0 if direction == "low" else 1.0
    prior_mean = {NOMINAL: 0.0, DEGRADING: sign * thr * 0.6,
                  FAULTED: sign * thr * 1.6, RECOVERING: sign * thr * 0.6}
    means = np.zeros(4); stds = np.zeros(4)
    floor = sh["min_emission_std"]
    for s in range(4):
        vals = obs[states == s]
        if len(vals) >= 5:
            means[s] = float(np.mean(vals)); stds[s] = max(float(np.std(vals)), floor)
        else:
            means[s] = prior_mean[s]; stds[s] = max(1.0, floor)

    # transition fit with Laplace smoothing
    lap = sh["transition_laplace"]
    A = np.full((4, 4), lap, dtype=np.float64)
    for t in range(1, len(states)):
        A[states[t - 1], states[t]] += 1.0
    A /= A.sum(axis=1, keepdims=True)

    pi = np.array([0.97, 0.01, 0.01, 0.01])
    return ChannelHMM(A=A, means=means, stds=stds, pi=pi, direction=direction)


def belief_for_channel(raw_signal: np.ndarray, hmm: ChannelHMM, cfg: dict) -> np.ndarray:
    alpha = cfg["fault_labeling"]["ema_alpha"]
    obs = _robust_z(_ema(raw_signal, alpha))
    return hmm.filter(obs)


def p_faulted(belief: np.ndarray) -> np.ndarray:
    """Probability mass on the 'bad' states (faulted + degrading)."""
    return belief[:, FAULTED] + belief[:, DEGRADING]


def fit_all_channels(df, cfg: dict) -> Dict[str, ChannelHMM]:
    out = {}
    for ch in ("blur", "illumination", "occlusion"):
        direction = cfg["fault_labeling"]["direction"][ch]
        out[ch] = fit_channel_hmm(df[ch].to_numpy(float),
                                  df[f"{ch}_fault"].to_numpy(bool), cfg, direction)
    return out
