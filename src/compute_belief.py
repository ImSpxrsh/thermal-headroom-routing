"""Phase 3: two-state compute-contention belief estimator.

A two-state HMM over the sliding-window latency feature vector [p95, p99,
queue_depth], with discrete latent state in {nominal, contended}. Diagonal-Gaussian
emissions, transitions fit from a labeled contention schedule (we know when the
contention generator was active, so emissions and transitions are fit, not assumed).
Output is a belief P(contended | features_{1:t}), a distribution, never a hard label.

Section 0a: the compute model has two states only. Contention is real but
non-thermal, so we do not add a 'throttled' state we cannot faithfully observe.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NOMINAL, CONTENDED = 0, 1


@dataclass
class ComputeHMM:
    A: np.ndarray          # 2x2
    means: np.ndarray      # 2 x d
    stds: np.ndarray       # 2 x d
    pi: np.ndarray         # 2

    def _emission_loglik(self, x: np.ndarray) -> np.ndarray:
        v = self.stds ** 2
        ll = -0.5 * (np.log(2 * np.pi * v) + (x[None, :] - self.means) ** 2 / v)
        return ll.sum(axis=1)

    def filter(self, feats: np.ndarray) -> np.ndarray:
        n = len(feats)
        belief = np.zeros((n, 2))
        logA = np.log(self.A + 1e-12)
        lp = np.log(self.pi + 1e-12) + self._emission_loglik(feats[0])
        lp -= lp.max(); p = np.exp(lp); p /= p.sum(); belief[0] = p
        for t in range(1, n):
            prev = np.log(belief[t - 1] + 1e-12)
            pred = np.array([np.logaddexp.reduce(prev + logA[:, j]) for j in range(2)])
            lp = pred + self._emission_loglik(feats[t])
            lp -= lp.max(); p = np.exp(lp); p /= p.sum(); belief[t] = p
        return belief


def fit_compute_hmm(feats: np.ndarray, labels: np.ndarray, cfg: dict) -> ComputeHMM:
    """feats: (n,d) standardized latency features. labels: bool contention-active."""
    d = feats.shape[1]
    means = np.zeros((2, d)); stds = np.ones((2, d))
    for s, mask in [(NOMINAL, ~labels), (CONTENDED, labels)]:
        if mask.sum() >= 3:
            means[s] = feats[mask].mean(axis=0)
            stds[s] = np.maximum(feats[mask].std(axis=0), 0.25)
        else:
            means[s] = feats.mean(axis=0) + (0.0 if s == NOMINAL else 1.0)
            stds[s] = np.maximum(feats.std(axis=0), 0.25)
    lap = 1.0
    A = np.full((2, 2), lap)
    for t in range(1, len(labels)):
        A[int(labels[t - 1]), int(labels[t])] += 1.0
    A /= A.sum(axis=1, keepdims=True)
    base = float(labels.mean())
    pi = np.array([1 - base, base]) if 0 < base < 1 else np.array([0.9, 0.1])
    return ComputeHMM(A=A, means=means, stds=stds, pi=pi)


def standardize(feats: np.ndarray, ref: np.ndarray = None):
    ref = feats if ref is None else ref
    mu = ref.mean(axis=0); sd = np.maximum(ref.std(axis=0), 1e-6)
    return (feats - mu) / sd, (mu, sd)


def p_contended(belief: np.ndarray) -> np.ndarray:
    return belief[:, CONTENDED]
