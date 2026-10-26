"""Learned routing baseline (Part 1b of the patch).

A small supervised MLP that maps the SAME belief vector the joint policy consumes
([b_sensor_blur, b_sensor_illum, b_sensor_occ, b_compute] plus the deadline) to a config
index. It is trained offline by cross-entropy against the per-frame ORACLE config label
(the config that, with full knowledge, met the deadline at maximum accuracy). No feature
the joint policy does not also have, so the comparison is fair.

The question it answers: does the belief-space joint policy's STRUCTURE matter, or does a
learned function on the same belief features match it? Both outcomes are valid and
reported plainly (this is a baseline, not a contribution).
"""
from __future__ import annotations

from typing import List

import numpy as np

import policies as pol


def oracle_config_labels(sub, accuracy_floor: float) -> np.ndarray:
    """Per-frame oracle config index: among configs whose REALIZED latency meets the
    deadline AND whose accuracy >= floor, the highest-accuracy one; if none meet the
    deadline, the fastest config (min realized latency). This is the best achievable
    per-frame choice with full knowledge."""
    keys = pol.CONFIG_KEYS
    T = sub.T
    labels = np.zeros(T, dtype=int)
    for t in range(T):
        feasible = [(c, sub.acc[c][t]) for c in keys
                    if sub.L[c][t] <= sub.deadline_s and np.nan_to_num(sub.acc[c][t]) >= accuracy_floor]
        if feasible:
            best = max(feasible, key=lambda ca: ca[1])[0]
        else:
            best = min(keys, key=lambda c: sub.L[c][t])  # nothing meets it -> fastest
        labels[t] = keys.index(best)
    return labels


def features(sub) -> np.ndarray:
    """T x 5 feature matrix: per-channel sensor belief, compute belief, deadline (s)."""
    ch = sub.s_belief_channels
    if ch is None:  # fallback: replicate the max sensor belief across 3 slots
        ch = np.repeat(sub.s_belief[:, None], 3, axis=1)
    dl = np.full((sub.T, 1), sub.deadline_s)
    return np.concatenate([ch, sub.c_belief[:, None], dl], axis=1).astype(np.float32)


class LearnedRouter:
    """Two-hidden-layer MLP classifier over configs. Small by design."""

    def __init__(self, in_dim: int, n_configs: int, hidden=(32, 16), seed: int = 0):
        import torch
        torch.manual_seed(seed)
        self.n = n_configs
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden[0]), torch.nn.ReLU(),
            torch.nn.Linear(hidden[0], hidden[1]), torch.nn.ReLU(),
            torch.nn.Linear(hidden[1], n_configs),
        )
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 300, lr: float = 1e-2):
        import torch
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Xs = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(Xs), yt)
            loss.backward(); opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        Xs = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        with torch.no_grad():
            return self.net(Xs).argmax(1).numpy()


def choices_from_indices(idx: np.ndarray) -> List[str]:
    return [pol.CONFIG_KEYS[i] for i in idx]
