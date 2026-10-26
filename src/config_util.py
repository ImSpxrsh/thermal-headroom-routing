"""Shared configuration loading and device detection.

All parameters live in config.yaml (overridable via .env). Nothing that belongs
in config.yaml is hardcoded in the pipeline code.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Environment overrides for the few path/runtime knobs in .env.
    paths = cfg.setdefault("paths", {})
    if os.getenv("RAW_BAG"):
        paths["raw_bag"] = os.environ["RAW_BAG"]
    if os.getenv("FRAMES_DIR"):
        paths["frames_dir"] = os.environ["FRAMES_DIR"]
    if os.getenv("OUTPUTS_DIR"):
        paths["outputs_dir"] = os.environ["OUTPUTS_DIR"]
    if os.getenv("DATA_ROOT"):
        paths["data_root"] = os.environ["DATA_ROOT"]
    if os.getenv("DEVICE"):
        cfg.setdefault("device", {})["mode"] = os.environ["DEVICE"]
    # Extended frontier: merge the extra configs into the profiled set (opt-in).
    if os.getenv("EXTENDED_FRONTIER") == "1" and cfg.get("configs_extended"):
        cfg["configs"] = {**cfg["configs"], **cfg["configs_extended"]}
    return cfg


def abspath(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p)


def resolve_device(cfg: dict) -> str:
    """Return 'cuda', 'mps', or 'cpu'. 'auto' prefers CUDA, then Apple MPS, then CPU."""
    mode = cfg.get("device", {}).get("mode", "auto")
    if mode == "cpu":
        return "cpu"
    try:
        import torch
        cuda = torch.cuda.is_available()
        mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if mode == "cuda":
            return "cuda" if cuda else "cpu"
        if mode == "mps":
            return "mps" if mps else "cpu"
        # auto
        if cuda:
            return "cuda"
        if mps:
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def half_supported(device: str) -> bool:
    """fp16 is a meaningful axis only on CUDA. On CPU it collapses to fp32, and on
    Apple MPS fp16 inference is unreliable for this study, so we treat the precision
    axis as CUDA-only and run fp32 on MPS/CPU (documented in the write-up)."""
    return device == "cuda"
