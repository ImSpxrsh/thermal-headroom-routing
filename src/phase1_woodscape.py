"""Track C phase 1 (WoodScape fisheye). LOCAL/UNCOMMITTED.

Takes the precomputed fisheye-correct observations from extract_woodscape.py
(polar blur + illumination on the raw fisheye, occlusion from soiling masks), applies
the SAME deterministic threshold-and-smooth fault labels used everywhere, then reuses
the existing contention generator and coupling regimes. Writes outputs/<track>/phase1/*
in the exact format phases 2-6 consume, so the rest of the pipeline runs unchanged.

No Track B here: the soiling masks are real ground-truth occlusion, so no controlled
injection is needed (and Track B is a normal-lens construct).

    OUTPUTS_DIR=outputs/trackC FRAMES_DIR=data/frames/woodscape_soiling \
      python src/phase1_woodscape.py --obs data/cache/trackC_observations_raw.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath
import degradation as deg
import phase1_run  # reuse contention_demo, regimes_demo, _shade_faults


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/cache/trackC_observations_raw.csv")
    args = ap.parse_args()
    cfg = load_config()
    OUT = abspath(cfg["paths"]["outputs_dir"]) / "phase1"
    OUT.mkdir(parents=True, exist_ok=True)
    phase1_run.OUT = OUT  # the reused contention/regimes helpers write to this global

    df = pd.read_csv(abspath(args.obs))
    print(f"[Track C] loaded {len(df)} fisheye-correct observations from {args.obs}")
    labeled, segs = deg.label_faults(df, cfg)
    labeled.to_csv(OUT / "trackA_observations.csv", index=False)  # name kept for downstream

    # overlay plot (real soiling-driven faults)
    fidx = labeled["frame_idx"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, ch, col in zip(axes, ["blur", "illumination", "occlusion"],
                           ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(fidx, labeled[ch], color=col, lw=0.9)
        phase1_run._shade_faults(ax, fidx, labeled[f"{ch}_fault"].to_numpy(), "red", "labeled fault")
        ax.set_ylabel(ch); ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("Track C (WoodScape fisheye): polar-blur / entropy / mask-occlusion "
                      "with threshold-and-smooth labels")
    axes[-1].set_xlabel("frame index (dataset order)")
    fig.tight_layout(); fig.savefig(OUT / "trackC_overlay.png", dpi=130); plt.close(fig)

    c = phase1_run.contention_demo(cfg)
    r = phase1_run.regimes_demo(labeled, cfg)
    summary = {"track": "C_woodscape_fisheye", "n_frames": int(len(df)),
               "fault_frac": float(labeled["any_fault"].mean()),
               "occlusion_is_mask_ground_truth": True,
               "contention": c, "regimes": r}
    with open(OUT / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Phase 1 (Track C) summary:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
