"""Track C orchestrator (WoodScape fisheye). LOCAL/UNCOMMITTED.

Runs: extract_woodscape -> phase1_woodscape -> phase2 -> phase3 (frontier+deadline)
-> phase4 -> phase5 (RQ-H) -> phase6 (RQ-A1/A2), all namespaced to outputs/trackC/.

    python src/run_woodscape.py --soiling-dir data/woodscape_soiling/train
    python src/run_woodscape.py --soiling-dir <dir> --max-frames 1500 --profile-n 0
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
FRAMES = "data/frames/woodscape_soiling"
OBS = "data/cache/trackC_observations_raw.csv"
OUTPUTS = "outputs/trackC"


def run(cmd, env=None):
    print(f"\n{'='*70}\n[run_woodscape] {' '.join(cmd)}\n{'='*70}", flush=True)
    r = subprocess.run([PY, "-u"] + cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {cmd} (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soiling-dir", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--profile-n", default="0", help="frames to profile (0=all extracted)")
    args = ap.parse_args()

    base = dict(os.environ, FRAMES_DIR=FRAMES, OUTPUTS_DIR=OUTPUTS, PROFILE_N=str(args.profile_n))

    ex = [str(HERE / "extract_woodscape.py"), "--soiling-dir", args.soiling_dir,
          "--out", FRAMES, "--obs-out", OBS, "--max-frames", str(args.max_frames)]
    if args.calib:
        ex += ["--calib", args.calib]
    run(ex, env=base)
    run([str(HERE / "phase1_woodscape.py"), "--obs", OBS], env=base)
    for script in ("phase2_run.py", "profile_frontier.py", "phase4_run.py",
                   "phase5_run.py", "phase6_run.py"):
        run([str(HERE / script)], env=base)
    print(f"\n[run_woodscape] complete. Results in {OUTPUTS}/")


if __name__ == "__main__":
    raise SystemExit(main())
