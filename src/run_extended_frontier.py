"""Extended frontier (Part 1a): re-profile a dense 9-config Pareto set and re-run routing,
reusing an already-profiled track's frames + labels (no re-extraction, no re-labeling).

It copies the track's existing phase1 (labels/observations are config-independent), then
profiles configs C1..C9 (EXTENDED_FRONTIER=1 merges configs_extended) and re-runs phases
4-6, all into a separate outputs/<track>_extended/ so the original 4-config results are
untouched.

    python src/run_extended_frontier.py --track-outputs outputs \
        --frames data/frames/turnpike_afternoon_fall_0
    python src/run_extended_frontier.py --track-outputs outputs/trackD_rain_4_0 \
        --frames data/frames/radiate_rain_4_0 --profile-n 1500
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(script, env):
    print(f"\n{'='*70}\n[extended] {script}\n{'='*70}", flush=True)
    r = subprocess.run([PY, "-u", str(HERE / script)], env=env)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-outputs", required=True, help="existing track outputs dir (has phase1/)")
    ap.add_argument("--frames", required=True, help="that track's frames dir (for re-profiling)")
    ap.add_argument("--out", default=None, help="extended outputs dir (default <track>_extended)")
    ap.add_argument("--profile-n", default="0", help="frames to profile (0=all)")
    args = ap.parse_args()

    src_out = Path(args.track_outputs)
    out = args.out or (str(src_out).rstrip("/\\") + "_extended")
    # reuse the track's labels/observations (config-independent)
    p1_src = src_out / "phase1"
    if not p1_src.exists():
        raise SystemExit(f"no phase1 in {src_out}; run that track's pipeline first.")
    shutil.copytree(str(p1_src), os.path.join(out, "phase1"), dirs_exist_ok=True)
    print(f"reused labels from {p1_src} -> {out}/phase1")

    env = dict(os.environ, EXTENDED_FRONTIER="1", FRAMES_DIR=args.frames,
               OUTPUTS_DIR=out, PROFILE_N=str(args.profile_n))
    run("profile_frontier.py", env)   # profiles C1..C9, dense frontier + deadline
    for s in ("phase4_run.py", "phase5_run.py", "phase6_run.py"):
        run(s, env)
    print(f"\n[extended] done. Dense 9-config frontier + routing in {out}/")
    print(f"  frontier table: {out}/phase3/frontier_table.csv ; plot: {out}/phase3/frontier.png")


if __name__ == "__main__":
    raise SystemExit(main())
