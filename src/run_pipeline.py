"""Run the full offline pipeline in order: extract frames (if needed), Phase 1
(degradation + labels + contention + regimes), Phase 2 (sensor belief), Phase 3
(frontier + deadline), Phase 4 (routing), Phase 5 (RQ-H centerpiece), Phase 6
(RQ-A1 + RQ-A2). Each phase is a subprocess; the run stops on the first failure.

This is the reproducibility entry point referenced in the README. For the
reportable GPU numbers use run_on_colab.ipynb instead of running this on a laptop.

    python src/run_pipeline.py            # all phases
    python src/run_pipeline.py --from 3   # resume from Phase 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("extract", "extract_frames.py"),
    ("phase1", "phase1_run.py"),
    ("phase2", "phase2_run.py"),
    ("phase3", "profile_frontier.py"),
    ("phase4", "phase4_run.py"),
    ("phase5", "phase5_run.py"),
    ("phase6", "phase6_run.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="extract",
                    help="step name or 1-based phase index to start from")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    names = [s[0] for s in STEPS]
    if args.start.isdigit():
        start_idx = int(args.start)  # 'phase N' -> index in STEPS (extract is 0)
        start_idx = names.index(f"phase{args.start}") if f"phase{args.start}" in names else int(args.start)
    else:
        start_idx = names.index(args.start)

    for i, (name, script) in enumerate(STEPS):
        if i < start_idx:
            continue
        if name == "extract" and args.skip_extract:
            continue
        print(f"\n{'='*70}\n[run_pipeline] {name}: {script}\n{'='*70}", flush=True)
        r = subprocess.run([PY, str(HERE / script)])
        if r.returncode != 0:
            print(f"[run_pipeline] {name} FAILED (exit {r.returncode}); stopping.")
            return r.returncode
    print("[run_pipeline] all steps complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
