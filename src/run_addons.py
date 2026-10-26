"""Run the CPU add-on analyses (learned router + RQ-H phase diagram) on a set of
already-profiled tracks, in one command. Defaults to the strong real tracks (fog, night).

    python src/run_addons.py                          # fog + night
    python src/run_addons.py --tracks outputs/trackD_rain_4_0 outputs/trackC
    python src/run_addons.py --only learned           # or --only sweep

These reuse each track's outputs/<track>/phase1 + phase3 (no re-profiling), and write
outputs/<track>/extras/{learned_router,phase_diagram}.{json,png}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
DEFAULT_TRACKS = ["outputs/trackD_fog_6_0", "outputs/trackD_night_1_0"]


def _run(script, track):
    print(f"\n{'='*70}\n[addons] {script} on {track}\n{'='*70}", flush=True)
    return subprocess.run([PY, "-u", str(HERE / script), "--track", track]).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", default=DEFAULT_TRACKS)
    ap.add_argument("--only", choices=["learned", "sweep"], default=None)
    args = ap.parse_args()

    for t in args.tracks:
        if not (Path(t) / "phase3" / "latency_distributions.npz").exists():
            print(f"[addons] SKIP {t}: no profiled frontier (phase3 missing)"); continue
        if args.only in (None, "learned"):
            _run("run_learned_router.py", t)
        if args.only in (None, "sweep"):
            _run("run_injection_sweep.py", t)

    # summary
    print(f"\n{'='*70}\n[addons] SUMMARY\n{'='*70}")
    for t in args.tracks:
        lr = Path(t) / "extras" / "learned_router.json"
        pd_ = Path(t) / "extras" / "phase_diagram.json"
        print(f"\n{t}:")
        if lr.exists():
            r = json.load(open(lr))
            print(f"  learned-router: joint miss {r['joint_miss']['mean']:.3f} vs learned "
                  f"{r['learned_miss']['mean']:.3f}  ({r['verdict'].split(':')[0]})")
        if pd_.exists():
            d = json.load(open(pd_))
            sig = [c for c in d["cells"] if c["significant"] and c["miss_reduction"] > 0]
            peak = max(d["cells"], key=lambda c: c["miss_reduction"])
            print(f"  phase-diagram: {len(sig)}/{len(d['cells'])} cells sig+; peak "
                  f"{peak['miss_reduction']*100:.1f}pp at coupling={peak['coupling']}, "
                  f"onset/min={peak['onset_per_min']}")


if __name__ == "__main__":
    raise SystemExit(main())
