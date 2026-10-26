"""Reviewer #2: real-silicon condition for HPEC. Run ONE hardware profile on an NVIDIA
Jetson (or any CUDA device with tegrastats) measuring what HPEC actually wants -- real
per-frame latency, throughput (FPS), ENERGY per frame, real co-running contention, and a
THERMAL/sustained-load check -- instead of the offline M5 traces + synthetic 2-state
competitor used in the main paper.

NOT runnable on the dev Windows box (no Jetson); this is turnkey for the Jetson. It reuses
detector.py, so the four frontier configs are identical to the main study.

What it measures, per config C1..C4:
  - latency p50/p95 (ms) and throughput (FPS), nominal;
  - energy/frame (mJ) via `tegrastats` power rails (VDD_GPU_SOC / VDD_CPU_GPU_CV) sampled
    during the run; degrades to latency/FPS only if tegrastats is absent;
  - REAL contention: relaunch the config with a genuine co-running GPU workload (a second
    detector process) and record the p95 latency shift -- a measured competitor, not a
    2-state model;
  - THERMAL check: run C1 continuously for --thermal-s seconds, logging latency and GPU
    temperature over time to detect throttling (the non-thermal caveat the paper flags).

    python src/jetson_harness.py --frames data/frames/radiate_fog_6_0 --n 300 --thermal-s 120
-> outputs/jetson/jetson_harness.json   (commit it; the paper's Table can then cite one
   real-silicon column instead of only dev-environment latencies)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def start_tegrastats(interval_ms=100):
    """Yield a subprocess streaming tegrastats, or None if unavailable."""
    if shutil.which("tegrastats") is None:
        return None
    return subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                            stdout=subprocess.PIPE, text=True, bufsize=1)


_PWR = re.compile(r"(VDD_GPU_SOC|VDD_CPU_GPU_CV|POM_5V_GPU|GPU)\s+(\d+)mW")
_TEMP = re.compile(r"(GPU|tj)@([\d.]+)C")


def _parse_tegrastats(line):
    pw = [int(m.group(2)) for m in _PWR.finditer(line)]
    tm = [float(m.group(2)) for m in _TEMP.finditer(line)]
    return (sum(pw) if pw else None), (max(tm) if tm else None)


def drain(proc):
    """Read all currently-buffered tegrastats lines -> (mean_mW, max_C)."""
    if proc is None:
        return None, None
    pw, tm = [], []
    # non-blocking-ish: read what's available quickly
    proc.stdout.flush()
    for _ in range(200):
        line = proc.stdout.readline()
        if not line:
            break
        p, t = _parse_tegrastats(line)
        if p is not None:
            pw.append(p)
        if t is not None:
            tm.append(t)
        if len(pw) > 50:
            break
    return (float(np.mean(pw)) if pw else None), (float(np.max(tm)) if tm else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="frame dir (any RGB sequence)")
    ap.add_argument("--n", type=int, default=300, help="frames per config")
    ap.add_argument("--thermal-s", type=int, default=120, help="sustained C1 seconds for throttle check")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import os
    os.environ["FRAMES_DIR"] = args.frames
    from config_util import load_config, resolve_device, abspath
    from data_harness import FrameSource
    from detector import build_detector

    cfg = load_config()
    device = resolve_device(cfg)
    if device != "cuda":
        print(f"WARNING: device={device}, not cuda. Run this ON THE JETSON for real numbers.")
    keys = list(cfg["configs"].keys())
    src = FrameSource(cfg=cfg)
    n = min(args.n, len(src))
    frames = [fr.image for fr in src.iter(stop=n)]
    conf = float(cfg["profiling"]["conf_threshold"])

    def time_config(det, imgs, teg):
        det.warmup(imgs[0])
        drain(teg)  # flush warmup power
        lat = []
        t0 = time.perf_counter()
        for im in imgs:
            a = time.perf_counter()
            det.predict(im, conf=conf)
            lat.append((time.perf_counter() - a) * 1e3)
        wall = time.perf_counter() - t0
        mean_mW, max_C = drain(teg)
        lat = np.array(lat)
        energy_mJ = (mean_mW * wall / len(imgs)) if mean_mW else None  # mW * s / frame = mJ/frame
        return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
                "fps": float(len(imgs) / wall), "mean_power_mW": mean_mW,
                "energy_per_frame_mJ": energy_mJ, "gpu_temp_C": max_C}

    teg = start_tegrastats()
    result = {"device": device, "frames": args.frames, "n": n,
              "tegrastats": teg is not None, "configs": {}}

    # nominal profile per config
    for k in keys:
        det = build_detector(cfg, k, device).load()
        result["configs"][k] = {"nominal": time_config(det, frames, teg)}
        del det

    # real co-running contention: launch a competitor detector process, re-time C1..C4
    comp = subprocess.Popen([sys.executable, "-c",
        "import sys;sys.path.insert(0,'src');from config_util import load_config,resolve_device;"
        "from detector import build_detector;import numpy as np;"
        "cfg=load_config();d=build_detector(cfg,'C1',resolve_device(cfg)).load();"
        "img=np.random.randint(0,255,(720,1280,3),'uint8');"
        "[d.predict(img) for _ in range(10**7)]"], cwd=str(abspath(".").parent))
    time.sleep(3)  # let the competitor saturate the GPU
    for k in keys:
        det = build_detector(cfg, k, device).load()
        result["configs"][k]["contended"] = time_config(det, frames, teg)
        p95n = result["configs"][k]["nominal"]["p95_ms"]
        p95c = result["configs"][k]["contended"]["p95_ms"]
        result["configs"][k]["p95_contention_shift"] = p95c / p95n if p95n else None
        del det
    comp.terminate()

    # thermal / sustained-load check on C1
    det = build_detector(cfg, keys[0], device).load()
    det.warmup(frames[0])
    series = []
    t_end = time.perf_counter() + args.thermal_s
    i = 0
    while time.perf_counter() < t_end:
        a = time.perf_counter()
        det.predict(frames[i % len(frames)], conf=conf)
        _, temp = drain(teg)
        series.append({"t_s": round(time.perf_counter() - (t_end - args.thermal_s), 1),
                       "lat_ms": (time.perf_counter() - a) * 1e3, "gpu_C": temp})
        i += 1
    if series:
        first = np.mean([s["lat_ms"] for s in series[:max(1, len(series)//10)]])
        last = np.mean([s["lat_ms"] for s in series[-max(1, len(series)//10):]])
        result["thermal"] = {"first_decile_lat_ms": float(first), "last_decile_lat_ms": float(last),
                             "throttle_ratio": float(last / first) if first else None,
                             "max_gpu_C": max((s["gpu_C"] for s in series if s["gpu_C"]), default=None),
                             "note": "throttle_ratio>1.05 indicates thermal throttling under sustained load"}
    if teg is not None:
        teg.terminate()

    out = abspath("outputs") / "jetson"; out.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out / "jetson_harness.json", "w"), indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    print(f"\n-> {out / 'jetson_harness.json'}  (commit it; cite one real-silicon column)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
