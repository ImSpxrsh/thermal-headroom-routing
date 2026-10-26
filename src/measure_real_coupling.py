"""Measure the REAL fault<->compute-load co-occurrence and run a DE-CIRCULARIZED RQ-H.

WHY: the headline experiment generates the contention schedule FROM the fault labels
(Pc(F)=0.85, Pc(N)=0.05 are hard-coded in config.yaml), so a reviewer can argue the
coupled router only detects a relationship we imposed. This script instead MEASURES the
coupling from real data and drives contention from a real, independently-measured
compute-load signal.

WHAT it does (needs the real frames -> run on the machine that has them):
  1. Run the reference config C1 over the real frames; record per-frame REAL end-to-end
     latency (incl. NMS/postproc) and detection count. Detection count is a genuine
     compute-demand proxy computed from YOLO OUTPUT, independent of the
     blur/illumination/occlusion fault signals.
  2. Measure Pc(F)=P(high-load | fault) and Pc(N)=P(high-load | nominal) from the data
     (high-load = top-quartile of the proxy). These are the MEASURED analogues of the
     paper's hard-coded 0.85 / 0.05, plus corr(fault, load) with a bootstrap CI. THIS is
     the "does the dependency exist in the real world" number (reviewer #6).
  3. DE-CIRCULARIZED RQ-H: build the contention schedule by thresholding the real load
     proxy (NOT the fault labels), feed it to the same routing substrate via
     state_override, and run joint vs decoupled. kappa is now EMPIRICAL. If the router
     still helps, the benefit is not an artifact of the imposed schedule.

Interpretation: if Pc(F) > Pc(N) and the de-circularized reduction is positive, the
coupling is REAL and the method has an empirical claim about the world (moves the paper
past "clean ablation of a constructed scenario"). If Pc(F) ~ Pc(N), that is an honest
null: in this data degradation and compute load do not co-occur, and the coupled regime
is imposed-only -- report it plainly.

    python src/measure_real_coupling.py --track outputs/trackD_rain_4_0 \
        --frames data/frames/radiate_rain_4_0 --load-proxy detections --max-frames 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def boot_corr_ci(fault, proxy, B=2000, seed=17):
    rng = np.random.default_rng(seed)
    n = len(fault)
    rs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        rs[i] = pearson(fault[idx], proxy[idx])
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help="outputs dir, e.g. outputs/trackD_rain_4_0")
    ap.add_argument("--frames", required=True, help="frames dir for this track")
    ap.add_argument("--load-proxy", choices=["detections", "latency"], default="detections")
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--high-quantile", type=float, default=0.75)
    args = ap.parse_args()
    os.environ["OUTPUTS_DIR"] = args.track
    os.environ["FRAMES_DIR"] = args.frames

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath, resolve_device
    from data_harness import FrameSource
    from detector import build_detector
    import simulate as sim
    import policies as pol
    import ci as cistats

    cfg = load_config()
    device = resolve_device(cfg)
    ref = cfg.get("reference_config", "C1")
    out_dir = abspath(args.track)

    # --- 1) run reference config over real frames: real latency + detection count ---
    src = FrameSource(cfg=cfg)
    n = min(args.max_frames, len(src))
    frames = [fr.image for fr in src.iter(stop=n)]
    print(f"[real-coupling] {args.track}: {n} frames on {device}, ref={ref}, proxy={args.load_proxy}")
    det = build_detector(cfg, ref, device).load()
    det.warmup(frames[0])
    lat, ndet = [], []
    for img in frames:
        t0 = time.perf_counter()
        d = det.predict(img, conf=float(cfg["profiling"]["conf_threshold"]))
        lat.append(time.perf_counter() - t0)
        ndet.append(int(len(d.cls)))
    lat = np.array(lat); ndet = np.array(ndet, float)
    proxy = ndet if args.load_proxy == "detections" else lat

    # --- 2) real fault<->load co-occurrence ---
    lab = pd.read_csv(out_dir / "phase1" / "trackA_observations.csv").iloc[:n]
    fault = lab["any_fault"].to_numpy(bool)[:len(proxy)]
    proxy = proxy[:len(fault)]
    hi = proxy > np.quantile(proxy, args.high_quantile)
    Pc_F = float(hi[fault].mean()) if fault.any() else float("nan")
    Pc_N = float(hi[~fault].mean()) if (~fault).any() else float("nan")
    r = pearson(fault.astype(float), proxy)
    r_lo, r_hi = boot_corr_ci(fault.astype(float), proxy)
    print(f"  MEASURED  Pc(F)={Pc_F:.3f}  Pc(N)={Pc_N:.3f}  (paper hard-codes 0.85 / 0.05)")
    print(f"  corr(fault, {args.load_proxy}) r={r:+.3f}  95%CI [{r_lo:+.3f},{r_hi:+.3f}]  "
          f"mean_ndet={ndet.mean():.2f}  mean_lat_ms={lat.mean()*1e3:.1f}")

    # --- 3) de-circularized RQ-H: contention from the REAL load proxy, not fault labels ---
    seeds = cfg["seeds"]
    rt = cfg["routing"]["reliability_target"]
    dwell = cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]
    jm, dm, kaps = [], [], []
    for s in seeds:
        sub = sim.build_substrate(cfg, "coupled", s, state_override=hi)
        kaps.append(sub.kappa)
        jm.append(sim.run_policy(pol.JointPolicy(sub.fm, sub.kappa, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"])
        dm.append(sim.run_policy(pol.DecoupledPolicy(sub.fm, rt, pol.Hysteresis(dwell)), sub)["deadline_miss_rate"])
    red = cistats.paired_diff_ci(dm, jm)
    emp_kappa = float(np.mean(kaps))

    # imposed-schedule reduction (committed phase5) for side-by-side
    imposed = None
    p5 = out_dir / "phase5" / "phase5_rqh.json"
    if p5.exists():
        imposed = json.load(open(p5))["summary"]["coupled"]["miss_reduction_decoupled_minus_joint"]["mean"] * 100

    print(f"  DE-CIRCULARIZED RQ-H (schedule from real {args.load_proxy}): empirical kappa={emp_kappa:.3f}, "
          f"reduction {red['mean']*100:+.2f}pp [{red['lo']*100:.2f},{red['hi']*100:.2f}] "
          f"(sig={red['significant']}); imposed-schedule was {imposed:+.2f}pp" if imposed is not None else "")

    verdict = ("REAL COUPLING PRESENT: high load co-occurs with faults and the router still "
               "helps under a real-load-driven schedule."
               if (Pc_F > Pc_N and red["significant"] and red["mean"] > 0) else
               "NO/WEAK REAL COUPLING: in this data, compute load does not co-occur with faults "
               "beyond chance; the coupled regime is imposed-only (report honestly).")
    print(f"  VERDICT: {verdict}")

    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump({
        "track": args.track, "load_proxy": args.load_proxy, "n_frames": int(len(fault)),
        "high_quantile": args.high_quantile,
        "measured_Pc_fault": Pc_F, "measured_Pc_nominal": Pc_N,
        "imposed_Pc_fault": cfg["regimes"]["coupled"]["contention_given_fault_prob"],
        "imposed_Pc_nominal": cfg["regimes"]["coupled"]["contention_given_nominal_prob"],
        "corr_fault_load": r, "corr_ci": [r_lo, r_hi],
        "mean_detections": float(ndet.mean()), "mean_latency_ms": float(lat.mean() * 1e3),
        "decirc_empirical_kappa": emp_kappa,
        "decirc_reduction_pp": red["mean"] * 100, "decirc_lo_pp": red["lo"] * 100,
        "decirc_hi_pp": red["hi"] * 100, "decirc_significant": red["significant"],
        "imposed_reduction_pp": imposed, "verdict": verdict,
    }, open(extras / "real_coupling.json", "w"), indent=2, default=str)
    print(f"  -> {extras / 'real_coupling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
