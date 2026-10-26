"""RQ-Coupling (highest-value addition): does a REAL, GENUINELY SEPARATE concurrent
workload naturally slow down under sensor-fault conditions, rather than reusing the
primary YOLO detector's own latency/detection-count as the load proxy
(measure_real_coupling.py)?

WHY this is a stronger test than measure_real_coupling.py's proxies: those proxies are
derived from the SAME network whose contention we are trying to predict, so a skeptic can
ask whether the "coupling" is just autocorrelation in one model's runtime. Here we run a
CREDIBLE CO-RUNNING WORKLOAD -- ORB feature detection + frame-to-frame descriptor matching,
a real, CPU-bound, compute-heavy step of the kind a concurrent tracking/mapping/SLAM
thread would perform -- entirely independent of YOLO, of the blur/illumination/occlusion
fault channels, and of any fault label. We time it for real (no simulation), then test
whether its timing co-occurs with the existing fault labels, exactly as
measure_real_coupling.py does for its two proxies.

    python src/measure_concurrent_workload.py --track outputs/trackD_fog_6_0 \
        --frames data/frames/radiate_fog_6_0 --max-frames 1500
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


def concurrent_workload_latency(frames, n_features=500) -> np.ndarray:
    """A real, independent concurrent CPU workload: ORB keypoint detection +
    frame-to-frame descriptor matching, timed per frame. This is the kind of
    compute a tracking/mapping thread genuinely performs; its cost depends on
    scene texture/motion, not on YOLO or the fault-labeling signals at all."""
    import cv2
    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    times = np.empty(len(frames))
    prev_des = None
    for i, img in enumerate(frames):
        t0 = time.perf_counter()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if prev_des is not None and des is not None and len(des) >= 2 and len(prev_des) >= 2:
            matches = bf.knnMatch(prev_des, des, k=2)
            _good = [m for pair in matches if len(pair) == 2
                     for m, n2 in [pair] if m.distance < 0.75 * n2.distance]
        times[i] = time.perf_counter() - t0
        prev_des = des
    return times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--high-quantile", type=float, default=0.75)
    args = ap.parse_args()
    os.environ["OUTPUTS_DIR"] = args.track
    os.environ["FRAMES_DIR"] = args.frames

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath
    from data_harness import FrameSource
    import simulate as sim
    import policies as pol
    import ci as cistats

    cfg = load_config()
    out_dir = abspath(args.track)

    src = FrameSource(cfg=cfg)
    n = min(args.max_frames, len(src))
    frames = [fr.image for fr in src.iter(stop=n)]
    print(f"[concurrent-workload] {args.track}: {n} frames, ORB detect+match (real, independent workload)")
    proxy = concurrent_workload_latency(frames)

    lab = pd.read_csv(out_dir / "phase1" / "trackA_observations.csv").iloc[:n]
    fault = lab["any_fault"].to_numpy(bool)[:len(proxy)]
    proxy = proxy[:len(fault)]
    hi = proxy > np.quantile(proxy, args.high_quantile)
    Pc_F = float(hi[fault].mean()) if fault.any() else float("nan")
    Pc_N = float(hi[~fault].mean()) if (~fault).any() else float("nan")
    r = pearson(fault.astype(float), proxy)
    r_lo, r_hi = boot_corr_ci(fault.astype(float), proxy)
    print(f"  MEASURED  Pc(F)={Pc_F:.3f}  Pc(N)={Pc_N:.3f}  (paper hard-codes 0.85 / 0.05)")
    print(f"  corr(fault, ORB-workload-latency) r={r:+.3f}  95%CI [{r_lo:+.3f},{r_hi:+.3f}]  "
          f"mean_orb_ms={proxy.mean()*1e3:.2f}")

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

    imposed = None
    p5 = out_dir / "phase5" / "phase5_rqh.json"
    if p5.exists():
        imposed = json.load(open(p5))["summary"]["coupled"]["miss_reduction_decoupled_minus_joint"]["mean"] * 100

    print(f"  DE-CIRCULARIZED RQ-H (schedule from real ORB-workload load): empirical kappa={emp_kappa:.3f}, "
          f"reduction {red['mean']*100:+.2f}pp [{red['lo']*100:.2f},{red['hi']*100:.2f}] "
          f"(sig={red['significant']})" + (f"; imposed-schedule was {imposed:+.2f}pp" if imposed is not None else ""))

    verdict = ("REAL COUPLING PRESENT: a genuinely separate concurrent workload runs slower "
               "under real sensor faults, and the router still helps under a schedule driven "
               "by that real, independent load."
               if (Pc_F > Pc_N and red["significant"] and red["mean"] > 0) else
               "NO/WEAK REAL COUPLING: the independent concurrent workload's timing does not "
               "co-occur with sensor faults beyond chance in this data (report honestly).")
    print(f"  VERDICT: {verdict}")

    extras = out_dir / "extras"; extras.mkdir(parents=True, exist_ok=True)
    json.dump({
        "track": args.track, "load_proxy": "concurrent_orb_workload", "n_frames": int(len(fault)),
        "high_quantile": args.high_quantile,
        "measured_Pc_fault": Pc_F, "measured_Pc_nominal": Pc_N,
        "corr_fault_load": r, "corr_ci": [r_lo, r_hi],
        "mean_orb_workload_ms": float(proxy.mean() * 1e3),
        "decirc_empirical_kappa": emp_kappa,
        "decirc_reduction_pp": red["mean"] * 100, "decirc_lo_pp": red["lo"] * 100,
        "decirc_hi_pp": red["hi"] * 100, "decirc_significant": red["significant"],
        # per-seed paired differences (decoupled - joint, pp) and the two-sided paired
        # t-test p computed DIRECTLY from them (for multiple-testing correction)
        "decirc_per_seed_reduction_pp": [(d - j) * 100 for d, j in zip(dm, jm)],
        "decirc_p_two_sided": red["p_two_sided"],
        "imposed_reduction_pp": imposed, "verdict": verdict,
    }, open(extras / "concurrent_workload_coupling.json", "w"), indent=2, default=str)
    print(f"  -> {extras / 'concurrent_workload_coupling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
