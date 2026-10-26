"""Phase 3 driver: profile the four-config frontier and set the deadline.

For each config (C1..C4) it measures:
  - task accuracy = agreement with the C1 reference detections (pseudo-GT), overall
    and bucketed by active fault type (from the Phase 1 Track A labels);
  - end-to-end per-frame latency distribution under nominal and contended compute.

It then sets the deadline to the median end-to-end latency of C1 under nominal
compute (locked rule, section 0a), writes it back to config.yaml, assembles the
frontier table, saves the per-frame accuracy profile + latency distributions for the
routing simulation, and draws the accuracy-vs-latency frontier plot.

Latency numbers here are environment-specific and dev-only. Reportable numbers come
from run_on_colab.ipynb.

Run after frames are extracted and Phase 1 has produced trackA_observations.csv:
    python src/profile_frontier.py
    PROFILE_N=20 python src/profile_frontier.py    # quick correctness pass
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath, resolve_device
from data_harness import FrameSource
from detector import build_detector, Detection
import agreement as ag
import contention as cont

CHANS = ["blur", "illumination", "occlusion"]
CONFIG_KEYS = ["C1", "C2", "C3", "C4"]


def _empty_cache(device: str):
    import torch
    try:
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def _is_oom(err: Exception) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "oom" in msg or "mps backend out of memory" in msg


def _run_config_over_frames(detector, frames, conf):
    """Return list[Detection] for the config over the given BGR frames, with
    per-config CPU fallback on GPU OOM (CUDA or MPS), flagged dev-only."""
    dets = []
    fell_back = False
    dev = detector.device
    for img in frames:
        try:
            dets.append(detector.predict(img, conf=conf))
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            _empty_cache(dev)
            detector.device = "cpu"; detector.half = False
            detector.load()
            fell_back = True
            dets.append(detector.predict(img, conf=conf))
    return dets, fell_back


def _latency_dist(detector, frames, cfg, device, contended):
    import torch
    # warmup
    for _ in range(5):
        detector.predict(frames[0])
    lat = []
    ctx = cont.make_contention(cfg, device) if contended else None
    if ctx is not None:
        ctx.__enter__()
        for _ in range(3):
            detector.predict(frames[0])
    try:
        for img in frames:
            t0 = time.perf_counter()
            detector.predict(img)
            lat.append(time.perf_counter() - t0)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return np.array(lat)


def main() -> int:
    import torch
    cfg = load_config()
    device = resolve_device(cfg)
    pcfg = cfg["profiling"]
    n_frames = int(os.getenv("PROFILE_N", pcfg["n_frames"]))
    n_lat = int(pcfg["latency_sample_frames"])
    conf = float(pcfg["conf_threshold"])
    iou = float(pcfg["iou_threshold"])

    # config set is dynamic (supports the extended frontier via configs_extended)
    global CONFIG_KEYS
    CONFIG_KEYS = list(cfg["configs"].keys())
    ref = cfg.get("reference_config", CONFIG_KEYS[0])
    print(f"Profiling configs: {CONFIG_KEYS} (reference={ref})")

    out = abspath(cfg["paths"]["outputs_dir"]) / "phase3"
    out.mkdir(parents=True, exist_ok=True)

    src = FrameSource(cfg=cfg)
    total = len(src)
    n_frames = total if n_frames <= 0 else min(n_frames, total)
    print(f"Profiling on {n_frames}/{total} frames, device={device}")
    frames = [fr.image for fr in src.iter(stop=n_frames)]

    # fault buckets from Phase 1 Track A labels
    labels_path = abspath(cfg["paths"]["outputs_dir"]) / "phase1" / "trackA_observations.csv"
    fault_masks = {}
    if labels_path.exists():
        lab = pd.read_csv(labels_path).iloc[:n_frames]
        for ch in CHANS:
            fault_masks[ch] = lab[f"{ch}_fault"].to_numpy(bool)
        fault_masks["nominal"] = ~lab["any_fault"].to_numpy(bool)
    else:
        print("WARN: no Track A labels; per-fault buckets skipped")

    n_lat = min(n_lat, n_frames)
    lat_frames = frames[:n_lat]

    # --- reference pass (the reference config = pseudo-GT) ---
    print(f"Reference pass {ref} (pseudo-GT)...")
    c1 = build_detector(cfg, ref, device).load()
    ref_dets, c1_fb = _run_config_over_frames(c1, frames, conf)
    lat_c1_nom = _latency_dist(c1, lat_frames, cfg, device, contended=False)
    lat_c1_con = _latency_dist(c1, lat_frames, cfg, device, contended=True)
    del c1; _empty_cache(device)

    deadline_ms = float(np.median(lat_c1_nom) * 1e3 * cfg["deadline"]["multiplier"])

    rows = []                 # frontier table rows
    per_frame_acc = []        # for routing sim: frame_idx, config, fault_type, f1
    lat_dists = {ref: {"nominal": lat_c1_nom, "contended": lat_c1_con}}
    fallbacks = {ref: c1_fb}

    for key in CONFIG_KEYS:
        print(f"Config {key}...")
        if key == ref:
            dets = ref_dets
            ln, lc = lat_c1_nom, lat_c1_con
            fb = c1_fb
        else:
            det = build_detector(cfg, key, device).load()
            dets, fb = _run_config_over_frames(det, frames, conf)
            ln = _latency_dist(det, lat_frames, cfg, device, contended=False)
            lc = _latency_dist(det, lat_frames, cfg, device, contended=True)
            lat_dists[key] = {"nominal": ln, "contended": lc}
            fallbacks[key] = fb
            del det; _empty_cache(device)

        # per-frame agreement vs C1
        f1s = np.array([ag.agreement_f1(dets[i], ref_dets[i], iou)["f1"]
                        for i in range(n_frames)])
        for i in range(n_frames):
            per_frame_acc.append({"frame_idx": i, "config": key, "f1": float(f1s[i])})

        row = {"config": key,
               "model": cfg["configs"][key]["model"],
               "imgsz": cfg["configs"][key]["imgsz"],
               "half": bool(cfg["configs"][key].get("half", False)) and device == "cuda",
               "cpu_fallback": fb,
               "acc_overall": float(f1s.mean()),
               "lat_median_nom_ms": float(np.median(ln) * 1e3),
               "lat_p95_nom_ms": float(np.percentile(ln, 95) * 1e3),
               "lat_median_con_ms": float(np.median(lc) * 1e3),
               "lat_p95_con_ms": float(np.percentile(lc, 95) * 1e3),
               "meets_deadline_nom": bool(np.median(ln) * 1e3 <= deadline_ms),
               "meets_deadline_con": bool(np.median(lc) * 1e3 <= deadline_ms)}
        for bucket, mask in fault_masks.items():
            m = mask[:n_frames]
            row[f"acc_{bucket}"] = float(f1s[m].mean()) if m.any() else float("nan")
        rows.append(row)

    frontier = pd.DataFrame(rows)
    frontier.to_csv(out / "frontier_table.csv", index=False)
    pd.DataFrame(per_frame_acc).to_csv(out / "per_frame_accuracy.csv", index=False)
    np.savez(out / "latency_distributions.npz",
             **{f"{k}_{st}": v for k, d in lat_dists.items() for st, v in d.items()})

    # write the deadline back into config.yaml (value_ms), reported not hand-set
    _update_config_deadline(deadline_ms)

    # frontier plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, r in frontier.iterrows():
        ax.scatter(r["lat_median_nom_ms"], r["acc_overall"], s=80)
        ax.annotate(r["config"], (r["lat_median_nom_ms"], r["acc_overall"]),
                    textcoords="offset points", xytext=(6, 6))
    ax.axvline(deadline_ms, color="red", ls="--", label=f"deadline {deadline_ms:.0f} ms")
    ax.set_xlabel("median end-to-end latency, nominal (ms, dev-env not reportable)")
    ax.set_ylabel("accuracy = agreement-with-C1 (pseudo-GT)")
    ax.set_title("Frontier: accuracy vs latency by config")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "frontier.png", dpi=130); plt.close(fig)

    # self-assessment. Degenerate = no routing pressure: either all configs meet the
    # deadline under contention, or none do (no separation to route around). The
    # interesting frontier has SOME configs feasible under contention and some not.
    order_ok = _check_ordering(frontier)
    con = frontier["meets_deadline_con"]
    degenerate = bool(con.all() or (~con).all())
    summary = {"device": device, "n_frames": n_frames, "deadline_ms": deadline_ms,
               "ordering_sensible": order_ok, "degenerate_all_meet_deadline": degenerate,
               "cpu_fallbacks": fallbacks,
               "frontier": frontier.to_dict(orient="records")}
    with open(out / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _check_ordering(frontier: pd.DataFrame) -> bool:
    """Sensible if the reference (most expensive) config is the slowest and there is real
    latency spread across the set. Works for any config-set size."""
    lat = frontier.set_index("config")["lat_median_nom_ms"]
    ref = frontier.iloc[0]["config"]  # reference is profiled first (row 0)
    return bool(lat[ref] >= lat.max() - 1e-6 and lat.max() > 1.2 * lat.min())


def _update_config_deadline(value_ms: float):
    p = abspath("config.yaml")
    txt = p.read_text(encoding="utf-8")
    import re
    txt2 = re.sub(r"value_ms:\s*\S+", f"value_ms: {value_ms:.3f}", txt, count=1)
    p.write_text(txt2, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
