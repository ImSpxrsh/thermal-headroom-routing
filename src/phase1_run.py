"""Phase 1 driver: degradation observations, fault labels, Track B injection,
contention-generator demo, and coupling regimes. Produces the acceptance plots
in outputs/phase1/ and appends a self-assessment to REPORT.md.

Run after frames are extracted:
    python src/phase1_run.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath, resolve_device, half_supported
import degradation as deg
import injection as inj
import regimes as reg
import contention as cont
from data_harness import FrameSource


OUT = None  # set in main


def _shade_faults(ax, frame_idx, fault_bool, color, label):
    inseg = False
    s = 0
    for k in range(len(fault_bool)):
        if fault_bool[k] and not inseg:
            inseg = True; s = k
        elif not fault_bool[k] and inseg:
            inseg = False
            ax.axvspan(frame_idx[s], frame_idx[k - 1], color=color, alpha=0.18,
                       label=label); label = None
    if inseg:
        ax.axvspan(frame_idx[s], frame_idx[len(fault_bool) - 1], color=color,
                   alpha=0.18, label=label)


def track_a(src, cfg) -> pd.DataFrame:
    print("[Track A] computing real degradation observations...")
    df = deg.compute_observations(src)
    labeled, segs = deg.label_faults(df, cfg)
    labeled.to_csv(OUT / "trackA_observations.csv", index=False)

    fidx = labeled["frame_idx"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, ch, col in zip(axes, ["blur", "illumination", "occlusion"],
                           ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(fidx, labeled[ch], color=col, lw=0.9)
        _shade_faults(ax, fidx, labeled[f"{ch}_fault"].to_numpy(), "red", "labeled fault")
        ax.set_ylabel(ch)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame index")
    axes[0].set_title("Track A (real): degradation signals with deterministic "
                      "threshold-and-smooth fault labels")
    fig.tight_layout(); fig.savefig(OUT / "trackA_overlay.png", dpi=130); plt.close(fig)
    print(f"[Track A] {len(segs)} fault segments; "
          f"{labeled['any_fault'].mean()*100:.1f}% of frames flagged faulted")
    return labeled


def track_b(src, cfg) -> dict:
    print("[Track B] controlled injection (validation aid only)...")
    n = len(src)
    plan = inj.make_injection_plan(n, cfg, seed=cfg["seeds"][0])
    intensity = float(cfg["tracks"]["B_injection"]["intensity"])
    rng = np.random.default_rng(cfg["seeds"][0])

    rows = []
    prev_gray = None
    import cv2
    for t, fr in enumerate(src.iter()):
        active = {c: bool(plan["active"][c][t]) for c in plan["active"]}
        img = inj.apply_injection(fr.image, active, intensity, rng)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = deg.blur_vol(gray)
        illum = deg.illumination_entropy(gray)
        occ = np.nan if prev_gray is None else deg.occlusion_track_survival(prev_gray, gray, cfg)
        rows.append(dict(frame_idx=fr.idx, blur=blur, illumination=illum, occlusion=occ))
        prev_gray = gray
    dfb = pd.DataFrame(rows)
    dfb["occlusion"] = dfb["occlusion"].bfill()
    dfb["gt_fault"] = plan["ground_truth"]
    for c in plan["active"]:
        dfb[f"gt_{c}"] = plan["active"][c]
    dfb.to_csv(OUT / "trackB_observations.csv", index=False)

    fidx = dfb["frame_idx"].to_numpy()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, ch, col in zip(axes, ["blur", "illumination", "occlusion"],
                           ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(fidx, dfb[ch], color=col, lw=0.9)
        _shade_faults(ax, fidx, plan["active"][ch], "purple", "injected (ground truth)")
        ax.set_ylabel(ch); ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame index")
    axes[0].set_title("Track B (VALIDATION ONLY): injected faults vs known timeline")
    fig.tight_layout(); fig.savefig(OUT / "trackB_overlay.png", dpi=130); plt.close(fig)

    with open(OUT / "trackB_plan.json", "w") as f:
        json.dump(inj.plan_to_dict(plan), f, indent=2)
    return {"events": len(plan["events"]),
            "gt_fault_frac": float(plan["ground_truth"].mean())}


def contention_demo(cfg) -> dict:
    device = resolve_device(cfg)
    print(f"[Contention] device={device}; measuring nominal vs contended latency...")
    step = cont.default_step_fn(device)
    n = int(cfg["contention"].get("demo_steps", 150))
    lat_nom = cont.measure_latency(step, n, cfg, device, contended=False)
    lat_con = cont.measure_latency(step, n, cfg, device, contended=True)
    win = cfg["contention"]["latency_window_frames"]
    fps = cfg["dataset"]["nominal_fps"]
    s_nom = cont.sliding_window_stats(lat_nom, win, fps)
    s_con = cont.sliding_window_stats(lat_con, win, fps)

    summary = {
        "device": device,
        "p95_nominal_ms": float(np.percentile(lat_nom, 95) * 1e3),
        "p95_contended_ms": float(np.percentile(lat_con, 95) * 1e3),
        "p99_nominal_ms": float(np.percentile(lat_nom, 99) * 1e3),
        "p99_contended_ms": float(np.percentile(lat_con, 99) * 1e3),
        "median_nominal_ms": float(np.median(lat_nom) * 1e3),
        "median_contended_ms": float(np.median(lat_con) * 1e3),
    }
    summary["p95_shift_ratio"] = summary["p95_contended_ms"] / max(summary["p95_nominal_ms"], 1e-6)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(lat_nom * 1e3, bins=40, alpha=0.6, label="nominal")
    ax[0].hist(lat_con * 1e3, bins=40, alpha=0.6, label="contended")
    ax[0].axvline(summary["p95_nominal_ms"], color="tab:blue", ls="--")
    ax[0].axvline(summary["p95_contended_ms"], color="tab:orange", ls="--")
    ax[0].set_xlabel("end-to-end step latency (ms)"); ax[0].set_ylabel("count")
    ax[0].set_title("Latency distribution (dev-env, NOT reportable)"); ax[0].legend()
    ax[1].plot(s_nom["p95_s"] * 1e3, label="p95 nominal")
    ax[1].plot(s_con["p95_s"] * 1e3, label="p95 contended")
    ax[1].plot(s_con["queue_depth"], label="queue depth (contended)", color="tab:red", alpha=0.6)
    ax[1].set_xlabel("step"); ax[1].set_title("Sliding p95 + queue depth"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "contention_shift.png", dpi=130); plt.close(fig)
    with open(OUT / "contention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Contention] p95 {summary['p95_nominal_ms']:.2f} -> "
          f"{summary['p95_contended_ms']:.2f} ms (x{summary['p95_shift_ratio']:.2f})")
    return summary


def regimes_demo(labeled, cfg) -> dict:
    print("[Regimes] building uncoupled and coupled schedules...")
    fault = labeled["any_fault"].to_numpy()
    n = len(fault)
    seed = cfg["seeds"][0]
    unc = reg.make_schedule("uncoupled", n, fault, cfg, seed)
    cou = reg.make_schedule("coupled", n, fault, cfg, seed)
    emp_u = reg.empirical_coupling(fault, unc)
    emp_c = reg.empirical_coupling(fault, cou)

    fidx = labeled["frame_idx"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for ax, sched, emp, name in zip(axes, [unc, cou], [emp_u, emp_c],
                                    ["uncoupled", "coupled"]):
        _shade_faults(ax, fidx, fault, "red", "sensor fault")
        ax.fill_between(fidx, 0, sched.astype(float), step="mid", alpha=0.5,
                        color="tab:purple", label="contention active")
        ax.set_ylabel(name)
        ax.set_title(f"{name}: corr(fault,contention)={emp['pearson_r']:.2f}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame index")
    fig.tight_layout(); fig.savefig(OUT / "regimes_schedule.png", dpi=130); plt.close(fig)
    return {"uncoupled": emp_u, "coupled": emp_c}


def main() -> int:
    global OUT
    cfg = load_config()
    OUT = abspath(cfg["paths"]["outputs_dir"]) / "phase1"
    OUT.mkdir(parents=True, exist_ok=True)
    src = FrameSource(cfg=cfg)
    print(f"Loaded {len(src)} frames")

    labeled = track_a(src, cfg)
    # Track B is the controlled-injection validation aid (a second optical-flow pass over
    # all frames). It is not part of the real-data headline, so it can be skipped to halve
    # Phase 1 cost on CPU-bound environments (e.g. Colab). SKIP_TRACK_B=1 to skip.
    if os.getenv("SKIP_TRACK_B") == "1":
        print("[Track B] SKIPPED (SKIP_TRACK_B=1)")
        b = {"skipped": True}
    else:
        b = track_b(src, cfg)
    c = contention_demo(cfg)
    r = regimes_demo(labeled, cfg)

    summary = {"n_frames": len(src), "trackA_fault_frac": float(labeled["any_fault"].mean()),
               "trackB": b, "contention": c, "regimes": r}
    with open(OUT / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Phase 1 summary:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
