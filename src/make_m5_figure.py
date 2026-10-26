"""Figure and table for the Apple M5 hardware check (outputs/m5_hardware/m5_hardware.json).

    python src/m5_hardware.py && python src/make_m5_figure.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.fontsize": 7, "legend.frameon": False})
COL = {"C1": "#c0392b", "C2": "#c58b2b", "C4": "#3b6fb6",
       "static_C1": "#9a9a9a", "thermostat": "#2c7a4b", "thr": "#c0392b"}
LBL = {"static_C1": "Static C1", "thermostat": "Thermostat", "thr": "THR"}


def main():
    """Heating curves come from the first pass (m5_unpaced.json: frames back to back, so
    each config runs the GPU flat out). Router runs and the paced C2 curve come from the
    second pass (m5_paced.json: one frame per C1 median latency, as in the simulation)."""
    d = json.load(open(ROOT / "outputs" / "m5_hardware" / "m5_unpaced.json"))
    pz = json.load(open(ROOT / "outputs" / "m5_hardware" / "m5_paced.json"))
    curves = dict(d["curves"]); curves["C2 paced"] = pz["curves"]["C2_paced"]
    COL["C2 paced"] = "#7b5ea7"
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    ax = axes[0]
    for cfg, c in curves.items():
        s_ = np.array(c["sensor"]); t = s_[:, 0] - c["t0"]
        ax.plot(t, s_[:, 1], color=COL[cfg], lw=0.8, alpha=0.5)
        f = c["fit_gpu"]
        tt = np.linspace(0, t.max(), 200)
        ax.plot(tt, f["T_ss"] + (f["T_0"] - f["T_ss"]) * np.exp(-tt / f["tau_s"]), color=COL[cfg], lw=1.6,
                ls="--" if cfg == "C2 paced" else "-",
                label=f"{cfg}: $\\tau$={f['tau_s']:.0f}s, $T_\\infty$={f['T_ss']:.1f}°C")
    ax.set_xlabel("time under load (s)"); ax.set_ylabel("M5 GPU temperature (°C)")
    ax.set_title("heating under sustained YOLO11 inference", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=6.3)
    ax = axes[1]
    for pol, r in pz["runs"].items():
        ax.plot(r["trace"]["t"], r["trace"]["gpu_temp"], color=COL[pol], lw=1.1,
                ls="--" if pol == "static_C1" else "-",
                label=f"{LBL[pol]} (C1 {r['frac_C1']*100:.0f}%, above cap {r['time_above_cap']*100:.0f}%)")
    ax.axhline(pz["cap_c"], color="k", lw=0.7)
    ax.text(2, pz["cap_c"] + 0.4, f"cap {pz['cap_c']:.1f}°C", fontsize=6.5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("M5 GPU temperature (°C)")
    ax.set_title(f"routers live on the M5, one frame per {pz['period_s']*1e3:.0f} ms", fontsize=8.5)
    ax.legend(loc="lower right", fontsize=6.3)
    fig.tight_layout(pad=0.4)
    out = ROOT / "paper" / "figures"
    fig.savefig(out / "fig_m5.pdf"); fig.savefig(out / "fig_m5.png", dpi=200)
    plt.close(fig)

    rows = []
    for cfg, c in curves.items():
        lat = c.get("lat_median_ms", float("nan"))
        rows.append(" & ".join([cfg, f"{lat:.0f}",
                                f"{c['load_power_w'] - c['idle_power_w']:.1f}", f"{c['fit_gpu']['tau_s']:.0f}",
                                f"{c['fit_gpu']['T_ss'] - c['idle_gpu_temp']:.1f}",
                                f"{(c['fit_gpu']['T_ss'] - c['idle_gpu_temp']) / (c['load_power_w'] - c['idle_power_w']):.2f}",
                                f"{c['fit_gpu']['rmse_c']:.2f}"]) + r" \\")
    (ROOT / "paper" / "tables" / "tab_m5.tex").write_text("\n".join(rows) + "\n")
    summ = {"cap": pz["cap_c"], "period_ms": pz["period_s"] * 1e3, "idle_T": d["curves"]["C1"]["idle_gpu_temp"],
            "runs": {p: {k: r[k] for k in ("frac_C1", "time_above_cap", "peak_gpu_temp", "mean_lat_ms", "miss_rate")}
                     for p, r in pz["runs"].items()},
            "curves": {c: {"tau": v["fit_gpu"]["tau_s"], "rise": v["fit_gpu"]["T_ss"] - v["idle_gpu_temp"],
                           "T_ss": v["fit_gpu"]["T_ss"], "dP": v["load_power_w"] - v["idle_power_w"],
                           "rmse": v["fit_gpu"]["rmse_c"], "lat": v.get("lat_median_ms"),
                           "lat_first": v.get("lat_first_ms"), "lat_last": v.get("lat_last_ms")}
                       for c, v in curves.items()},
            "unpaced_runs": {p: {k: r[k] for k in ("frac_C1", "time_above_cap", "peak_gpu_temp")}
                             for p, r in d["hil"]["runs"].items()}}
    (ROOT / "paper" / "tables" / "m5_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
