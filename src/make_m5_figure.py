"""Figure and table for the Apple M5 heating measurement
(outputs/m5_hardware/m5_unpaced.json, written by src/m5_hardware.py).

    python src/m5_hardware.py && python src/make_m5_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.fontsize": 7, "legend.frameon": False})
COL = {"C1": "#c0392b", "C2": "#c58b2b", "C4": "#3b6fb6"}


def main():
    d = json.load(open(ROOT / "outputs" / "m5_hardware" / "m5_unpaced.json"))
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for cfg, c in d["curves"].items():
        s = np.array(c["sensor"]); t = s[:, 0] - c["t0"]
        ax.plot(t, s[:, 1], color=COL[cfg], lw=0.7, alpha=0.45)
        f = c["fit_gpu"]
        tt = np.linspace(0, t.max(), 200)
        ax.plot(tt, f["T_ss"] + (f["T_0"] - f["T_ss"]) * np.exp(-tt / f["tau_s"]), color=COL[cfg], lw=1.6,
                label=f"{cfg}: $\\tau$={f['tau_s']:.0f}s, $T_\\infty$={f['T_ss']:.1f}°C")
    ax.set_xlabel("time under load (s)"); ax.set_ylabel("M5 GPU temperature (°C)")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    out = ROOT / "paper" / "figures"
    fig.savefig(out / "fig_m5.pdf"); fig.savefig(out / "fig_m5.png", dpi=200)
    plt.close(fig)

    rows, summ = [], {}
    for cfg, c in d["curves"].items():
        dP = c["load_power_w"] - c["idle_power_w"]
        rise = c["fit_gpu"]["T_ss"] - c["idle_gpu_temp"]
        rows.append(" & ".join([cfg, f"{c['lat_median_ms']:.0f}", f"{c['lat_first_ms']:.0f}", f"{c['lat_last_ms']:.0f}",
                                f"{dP:.1f}", f"{c['fit_gpu']['tau_s']:.0f}", f"{rise:.1f}", f"{rise / dP:.2f}",
                                f"{c['fit_gpu']['rmse_c']:.2f}"]) + r" \\")
        summ[cfg] = {"lat": c["lat_median_ms"], "dP": dP, "tau": c["fit_gpu"]["tau_s"], "T_ss": c["fit_gpu"]["T_ss"],
                     "rise": rise, "R": rise / dP, "rmse": c["fit_gpu"]["rmse_c"], "idle_T": c["idle_gpu_temp"],
                     "idle_P": c["idle_power_w"], "fan_end": c["fan_rpm_end"]}
    (ROOT / "paper" / "tables" / "tab_m5.tex").write_text("\n".join(rows) + "\n")
    (ROOT / "paper" / "tables" / "m5_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
