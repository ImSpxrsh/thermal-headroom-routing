"""Figures and LaTeX tables for the extreme-heat paper, generated directly from
outputs/extreme_env/extreme_env.json so no number is transcribed by hand.

    python src/run_extreme_env.py && python src/make_extreme_figures.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
SHORT = {"trackA": "Off-road (clean)", "trackA_309": "Off-road seg.", "trackD_rain_4_0": "Rain",
         "trackD_snow_1_0": "Snow", "trackD_fog_6_0": "Fog", "trackD_night_1_0": "Night",
         "trackC": "Soiling (fisheye)"}
STYLE = {"static_C1": ("Static C1", "#9a9a9a", "--"), "reactive": ("Reactive", "#c58b2b", ":"),
         "decoupled": ("Belief (no thermal)", "#3b6fb6", "-"),
         "duty_matched": ("Matched budget", "#7b5ea7", "-."),
         "thermal": ("THR (ours)", "#c0392b", "-")}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "legend.fontsize": 7.5, "legend.frameon": False})


def pp(m, sig=True):
    s = f"{m['mean']*100:+.1f}"
    if sig and m.get("significant"):
        s += "$^\\ast$"
    return s


def fig_sweep(d):
    tracks = list(d["tracks"])
    ambs = sorted(int(a) for a in d["tracks"][tracks[0]]["E2_thermal"]["by_ambient"])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for p, (lbl, col, ls) in STYLE.items():
        for ax, met, scale in zip(axes, ["deadline_miss_rate", "throttled_frac", "max_miss_run"], [100, 100, 0.1]):
            v = np.array([[d["tracks"][t]["E2_thermal"]["by_ambient"][str(a)][p][met]["mean"] * scale
                           for a in ambs] for t in tracks])
            ax.plot(ambs, v.mean(0), ls, color=col, lw=1.6, label=lbl, marker="o", ms=2.5)
            ax.fill_between(ambs, v.min(0), v.max(0), color=col, alpha=0.12, lw=0)
    axes[0].set_ylabel("deadline-miss rate (%)")
    axes[1].set_ylabel("time throttled (%)")
    axes[2].set_ylabel("longest blackout (s)"); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("ambient temperature (°C)")
        ax.axvline(45, color="k", lw=0.6, alpha=0.4)
    axes[0].text(45.5, axes[0].get_ylim()[1] * 0.95, "Phoenix\nsummer", fontsize=6.5, va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_heat_sweep.pdf"); fig.savefig(FIG / "fig_heat_sweep.png", dpi=200)
    plt.close(fig)


def fig_trace():
    """One 2-minute mission at 45 degC on the fog trace: SoC temperature per policy."""
    os.environ["OUTPUTS_DIR"] = "outputs/trackD_fog_6_0"
    import config_util, simulate as sim, thermal as th
    cfg = config_util.load_config()
    out = config_util.abspath(cfg["paths"]["outputs_dir"])
    lat = np.load(out / "phase3" / "latency_distributions.npz")
    lat_dists = {c: {"nominal": lat[f"{c}_nominal"], "contended": lat[f"{c}_contended"]} for c in cfg["configs"]}
    model = th.ThermalModel.from_cfg(cfg)
    sub = sim.build_substrate(cfg, regime="uncoupled", seed=0)
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    tt = np.arange(1200) * model.dt_s
    for p in ("static_C1", "decoupled", "thermal"):
        r = th.run_closed_loop(p, sub, lat_dists, model, 45.0, np.random.default_rng(50_000 + 45),
                               1200, return_trace=True)
        lbl, col, ls = STYLE[p]
        ax.plot(tt, r["trace"]["temp_c"], ls, color=col, lw=1.2, label=lbl)
    ax.axhline(model.t_hi, color="k", lw=0.6); ax.text(1, model.t_hi + 0.4, "throttle", fontsize=6.5)
    ax.axhline(model.t_lo, color="k", lw=0.6, ls=":"); ax.text(1, model.t_lo - 1.8, "release", fontsize=6.5)
    ax.set_xlabel("mission time (s)"); ax.set_ylabel("SoC temperature (°C)")
    ax.set_ylim(60, 93); ax.legend(loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_heat_trace.pdf"); fig.savefig(FIG / "fig_heat_trace.png", dpi=200)
    plt.close(fig)


def fig_compound(d, amb="45"):
    tracks = list(d["tracks"])
    pols = [("joint", "Fault coupling only", "#3b6fb6"), ("thermal", "THR only", "#c0392b"),
            ("joint_thermal", "Coupling + THR", "#2c7a4b")]
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    x = np.arange(len(tracks)); w = 0.26
    for i, (p, lbl, col) in enumerate(pols):
        ms = [d["tracks"][t]["E3_compound"]["by_ambient"][amb][p]["miss_reduction_vs_decoupled"] for t in tracks]
        m = np.array([q["mean"] for q in ms]) * 100
        err = np.array([[q["mean"] - q["lo"], q["hi"] - q["mean"]] for q in ms]).T * 100
        ax.bar(x + (i - 1) * w, m, w, yerr=err, color=col, label=lbl, capsize=1.5, error_kw={"lw": 0.6})
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in tracks], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("miss reduction (pp)"); ax.axhline(0, color="k", lw=0.5)
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "fig_compound.pdf"); fig.savefig(FIG / "fig_compound.png", dpi=200)
    plt.close(fig)


def tables(d):
    TAB.mkdir(exist_ok=True)
    rows = []
    for t, r in d["tracks"].items():
        cells = [SHORT[t]]
        for amb in ("45", "55"):
            e = r["E2_thermal"]["by_ambient"][amb]
            cells += [f"{e['static_C1']['max_miss_run']['mean']/10:.1f}",
                      f"{e['decoupled']['throttled_frac']['mean']*100:.0f}",
                      f"{e['thermal']['throttled_frac']['mean']*100:.0f}",
                      pp(e["thermal"]["miss_reduction_vs_decoupled"]),
                      pp(e["duty_matched"]["miss_reduction_vs_decoupled"]),
                      pp(e["thermal_vs_duty_matched_miss"])]
        rows.append(" & ".join(cells) + r" \\")
    (TAB / "tab_heat_wide.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for t, r in d["tracks"].items():
        cells = [SHORT[t]]
        for amb in ("25", "45", "55"):
            e = r["E3_compound"]["by_ambient"][amb]
            cells += [pp(e["joint"]["miss_reduction_vs_decoupled"]),
                      pp(e["joint_thermal"]["miss_reduction_vs_decoupled"])]
        rows.append(" & ".join(cells) + r" \\")
    (TAB / "tab_compound.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for name, lbl in [("true", "true model"), ("R_x0.7", r"$R_{\mathrm{th}}\times0.7$"),
                      ("R_x1.3", r"$R_{\mathrm{th}}\times1.3$"), ("tau_x0.5", r"$\tau\times0.5$"),
                      ("tau_x2", r"$\tau\times2$")]:
        ds = [r["E4_misspec"][name]["thermal"]["miss_reduction_vs_decoupled"] for r in d["tracks"].values()]
        acc = [r["E4_misspec"][name]["thermal"]["mean_accuracy"]["mean"] for r in d["tracks"].values()]
        thr = max(r["E4_misspec"][name]["thermal"]["throttled_frac"]["mean"] for r in d["tracks"].values())
        nsig = sum(q["significant"] for q in ds)
        rows.append(f"{lbl} & {min(q['mean'] for q in ds)*100:+.1f} to {max(q['mean'] for q in ds)*100:+.1f}"
                    f" & {nsig}/{len(ds)} & {min(acc):.3f} to {max(acc):.3f} & {thr*100:.0f} \\\\")
    (TAB / "tab_misspec.tex").write_text("\n".join(rows) + "\n")

    rows = []
    for t, r in d["tracks"].items():
        e = r["E1_bursts"]
        rows.append(" & ".join([SHORT[t],
                                f"{e['decoupled']['burst_frac_k3']['mean']*100:.1f}",
                                f"{e['joint']['burst_frac_k3']['mean']*100:.1f}",
                                pp(e["burst_frac_reduction_decoupled_minus_joint"]),
                                f"{e['decoupled']['max_miss_run']['mean']:.1f}",
                                f"{e['joint']['max_miss_run']['mean']:.1f}"]) + r" \\")
    (TAB / "tab_bursts.tex").write_text("\n".join(rows) + "\n")


def tab_efficiency(d):
    """Efficiency-quality-cost summary per policy, mean over tracks (min--max in text)."""
    pols = [("static_C1", "Static C1"), ("static_C4", "Static C4"), ("reactive", "Reactive"),
            ("decoupled", "Belief"), ("duty_matched", "Matched budget"), ("thermal", "THR (ours)")]
    rows = []
    for amb in ("45", "55"):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Ambient {amb}\\,\\textdegree C}}}} \\\\")
        for p, lbl in pols:
            es = [r["E2_thermal"]["by_ambient"][amb][p] for r in d["tracks"].values()]
            mean = lambda k: float(np.mean([e[k]["mean"] for e in es]))
            miss, pw = mean("deadline_miss_rate"), mean("mean_power_w")
            # energy per ON-TIME frame: J per frame / fraction of frames delivered on time
            j_ontime = float(np.mean([e["mean_power_w"]["mean"] * 0.1 / max(1 - e["deadline_miss_rate"]["mean"], 1e-9)
                                      for e in es]))
            rows.append(" & ".join([lbl, f"{miss*100:.1f}", f"{mean('throttled_frac')*100:.1f}",
                                    f"{mean('max_miss_run')/10:.1f}", f"{pw:.1f}", f"{j_ontime:.2f}",
                                    f"{mean('mean_accuracy'):.3f}"]) + r" \\")
    (TAB / "tab_efficiency.tex").write_text("\n".join(rows) + "\n")


def main():
    d = json.load(open(ROOT / "outputs" / "extreme_env" / "extreme_env.json"))
    FIG.mkdir(parents=True, exist_ok=True)
    fig_sweep(d); fig_compound(d); tables(d); tab_efficiency(d); fig_trace()
    print("figures ->", FIG, "| tables ->", TAB)


if __name__ == "__main__":
    main()
