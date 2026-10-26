"""Figures and LaTeX tables for the Thermal-Headroom Routing paper, generated directly
from outputs/extreme_env/extreme_env.json so no number is transcribed by hand.

Significance marks ($^\\ast$) are Holm-corrected across the seven traces of each
family (one family = one comparison at one ambient / setting), at alpha = 0.05.

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
STYLE = {"static_C1": ("Static C1", "#9a9a9a", "--"),
         "decoupled": ("Belief", "#3b6fb6", "-"),
         "thermostat": ("Thermostat", "#2c7a4b", ":"),
         "cap45": ("Cap tuned at 45°C", "#c58b2b", (0, (5, 1, 1, 1))),
         "duty_matched": ("Matched budget", "#7b5ea7", "-."),
         "thermal": ("THR (ours)", "#c0392b", "-")}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "legend.fontsize": 7.5, "legend.frameon": False})


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------
def holm(ps, alpha=0.05):
    """Boolean list: which hypotheses are rejected under Holm's step-down procedure."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m = len(ps)
    rej = np.zeros(m, bool)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (m - k):
            rej[i] = True
        else:
            break
    return rej.tolist()


def signed_count(tests):
    """'b/w': Holm-significant traces where the comparison favours THR / the other policy."""
    sig = family(tests)
    b = sum(1 for t, g in zip(tests, sig) if g and t["mean"] > 0)
    w = sum(1 for t, g in zip(tests, sig) if g and t["mean"] < 0)
    return f"{b}/{w}"


def family(tests):
    """tests: list of paired_diff_ci dicts (one per trace). Returns Holm-significance flags."""
    return holm([t["p_two_sided"] for t in tests])


def fmt(m, sig, scale=100, nd=1):
    s = f"{m['mean']*scale:+.{nd}f}"
    return s + "$^\\ast$" if sig else s


def pp_family(tests, scale=100, nd=1):
    sig = family(tests)
    return [fmt(t, s_, scale, nd) for t, s_ in zip(tests, sig)]


def rng(vals, scale=100, nd=1, signed=True):
    lo, hi = min(vals) * scale, max(vals) * scale
    f = f"{{:+.{nd}f}}" if signed else f"{{:.{nd}f}}"
    return f"{f.format(lo)} to {f.format(hi)}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_sweep(d):
    tracks = list(d["tracks"])
    ambs = sorted(int(a) for a in d["tracks"][tracks[0]]["E2_thermal"]["by_ambient"])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35))
    for p, (lbl, col, ls) in STYLE.items():
        for ax, met, scale in zip(axes, ["deadline_miss_rate", "throttled_frac", "max_miss_run"], [100, 100, 0.1]):
            v = np.array([[d["tracks"][t]["E2_thermal"]["by_ambient"][str(a)][p][met]["mean"] * scale
                           for a in ambs] for t in tracks])
            ax.plot(ambs, v.mean(0), linestyle=ls, color=col, lw=1.5, label=lbl, marker="o", ms=2.3)
            ax.fill_between(ambs, v.min(0), v.max(0), color=col, alpha=0.10, lw=0)
    axes[0].set_ylabel("deadline-miss rate (%)")
    axes[1].set_ylabel("time throttled (%)")
    axes[2].set_ylabel("longest blackout (s)"); axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("ambient temperature (°C)")
        ax.axvline(45, color="k", lw=0.6, alpha=0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0))
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
        ax.plot(tt, r["trace"]["temp_c"], linestyle=ls, color=col, lw=1.2, label=lbl)
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


def fig_pareto(d):
    """Accuracy vs deadline-miss, mean over traces. The fixed-cap family traces the
    frontier a duty cap can reach at one ambient."""
    tracks = list(d["tracks"])
    caps = sorted(d["tracks"][tracks[0]]["E9_caps"]["45"], key=float)
    pts = [("static_C1", "o"), ("static_C4", "o"), ("decoupled", "s"), ("thermostat", "D"),
           ("cap45", "v"), ("duty_matched", "P"), ("thermal", "*")]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for ax, amb in zip(axes, ("45", "55")):
        mean = lambda getter: float(np.mean([getter(d["tracks"][t]) for t in tracks]))
        cx = [mean(lambda r: r["E9_caps"][amb][c]["fixed_cap"]["mean_accuracy"]["mean"]) for c in caps]
        cy = [mean(lambda r: r["E9_caps"][amb][c]["fixed_cap"]["deadline_miss_rate"]["mean"]) * 100 for c in caps]
        ax.plot(cx, cy, "-", color="#b0a89a", lw=1.2, marker=".", ms=5, label="fixed C1 cap")
        for c, x_, y_ in zip(caps, cx, cy):
            if c in ("0.2", "0.6", "1"):
                ax.annotate(c, (x_, y_), fontsize=6, xytext=(3, -8), textcoords="offset points", color="#8a8276")
        for p, mk in pts:
            e = lambda r: r["E2_thermal"]["by_ambient"][amb][p]
            x_ = mean(lambda r: e(r)["mean_accuracy"]["mean"])
            y_ = mean(lambda r: e(r)["deadline_miss_rate"]["mean"]) * 100
            lbl, col = (STYLE[p][0], STYLE[p][1]) if p in STYLE else ("Static C4", "#555555")
            ax.scatter([x_], [y_], marker=mk, s=60 if mk == "*" else 26, color=col, label=lbl, zorder=3)
        ax.set_title(f"ambient {amb} °C"); ax.set_xlabel("pseudo-GT agreement (higher is better)")
    axes[0].set_ylabel("deadline-miss rate (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(0.80, 0.5), fontsize=7)
    fig.tight_layout(pad=0.4, rect=(0, 0, 0.80, 1))
    fig.savefig(FIG / "fig_pareto.pdf"); fig.savefig(FIG / "fig_pareto.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tab_heat(d):
    tracks = list(d["tracks"])
    cols = {}
    for amb in ("45", "55"):
        E = [d["tracks"][t]["E2_thermal"]["by_ambient"][amb] for t in tracks]
        cols[amb] = {
            "blk": [f"{e['static_C1']['max_miss_run']['mean']/10:.1f}" for e in E],
            "thr_b": [f"{e['decoupled']['throttled_frac']['mean']*100:.0f}" for e in E],
            "thr_t": [f"{e['thermal']['throttled_frac']['mean']*100:.0f}" for e in E],
            "d_thr": pp_family([e["thermal"]["miss_reduction_vs_decoupled"] for e in E]),
            "d_tst": pp_family([e["thr_vs_thermostat_miss"] for e in E]),
            "d_mb": pp_family([e["thr_vs_duty_matched_miss"] for e in E]),
        }
    rows = []
    for i, t in enumerate(tracks):
        cells = [SHORT[t]]
        for amb in ("45", "55"):
            c = cols[amb]
            cells += [c["blk"][i], c["thr_b"][i], c["thr_t"][i], c["d_thr"][i], c["d_tst"][i], c["d_mb"][i]]
            if amb == "45":
                cells.append("")
        rows.append(" & ".join(cells) + r" \\")
    (TAB / "tab_heat_wide.tex").write_text("\n".join(rows) + "\n")


def tab_efficiency(d):
    pols = [("static_C1", "Static C1"), ("static_C4", "Static C4"), ("reactive", "Reactive"),
            ("decoupled", "Belief"), ("thermostat", "Thermostat"), ("cap45", "Cap (45\\degc)"),
            ("cap55", "Cap (55\\degc)"), ("duty_matched", "Matched budget"), ("thermal", "THR (ours)")]
    rows = []
    for amb in ("45", "55"):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Ambient {amb}\\degc}}}} \\\\")
        for p, lbl in pols:
            es = [r["E2_thermal"]["by_ambient"][amb][p] for r in d["tracks"].values()]
            mean = lambda k: float(np.mean([e[k]["mean"] for e in es]))
            rows.append(" & ".join([lbl, f"{mean('deadline_miss_rate')*100:.1f}",
                                    f"{mean('throttled_frac')*100:.1f}", f"{mean('max_miss_run')/10:.1f}",
                                    f"{mean('mean_power_w'):.1f}", f"{mean('energy_per_ontime_j'):.2f}",
                                    f"{mean('mean_accuracy'):.3f}"]) + r" \\")
    (TAB / "tab_efficiency.tex").write_text("\n".join(rows) + "\n")


def tab_caps(d):
    """Fixed cap tuned at 45 degC vs THR, evaluated at every ambient (mean over traces)."""
    rows = []
    for amb in ("35", "40", "45", "50", "55"):
        cells = [f"{amb}\\degc"]
        for p in ("cap45", "thermal"):
            es = [r["E2_thermal"]["by_ambient"][amb][p] for r in d["tracks"].values()]
            mean = lambda k: float(np.mean([e[k]["mean"] for e in es]))
            cells += [f"{mean('throttled_frac')*100:.1f}", f"{mean('deadline_miss_rate')*100:.1f}",
                      f"{mean('mean_accuracy'):.3f}"]
        rows.append(" & ".join(cells) + r" \\")
    (TAB / "tab_caps.tex").write_text("\n".join(rows) + "\n")


def tab_compound(d):
    tracks = list(d["tracks"])
    cols = []
    for amb in ("25", "45", "55"):
        E = [d["tracks"][t]["E3_compound"]["by_ambient"][amb] for t in tracks]
        cols.append(pp_family([e["joint"]["miss_reduction_vs_decoupled"] for e in E]))
        cols.append(pp_family([e["joint_thermal"]["miss_reduction_vs_decoupled"] for e in E]))
    rows = [" & ".join([SHORT[t]] + [c[i] for c in cols]) + r" \\" for i, t in enumerate(tracks)]
    (TAB / "tab_compound.tex").write_text("\n".join(rows) + "\n")


def tab_misspec(d):
    rows = []
    for name, lbl in [("true", "true model"), ("R_x0.7", r"$R_{\mathrm{th}}\times0.7$"),
                      ("R_x1.3", r"$R_{\mathrm{th}}\times1.3$"), ("tau_x0.5", r"$\tau\times0.5$"),
                      ("tau_x2", r"$\tau\times2$")]:
        E = [r["E4_misspec"][name] for r in d["tracks"].values()]
        ds = [e["thermal"]["miss_reduction_vs_decoupled"] for e in E]
        acc = [e["thermal"]["mean_accuracy"]["mean"] for e in E]
        thr = max(e["thermal"]["throttled_frac"]["mean"] for e in E)
        nsig = sum(family(ds))
        rows.append(f"{lbl} & {rng([q['mean'] for q in ds])} & {nsig}/{len(ds)} & "
                    f"{min(acc):.3f} to {max(acc):.3f} & {thr*100:.1f} \\\\")
    (TAB / "tab_misspec.tex").write_text("\n".join(rows) + "\n")


def tab_deadline(d):
    """E5: deadline multiplier x ambient; mean over traces + Holm sig counts (THR vs Belief)."""
    rows = []
    for mult in ("1.0", "1.25", "1.5"):
        for amb in ("35", "45", "55"):
            E = [r["E5_deadline"][mult][amb] for r in d["tracks"].values()]
            mean = lambda p, k: float(np.mean([e[p][k]["mean"] for e in E]))
            d_thr = [e["thermal"]["miss_reduction_vs_decoupled"] for e in E]
            d_u = [e["thermal"]["utility_gain_vs_decoupled"] for e in E]
            rows.append(" & ".join([
                f"{mult}$\\times$" if amb == "35" else "", f"{amb}\\degc",
                f"{mean('decoupled','deadline_miss_rate')*100:.1f}",
                f"{mean('thermostat','deadline_miss_rate')*100:.1f}",
                f"{mean('thermal','deadline_miss_rate')*100:.1f}",
                f"{mean('decoupled','throttled_frac')*100:.1f}",
                f"{mean('thermostat','throttled_frac')*100:.1f}",
                f"{mean('thermal','throttled_frac')*100:.1f}",
                f"{mean('decoupled','on_time_utility'):.3f}",
                f"{mean('thermostat','on_time_utility'):.3f}",
                f"{mean('thermal','on_time_utility'):.3f}",
                signed_count(d_thr), signed_count(d_u)]) + r" \\")
        if mult != "1.5":
            rows.append(r"\midrule")
    (TAB / "tab_deadline.tex").write_text("\n".join(rows) + "\n")


def tab_plant(d):
    rows = []
    names = [("R_x0.8", r"$R_{\mathrm{th}}\times0.8$"), ("R_x1.2", r"$R_{\mathrm{th}}\times1.2$"),
             ("tau_x0.5", r"$\tau\times0.5$"), ("tau_x2", r"$\tau\times2$"),
             ("throttle_x0.8", "throttle power $0.8\\times$")]
    for name, lbl in names:
        cells = [lbl]
        for amb in ("45", "55"):
            E = [r["E6_plant"][name][amb] for r in d["tracks"].values()]
            ds = [e["thermal"]["miss_reduction_vs_decoupled"] for e in E]
            mx = lambda p: f"{max(e[p]['throttled_frac']['mean'] for e in E)*100:.1f}"
            cells += [rng([q["mean"] for q in ds]), f"{sum(family(ds))}/7",
                      mx("decoupled"), mx("thermostat"), mx("thermal")]
            if amb == "45":
                cells.append("")
        rows.append(" & ".join(cells) + r" \\")
    (TAB / "tab_plant.tex").write_text("\n".join(rows) + "\n")


def tab_bursts(d):
    tracks = list(d["tracks"])
    E = [d["tracks"][t]["E1_bursts"] for t in tracks]
    dl = pp_family([e["burst_frac_reduction_decoupled_minus_joint"] for e in E])
    rows = []
    for i, t in enumerate(tracks):
        e = E[i]
        rows.append(" & ".join([SHORT[t], f"{e['decoupled']['burst_frac_k3']['mean']*100:.1f}",
                                f"{e['joint']['burst_frac_k3']['mean']*100:.1f}", dl[i],
                                f"{e['decoupled']['max_miss_run']['mean']:.1f}",
                                f"{e['joint']['max_miss_run']['mean']:.1f}"]) + r" \\")
    (TAB / "tab_bursts.tex").write_text("\n".join(rows) + "\n")


def _rg(E, f):
    v = [f(e) for e in E]
    return [min(v), max(v)]


def summary(d):
    """Numbers quoted in the prose, in one place (paper/tables/summary.json)."""
    T = d["tracks"]
    tr = list(T)
    S = {"n_seeds": len(d["seeds"])}
    for amb in ("45", "50", "55"):
        E = [T[t]["E2_thermal"]["by_ambient"][amb] for t in tr]
        S[f"E2_{amb}"] = {
            "thr_vs_belief_miss": _rg(E, lambda e: e["thermal"]["miss_reduction_vs_decoupled"]["mean"]),
            "thr_vs_belief_sig": sum(family([e["thermal"]["miss_reduction_vs_decoupled"] for e in E])),
            "thr_vs_thermostat_miss": _rg(E, lambda e: e["thr_vs_thermostat_miss"]["mean"]),
            "thr_vs_thermostat_sig": sum(family([e["thr_vs_thermostat_miss"] for e in E])),
            "thr_vs_thermostat_throttle_sig": sum(family([e["thr_vs_thermostat_throttle"] for e in E])),
            "thr_vs_thermostat_acc_mean": float(np.mean([e["thr_vs_thermostat_accuracy"]["mean"] for e in E])),
            "thr_vs_thermostat_util_sig": sum(family([e["thr_vs_thermostat_utility"] for e in E])),
            "thr_vs_thermostat_util": _rg(E, lambda e: e["thr_vs_thermostat_utility"]["mean"]),
            "thermostat_throttle": _rg(E, lambda e: e["thermostat"]["throttled_frac"]["mean"]),
            "thr_vs_mb_miss": _rg(E, lambda e: e["thr_vs_duty_matched_miss"]["mean"]),
            "thr_vs_mb_sig": sum(family([e["thr_vs_duty_matched_miss"] for e in E])),
            "thr_vs_mb_throttle_sig": sum(family([e["thr_vs_duty_matched_throttle"] for e in E])),
            "mb_throttle": _rg(E, lambda e: e["duty_matched"]["throttled_frac"]["mean"]),
            "thr_throttle_max": max(e["thermal"]["throttled_frac"]["mean"] for e in E),
            "belief_throttle": _rg(E, lambda e: e["decoupled"]["throttled_frac"]["mean"]),
            "thr_energy_rel_red": float(np.mean([e["thermal"]["energy_reduction_vs_decoupled"]["mean"] /
                                                 e["decoupled"]["energy_per_ontime_j"]["mean"] for e in E])),
            "thr_energy_sig": sum(family([e["thermal"]["energy_reduction_vs_decoupled"] for e in E])),
            "thr_util_vs_belief_sig": sum(family([e["thermal"]["utility_gain_vs_decoupled"] for e in E])),
            "thr_util_vs_belief": _rg(E, lambda e: e["thermal"]["utility_gain_vs_decoupled"]["mean"]),
            "thr_vs_cap45_miss": _rg(E, lambda e: e["thr_vs_cap45_miss"]["mean"]),
            "thr_vs_cap45_sig": sum(family([e["thr_vs_cap45_miss"] for e in E])),
            "thr_vs_cap45_throttle_sig": sum(family([e["thr_vs_cap45_throttle"] for e in E])),
            "thr_vs_cap55_acc_mean": float(np.mean([e["thr_vs_cap55_accuracy"]["mean"] for e in E])),
            "thr_vs_cap55_acc_sig": sum(family([e["thr_vs_cap55_accuracy"] for e in E])),
            "cap45_throttle": _rg(E, lambda e: e["cap45"]["throttled_frac"]["mean"]),
            "fracC1_thr": _rg(E, lambda e: e["thermal"]["frac_C1"]["mean"]),
            "static_c1_blackout_s": _rg(E, lambda e: e["static_C1"]["max_miss_run"]["mean"] / 10),
            "static_c1_throttle": _rg(E, lambda e: e["static_C1"]["throttled_frac"]["mean"]),
        }
    for amb in ("20", "25", "30", "35", "40"):
        E = [T[t]["E2_thermal"]["by_ambient"][amb] for t in tr]
        S[f"E2_{amb}"] = {"max_abs_thr_vs_belief": max(abs(e["thermal"]["miss_reduction_vs_decoupled"]["mean"]) for e in E),
                          "belief_throttle_max": max(e["decoupled"]["throttled_frac"]["mean"] for e in E),
                          "thr_fracC1": _rg(E, lambda e: e["thermal"]["frac_C1"]["mean"]),
                          "thr_acc": _rg(E, lambda e: e["thermal"]["mean_accuracy"]["mean"]),
                          "cap45_acc_minus_thr": float(np.mean([-e["thr_vs_cap45_accuracy"]["mean"] for e in E]))}
    S["caps"] = {t: T[t]["E2_thermal"]["caps"] for t in tr}
    for key in ("E7_thr_settings",):
        for amb in ("45", "55"):
            res = {}
            for hm in T[tr[0]][key][amb]:
                E = [T[t][key][amb][hm] for t in tr]
                res[hm] = {"miss": _rg(E, lambda e: e["thermal"]["miss_reduction_vs_decoupled"]["mean"]),
                           "sig": sum(family([e["thermal"]["miss_reduction_vs_decoupled"] for e in E])),
                           "throttle_max": max(e["thermal"]["throttled_frac"]["mean"] for e in E),
                           "acc_mean": float(np.mean([e["thermal"]["mean_accuracy"]["mean"] for e in E]))}
            S[f"E7_{amb}"] = res
    for amb in ("45", "55"):
        res = {}
        for ts in T[tr[0]]["E8_thermostat"][amb]:
            E = [T[t]["E8_thermostat"][amb][ts] for t in tr]
            res[ts] = {"thr_minus_tst_miss_mean": float(np.mean([e["thr_vs_thermostat_miss"]["mean"] for e in E])),
                       "thr_better_sig": sum(family([e["thr_vs_thermostat_miss"] for e in E])),
                       "tst_throttle_max": max(e["thermostat"]["throttled_frac"]["mean"] for e in E),
                       "tst_miss_mean": float(np.mean([e["thermostat"]["deadline_miss_rate"]["mean"] for e in E])),
                       "tst_acc_mean": float(np.mean([e["thermostat"]["mean_accuracy"]["mean"] for e in E])),
                       "thr_miss_mean": float(np.mean([e["thermal"]["deadline_miss_rate"]["mean"] for e in E])),
                       "thr_acc_mean": float(np.mean([e["thermal"]["mean_accuracy"]["mean"] for e in E]))}
        S[f"E8_{amb}"] = res
    E = [T[t]["E1_bursts"] for t in tr]
    S["E1"] = {"burst_sig": sum(family([e["burst_frac_reduction_decoupled_minus_joint"] for e in E])),
               "maxrun_sig": sum(family([e["max_run_reduction_decoupled_minus_joint"] for e in E])),
               "burst": {SHORT[t]: e["burst_frac_reduction_decoupled_minus_joint"]["mean"] for t, e in zip(tr, E)}}
    for amb in ("25", "45", "55"):
        E = [T[t]["E3_compound"]["by_ambient"][amb] for t in tr]
        S[f"E3_{amb}"] = {"jt": _rg(E, lambda e: e["joint_thermal"]["miss_reduction_vs_decoupled"]["mean"]),
                          "jt_sig": sum(family([e["joint_thermal"]["miss_reduction_vs_decoupled"] for e in E])),
                          "j_sig": sum(family([e["joint"]["miss_reduction_vs_decoupled"] for e in E])),
                          "thr_clean": E[0]["thermal"]["miss_reduction_vs_decoupled"]["mean"],
                          "j_clean": E[0]["joint"]["miss_reduction_vs_decoupled"]["mean"],
                          "jt_minus_j_min": min(e["joint_thermal"]["miss_reduction_vs_decoupled"]["mean"] -
                                                e["joint"]["miss_reduction_vs_decoupled"]["mean"] for e in E)}
    for mult in ("1.0", "1.25", "1.5"):
        for amb in ("45", "55"):
            E = [T[t]["E5_deadline"][mult][amb] for t in tr]
            S[f"E5_{mult}_{amb}"] = {
                "thr_vs_belief_miss": _rg(E, lambda e: e["thermal"]["miss_reduction_vs_decoupled"]["mean"]),
                "thr_vs_belief_sig": sum(family([e["thermal"]["miss_reduction_vs_decoupled"] for e in E])),
                "thr_util_vs_belief": _rg(E, lambda e: e["thermal"]["utility_gain_vs_decoupled"]["mean"]),
                "thr_util_vs_belief_sig": sum(family([e["thermal"]["utility_gain_vs_decoupled"] for e in E])),
                "thr_vs_tst_util": _rg(E, lambda e: e["thr_vs_thermostat_utility"]["mean"]),
                "thr_vs_tst_util_sig": sum(family([e["thr_vs_thermostat_utility"] for e in E])),
                "thr_vs_tst_miss_sig": sum(family([e["thr_vs_thermostat_miss"] for e in E])),
                "thr_vs_mb_util": _rg(E, lambda e: e["thr_vs_duty_matched_utility"]["mean"]),
                "thr_vs_mb_util_sig": sum(family([e["thr_vs_duty_matched_utility"] for e in E])),
                "acc": {p: float(np.mean([e[p]["mean_accuracy"]["mean"] for e in E])) for p in ("decoupled", "thermostat", "thermal", "duty_matched")},
                "miss": {p: float(np.mean([e[p]["deadline_miss_rate"]["mean"] for e in E])) for p in ("decoupled", "thermostat", "thermal", "duty_matched")},
                "thr": {p: float(np.mean([e[p]["throttled_frac"]["mean"] for e in E])) for p in ("decoupled", "thermostat", "thermal", "duty_matched")},
            }
    for name in T[tr[0]]["E6_plant"]:
        for amb in ("45", "55"):
            E = [T[t]["E6_plant"][name][amb] for t in tr]
            S[f"E6_{name}_{amb}"] = {"thr_throttle_max": max(e["thermal"]["throttled_frac"]["mean"] for e in E),
                                     "tst_throttle_max": max(e["thermostat"]["throttled_frac"]["mean"] for e in E),
                                     "belief_throttle_max": max(e["decoupled"]["throttled_frac"]["mean"] for e in E),
                                     "thr_vs_belief_sig": sum(family([e["thermal"]["miss_reduction_vs_decoupled"] for e in E])),
                                     "thr_vs_tst_miss_sig": sum(family([e["thr_vs_thermostat_miss"] for e in E]))}
    (TAB / "summary.json").write_text(json.dumps(S, indent=1))
    return S


def main():
    d = json.load(open(ROOT / "outputs" / "extreme_env" / "extreme_env.json"))
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(exist_ok=True)
    fig_sweep(d); fig_compound(d); fig_pareto(d); fig_trace()
    tab_heat(d); tab_efficiency(d); tab_caps(d); tab_compound(d); tab_misspec(d)
    tab_deadline(d); tab_plant(d); tab_bursts(d)
    summary(d)
    print("figures ->", FIG, "| tables ->", TAB)


if __name__ == "__main__":
    main()
