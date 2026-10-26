"""Regenerate the headline RQ-H summary figure (Fig. 1) from the committed phase5
outputs, with a SIGN-CONSISTENT axis and print-legible fonts. Addresses the reviewer
notes: the axis is explicitly 'decoupled - coupled' (positive = coupling helps, matching
Table II), fonts are enlarged, and the layout is padded so nothing clips at the column
edge.

    python src/make_summary_figure.py
-> paper/figures/fig_rqh_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# (outputs dir, short label) ordered by effect size
TRACKS = [
    ("trackD_fog_6_0", "fog 6-0"),
    ("trackD_night_1_0", "night"),
    ("trackD_fog_8_0", "fog 8-0"),
    ("trackD_fog_8_1", "fog 8-1"),
    ("trackA_309", "off-road seg."),
    ("trackD_snow_1_0", "snow"),
    ("trackD_rain_2_0", "rain 2-0"),
    ("trackD_rain_3_0", "rain 3-0"),
    ("trackC", "soiling"),
    ("trackD_rain_4_0", "rain 4-0"),
    ("trackA", "off-road full"),
]


def load(t):
    p = OUT / t / "phase5" / "phase5_rqh.json"
    if not p.exists():
        return None
    s = json.load(open(p))["summary"]
    c = s["coupled"]["miss_reduction_decoupled_minus_joint"]
    u = s["uncoupled"]["miss_reduction_decoupled_minus_joint"]
    return c, u


def main():
    rows = [(lbl, *load(t)) for t, lbl in TRACKS if load(t)]
    labels = [r[0] for r in rows]
    means = [r[1]["mean"] * 100 for r in rows]
    los = [(r[1]["mean"] - r[1]["lo"]) * 100 for r in rows]
    his = [(r[1]["hi"] - r[1]["mean"]) * 100 for r in rows]
    sig = [r[1]["significant"] for r in rows]
    unc = [r[2]["mean"] * 100 for r in rows]

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13})
    fig, ax = plt.subplots(figsize=(9, 5.0))
    x = range(len(rows))
    for i, (m, lo, hi, s) in enumerate(zip(means, los, his, sig)):
        marker = "o" if s else "s"
        ax.errorbar(i, m, yerr=[[lo], [hi]], fmt=marker, ms=11, capsize=6, lw=2,
                    color="tab:blue" if s else "tab:gray",
                    mfc="tab:blue" if s else "white", mec="tab:blue" if s else "tab:gray")
    ax.scatter(list(x), unc, marker="x", s=80, color="0.4", zorder=5,
               label="uncoupled control")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Deadline-miss reduction:\ndecoupled − coupled (pp)")
    ax.set_title("RQ-H: coupling-aware routing reduces deadline misses\nunder controlled, fault-conditioned contention")
    ax.legend(loc="upper right", frameon=False)
    ax.margins(x=0.08)
    fig.tight_layout(pad=1.4)
    figs = ROOT / "paper" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "fig_rqh_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figs / 'fig_rqh_summary.png'}")
    for lbl, m, s in zip(labels, means, sig):
        print(f"  {lbl:16s} {m:+.2f}pp {'(sig)' if s else '(n.s.)'}")


if __name__ == "__main__":
    raise SystemExit(main())
