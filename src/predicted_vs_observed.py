"""Predicted-vs-observed validation: does the phase diagram QUANTITATIVELY predict the
routing benefit at each sequence's coupling operating point?

The phase diagram (run_injection_sweep.py) maps (coupling strength x fault-onset rate) ->
joint-vs-decoupled miss reduction, on each track's real latency/accuracy profile. Every
evaluated sequence occupies TWO operating points on that map:

  imposed:  coupling = Pc(F)-Pc(N) from config (0.85-0.05=0.80), onset rate measured from
            the track's real fault timeline. Observed benefit = the committed RQ-H.
  measured: coupling = measured Pc(F)-Pc(N) from measure_real_coupling.py (real load),
            same onset rate. Observed benefit = the de-circularized RQ-H.

If predictions track observations at BOTH points -- including predicting ~0 where the
measured real coupling is ~0 -- the phase diagram is a validated predictive model, and the
real-data null becomes a CONFIRMED prediction rather than a failure. That is the
scientifically strongest framing of the Experiment A result.

    python src/predicted_vs_observed.py --tracks outputs/trackD_rain_4_0 outputs/trackD_fog_6_0 ...
Writes outputs/multitrace/predicted_vs_observed.{json,png}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def jload(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def interp_grid(pd_json, coupling, onset_rate):
    """Bilinear interpolation of the phase-diagram grid at (coupling, onset_rate),
    clamped to the grid edges."""
    cs = np.asarray(pd_json["couplings"], float)
    rs = np.asarray(pd_json["onset_rates"], float)
    grid = np.asarray(pd_json["grid_miss_reduction_pp"], float)  # [coupling, onset]
    c = float(np.clip(coupling, cs.min(), cs.max()))
    r = float(np.clip(onset_rate, rs.min(), rs.max()))
    i = int(np.clip(np.searchsorted(cs, c) - 1, 0, len(cs) - 2))
    j = int(np.clip(np.searchsorted(rs, r) - 1, 0, len(rs) - 2))
    tc = (c - cs[i]) / (cs[i + 1] - cs[i]) if cs[i + 1] > cs[i] else 0.0
    tr = (r - rs[j]) / (rs[j + 1] - rs[j]) if rs[j + 1] > rs[j] else 0.0
    top = grid[i, j] * (1 - tr) + grid[i, j + 1] * tr
    bot = grid[i + 1, j] * (1 - tr) + grid[i + 1, j + 1] * tr
    return float(top * (1 - tc) + bot * tc)


def onset_rate_per_min(track_dir, fps):
    lab = pd.read_csv(track_dir / "phase1" / "trackA_observations.csv")
    fa = lab["any_fault"].to_numpy(bool)
    onsets = int(np.sum(fa[1:] & ~fa[:-1]) + (1 if fa[0] else 0))
    minutes = len(fa) / fps / 60.0
    return onsets / minutes if minutes > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", required=True)
    ap.add_argument("--fps", type=float, default=10.0, help="nominal fps for onset-rate calc")
    args = ap.parse_args()

    points = []
    for t in args.tracks:
        d = Path(t)
        if not d.is_absolute() and not (d / "phase1").exists():
            d = ROOT / t
        pdg = jload(d / "extras" / "phase_diagram.json")
        rqh = jload(d / "phase5" / "phase5_rqh.json")
        if not pdg or not rqh:
            print(f"  SKIP {t}: needs extras/phase_diagram.json + phase5_rqh.json"); continue
        rate = onset_rate_per_min(d, args.fps)
        name = d.name

        # imposed operating point (the headline RQ-H)
        c_imp = (rqh["summary"]["coupled"]["kappa_mean"])
        obs_imp = rqh["summary"]["coupled"]["miss_reduction_decoupled_minus_joint"]["mean"] * 100
        points.append({"track": name, "kind": "imposed", "coupling": c_imp,
                       "onset_per_min": rate, "predicted_pp": interp_grid(pdg, c_imp, rate),
                       "observed_pp": obs_imp})

        # uncoupled control point (coupling ~ 0, observed 0)
        c_unc = rqh["summary"]["uncoupled"]["kappa_mean"]
        obs_unc = rqh["summary"]["uncoupled"]["miss_reduction_decoupled_minus_joint"]["mean"] * 100
        points.append({"track": name, "kind": "uncoupled-control", "coupling": c_unc,
                       "onset_per_min": rate, "predicted_pp": interp_grid(pdg, c_unc, rate),
                       "observed_pp": obs_unc})

        # measured real-load operating point (Experiment A), if present
        rc = jload(d / "extras" / "real_coupling.json")
        if rc and rc.get("measured_Pc_fault") is not None:
            c_meas = max(0.0, rc["measured_Pc_fault"] - rc["measured_Pc_nominal"])
            points.append({"track": name, "kind": "measured-real-load", "coupling": c_meas,
                           "onset_per_min": rate,
                           "predicted_pp": interp_grid(pdg, c_meas, rate),
                           "observed_pp": rc.get("decirc_reduction_pp")})

    if not points:
        print("no usable tracks"); return 2

    pred = np.array([p["predicted_pp"] for p in points if p["observed_pp"] is not None])
    obs = np.array([p["observed_pp"] for p in points if p["observed_pp"] is not None])
    r = float(np.corrcoef(pred, obs)[0, 1]) if pred.std() > 1e-9 and obs.std() > 1e-9 else float("nan")
    mae = float(np.mean(np.abs(pred - obs)))

    outdir = ROOT / "outputs" / "multitrace"; outdir.mkdir(parents=True, exist_ok=True)
    json.dump({"points": points, "pearson_r": r, "mae_pp": mae},
              open(outdir / "predicted_vs_observed.json", "w"), indent=2, default=str)

    # scatter with y=x
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(figsize=(7, 6))
    styles = {"imposed": ("o", "tab:blue"), "uncoupled-control": ("x", "0.4"),
              "measured-real-load": ("D", "tab:red")}
    for kind, (mk, col) in styles.items():
        xs = [p["predicted_pp"] for p in points if p["kind"] == kind and p["observed_pp"] is not None]
        ys = [p["observed_pp"] for p in points if p["kind"] == kind and p["observed_pp"] is not None]
        if xs:
            ax.scatter(xs, ys, marker=mk, s=90, color=col, label=kind, zorder=5)
    lim = max(1.0, float(np.max(np.abs(np.concatenate([pred, obs]))))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.6)
    ax.set_xlim(-lim * 0.15, lim); ax.set_ylim(-lim * 0.15, lim)
    ax.set_xlabel("Phase-diagram PREDICTED reduction (pp)")
    ax.set_ylabel("OBSERVED reduction (pp)")
    ax.set_title(f"Phase diagram as a predictive model\n(r={r:.2f}, MAE={mae:.1f}pp; "
                 "diamonds = measured real-load points)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "predicted_vs_observed.png", dpi=170)
    plt.close(fig)

    print(f"\n=== predicted vs observed ({len(points)} points) ===")
    for p in points:
        print(f"  {p['track']:24s} {p['kind']:20s} coupling={p['coupling']:.2f} "
              f"onset/min={p['onset_per_min']:.1f}  pred {p['predicted_pp']:+.2f}pp  "
              f"obs {p['observed_pp'] if p['observed_pp'] is None else format(p['observed_pp'], '+.2f')}pp")
    print(f"  Pearson r={r:.3f}  MAE={mae:.2f}pp")
    print(f"  -> {outdir / 'predicted_vs_observed.json'} / .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
