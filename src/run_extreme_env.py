"""Extreme-environment experiments (SWRS 2026, "Robotics in Extreme Environments").

Runs entirely from the cached per-track outputs (phase1 observations, phase3
accuracy + latency distributions); no frames or GPU needed.

  E1  Burst analysis. In the original coupled regime, does coupling-aware routing
      shorten perception BLACKOUTS (runs of consecutive deadline misses), not just
      the average miss rate?
  E2  Thermal throttling. Contention is endogenous: the router's own config choices
      heat the SoC (RC thermal model, src/thermal.py) until DVFS throttles it. Sweep
      ambient temperature 20-55 degC and compare static, reactive, belief (decoupled)
      and a headroom-aware thermal policy.
  E3  Compound regime. Fault-coupled contention (the original imposed regime) PLUS
      thermal throttling at the same time; adds the Joint+Thermal policy.
      duty_matched spends the thermal policy's C1 budget without temperature knowledge.
  E4  Thermal-model misspecification. The headroom-aware policy holds a wrong
      thermal model (R_th, tau off) while the plant uses the true one.

    python src/run_extreme_env.py            # all tracks, all parts
    python src/run_extreme_env.py --quick    # 2 seeds, 3 ambients (smoke test)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

TRACKS = [
    ("trackA", "TartanDrive full (clean)"),
    ("trackA_309", "TartanDrive off-road seg."),
    ("trackD_rain_4_0", "RADIATE rain"),
    ("trackD_snow_1_0", "RADIATE snow"),
    ("trackD_fog_6_0", "RADIATE fog"),
    ("trackD_night_1_0", "RADIATE night"),
    ("trackC", "WoodScape soiling"),
]
AMBIENTS = [20, 25, 30, 35, 40, 45, 50, 55]
THERMAL_POLICIES = ["static_C1", "static_C4", "reactive", "decoupled", "thermal", "duty_matched"]
COMPOUND_POLICIES = ["decoupled", "joint", "thermal", "joint_thermal"]
METRICS = ["deadline_miss_rate", "mean_accuracy", "on_time_utility", "throttled_frac",
           "peak_temp_c", "mean_power_w", "frac_C1", "max_miss_run", "burst_frac_k3"]
N_FRAMES = 1200   # 2-minute mission at 10 fps (real trace tiled)
PHOENIX_C = 45    # typical Phoenix summer afternoon ambient


def _load_track(track: str):
    os.environ["OUTPUTS_DIR"] = f"outputs/{track}"
    import importlib
    import config_util, simulate, policies
    importlib.reload(config_util)
    cfg = config_util.load_config()
    out = config_util.abspath(cfg["paths"]["outputs_dir"])
    lat = np.load(out / "phase3" / "latency_distributions.npz")
    keys = list(cfg["configs"].keys())
    lat_dists = {c: {"nominal": lat[f"{c}_nominal"], "contended": lat[f"{c}_contended"]} for c in keys}
    return cfg, lat_dists


def _agg(vals_by_policy, base=None):
    import ci as cistats
    res = {}
    for p, rows in vals_by_policy.items():
        res[p] = {m: cistats.mean_ci([r[m] for r in rows]) for m in METRICS}
    if base:
        for p, rows in vals_by_policy.items():
            if p == base:
                continue
            res[p]["miss_reduction_vs_" + base] = cistats.paired_diff_ci(
                [r["deadline_miss_rate"] for r in vals_by_policy[base]],
                [r["deadline_miss_rate"] for r in rows])
            res[p]["max_run_reduction_vs_" + base] = cistats.paired_diff_ci(
                [r["max_miss_run"] for r in vals_by_policy[base]],
                [r["max_miss_run"] for r in rows])
            res[p]["utility_gain_vs_" + base] = cistats.paired_diff_ci(
                [r["on_time_utility"] for r in rows],
                [r["on_time_utility"] for r in vals_by_policy[base]])
    if "thermal" in vals_by_policy and "duty_matched" in vals_by_policy:
        # matched-budget control: does knowing the temperature beat spending the same C1 budget blind?
        res["thermal_vs_duty_matched_miss"] = cistats.paired_diff_ci(
            [r["deadline_miss_rate"] for r in vals_by_policy["duty_matched"]],
            [r["deadline_miss_rate"] for r in vals_by_policy["thermal"]])
        res["thermal_vs_duty_matched_throttle"] = cistats.paired_diff_ci(
            [r["throttled_frac"] for r in vals_by_policy["duty_matched"]],
            [r["throttled_frac"] for r in vals_by_policy["thermal"]])
    return res


def e1_bursts(track, seeds):
    import simulate as sim
    import policies as pol
    import thermal as th
    import ci as cistats
    cfg, _ = _load_track(track)
    rt = cfg["routing"]["reliability_target"]
    dwell = cfg["routing"]["hysteresis"]["fixed"]["dwell_frames"]
    kappa = sim.calibration_kappa(cfg, "coupled", [s + 100 for s in seeds])
    rows = {k: [] for k in ("joint", "decoupled", "oracle", "reactive")}
    for s in seeds:
        sub = sim.build_substrate(cfg, regime="coupled", seed=s)
        runs = {
            "joint": sim.run_policy(pol.JointPolicy(sub.fm, kappa, rt, pol.Hysteresis(dwell)), sub),
            "decoupled": sim.run_policy(pol.DecoupledPolicy(sub.fm, rt, pol.Hysteresis(dwell)), sub),
            "oracle": sim.run_oracle_policy(pol.OracleContentionPolicy(sub.fm, rt, pol.Hysteresis(dwell)), sub),
            "reactive": sim.run_reactive_policy(pol.ReactiveLatencyPolicy(sub.fm, rt, pol.Hysteresis(dwell)), sub),
        }
        for k, r in runs.items():
            miss = np.array([c != pol.ABSTAIN and sub.L[c][t] > sub.deadline_s
                             for t, c in enumerate(r["choices"])])
            rows[k].append({"deadline_miss_rate": r["deadline_miss_rate"], **th.burst_stats(miss)})
    out = {"kappa": kappa}
    for k, v in rows.items():
        out[k] = {m: cistats.mean_ci([x[m] for x in v]) for m in ("deadline_miss_rate", "max_miss_run", "burst_frac_k3")}
    out["max_run_reduction_decoupled_minus_joint"] = cistats.paired_diff_ci(
        [x["max_miss_run"] for x in rows["decoupled"]], [x["max_miss_run"] for x in rows["joint"]])
    out["burst_frac_reduction_decoupled_minus_joint"] = cistats.paired_diff_ci(
        [x["burst_frac_k3"] for x in rows["decoupled"]], [x["burst_frac_k3"] for x in rows["joint"]])
    return out


def _closed_loop_suite(track, seeds, ambients, policies, compound, model_override=None):
    import simulate as sim
    import regimes as reg
    import thermal as th
    cfg, lat_dists = _load_track(track)
    model = th.ThermalModel.from_cfg(cfg)
    kappa = sim.calibration_kappa(cfg, "coupled", [s + 100 for s in seeds]) if compound else 0.0
    res = {}
    for t_amb in ambients:
        vals = {p: [] for p in policies}
        for s in seeds:
            sub = sim.build_substrate(cfg, regime="uncoupled", seed=s)  # beliefs/acc/frontier only
            exo = None
            if compound:
                fault = sub.fault_active[np.arange(N_FRAMES) % sub.T]
                exo = reg.coupled_schedule(N_FRAMES, fault, cfg, seed=s)
            for p in policies:
                # same rng seed for every policy -> identical latency/temperature-noise draws
                rng = np.random.default_rng(50_000 + 97 * s + int(t_amb))
                budget = vals["thermal"][-1]["frac_C1"] if p == "duty_matched" else 1.0
                vals[p].append(th.run_closed_loop(p, sub, lat_dists, model, float(t_amb), rng,
                                                  N_FRAMES, kappa=kappa, exo_contention=exo,
                                                  policy_model=model_override, c1_budget=budget))
        res[str(t_amb)] = _agg(vals, base="decoupled")
    return {"kappa": kappa, "model": model.__dict__, "by_ambient": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    ambients = [25, 45, 55] if args.quick else AMBIENTS
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "outputs" / "extreme_env"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validation-locked protocol (after Zhou & Roy, arXiv:2609.23924): every tunable
    # (thermal parameters, forecast horizon, margin, dwell, slow-ratio) is fixed a priori
    # in code/config before any run; nothing is selected on evaluation outcomes.
    results = {"seeds": seeds, "n_frames": N_FRAMES, "tracks": {},
               "protocol": {"eval_labels_used_for_selection": False,
                            "tunables_fixed_a_priori": ["thermal.model.*", "horizon_frames=50",
                                                        "margin_c=2.0", "dwell=10", "slow_ratio=1.5"]}}
    for track, label in TRACKS:
        print(f"== {track}", flush=True)
        r = {"label": label}
        r["E1_bursts"] = e1_bursts(track, seeds)
        r["E2_thermal"] = _closed_loop_suite(track, seeds, ambients, THERMAL_POLICIES, compound=False)
        r["E3_compound"] = _closed_loop_suite(track, seeds, ambients, COMPOUND_POLICIES, compound=True)
        import thermal as th
        cfg, _ = _load_track(track)
        m = th.ThermalModel.from_cfg(cfg)
        r["E4_misspec"] = {}
        for name, mm in {"true": m, "R_x0.7": replace(m, r_th=m.r_th * 0.7),
                         "R_x1.3": replace(m, r_th=m.r_th * 1.3),
                         "tau_x0.5": replace(m, tau_s=m.tau_s * 0.5),
                         "tau_x2": replace(m, tau_s=m.tau_s * 2.0)}.items():
            r["E4_misspec"][name] = _closed_loop_suite(track, seeds, [PHOENIX_C], ["decoupled", "thermal"],
                                                       compound=False, model_override=mm)["by_ambient"][str(PHOENIX_C)]
        results["tracks"][track] = r
        e2 = r["E2_thermal"]["by_ambient"][str(PHOENIX_C if not args.quick else 45)]
        print(f"   E1 max-run red. {r['E1_bursts']['max_run_reduction_decoupled_minus_joint']['mean']:+.2f} | "
              f"E2@45C miss dec {e2['decoupled']['deadline_miss_rate']['mean']:.3f} "
              f"thermal {e2['thermal']['deadline_miss_rate']['mean']:.3f}", flush=True)

    name = "extreme_env_quick.json" if args.quick else "extreme_env.json"
    with open(out_dir / name, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("wrote", out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
