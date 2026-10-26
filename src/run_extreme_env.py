"""Extreme-heat experiments for the Thermal-Headroom Routing paper.

Runs entirely from the cached per-track outputs (phase1 observations, phase3
accuracy + latency distributions); no frames or GPU needed.

  E1  Bursts. In the original exogenous fault-coupled regime, does coupling-aware
      routing shorten runs of consecutive deadline misses?
  E2  Heat. Contention is endogenous: the router's own config choices heat the SoC
      (RC model, src/thermal.py) until DVFS throttles it. Ambient sweep 20-55 degC.
      Baselines: static C1/C4, reactive, belief (decoupled), a thermostat that bars C1
      above 88 degC, fixed C1 caps tuned offline at 45 and at 55 degC, and the
      matched-budget control (THR's own C1 share, no temperature input).
  E3  Heat plus fault-coupled exogenous contention; adds Joint and Joint+THR.
  E4  THR holds a wrong thermal model while the plant keeps the true one.
  E5  Deadline sweep: deadline = 1.0, 1.25, 1.5 x median C1 latency, so at 1.25+ C1
      meets the deadline on a cool chip and throttling is the only source of misses.
  E6  Plant sensitivity: the TRUE chip differs (R_th, tau, throttle power); THR is
      given the matching model (an identified chip).
  E7  THR setting sensitivity: forecast horizon H and margin m.
  E8  Thermostat threshold sweep (best-tuned thermostat, tuned on the test seeds).
  E9  Pareto: fixed C1 caps from 0.2 to 1.0 against every policy.

Every tunable used for the headline (plant parameters, H=50, m=2, dwell=10, slow
ratio 1.5, thermostat 88 degC) was fixed before the first evaluation run; E7/E8 are
reported as sensitivity analyses, never used to pick the headline settings.

    python src/run_extreme_env.py            # full suite, 20 seeds (~15 min)
    python src/run_extreme_env.py --quick    # 2 seeds, smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
THERMAL_POLICIES = ["static_C1", "static_C4", "reactive", "decoupled", "thermostat",
                    "thermal", "duty_matched", "cap45", "cap55"]
COMPOUND_POLICIES = ["decoupled", "joint", "thermal", "joint_thermal"]
SWEEP_POLICIES = ["decoupled", "thermostat", "thermal", "duty_matched"]
METRICS = ["deadline_miss_rate", "mean_accuracy", "on_time_utility", "throttled_frac",
           "peak_temp_c", "mean_power_w", "frac_C1", "max_miss_run", "burst_frac_k3",
           "energy_per_ontime_j"]
N_FRAMES = 1200   # 2-minute mission at 10 fps (real trace tiled)
PHOENIX_C = 45
THERMOSTAT_C = 88.0           # a priori: t_hi - margin, the same line THR forecasts against
DEADLINE_MULTS = [1.0, 1.25, 1.5]
HM_GRID = [(h, m) for h in (20, 50, 100) for m in (0.0, 2.0, 4.0)]
THERMOSTAT_GRID = [84.0, 85.0, 86.0, 87.0, 88.0, 89.0]
CAP_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _load_track(track: str):
    os.environ["OUTPUTS_DIR"] = f"outputs/{track}"
    import importlib
    import config_util
    importlib.reload(config_util)
    cfg = config_util.load_config()
    out = config_util.abspath(cfg["paths"]["outputs_dir"])
    lat = np.load(out / "phase3" / "latency_distributions.npz")
    keys = list(cfg["configs"].keys())
    lat_dists = {c: {"nominal": lat[f"{c}_nominal"], "contended": lat[f"{c}_contended"]} for c in keys}
    return cfg, lat_dists


def _row(r, model):
    r["energy_per_ontime_j"] = r["mean_power_w"] * model.dt_s / max(1.0 - r["deadline_miss_rate"], 1e-9)
    return r


def _agg(vals_by_policy, base="decoupled"):
    import ci as cistats
    res = {}
    for p, rows in vals_by_policy.items():
        res[p] = {m: cistats.mean_ci([r[m] for r in rows]) for m in METRICS}
    if base in vals_by_policy:
        b = vals_by_policy[base]
        for p, rows in vals_by_policy.items():
            if p == base:
                continue
            res[p]["miss_reduction_vs_" + base] = cistats.paired_diff_ci(
                [r["deadline_miss_rate"] for r in b], [r["deadline_miss_rate"] for r in rows])
            res[p]["throttle_reduction_vs_" + base] = cistats.paired_diff_ci(
                [r["throttled_frac"] for r in b], [r["throttled_frac"] for r in rows])
            res[p]["utility_gain_vs_" + base] = cistats.paired_diff_ci(
                [r["on_time_utility"] for r in rows], [r["on_time_utility"] for r in b])
            res[p]["energy_reduction_vs_" + base] = cistats.paired_diff_ci(
                [r["energy_per_ontime_j"] for r in b], [r["energy_per_ontime_j"] for r in rows])
    # THR against each competitor (positive = THR better)
    if "thermal" in vals_by_policy:
        t = vals_by_policy["thermal"]
        for p in ("duty_matched", "thermostat", "cap45", "cap55", "reactive"):
            if p not in vals_by_policy:
                continue
            o = vals_by_policy[p]
            res[f"thr_vs_{p}_miss"] = cistats.paired_diff_ci(
                [r["deadline_miss_rate"] for r in o], [r["deadline_miss_rate"] for r in t])
            res[f"thr_vs_{p}_throttle"] = cistats.paired_diff_ci(
                [r["throttled_frac"] for r in o], [r["throttled_frac"] for r in t])
            res[f"thr_vs_{p}_utility"] = cistats.paired_diff_ci(
                [r["on_time_utility"] for r in t], [r["on_time_utility"] for r in o])
            res[f"thr_vs_{p}_accuracy"] = cistats.paired_diff_ci(
                [r["mean_accuracy"] for r in t], [r["mean_accuracy"] for r in o])
    return res


class Track:
    """Caches per-seed substrates for one track."""
    def __init__(self, track, seeds):
        import simulate as sim
        import thermal as th
        self.cfg, self.lat = _load_track(track)
        self.model = th.ThermalModel.from_cfg(self.cfg)
        self.subs = {s: sim.build_substrate(self.cfg, regime="uncoupled", seed=s) for s in seeds}
        self.kappa = sim.calibration_kappa(self.cfg, "coupled", [s + 100 for s in seeds])
        self.base_duty = th.config_duty(self.subs[seeds[0]].fm, self.lat)
        self.seeds = seeds

    def run(self, policy, t_amb, seed, *, sub=None, model=None, exo=None, **kw):
        import thermal as th
        rng = np.random.default_rng(50_000 + 97 * seed + int(t_amb))   # identical draws across policies
        m = model or self.model
        r = th.run_closed_loop(policy, sub or self.subs[seed], self.lat, m, float(t_amb), rng, N_FRAMES,
                               kappa=self.kappa, exo_contention=exo, duty=self.base_duty, **kw)
        return _row(r, m)


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


def run_track(track, seeds, ambients):
    import policies as pol
    import regimes as reg
    import ci as cistats
    T = Track(track, seeds)
    out = {}

    # E2: the C1 caps tuned "offline" at 45 / 55 degC are THR's mean C1 share there
    # (tuned on the same seeds, which favours the caps).
    caps = {}
    for a in (45, 55):
        caps[a] = float(np.mean([T.run("thermal", a, s)["frac_C1"] for s in seeds]))
    e2 = {}
    for a in ambients:
        vals = {p: [] for p in THERMAL_POLICIES}
        for s in seeds:
            for p in THERMAL_POLICIES:
                if p == "duty_matched":
                    r = T.run("duty_matched", a, s, c1_budget=vals["thermal"][-1]["frac_C1"])
                elif p in ("cap45", "cap55"):
                    r = T.run("fixed_cap", a, s, c1_budget=caps[int(p[3:])])
                elif p == "thermostat":
                    r = T.run("thermostat", a, s, t_set=THERMOSTAT_C)
                else:
                    r = T.run(p, a, s)
                vals[p].append(r)
        e2[str(a)] = _agg(vals)
    out["E2_thermal"] = {"kappa": T.kappa, "model": T.model.__dict__, "caps": caps, "by_ambient": e2}

    # E3: compound
    e3 = {}
    for a in ambients:
        vals = {p: [] for p in COMPOUND_POLICIES}
        for s in seeds:
            sub = T.subs[s]
            fault = sub.fault_active[np.arange(N_FRAMES) % sub.T]
            exo = reg.coupled_schedule(N_FRAMES, fault, T.cfg, seed=s)
            for p in COMPOUND_POLICIES:
                vals[p].append(T.run(p, a, s, exo=exo))
        e3[str(a)] = _agg(vals)
    out["E3_compound"] = {"kappa": T.kappa, "by_ambient": e3}

    # E4: THR holds a wrong model
    m = T.model
    variants = {"true": m, "R_x0.7": replace(m, r_th=m.r_th * 0.7), "R_x1.3": replace(m, r_th=m.r_th * 1.3),
                "tau_x0.5": replace(m, tau_s=m.tau_s * 0.5), "tau_x2": replace(m, tau_s=m.tau_s * 2.0)}
    out["E4_misspec"] = {}
    for name, mm in variants.items():
        vals = {"decoupled": [T.run("decoupled", PHOENIX_C, s) for s in seeds],
                "thermal": [T.run("thermal", PHOENIX_C, s, policy_model=mm) for s in seeds]}
        out["E4_misspec"][name] = _agg(vals)

    # E5: deadline sweep (power model unchanged; only D moves)
    out["E5_deadline"] = {}
    for mult in DEADLINE_MULTS:
        subs = {}
        for s in seeds:
            sub = T.subs[s]
            dl = sub.deadline_s * mult
            fm = pol.build_frontier_model(dl, {c: T.lat[c] for c in pol.CONFIG_KEYS},
                                          sub.fm.acc_nominal, sub.fm.acc_faulted)
            subs[s] = replace(sub, deadline_s=dl, fm=fm)
        res = {}
        for a in (35, 45, 55):
            vals = {p: [] for p in SWEEP_POLICIES}
            for s in seeds:
                for p in SWEEP_POLICIES:
                    kw = {"c1_budget": vals["thermal"][-1]["frac_C1"]} if p == "duty_matched" else {}
                    vals[p].append(T.run(p, a, s, sub=subs[s], **kw))
            res[str(a)] = _agg(vals)
        out["E5_deadline"][str(mult)] = res

    # E6: the true plant differs; THR uses the matching (identified) model
    plants = {"R_x0.8": replace(m, r_th=m.r_th * 0.8), "R_x1.2": replace(m, r_th=m.r_th * 1.2),
              "tau_x0.5": replace(m, tau_s=m.tau_s * 0.5), "tau_x2": replace(m, tau_s=m.tau_s * 2.0),
              "throttle_x0.8": replace(m, throttle_power_scale=0.8)}
    out["E6_plant"] = {}
    for name, pm in plants.items():
        res = {}
        for a in (45, 55):
            vals = {p: [] for p in SWEEP_POLICIES}
            for s in seeds:
                for p in SWEEP_POLICIES:
                    kw = {"c1_budget": vals["thermal"][-1]["frac_C1"]} if p == "duty_matched" else {}
                    vals[p].append(T.run(p, a, s, model=pm, **kw))
            res[str(a)] = _agg(vals)
        out["E6_plant"][name] = res

    # E7: THR horizon / margin sensitivity
    out["E7_thr_settings"] = {}
    for a in (45, 55):
        base = [T.run("decoupled", a, s) for s in seeds]
        res = {}
        for h, mg in HM_GRID:
            vals = {"decoupled": base,
                    "thermal": [T.run("thermal", a, s, thr_params={"horizon_frames": h, "margin_c": mg}) for s in seeds]}
            res[f"H{h}_m{mg:g}"] = _agg(vals)
        out["E7_thr_settings"][str(a)] = res

    # E8: thermostat threshold sweep
    out["E8_thermostat"] = {}
    for a in (45, 55):
        thr = [T.run("thermal", a, s) for s in seeds]
        res = {}
        for ts in THERMOSTAT_GRID:
            vals = {"thermal": thr, "thermostat": [T.run("thermostat", a, s, t_set=ts) for s in seeds]}
            res[f"{ts:g}"] = _agg(vals)
        out["E8_thermostat"][str(a)] = res

    # E9: Pareto of fixed C1 caps
    out["E9_caps"] = {}
    for a in (45, 55):
        res = {}
        for cap in CAP_GRID:
            vals = {"fixed_cap": [T.run("fixed_cap", a, s, c1_budget=cap) for s in seeds]}
            res[f"{cap:g}"] = _agg(vals)
        out["E9_caps"][str(a)] = res
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = [0, 1] if args.quick else list(range(20))
    ambients = [25, 45, 55] if args.quick else AMBIENTS
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "outputs" / "extreme_env"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validation-locked protocol (after Zhou & Roy, arXiv:2609.23924).
    results = {"seeds": seeds, "n_frames": N_FRAMES, "tracks": {},
               "protocol": {"eval_labels_used_for_selection": False,
                            "tunables_fixed_a_priori": ["thermal.model.*", "horizon_frames=50",
                                                        "margin_c=2.0", "dwell=10", "slow_ratio=1.5",
                                                        "thermostat_c=88.0"],
                            "sensitivity_only": ["E7 horizon/margin grid", "E8 thermostat grid"]}}
    t0 = time.time()
    for track, label in TRACKS:
        print(f"== {track}", flush=True)
        r = {"label": label, "E1_bursts": e1_bursts(track, seeds), **run_track(track, seeds, ambients)}
        results["tracks"][track] = r
        e2 = r["E2_thermal"]["by_ambient"]["45"]
        print(f"   45C miss: belief {e2['decoupled']['deadline_miss_rate']['mean']:.3f} "
              f"thermostat {e2['thermostat']['deadline_miss_rate']['mean']:.3f} "
              f"THR {e2['thermal']['deadline_miss_rate']['mean']:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    name = "extreme_env_quick.json" if args.quick else "extreme_env.json"
    with open(out_dir / name, "w") as f:
        json.dump(results, f, indent=1, default=float)
    print("wrote", out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
