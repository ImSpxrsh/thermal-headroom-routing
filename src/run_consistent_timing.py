"""E11: the heat experiment with physically consistent frame timing.

In E2 each recorded 10 Hz frame is one decision and the thermal model advances 0.1 s
per decision, although C1 takes about one deadline D (211-341 ms) to run. Here each
decision lasts exactly D of wall-clock time:
  - the thermal plant integrates over dt = D per decision;
  - the router takes the newest recorded frame at each decision (latest-frame
    scheduling), so decision k uses the trace frame at time k*D;
  - inference still running at the next boundary is abandoned (no queue, no overlap),
    and power is charged by duty = median latency / D, capped at 1;
  - THR's forecast horizon stays 5 s of wall-clock time (H = round(5 / D) decisions).
Everything else (policies, seeds, thermostat at 88 degC, caps) is as in E2. Missions
are 1200 decisions, i.e. 4.2 to 6.8 minutes of wall-clock time.

    python src/run_consistent_timing.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_extreme_env as rx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AMBIENTS = [35, 40, 45, 50, 55]
POLICIES = ["static_C1", "decoupled", "thermostat", "thermal", "duty_matched", "cap45"]
PLANTS = {"R_x1.2": {"r_th": 1.2}, "tau_x0.5": {"tau_s": 0.5}}


def main():
    seeds = list(range(20))
    out = {"seeds": seeds, "ambients": AMBIENTS, "tracks": {}}
    for track, label in rx.TRACKS:
        T = rx.Track(track, seeds)
        D = T.subs[seeds[0]].deadline_s
        stride = D / 0.1
        base = replace(T.model, dt_s=D)
        thr = {"horizon_frames": max(1, int(round(5.0 / D))), "margin_c": 2.0}

        def run(p, a, s, model, **kw):
            if p == "thermal":
                kw["thr_params"] = thr
            if p == "thermostat":
                kw["t_set"] = rx.THERMOSTAT_C
            return T.run(p, a, s, model=model, frame_stride=stride, **kw)

        # cap tuned at 45 degC under the same timing: THR's mean C1 share there
        cap45 = float(np.mean([run("thermal", 45, s, base)["frac_C1"] for s in seeds]))
        res = {"D_s": D, "H": thr["horizon_frames"], "cap45": cap45, "base": {}, "plants": {}}
        for a in AMBIENTS:
            vals = {p: [] for p in POLICIES}
            for s in seeds:
                for p in POLICIES:
                    kw = {}
                    if p == "duty_matched":
                        kw["c1_budget"] = vals["thermal"][-1]["frac_C1"]
                    if p == "cap45":
                        p_run, kw["c1_budget"] = "fixed_cap", cap45
                    else:
                        p_run = p
                    vals[p].append(run(p_run, a, s, base, **kw))
            res["base"][str(a)] = rx._agg(vals)
        for name, mult in PLANTS.items():
            pm = replace(base, **{k: getattr(base, k) * v for k, v in mult.items()})
            vals = {p: [] for p in ("decoupled", "thermostat", "thermal")}
            for s in seeds:
                for p in vals:
                    vals[p].append(run(p, 55, s, pm))
            res["plants"][name] = rx._agg(vals)
        out["tracks"][track] = res
        e = res["base"]["45"]
        print(f"{track}: D {D*1e3:.0f} ms | 45C thr% belief {e['decoupled']['throttled_frac']['mean']*100:.0f} "
              f"tst {e['thermostat']['throttled_frac']['mean']*100:.0f} THR {e['thermal']['throttled_frac']['mean']*100:.0f} "
              f"| miss red {e['thermal']['miss_reduction_vs_decoupled']['mean']*100:+.1f}", flush=True)
    path = ROOT / "outputs" / "extreme_env" / "consistent_timing.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
