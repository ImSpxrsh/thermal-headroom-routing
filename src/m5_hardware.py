"""Real-hardware check on an Apple M5 (MacBook Pro): measure how the SoC heats under
sustained YOLO11 inference, fit the RC thermal model of src/thermal.py to it, and run
THR in the loop against real inference and real temperature readings.

Temperature and power come from `macmon` (sudoless IOReport reader for Apple
silicon; `brew install macmon`). Inference runs on the M5 GPU through MPS.

  part 1  heating curves: for each config, idle -> run continuously -> cool down,
          logging per-frame latency, GPU/CPU temperature, power and fan speed.
  part 2  RC fit: T(t) = T_ss + (T_0 - T_ss) exp(-t / tau) on the loaded segment;
          R_th = (T_ss - T_idle) / (P_load - P_idle). Gives measured tau, R_th and
          per-config power for this chip.
  part 3  hardware in the loop: the MacBook's fans keep the M5 well below its real
          throttle point, so we impose a temperature CAP (fixed from part 1 before any
          policy run) and ask whether each policy keeps the real chip under it while
          serving as much C1 as it can. Policies: static C1, thermostat (bar C1 above
          cap - margin), THR (forecast with the fitted RC model).

    python src/m5_hardware.py                 # full protocol (~1 hour)
    python src/m5_hardware.py --smoke         # 20 s per phase, checks the plumbing
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "m5_hardware"
CONFIGS = {"C1": ("yolo11x.pt", 1280), "C2": ("yolo11x.pt", 640),
           "C3": ("yolo11n.pt", 1280), "C4": ("yolo11n.pt", 640)}


class Sensor:
    """Background reader for `macmon pipe`; rows are (t, gpu_temp, cpu_temp, gpu_W, cpu_W, sys_W, fan_rpm)."""
    def __init__(self, interval_ms=250):
        self.rows = []
        self._p = subprocess.Popen(["macmon", "pipe", "-i", str(interval_ms)],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self._p.stdout:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            fan = d.get("fans") or [{}]
            self.rows.append((time.time(), d["temp"]["gpu_temp_avg"], d["temp"]["cpu_temp_avg"],
                              d.get("gpu_power", 0.0), d.get("cpu_power", 0.0), d.get("sys_power", 0.0),
                              fan[0].get("rpm", 0)))

    def latest(self):
        return self.rows[-1] if self.rows else None

    def since(self, t0):
        return [r for r in self.rows if r[0] >= t0]

    def stop(self):
        self._p.terminate()


def load_models(device):
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS
    import cv2
    models = {w: YOLO(w) for w in {"yolo11x.pt", "yolo11n.pt"}}
    imgs = [cv2.resize(cv2.imread(str(p)), (1024, 544)) for p in sorted(ASSETS.glob("*.jpg"))]
    for c, (w, sz) in CONFIGS.items():          # warm up every config on the GPU
        for _ in range(3):
            models[w].predict(imgs[0], imgsz=sz, device=device, verbose=False)
    return models, imgs


def infer(models, imgs, cfg, i, device):
    w, sz = CONFIGS[cfg]
    t = time.perf_counter()
    models[w].predict(imgs[i % len(imgs)], imgsz=sz, device=device, verbose=False)
    return time.perf_counter() - t


def wait_cool(sensor, idle_s, log):
    time.sleep(idle_s)
    r = sensor.latest()
    log(f"   idle {idle_s:.0f}s -> gpu {r[1]:.1f} C cpu {r[2]:.1f} C fan {r[6]:.0f}")


def heating_curve(models, imgs, sensor, cfg, load_s, device, period_s=0.0):
    """period_s > 0 paces frames at a fixed rate (one frame per period, idle for the
    rest), which is how the simulation charges power: duty = latency / period."""
    t0 = time.time(); lat = []; i = 0
    while time.time() - t0 < load_s:
        ts = time.perf_counter()
        lat.append((time.time() - t0, infer(models, imgs, cfg, i, device))); i += 1
        if period_s:
            time.sleep(max(0.0, period_s - (time.perf_counter() - ts)))
    return {"t0": t0, "lat": lat, "sensor": sensor.since(t0), "period_s": period_s}


def fit_rc(seg, t0, idx=1):
    """Fit T(t) = T_ss + (T0 - T_ss) exp(-t/tau) to the loaded segment."""
    from scipy.optimize import curve_fit
    t = np.array([r[0] - t0 for r in seg]); T = np.array([r[idx] for r in seg])
    f = lambda t, Tss, T0, tau: Tss + (T0 - Tss) * np.exp(-t / tau)
    p, _ = curve_fit(f, t, T, p0=[T[-1], T[0], 30.0], bounds=([0, 0, 1], [150, 150, 2000]), maxfev=20000)
    resid = T - f(t, *p)
    return {"T_ss": float(p[0]), "T_0": float(p[1]), "tau_s": float(p[2]),
            "rmse_c": float(np.sqrt(np.mean(resid ** 2)))}


def run_policy(models, imgs, sensor, policy, dur_s, device, cap, margin, rc, idle_T, power, period_s=0.0):
    """policy in {static_C1, thermostat, thr}. Serves C1 when allowed, else C2. With
    period_s > 0 frames arrive at a fixed rate and the GPU idles after each one."""
    t0 = time.time(); log = []; i = 0
    held, cur = 0, "C1"
    while time.time() - t0 < dur_s:
        T = sensor.latest()[1]
        if policy == "static_C1":
            want = "C1"
        elif policy == "thermostat":
            want = "C2" if T >= cap - margin else "C1"
        else:
            # THR: forecast the GPU temperature after holding C1 for H seconds (fitted RC model)
            T_inf = idle_T + rc["R_th"] * (power["C1"] - power["idle"])
            T_h = T_inf + (T - T_inf) * np.exp(-rc["H_s"] / rc["tau_s"])
            want = "C1" if T_h < cap - margin else "C2"
        if want != cur and held >= 10:          # 10-frame dwell, as in the simulation
            cur, held = want, 0
        else:
            held += 1
        ts = time.perf_counter()
        lat = infer(models, imgs, cur, i, device); i += 1
        log.append((time.time() - t0, cur, lat))
        if period_s:
            time.sleep(max(0.0, period_s - (time.perf_counter() - ts)))
    s = sensor.since(t0)
    temps = np.array([x[1] for x in s]); ts = np.array([x[0] - t0 for x in s])
    return {"policy": policy, "period_s": period_s, "miss_rate": float(np.mean([l > period_s for _, _, l in log])) if period_s else None,
            "frames": len(log), "frac_C1": float(np.mean([c == "C1" for _, c, _ in log])),
            "time_above_cap": float(np.mean(temps > cap)), "peak_gpu_temp": float(temps.max()),
            "mean_lat_ms": float(np.mean([l for _, _, l in log]) * 1e3), "fps": len(log) / dur_s,
            "trace": {"t": ts.tolist(), "gpu_temp": temps.tolist(),
                      "choice": [(round(t, 2), c) for t, c, _ in log]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--paced-rerun", action="store_true",
                    help="reuse the unpaced heating curves and cap from m5_unpaced.json; measure a paced "
                         "C2 curve and rerun the three routers with frames paced at C1's median latency")
    a = ap.parse_args()
    if a.paced_rerun:
        return paced_rerun()
    idle_s, load_s, cool_s, pol_s = (15, 20, 15, 20) if a.smoke else (180, 420, 300, 300)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps"
    log = lambda m: print(m, flush=True)
    sensor = Sensor(250)
    time.sleep(2)
    models, imgs = load_models(device)
    res = {"device": "Apple M5 (MacBook Pro, Mac17,2), MPS",
           "protocol": {"idle_s": idle_s, "load_s": load_s, "cool_s": cool_s, "policy_s": pol_s},
           "curves": {}}

    for cfg in ("C1", "C2", "C4"):
        log(f"== heating curve {cfg}")
        wait_cool(sensor, idle_s if cfg == "C1" else cool_s, log)
        idle = sensor.since(time.time() - 30)
        cur = heating_curve(models, imgs, sensor, cfg, load_s, device)
        seg = cur["sensor"]
        cur["idle_gpu_temp"] = float(np.mean([r[1] for r in idle]))
        cur["idle_power_w"] = float(np.mean([r[3] + r[4] for r in idle]))
        tail = [r for r in seg if r[0] - cur["t0"] > 0.6 * load_s]
        cur["load_power_w"] = float(np.mean([r[3] + r[4] for r in tail]))
        cur["lat_median_ms"] = float(np.median([l for _, l in cur["lat"]]) * 1e3)
        first = [l for t, l in cur["lat"] if t < min(30, load_s / 3)]
        last = [l for t, l in cur["lat"] if t > load_s - min(30, load_s / 3)]
        cur["lat_first_ms"], cur["lat_last_ms"] = float(np.median(first) * 1e3), float(np.median(last) * 1e3)
        cur["fit_gpu"] = fit_rc(seg, cur["t0"], 1)
        cur["fit_cpu"] = fit_rc(seg, cur["t0"], 2)
        dP = cur["load_power_w"] - cur["idle_power_w"]
        cur["R_th_gpu"] = (cur["fit_gpu"]["T_ss"] - cur["idle_gpu_temp"]) / dP if dP > 0.5 else float("nan")
        cur["fan_rpm_end"] = float(seg[-1][6])
        log(f"   {cfg}: lat {cur['lat_median_ms']:.0f} ms (start {cur['lat_first_ms']:.0f} / end "
            f"{cur['lat_last_ms']:.0f}), P {cur['idle_power_w']:.1f}->{cur['load_power_w']:.1f} W, "
            f"T_ss {cur['fit_gpu']['T_ss']:.1f} C tau {cur['fit_gpu']['tau_s']:.0f} s "
            f"R {cur['R_th_gpu']:.2f} C/W rmse {cur['fit_gpu']['rmse_c']:.2f} fan {cur['fan_rpm_end']:.0f}")
        cur["sensor"] = [list(r) for r in seg]
        res["curves"][cfg] = cur

    # Hardware in the loop. The cap is fixed here from part 1, before any policy run:
    # 60% of the way from idle to C1's fitted steady state.
    c1, c2 = res["curves"]["C1"], res["curves"]["C2"]
    idle_T = c1["idle_gpu_temp"]
    cap = idle_T + 0.6 * (c1["fit_gpu"]["T_ss"] - idle_T)
    rc = {"tau_s": c1["fit_gpu"]["tau_s"], "R_th": c1["R_th_gpu"], "H_s": 5.0}
    power = {"idle": c1["idle_power_w"], "C1": c1["load_power_w"], "C2": c2["load_power_w"]}
    res["hil"] = {"cap_c": cap, "margin_c": 1.0, "rc": rc, "power": power, "runs": {}}
    log(f"== hardware in the loop, cap {cap:.1f} C")
    for pol in ("static_C1", "thermostat", "thr"):
        wait_cool(sensor, cool_s, log)
        r = run_policy(models, imgs, sensor, pol, pol_s, device, cap, 1.0, rc, idle_T, power)
        log(f"   {pol}: C1 share {r['frac_C1']:.2f}, time above cap {r['time_above_cap']*100:.1f}%, "
            f"peak {r['peak_gpu_temp']:.1f} C, {r['fps']:.1f} fps")
        res["hil"]["runs"][pol] = r
    sensor.stop()
    name = "m5_smoke.json" if a.smoke else "m5_hardware.json"
    json.dump(res, open(OUT / name, "w"), indent=1)
    log(f"wrote {OUT / name}")


def paced_rerun():
    """Second pass. The first pass fed frames back to back, so every config kept the GPU
    saturated and falling back to C2 could not cool the chip. Here frames arrive once per
    C1 median latency (the simulation's deadline rule) and the GPU idles after a cheap
    frame. The cap is the one fixed in the first pass, before any router ran."""
    prev = json.load(open(OUT / "m5_unpaced.json"))
    device = "mps"
    log = lambda m: print(m, flush=True)
    sensor = Sensor(250)
    time.sleep(2)
    models, imgs = load_models(device)
    c1 = prev["curves"]["C1"]
    period = c1["lat_median_ms"] / 1e3
    cap, idle_T = prev["hil"]["cap_c"], c1["idle_gpu_temp"]
    res = {"device": prev["device"], "period_s": period, "cap_c": cap, "curves": {}, "runs": {}}
    log(f"== paced C2 heating curve, period {period*1e3:.0f} ms")
    wait_cool(sensor, 300, log)
    idle = sensor.since(time.time() - 30)
    cur = heating_curve(models, imgs, sensor, "C2", 420, device, period_s=period)
    seg = cur["sensor"]
    cur["idle_gpu_temp"] = float(np.mean([r[1] for r in idle]))
    cur["idle_power_w"] = float(np.mean([r[3] + r[4] for r in idle]))
    tail = [r for r in seg if r[0] - cur["t0"] > 0.6 * 420]
    cur["load_power_w"] = float(np.mean([r[3] + r[4] for r in tail]))
    cur["fit_gpu"] = fit_rc(seg, cur["t0"], 1)
    cur["lat_median_ms"] = float(np.median([l for _, l in cur["lat"]]) * 1e3)
    log(f"   paced C2: P {cur['idle_power_w']:.1f}->{cur['load_power_w']:.1f} W, T_ss {cur['fit_gpu']['T_ss']:.1f} C "
        f"tau {cur['fit_gpu']['tau_s']:.0f} s")
    cur["sensor"] = [list(r) for r in seg]
    res["curves"]["C2_paced"] = cur
    rc = prev["hil"]["rc"]
    power = {"idle": c1["idle_power_w"], "C1": c1["load_power_w"], "C2": cur["load_power_w"]}
    res["rc"], res["power"] = rc, power
    log(f"== hardware in the loop (paced), cap {cap:.1f} C")
    for pol in ("static_C1", "thermostat", "thr"):
        wait_cool(sensor, 300, log)
        r = run_policy(models, imgs, sensor, pol, 300, device, cap, 1.0, rc, idle_T, power, period_s=period)
        log(f"   {pol}: C1 share {r['frac_C1']:.2f}, time above cap {r['time_above_cap']*100:.1f}%, "
            f"peak {r['peak_gpu_temp']:.1f} C, miss {r['miss_rate']*100:.1f}%")
        res["runs"][pol] = r
    sensor.stop()
    json.dump(res, open(OUT / "m5_paced.json", "w"), indent=1)
    log(f"wrote {OUT / 'm5_paced.json'}")


if __name__ == "__main__":
    main()
