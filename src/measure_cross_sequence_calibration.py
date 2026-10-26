"""RQ-Coupling (c): cross-SEQUENCE calibration transfer -- FULL model, not just kappa.

Phase 5's kappa is fit on calibration draws of the SAME trace being tested (different
contention-schedule seeds, same real fault timeline): a temporal/seed separation, not a
sequence separation. This script fits the fault-belief model on a DIFFERENT sequence and
tests it on the held-out sequence. Two things transfer, addressing the reviewer's request
to "calibrate the HMM and kappa on one sequence and test on another":

  1. kappa  -- the scalar coupling coefficient (fit on the calib sequence).
  2. the sensor-fault switching HMMs -- the entire belief estimator (transition matrix +
     per-state Gaussian emissions per channel), fit on the calib sequence and used to
     filter the TEST sequence's observations.

It reports, on the test sequence: (a) fault-detection balanced accuracy of the transferred
HMM vs the test sequence's own natively-fit HMM (does the belief model generalize?), and
(b) RQ-H deadline-miss reduction with the fully-transferred (HMM + kappa) estimator vs the
native one (does the routing benefit survive a model trained on another trajectory?).

    python src/measure_cross_sequence_calibration.py \
        --calib-track outputs/trackD_rain_2_0 --test-track outputs/trackD_rain_4_0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def balanced_accuracy(pred_bool: np.ndarray, true_bool: np.ndarray) -> float:
    p = np.asarray(pred_bool, bool); t = np.asarray(true_bool, bool)
    tp = int((p & t).sum()); tn = int((~p & ~t).sum())
    fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    vals = [v for v in (tpr, tnr) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-track", required=True, help="sequence to FIT the HMM + kappa on")
    ap.add_argument("--test-track", required=True, help="sequence to EVALUATE on")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config_util import load_config, abspath
    import simulate as sim
    import policies as pol
    import sensor_belief as sb
    import ci as cistats

    for t in (args.calib_track, args.test_track):
        if not (abspath(t) / "phase3" / "phase3_summary.json").exists():
            print(f"ERROR: {t} has no profiled frontier."); return 2

    # --- fit the transferable model on the CALIB sequence ---
    os.environ["OUTPUTS_DIR"] = args.calib_track
    cfg_c = load_config()
    calib_lab = pd.read_csv(abspath(args.calib_track) / "phase1" / "trackA_observations.csv")
    calib_hmms = sb.fit_all_channels(calib_lab, cfg_c)                      # the belief estimator
    kappa_transfer = sim.calibration_kappa(cfg_c, "coupled", [s + 100 for s in [1, 2, 3, 4, 5]])

    # --- evaluate on the TEST sequence ---
    os.environ["OUTPUTS_DIR"] = args.test_track
    cfg_t = load_config()
    seeds = cfg_t["seeds"]
    rt = cfg_t["routing"]["reliability_target"]
    dwell = cfg_t["routing"]["hysteresis"]["fixed"]["dwell_frames"]
    kappa_native = sim.calibration_kappa(cfg_t, "coupled", [s + 100 for s in seeds])

    subs_native = [sim.build_substrate(cfg_t, "coupled", s) for s in seeds]
    subs_xfer = [sim.build_substrate(cfg_t, "coupled", s, sensor_hmms=calib_hmms) for s in seeds]

    # (a) fault-detection balanced accuracy of native vs transferred HMM on the TEST track
    def mean_bacc(subs):
        return float(np.mean([balanced_accuracy(su.s_belief > 0.5, su.fault_active) for su in subs]))
    bacc_native = mean_bacc(subs_native)
    bacc_xfer = mean_bacc(subs_xfer)

    # (b) RQ-H reduction: native (native HMM + native kappa) vs full transfer (xfer HMM + xfer kappa)
    def rqh(subs, kappa):
        jl, dl = [], []
        for su in subs:
            jl.append(sim.run_policy(pol.JointPolicy(su.fm, kappa, rt, pol.Hysteresis(dwell)), su)["deadline_miss_rate"])
            dl.append(sim.run_policy(pol.DecoupledPolicy(su.fm, rt, pol.Hysteresis(dwell)), su)["deadline_miss_rate"])
        return cistats.paired_diff_ci(dl, jl)

    native = rqh(subs_native, kappa_native)
    transferred = rqh(subs_xfer, kappa_transfer)          # FULL transfer: HMM + kappa

    def pack(d):
        return {"mean": d["mean"] * 100, "lo": d["lo"] * 100, "hi": d["hi"] * 100,
                "significant": d["significant"]}

    summary = {
        "calib_track": args.calib_track, "test_track": args.test_track,
        "transfers": "sensor_fault_HMM + kappa (full belief model, not just the scalar)",
        "kappa_native_on_test_track": kappa_native,
        "kappa_transferred_from_calib_track": kappa_transfer,
        "fault_bacc_native_hmm_on_test": bacc_native,
        "fault_bacc_transferred_hmm_on_test": bacc_xfer,
        "fault_bacc_drop_from_transfer": bacc_native - bacc_xfer,
        "rqh_reduction_native_pp": pack(native),
        "rqh_reduction_full_transfer_pp": pack(transferred),
    }
    out_dir = abspath(args.test_track) / "extras"; out_dir.mkdir(parents=True, exist_ok=True)
    calib_name = Path(args.calib_track).name
    json.dump(summary, open(out_dir / f"cross_seq_calib_from_{calib_name}.json", "w"), indent=2, default=str)

    print(f"\n{'='*72}\n[cross-sequence transfer] calib={args.calib_track} -> test={args.test_track}\n{'='*72}")
    print(f"transfers: sensor-fault HMM + kappa (whole belief model)")
    print(f"kappa native {kappa_native:.3f} | transferred {kappa_transfer:.3f}")
    print(f"fault-detection balanced accuracy on test: native HMM {bacc_native:.3f} | "
          f"transferred HMM {bacc_xfer:.3f}  (drop {bacc_native - bacc_xfer:+.3f})")
    print(f"RQ-H reduction native (native HMM+kappa):        {native['mean']*100:+.2f}pp "
          f"[{native['lo']*100:.2f},{native['hi']*100:.2f}] sig={native['significant']}")
    print(f"RQ-H reduction FULL transfer (xfer HMM+kappa):   {transferred['mean']*100:+.2f}pp "
          f"[{transferred['lo']*100:.2f},{transferred['hi']*100:.2f}] sig={transferred['significant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
