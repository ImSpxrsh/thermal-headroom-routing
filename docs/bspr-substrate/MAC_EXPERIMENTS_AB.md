# Experiments A & B on the Mac (test against ANY data)

Two experiments that add REAL evidence and move the paper from ~3/5 toward 4/5 (see
`REVIEWER_FIXES.md` Tier 2 for why). Both run on Apple Silicon (MPS); neither needs a
rewrite of the pipeline. They work on RADIATE **or any dataset you can turn into frames**.

- **Experiment A** measures whether sensor faults and compute load actually co-occur in the
  data (the coupling the paper currently *imposes*), and re-runs the routing with contention
  driven by that real load. This is the ceiling-breaker.
- **Experiment B** runs several sequences per condition and reports a confidence interval
  **over trajectories**, not just seeds. This is the safe generalization win.

Everything here uses the venv + torch(MPS) setup from `MAC_RUN_GUIDE.md` section 1. Run all
commands from the repo root with the venv active.

---

## Prereqs (once)
```bash
source .venv/bin/activate                      # from MAC_RUN_GUIDE.md section 1
python -c "import torch; print('mps:', torch.backends.mps.is_available())"   # expect True
```
Both experiments assume a track has already been through the pipeline (Phase 1 labels +
Phase 3 profiling), because they reuse `outputs/<track>/phase1` and `phase3`. If a sequence
is new, run the pipeline on it first (see "Test against ANY data" below).

Tip: also do the item-0 re-profile first (`profiling.n_frames: 1500` in `config.yaml`) so
these run on representative windows, not the first 300 frames.

---

## Experiment A -- measure the real coupling + de-circularize

`src/measure_real_coupling.py` runs the reference config (C1) over the real frames, records
per-frame **real latency** (incl. NMS/postproc) and **detection count** (a compute-load
proxy from YOLO output, independent of the blur/illumination/occlusion fault signals), then:
1. measures `Pc(fault)=P(high-load | fault)` and `Pc(nominal)=P(high-load | nominal)` -- the
   real analogue of the paper's hard-coded `0.85 / 0.05` -- with a bootstrap correlation CI;
2. re-runs RQ-H with the contention schedule driven by the **real load proxy** (via the new
   `state_override` in `simulate.build_substrate`), so kappa is EMPIRICAL, not imposed.

Run it per sequence, both proxies:
```bash
for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/measure_real_coupling.py --track outputs/trackD_$SEQ \
     --frames data/frames/radiate_$SEQ --load-proxy detections --max-frames 1500
  python src/measure_real_coupling.py --track outputs/trackD_$SEQ \
     --frames data/frames/radiate_$SEQ --load-proxy latency --max-frames 1500
done
```
Writes `outputs/<track>/extras/real_coupling.json` and prints a verdict.

**How to read it (this decides the paper's ceiling):**
- `Pc(fault) > Pc(nominal)` and `corr(fault,load)` CI excludes 0 -> the coupling is REAL in
  this data. Report the measured `Pc(F)/Pc(N)` in place of (or beside) the hard-coded values.
- de-circularized `reduction_pp` still positive/significant -> the router helps even when
  contention is driven by real load, not by the fault labels. **This is the 4/5 result.**
- `Pc(fault) ~ Pc(nominal)` (or negative corr, e.g. foggy frames have *fewer* detections)
  -> honest null: in this data load does not track faults. Report it plainly and reframe the
  claim as "given a coupled regime, the estimator exploits it." Better you find this than a
  reviewer. Try the other `--load-proxy`, and re-check after the item-0 re-profile.

Flags: `--load-proxy {detections,latency}`, `--high-quantile 0.75` (what counts as
"high load"), `--max-frames N`.

---

## Experiment B -- more than one trace per condition

Today the CIs are 5 seeds over ONE trace per condition (sensitivity to injected noise, not
generalization). RADIATE has many sequences per weather; you have Dropbox access.

### B1. Get + run 3-5 sequences per condition
```bash
# example: rain. Download rain_1_0, rain_2_0, rain_3_0 (+ your existing rain_4_0) and:
for SEQ in rain_1_0 rain_2_0 rain_3_0 rain_4_0; do
  python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
  FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ \
    python src/run_pipeline.py --skip-extract
  # optional but recommended: Experiment A on each sequence too
  python src/measure_real_coupling.py --track outputs/trackD_$SEQ \
     --frames data/frames/radiate_$SEQ --load-proxy detections --max-frames 1500
done
```

### B2. Aggregate across sequences (CI over trajectories)
```bash
python src/aggregate_multitrace.py --condition rain \
  --tracks outputs/trackD_rain_1_0 outputs/trackD_rain_2_0 \
           outputs/trackD_rain_3_0 outputs/trackD_rain_4_0
# repeat --condition snow / fog / night with their sequence dirs
```
Writes `outputs/multitrace/<condition>.json` and prints:
- per-sequence RQ-H reduction,
- **cross-sequence mean + t-interval + cluster (resample-sequences) bootstrap CI**,
- `generalizes: true/false` (does the trajectory-level CI exclude zero),
- if Experiment A was run per sequence, the pooled measured `Pc(fault)/Pc(nominal)` and
  `corr(fault,load)`.

**How to read it:** if the cross-sequence CI excludes zero, the effect **generalizes across
independent traces** -- that is the honest statistic to put in the abstract, replacing the
seed CI. If it does not, say so; that is the real state of the evidence.

---

## Test against ANY data (not just RADIATE)

The pipeline only needs a directory of image frames. For any dataset (a new robot log, a
different weather set, your own captures):
```bash
# 1. get frames into a dir, one image per frame, in temporal order:
#      data/frames/<name>/000001.png, 000002.png, ...
#    (RADIATE: src/extract_radiate.py ; TartanDrive bag: src/extract_frames.py ;
#     WoodScape fisheye: src/extract_woodscape.py ; anything else: just drop PNGs in the dir)

# 2. run the pipeline (Phase 1 labels + Phase 3 profiling + routing) on it:
FRAMES_DIR=data/frames/<name> OUTPUTS_DIR=outputs/<name> python src/run_pipeline.py --skip-extract

# 3. Experiment A on it:
python src/measure_real_coupling.py --track outputs/<name> --frames data/frames/<name> \
   --load-proxy detections --max-frames 1500

# 4. Experiment B: repeat 1-3 for several sequences of the same condition, then:
python src/aggregate_multitrace.py --condition <name> --tracks outputs/<seq1> outputs/<seq2> ...
```
Notes:
- The fault labels come from the image health signals (blur / illumination / optical-flow
  occlusion), so any RGB frame sequence works. Fisheye needs the WoodScape path
  (undistort + soiling-mask occlusion) -- see `MAC_RUN_GUIDE.md` section 6.
- `measure_real_coupling.py` reads `outputs/<name>/phase1/trackA_observations.csv` for the
  fault labels and `phase3` for the routing frontier, so step 2 must finish first.
- Latencies are your Mac's (dev-environment); the co-occurrence (`Pc(F)/Pc(N)`, corr) and the
  deadline-miss comparisons are the reportable results, not absolute ms.

---

## What to put in the paper from these

| outcome | what it earns |
|---|---|
| Exp A: `Pc(F) > Pc(N)`, de-circularized reduction significant | real empirical coupling -> **4/5**; report measured Pc(F)/Pc(N), drop "imposed only" caveat |
| Exp A: `Pc(F) ~ Pc(N)` | honest null -> stay ~3 but bullet-proof; reframe to "exploits a coupled regime" |
| Exp B: cross-sequence CI excludes 0 | generalization across traces -> replace seed CI in the abstract |
| both done | strongest version: measured coupling + trajectory-level CIs |

Also (free, from `REVIEWER_FIXES.md` Tier 2): retitle around the coupling (not the noisy-OR,
which is a statistical wash), and lean the framing toward the HPEC systems contribution.
