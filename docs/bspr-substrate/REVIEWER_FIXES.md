# Reviewer Response: what to fix and run

Action list from the IEEE-HPEC reviewer feedback. Items are tagged
**[done-in-code]** (already implemented + committed, just needs a re-run / paper text),
**[run]** (you run it on the Mac with the data/GPU), or **[paper]** (text edit only).

Priorities in order: 0 (critical) > 1 > 3 > 2 > 5 > presentation.

---

## POST-RESULTS REFRAME (Experiments A & B are in) -- the higher-tier version

Results: **A = clean null** (real compute load does not co-occur with sensor faults on any
of the 8 RADIATE sequences; the coupling the paper studies is imposed, not observed there).
**B = all-positive but per-condition t-CIs include zero** at n=3 traces; cluster bootstrap
stays positive. Here is how that becomes a STRONGER paper, not a weaker one.

### The claim structure that survives (and is more interesting)
The paper is no longer "coupling helps" -- it is a **validated conditional**:

1. **Sufficiency** (imposed regime): when fault-load coupling is present (kappa~0.8), the
   coupled router cuts deadline misses in EVERY independent sequence tested -- pooled exact
   sign test across all sequences (see below), on top of the per-sequence seed CIs.
2. **Necessity, verified on real data** (the new part): when coupling is absent, the benefit
   is zero -- shown three independent ways: (a) the uncoupled schedule control, (b) the
   kappa=0 phase-diagram row, and now (c) **measured real load on 8 real sequences**, where
   the coupling is genuinely absent and the router is correctly inert (never negative).
   (c) is what no rewrite could buy: the zero-prediction was made by the model FIRST and
   confirmed on real data. "We measured it and found the regime absent, exactly as the
   phase diagram predicts at coupling~0" is falsification-grade evidence; reviewers reward
   it far more than silence.
3. **Deployment rule**: measure Pc(F)-Pc(N) on your platform (the measurement procedure is
   now part of the method -- measure_real_coupling.py); deploy the coupled router iff it is
   positive. The phase diagram tells you the expected benefit region. This turns the null
   into an actionable operating map instead of a dead end.

Honesty constraints (do NOT oversell): the phase diagram's zero-point and sign are
validated; its MAGNITUDES are not (it under-predicts severe-fault conditions like fog,
r~0.23) -- say "predicts when, not how much." And RADIATE's null does not mean coupling
never exists (single-SoC robots with shared perception/planning/control budgets, thermal
throttling under sustained load, detection-count-scaled downstream trackers are the
plausible real cases) -- it means it must be MEASURED, which the method now includes.

### New statistics that make Experiment B land (BUILT, run these)
Per-condition t-CIs at n=3 are the wrong test. Pool all sequences and use exact tests:
```bash
# pooled across ALL evaluated sequences (any condition):
python src/aggregate_multitrace.py --condition all --pool --tracks \
  outputs/trackD_rain_1_0 outputs/trackD_rain_2_0 outputs/trackD_rain_3_0 outputs/trackD_rain_4_0 \
  outputs/trackD_fog_1_0 outputs/trackD_fog_2_0 outputs/trackD_fog_6_0 \
  outputs/trackD_snow_1_0 outputs/trackD_night_1_0 outputs/trackC outputs/trackA_309
```
Reports the exact one-sided sign test ("positive in k/n independent sequences, p=1/2^n
when k=n": 6/6 -> p=0.016, 8/8 -> p=0.004, 10/10 -> p=0.001), Wilcoxon signed-rank, the
trajectory-level t-CI + cluster bootstrap, and the pooled de-circularized (real-load)
reductions with a does-no-harm count. Sanity: on the 6 committed conditions alone this
already gives mean +3.96pp, t-CI [0.67, 7.25], sign p=0.016 -- ACROSS heterogeneous
conditions, datasets, and camera geometries, which is itself a generalization statement.

### Zero-point validation figure (BUILT, run this)
```bash
python src/predicted_vs_observed.py --tracks <all sequence output dirs>
python src/gen_results.py     # picks up sections 8-10 automatically
```
Plots phase-diagram-predicted vs observed reduction for every sequence at three operating
points (imposed / uncoupled control / measured real load). The measured points cluster at
(pred~0, obs~0): the confirmed prediction. Put this figure in the paper next to the phase
diagram.

### Rewrite the abstract around the conditional (sketch)
Title: "When Does Coupling Sensor-Fault Belief to Compute-Contention Belief Help?
Deadline-Aware Perception Routing with a Measured Operating Map"
Abstract skeleton: (i) couple fault belief into contention anticipation; (ii) under
fault-conditioned contention, miss reduction in every one of N independent sequences
(sign test p=...), robust to fusion form; (iii) a coupling x onset-rate map predicts where
the benefit lives; its zero point is CONFIRMED on real data: measured fault-load
co-occurrence on 8 real adverse-weather sequences is absent, and the router is correctly
inert there (never harmful); (iv) the measurement procedure is part of the method: measure
Pc(F)-Pc(N) on target hardware, deploy iff positive. -> The contribution is the validated
conditional + the operating map + the measurement recipe, not a universal speedup.

### Friend checklist (in order)
1. **Commit the Experiment A/B outputs** (they are not on main yet!):
   `git add outputs/*/extras/real_coupling.json outputs/multitrace outputs/trackD_*` + push.
2. Run the pooled aggregation + predicted_vs_observed commands above; run
   `python src/gen_results.py`; commit RESULTS.md + outputs/multitrace.
3. Paper edits: reframe per this section; move the real-coupling measurement from
   "limitations" into a first-class results subsection ("Is the coupling real?"); report
   the sign test as the headline generalization statistic; keep per-condition numbers as
   supporting detail. State plainly that RADIATE shows no natural fault-load coupling and
   what platforms plausibly would.

---

## TIER 2 -- what actually moves this from 3/5 to 4-4.5/5 (real evidence)

Blunt truth: the rewrite/robustness fixes below (Tier 1) remove reject-risk and earn
"honest, well-controlled, clearly scoped" credit -- that is a **safer 3, not a 4**. You
cannot write your way past the fact that the contention schedule is generated from the
fault labels (Pc(F)=0.85, Pc(N)=0.05 are hard-coded), so the current strongest true claim
is "our estimator detects a dependency we imposed." Reviewers score the strength of that
claim, not the polish. To reach 4/4.5 you need **new evidence that the dependency is real**,
or a claim that does not depend on it. Two experiments do that; A is the ceiling-breaker,
B is the safest ROI. Both are built and have turnkey Mac instructions in
**`MAC_EXPERIMENTS_AB.md`** (works against RADIATE or any dataset).

### Experiment A -- measure the real coupling + de-circularize the schedule (BUILT)
`src/measure_real_coupling.py` is written and ready. It (1) runs C1 over the real frames and
records per-frame REAL latency + detection count (a compute-load proxy from YOLO output,
independent of the blur/illum/occlusion fault signals); (2) MEASURES Pc(F)=P(high-load|fault)
and Pc(N)=P(high-load|nominal) -- the real analogues of the hard-coded 0.85/0.05 -- plus
corr(fault, load) with a bootstrap CI; (3) re-runs RQ-H with the contention schedule driven
by the REAL load proxy (via the new `state_override` in `simulate.build_substrate`), so
kappa is EMPIRICAL, not imposed.
```bash
for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/measure_real_coupling.py --track outputs/trackD_$SEQ \
     --frames data/frames/radiate_$SEQ --load-proxy detections --max-frames 1500
done
# also try --load-proxy latency (real end-to-end latency incl. NMS/postproc)
```
- If **Pc(F) > Pc(N)** and the de-circularized reduction stays positive -> the coupling is
  REAL: you now have an empirical claim about the world, and the coupled regime is justified
  by data, not assumption. That is a genuine 4.
- If **Pc(F) ~ Pc(N)** -> honest null: degradation and compute load do not co-occur in this
  data; report it and reframe the paper as "given a coupled regime, the estimator exploits
  it" (still a clean 3, and you avoid a reviewer discovering the null for you).
- Best load proxy is real per-frame latency incl. NMS/postproc (scales with detections);
  detection count is the cleanest independent proxy. Report both.

### Experiment B -- more than one trace per condition (safest ROI)
The CIs today are 5 seeds over ONE frozen trace per condition (and only 300 frames for
RADIATE -- item 0). That is sensitivity to injected noise, not generalization. RADIATE has
many sequences per weather (rain_1..rain_7, etc.) and you have Dropbox access. Run 3-5
sequences per condition and report a bootstrap over TRAJECTORIES, not just seeds:
```bash
# download e.g. rain_1_0, rain_2_0, rain_3_0 ... then, per sequence:
python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ python src/run_pipeline.py --skip-extract
# then aggregate per-condition across sequences (CI over TRAJECTORIES):
python src/aggregate_multitrace.py --condition rain \
  --tracks outputs/trackD_rain_1_0 outputs/trackD_rain_2_0 outputs/trackD_rain_3_0 outputs/trackD_rain_4_0
```
This turns "significant on a frozen trace" into "generalizes across independent weather
episodes" -- it lifts the statistics score regardless of Experiment A's outcome. Do this
together with item 0 (re-profiling at 1500 frames is the same run).

### The noisy-OR is not the contribution -- retitle (do not try to save it)
The threshold rule `max(b_c, 1[b_f>0.5])` matches the noisy-OR on every condition (all CIs
include zero, +/-0.5pp), and stays a wash even under heavy observation noise (+/-1.2pp up to
sigma=0.75). So the specific fusion math the title sells is NOT what drives the result. Do
not add experiments to defend it -- the evidence says it does not matter. Instead:
- **Retitle around the coupling / belief-space idea**, not the noisy-OR (e.g. "Coupling
  Sensor-Fault Belief into Compute-Contention Anticipation for Deadline-Aware Inference").
- Present the noisy-OR as one sufficient instantiation and the threshold as an equally good
  realization. The contribution is "use fault belief to anticipate contention," demonstrated
  with the simplest rule that works.

### Venue lever (IEEE-HPEC is a compute/systems venue, not robotics)
HPEC reviewers weight the deadline-aware, contention-belief, accuracy-latency-frontier
engineering more than the "is the robot-world coupling real" question a robotics reviewer
raises. Lean the framing toward the systems contribution (device-appropriate contention
characterization, self-calibrated deadline, measured frontier, belief-driven routing under a
compute budget) and treat the sensor coupling as one application. This does not manufacture
evidence, but it aligns the claim with what your actual reviewers reward.

---

## 0. CRITICAL (not in the review, found while auditing): RADIATE RQ-H ran on only 300 frames

`profiling.n_frames: 300` in `config.yaml` bounded the RADIATE profiling to the first
**300 frames** of each ~2,600-frame sequence. Because the simulation length is the number
of profiled frames, **the RADIATE RQ-H deltas were computed on 300 frames**, while Table II
reports full-sequence frame counts and fault fractions. They do not match:

| track | Table II (frames / fault%) | actual RQ-H window | actual in-window fault% |
|---|---|---|---|
| fog   | 2659 / 1.9%  | **300** | **4.0%** |
| rain  | 2651 / 44.4% | **300** | **27.0%** |
| snow  | 2589 / 18.5% | **300** | **11.3%** |
| night | 2644 / 12.7% | **300** | **8.0%** |
| Track C (soiling) | 3303 / 8.1% | 3303 (full) | 8.1% (matches) |
| Track A off-road seg. | 309 / 30.4% | 300 | 31.3% (matches) |

Track C and Track A are fine; the four RADIATE tracks are not. This is the single biggest
validity gap (it also makes the "5 seeds reuse the same frames" point worse: it's 300
frames, 5 seeds). **Fix by re-profiling RADIATE on more frames**, then the reported
frames/fault% become representative and the CIs/temporal-split get more power.

**[run]** Re-profile each RADIATE sequence on >=1500 frames:
```bash
# bump the profiling window (use 0 for ALL frames if the Mac can take it)
# edit config.yaml:  profiling.n_frames: 1500     (was 300)

for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
  FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ \
    python src/run_pipeline.py --skip-extract
done

# then re-run the add-ons on the new outputs
python src/run_addons.py --tracks outputs/trackD_rain_4_0 outputs/trackD_snow_1_0 \
                                   outputs/trackD_fog_6_0 outputs/trackD_night_1_0
for t in trackD_fog_6_0 trackD_rain_4_0 trackD_snow_1_0 trackD_night_1_0; do
  SW=""; [ "$t" = "trackD_fog_6_0" ] && SW="--sweeps"
  python src/run_robustness.py --track outputs/$t $SW
done
python src/gen_results.py
python src/make_summary_figure.py
git add outputs RESULTS.md paper/figures && git commit -m "Re-profile RADIATE on 1500-frame windows" && git push
```
**[paper]** After re-profiling, update Table II frames/fault% to the new numbers and check
the RQ-H deltas (they should stay the same sign; report whatever they are). If you instead
keep 300 frames, you MUST report 300 and the in-window fault% in Table II, not the
full-sequence numbers.

---

## 1. Contention-schedule circularity (reviewer #1) -- mostly already safe

The strong version ("schedule uses the router's own detected onset/HMM output") is **false
in the code**, and you should say so explicitly:

- The contention schedule is built from `fault_active = lab["any_fault"]` -- the **Phase-1
  threshold-and-smooth labels** (precomputed, deterministic), see `regimes.coupled_schedule`
  and `simulate.build_substrate`. It is **not** the router's HMM output.
- The online router receives only `s_belief` (sensor HMM filtering **noisy** observations,
  per-seed noise `observation_noise_std=0.15`) and `c_belief` (compute HMM on probe
  latency). **It never sees `state`** (the schedule variable).
- The fault HMM must infer the event from the noisy observations independently.

**[paper]** State the four bullets above in the Experimental Setup, plus: kappa is fit on
**independent calibration contention draws** (different seeds), never the test draw
(`simulate.calibration_kappa`), and is a structural regime constant.

**[done-in-code]** Requirement #4 (calibration/test temporal separation) is now implemented:
`run_robustness.py` adds a `calibration_split` -- fit kappa on the first half of the trace,
evaluate RQ-H on the held-out second half. Results (see `RESULTS.md` 7d):
- kappa is stable where the calibration half has faults (0.78-0.86 vs full ~0.80).
- The held-out test is significant on the **long Track C** sequence (+1.53 pp, CI excludes 0).
- On the 300-frame RADIATE windows the held-out half is fault-sparse (the split is
  underpowered) -- another reason to re-profile longer (item 0). After item 0 this becomes a
  clean rebuttal on all tracks.

---

## 2. Pseudo-GT accuracy validation (reviewer #2)

**[run]** `src/validate_pseudo_gt.py` is a scaffold. Wire up `load_radiate_gt()` (parse the
RADIATE `annotations.json` and rectify boxes through the same calib as
`src/extract_radiate.py`) and `detect()` (mirror `profile_frontier.py`), then:
```bash
python src/validate_pseudo_gt.py --seq-dir data/radiate/rain_4_0 --max-frames 500
```
It correlates per-frame pseudo-GT (agreement-with-C1) against real F1 vs the RADIATE
labels. One number (high Pearson/Spearman) shows agreement-with-reference tracks real
detection quality. **[paper]** Add it as a validation paragraph; keep the rest of the
experiments on pseudo-GT.

---

## 3. Deadline looks convenient (reviewer #3) -- DONE in code

**[done-in-code]** `run_robustness.py` now sweeps the deadline for **every** condition (not
just fog) and adds **externally-motivated fixed deadlines** (10 Hz / 100 ms and 5 Hz /
200 ms), not self-calibrated. See `RESULTS.md` 7b (all-condition sweep) and 7c (fixed Hz).
Pattern: the benefit peaks at 1.0x for all conditions and shrinks at looser deadlines but
stays positive for most (off-road stays +2.53 pp at 1.2x; fog concentrates near 1.0x).
**[paper]** Replace the fog-only sweep with the all-condition table; use "self-calibrated
deadline" consistently and frame it as an operating-point experiment, not a robot
requirement. Cite the fixed-Hz result as the externally-motivated point.

---

## 4. Statistics (reviewer #4)

- **[paper]** Promote the moving-block bootstrap to the **primary** sequential analysis (it's
  already computed for all tracks, `RESULTS.md` 7). Keep the seed CI as secondary.
- **[run, optional]** More noise seeds: edit `seeds:` in `config.yaml` (e.g. 0-9) and re-run
  phases 5-6. This tightens the observation-noise CI only; it does NOT add cross-environment
  generalization, so keep the honest disclosure. Best done together with item 0.
- **[paper]** Tone down significance language so it does not sound like cross-route/vehicle
  generalization; the CIs are sensitivity to injected sensor noise on fixed frames.

---

## 5. Method under-specified (reviewer #5) -- concrete values to put in the paper

All from `config.yaml` + the named functions; add these so reviewers do not need the repo:

- **Health signals**: blur = variance of Laplacian; illumination = grayscale histogram
  entropy; occlusion = sparse optical-flow track-survival rate (fisheye: real soiling-mask
  pixel fraction; blur computed after polar-unwrap). Each robust-z normalized on an
  EMA(alpha=0.3) baseline.
- **Fault labels** (threshold-and-smooth): robust-z threshold 2.5, min segment 5 frames,
  recovery hysteresis 8 frames; per-channel "low tail = bad".
- **Sensor HMM**: 4 states {nominal, degrading, faulted, recovering}, Gaussian emissions
  (per-state std floor 0.3), transition init stay_nominal=0.92, stay_faulted=0.90, Laplace
  smoothing 1.0 on fitted counts; labels expanded with 3 onset / 5 recovery frames.
  `b_f^t = max_k P(faulted | o_{1:t})` (forward filter), see `sensor_belief.py`.
- **Compute HMM**: 2 states, features [p95, p99, queue_depth] over a 30-frame window,
  standardized; transition init stay_nominal=0.95, stay_contended=0.90;
  `b_c^t = P(contended | l_{1:t})`, see `compute_belief.py`.
- **E[acc_c | b_f]** = (1 - b_f)·acc_nominal[c] + b_f·acc_faulted[c] where acc_nominal/faulted
  are per-config mean agreement on non-fault/fault frames (`policies.FrontierModel.exp_accuracy`).
- **q_c (feasibility)**: a config is feasible in the predicted compute state s if its median
  latency in s meets the deadline, i.e. p_meet[c][s] >= 0.5 (`policies._choose`).
- **kappa fit**: kappa = clip(P(cont | fault_{t-lag}) - P(cont | nominal_{t-lag}), 0, 1),
  lag = 3 frames, fit on independent calibration draws (`policies.fit_coupling`,
  `simulate.calibration_kappa`). Coupled-regime generator uses P(cont|fault)=0.85,
  P(cont|nominal)=0.05.
- **Deadline**: median end-to-end latency of C1 under nominal compute (locked rule).
- **Hysteresis dwell**: fixed = 10 frames; model-derived from target false-transition rate
  0.02 (`policies.model_derived_dwell`). Reliability target 0.85.
- **Seeds**: 5; per-seed observation noise std 0.15 (sole source of seed variance).

---

## Presentation fixes (reviewer)

- **[done-in-code]** Figure 1 sign + fonts + clipping: `fig_rqh_summary.png` has been
  regenerated with `python src/make_summary_figure.py`. Axis is now "Deadline-miss
  reduction: decoupled - coupled (pp)" (positive = coupling helps, matches Table II), large
  fonts, padded layout. The previous version had the sign-inconsistent axis the reviewer
  flagged; re-run this script if you re-profile (item 0) so the figure matches the numbers.
- **[paper]** All 8 figures are committed under `paper/figures/`. After re-profiling
  (item 0) the RADIATE-derived ones (`fig_contention`, `fig_frontier`, `fig_regimes`,
  `fig_belief_occlusion`, `fig_rqh_fog`, `fig_phase_diagram`, `fig_rqa1_night`) will be
  stale; regenerate them from the new `outputs/<track>/phase*` and `extras/`.
- **[paper]** Standardize terminology: use **"coupled"** everywhere (or "joint" everywhere),
  not both.
- **[paper]** Define RQ-H and RQ-A1 in the intro before the labels are used.
- **[paper]** The learned-router / sequence-split table (rain miss 0.000): label clearly that
  it uses different temporal test partitions than Table II, or drop it. Given the MLP result
  is a wash, consider cutting it for space and using the room for items 2 and 5.
- **[paper]** State that "per-channel" = per health signal (blur / illumination / occlusion),
  not per camera or per modality.

---

## One-shot run sequence (after editing config.yaml: profiling.n_frames: 1500)
```bash
# 0+3+1: re-profile RADIATE, re-run add-ons + robustness, regenerate results + figure
for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
  FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ python src/run_pipeline.py --skip-extract
done
for t in trackD_fog_6_0 trackD_rain_4_0 trackD_snow_1_0 trackD_night_1_0 trackC trackA_309; do
  SW=""; [ "$t" = "trackD_fog_6_0" ] && SW="--sweeps"
  python src/run_robustness.py --track outputs/$t $SW
done
python src/gen_results.py && python src/make_summary_figure.py
# 2: wire up + run the pseudo-GT validation (needs RADIATE annotations)
# python src/validate_pseudo_gt.py --seq-dir data/radiate/rain_4_0 --max-frames 500
git add -A && git commit -m "Re-profile RADIATE + reviewer robustness add-ons" && git push
```
