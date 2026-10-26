# Belief-Space Perception Routing under Coupled Sensor-Fault and Compute-Contention Uncertainty

Offline study on the TartanDrive 2.0 off-road driving dataset. This document is the
results-and-limitations draft. Every quantitative claim traces to a specific plot or
table in `outputs/`. This is a falsification study: negative results are reported as
such.

## 1. Method

**Task and reference.** We route per frame among four YOLO11 perception configs that
span model size x input resolution x precision:
C1 = YOLO11x@1280 fp16 (reference), C2 = YOLO11x@640 fp16, C3 = YOLO11n@1280 fp16,
C4 = YOLO11n@640 fp16. C1 run on every frame is the pseudo-ground-truth. A cheaper
config's task accuracy is the IoU-matched, per-class agreement (F1) of its detections
with C1 on the same frame. This is agreement-with-reference, not external truth: any
systematic error in C1 is invisible. We state this wherever accuracy is used.

**Deadline.** Not hand-set. After profiling, the deadline is the median end-to-end
latency of C1 under nominal compute (latency value reported in the Results). C1 is then
borderline-feasible uncontended and misses under contention, while cheaper configs meet
the deadline under contention, creating real routing pressure.

**Degradation, two tracks.** Track A (headline, real) uses real blur (variance of
Laplacian), illumination (histogram entropy), and occlusion (sparse optical-flow track
survival) drawn from TartanDrive. Fault segments come from a single deterministic
threshold-and-smooth rule (robust-z on an EMA-smoothed signal, with minimum-duration
and asymmetric recovery hysteresis), reported as part of the method, never hand-labeled.
Track B (validation only) overlays controlled persistent faults with a known onset and
duration, used solely to measure belief-tracking lag against a ground-truth timeline.

**Compute contention.** Two states, {nominal, contended}, induced by a
device-appropriate generator: on GPU, a co-running GPU workload competing for SMs and
memory bandwidth plus host-to-device transfer pressure; on CPU, CPU-pinned compute plus
memory pressure. We measure end-to-end per-frame latency (preprocess + inference +
postprocess analogue) and a sliding-window p95/p99 plus queue-depth signal. Latency is
environment-specific; reportable numbers come from a GPU via `run_on_colab.ipynb`.

**Belief estimators.** Per sensor channel, a switching HMM over the continuous
(robust-z) observation with latent state in {nominal, degrading, faulted, recovering}
and Gaussian emissions, fit from labeled segments with asymmetric onset/recovery, output
a belief b_t = P(z_t | o_{1:t}). A two-state compute HMM over [p95, p99, queue-depth]
outputs P(contended | features). Both are recursive Bayesian filters, so they carry
persistence -- the design choice RQ-A1 puts on trial.

**Coupling regimes.** Uncoupled: contention follows a fixed periodic schedule,
independent of faults. Coupled: contention correlates with detected fault onset (with a
lag). The realized fault-contention correlation is reported for each regime.

**Routing.** Both the joint policy and the decoupled baseline maximize the same
objective, expected reliability = expected accuracy (under the sensor belief) x P(meet
deadline | a contention probability p_cont). The ONLY difference is how p_cont is
formed: the joint fuses the compute belief with a coupling prediction from the sensor
belief (noisy-OR with a coefficient kappa fit on a held-out calibration draw); the
decoupled uses the compute belief alone. Thus the decoupled is a fair union that uses
both signals (sensor for accuracy, compute for the deadline) but never models the
coupling. The novel claim is exactly the delta between them on deadline-miss rate.
Hysteresis comes in two variants (fixed dwell, and a dwell derived from the belief
model's transition structure). Abstention emits ABSTAIN when no config can meet the
deadline at the required reliability, scored by precision/recall against a post-hoc
oracle-infeasibility label. Abstention is OFF for the RQ-H headline (so deadline-miss is
measured over all frames, never confounded by differential abstention) and evaluated
separately.

**Statistics.** Every cross-method comparison reports a 95% confidence interval (t
interval) over at least five seeds, where a seed draws the contention schedule, the
realized latencies, and per-seed sensor measurement noise.

## 2. Results

All numbers below are from the local dev slice: **309 real TartanDrive 2.0 frames**
(first ~31 s of `turnpike_afternoon_fall`, extracted from a partial bag download via a
streaming ROS1 reader), profiled on an NVIDIA RTX A500 4 GB laptop GPU. Latencies are
dev-environment and not reportable; the reportable GPU version comes from
`run_on_colab.ipynb`. The deadline is 772 ms (median C1-nominal latency, locked rule).
Every comparison is over 5 seeds with 95% CIs. Real fault fraction: 30.4%
(illumination-dominated). Regime separation: coupled Pearson r=0.74, uncoupled r=0.06.

### 2.1 RQ-H (headline): joint vs decoupled, the coupling ablation

The centerpiece figure is `outputs/phase5/rqh_centerpiece.png`.

| regime    | kappa | joint deadline-miss | decoupled deadline-miss | reduction (decoupled - joint), 95% CI | accuracy (joint / decoupled) |
|-----------|-------|---------------------|-------------------------|---------------------------------------|------------------------------|
| coupled   | 0.79  | 0.405               | 0.423                   | **+1.73 pp [0.48, 2.99], significant** | 0.891 / 0.893               |
| uncoupled | 0.09  | 0.384               | 0.384                   | 0.00 pp [0.00, 0.00], n.s.            | 0.843 / 0.843               |

Modeling the sensor-compute coupling reduces the deadline-miss rate by **1.73
percentage points** in the coupled regime (95% CI excludes 0) and by **exactly 0** in
the uncoupled regime, at **matched accuracy** (0.891 vs 0.893). The uncoupled regime is
a clean control: the benefit appears only where the coupling is real, which is direct
evidence that the coupling -- not a confound -- drives the effect. We confirm RQ-H
directionally.

We do not oversell the magnitude, which is modest and near the 1-2% materiality
threshold we set in advance. Why modest: about 18% of frames are oracle-infeasible
(under heavy contention even the cheapest config's realized latency exceeds the
deadline), so many misses are unavoidable by any policy and the achievable gain is
capped; the joint's advantage is concentrated on the short onset-to-contention
preemption window; and on this slice the compute belief is not extremely laggy. A
longer trajectory (more onsets), a longer coupling lag, or denser task-relevant
detections would be expected to enlarge the effect. These are scope statements, not
post-hoc excuses, and the uncoupled control rules out an artifact either way.

#### 2.1b RQ-H on REAL adverse weather (Track D: RADIATE rain) — the real-data companion

TartanDrive (Track A) is real but fault-sparse (one labeled segment, ~1 onset), so its
RQ-H point, while directionally positive (+1.73 pp), rests on a CI-fragile single event.
RADIATE rain is the fault-dense real-data regime that fixes this. On `rain_4_0` (2,651
real rain frames, left ZED rectified to camera_left_rect; 1,500-frame window; 5 seeds;
`outputs/trackD_rain_4_0/`), the window has **13 fault-onset events** (vs ~1 on
TartanDrive), all occlusion-channel (rain droplets on the lens). Coupled r=0.76,
uncoupled r=0.01; contention p95 shift 3.73x.

| regime    | kappa | joint miss | decoupled miss | reduction 95% CI            |
|-----------|-------|-----------|----------------|-----------------------------|
| coupled   | 0.82  | 0.453     | 0.467          | **+1.39 pp [0.98, 1.80], sig** |
| uncoupled | 0.02  | 0.573     | 0.573          | 0.00 pp [0.00, 0.00], n.s.  |

The coupling reduces deadline-miss by **1.39 pp on real rain** (95% CI excludes 0) and by
exactly 0 in the uncoupled control. This is the first real-data RQ-H test with more than
one onset, and it agrees with both TartanDrive's coupled direction and the synthetic
result: the coupling effect is small but real and, critically, vanishes without the
coupling. Magnitude is modest and reported as such; 13 onsets beats 1 but is still
modest, so snow/fog/night and longer/combined windows are the path to a more powered
number. Latencies are dev-env (RTX A500), not reportable; the comparison is the result.

Triangulation so far: the coupling helps in (i) the powered synthetic regime, (ii)
TartanDrive's sparse real point (directional), and (iii) RADIATE real rain (significant,
clean control). TartanDrive's earlier null is now an explained point: too few onsets to
power the test, not evidence against the coupling.

### 2.2 RQ-A1 (supporting): persistence vs memoryless at matched accuracy

Figure: `outputs/phase6/rqa1_switching.png`. At matched detection balanced-accuracy
(belief 0.950 vs memoryless 0.950), the persistent belief detector halves the
config-switch rate in point estimate (**0.011 vs 0.022**), but the paired difference is
**0.011, 95% CI [-0.012, 0.033], not significant**.

Verdict: not supported on this slice. The direction favors persistence, but the effect
is not statistically distinguishable from zero because the slice has few fault onsets
(hence few switches, high relative variance over five seeds). This is an underpowered
test, not a refutation of the persistence premise; the full ~1198-frame trajectory is
the natural resolution. Reported honestly as a negative; RQ-H remains the headline.

On **real RADIATE rain** RQ-A1 is also not supported, and more pointedly: at matched
detection accuracy the persistent belief detector switches *more* than memoryless (0.030
vs 0.018, CI [-0.013, -0.011]). Diagnosis: rain's rapidly fluctuating droplet-occlusion
signal makes the noise-driven HMM belief cross the 0.5 boundary more often than the
instantaneous signal at its matched operating point. So persistence does not reduce
chattering on this real regime. RQ-A1 is the patch's clearest negative and is reported as
such; it does not affect RQ-H, which is a routing-feasibility result, not a detector one.

### 2.3 RQ-A2 (supporting): model-derived vs fixed hysteresis

Figure: `outputs/phase6/rqa2_hysteresis.png`. Controlled fault arrival (labeled as a
validation aid), stationary and non-stationary.

| arrival        | fixed switch | model-derived switch | onset-to-reconfig lag (fixed / md) |
|----------------|--------------|----------------------|------------------------------------|
| stationary     | 0.029        | 0.033 (n.s.)         | 1.40 / 1.10                        |
| non-stationary | 0.034        | 0.045 (sig)          | 6.60 / **1.60**                    |

Verdict: a tradeoff, not dominance. Under non-stationary arrival the model-derived
dwell is 4x more responsive (onset-to-reconfiguration 1.6 vs 6.6 frames) but switches
about 30% more; under stationary arrival the two are indistinguishable. The
model-derived hysteresis adapts to non-stationarity as intended but does not
Pareto-dominate a hand-tuned fixed dwell.

### 2.4 Abstention

Abstention did not trigger on this slice: under predicted contention the cheapest
config still reaches the reliability target, so there was no frame where the policy
judged no config feasible. The post-hoc oracle nonetheless labels 53/300 frames
infeasible (realized latency exceeds the deadline for all configs under heavy
contention). The gap means abstention as configured is conservative here; this is a
property of the slice (frequent but survivable contention), reported rather than tuned
away.

## 3. Relation to prior art

We distinguish from the ApproxDet and ApproxNet lineage (and Virtuoso, LiteReconfig,
SmartAdapt) on three axes. First, belief-state and persistence tracking versus their
deterministic nearest-neighbor contention sensing: we represent both sensor fault and
compute contention as latent belief over discrete states, updated recursively, rather
than as deterministic instantaneous inputs to a latency-regression scheduler. Second,
the explicit decoupled-versus-joint coupling ablation they never ran: our entire novel
claim is the delta between a joint policy and a fair decoupled union of the same two
signals. Third, an off-road persistent-fault regime versus their mobile and AR setting.
We cite Wang et al. 2016 (transient-versus-persistent POMDP alarm, Naval Research
Logistics) as the formal precedent that persistence reduces false alarms, and position
RQ-A1 as applying that logic to perception-configuration chattering rather than alarms.

## 4. Limitations (scope honesty)

- **No hardware-in-the-loop.** Fully offline, trace-driven evaluation.
- **Non-thermal, commodity contention.** Contention is induced co-running load on
  hardware we control, not embedded-SoC thermal throttling; hence a two-state compute
  model with no "throttled" state we cannot faithfully observe. The exact environment is
  disclosed (dev: NVIDIA RTX A500 4 GB laptop GPU; reportable: Colab GPU).
- **Designed, not measured, sensor-to-compute coupling.** The coupled regime imposes a
  correlation between fault onset and contention; we do not claim this correlation
  occurs naturally in TartanDrive. RQ-H tests whether, GIVEN such coupling, modeling it
  helps; the uncoupled regime is the control.
- **Pseudo-GT is agreement-with-reference.** Accuracy is agreement with C1, not truth;
  systematic C1 errors are invisible. Off-road scenes contain few COCO-class objects, so
  the agreement signal is sparse (see Results for the measured per-config accuracy
  spread and any consequence for the strength of the RQ-H test).
- **Track B is validation only.** Controlled injection is used solely to measure
  belief-tracking lag and never reported as a headline result.
- **Dev slice.** Local results use a real 309-frame slice extracted from a partial bag
  download; the full chunk and reportable latencies come from `run_on_colab.ipynb`.
- **In-sample HMM calibration.** The belief HMMs are fit and filtered on the same trace
  (shared by all methods, so it does not bias the RQ-H delta; it does inflate absolute
  belief calibration). The coupling coefficient is fit on a held-out calibration draw.
- **Track D (RADIATE) specifics.** Low camera resolution (672x376) upscaled to the
  reference imgsz (1280) for pseudo-GT; radar/lidar and RADIATE's own annotations are
  unused (camera-only, evaluated via YOLO11x agreement). Coupling on Track D is still
  designed, not measured, like the other tracks. The rain result uses one sequence
  (`rain_4_0`, 13 onsets in the window) on dev-env latencies; more conditions
  (snow/fog/night) and longer/combined windows would strengthen the power.

## 5. Most promising follow-up

- **More real adverse-weather conditions and longer windows on RADIATE** (snow, fog,
  night, and combined sequences) to push the real-data RQ-H from "small but significant"
  to well-powered, and to place each condition's operating point on the (rho, lambda)
  phase diagram next to TartanDrive's sparse point.
- **Real embedded contention on a borrowed Jetson** (thermal + memory-bandwidth
  contention that is measured rather than designed), so the coupling is observed rather
  than imposed.
- **Closed-loop evaluation** (e.g., a MuJoCo navigation loop with rendered fault
  injection measuring downstream task performance rather than detector agreement),
  scoped as the natural next step, not attempted here.
- A second dataset with denser task-relevant objects so the pseudo-GT accuracy axis
  differentiates configs more strongly.
