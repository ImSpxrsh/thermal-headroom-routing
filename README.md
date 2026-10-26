# Belief-Space Perception Routing under Coupled Sensor-Fault and Compute-Contention Uncertainty

An offline study of whether a **belief-space** perception router, one that tracks
probabilistic belief over *persistent* sensor faults and over compute contention
and that models the **coupling** between the two, beats memoryless and independent
baselines for adaptive robot perception under a fixed inference deadline.
Evaluation is offline on the **TartanDrive 2.0** off-road driving dataset.

This is a falsification project, not a confirmation project. A negative coupling
result is a valid, reported outcome. No tuning, leakage, or reframing is used to
manufacture a positive.

## Headline contribution

Prior adaptive-perception schedulers (ApproxDet, ApproxNet, Virtuoso,
LiteReconfig, SmartAdapt) combine video content and compute contention as parallel
deterministic inputs to a latency-regression scheduler. None represents either
signal as a belief state, and none runs a decoupled-versus-joint ablation. That
gap is our headline: is the **coupling** between sensor-fault belief and
compute-contention belief informative beyond the sum of its parts?

## Research questions (ordered by contribution weight)

- **RQ-H (headline).** Does jointly modeling sensor-fault belief and
  compute-contention belief reduce deadline-miss rate versus handling the two
  independently (a fair union of the same two signals combined by a fixed rule)?
  The decoupled-versus-joint ablation is the centerpiece.
- **RQ-A1 (supporting ablation and kill-switch).** Does persistence-aware belief
  tracking reduce configuration chattering versus a memoryless detector with the
  *same* features at matched detection accuracy? This is a falsification gate for
  the persistence premise.
- **RQ-A2 (supporting ablation).** Does a hysteresis dwell-time derived from the
  fault model's own transition and uncertainty structure dominate a fixed
  hand-tuned hysteresis under non-stationary fault arrival?

## Relation to prior art

We distinguish from the ApproxDet and ApproxNet lineage on three axes:

1. **Belief-state and persistence tracking** versus their deterministic
   nearest-neighbor contention sensing. We represent both sensor fault and compute
   contention as latent belief over discrete states, updated recursively.
2. **The explicit decoupled-versus-joint coupling ablation** they never ran. Our
   novel claim lives entirely in the delta between a joint policy and a fair
   decoupled union of the same two signals, measured on deadline-miss rate.
3. **An off-road persistent-fault regime** versus their mobile and AR setting.

We cite Wang et al. 2016 (transient-versus-persistent POMDP alarm, *Naval Research
Logistics*) as the formal precedent that persistence reduces false alarms, and we
position RQ-A1 as applying that logic to perception-configuration chattering rather
than to alarms.

## Locked methodological choices

- **Reference and pseudo-ground-truth.** The highest-fidelity config, YOLO11x at
  1280px fp32, run per frame, is the reference (C1). Task accuracy for cheaper
  configs is agreement of their detections with C1 on the same frame. This keeps
  the study about systems tradeoffs without external labels. The known bias:
  "accuracy" here is agreement-with-reference, so any systematic error in the
  reference is invisible. This is stated wherever the metric is used.
- **Model family.** Ultralytics YOLO11. Size gives the model axis, `imgsz` the
  resolution axis, `half=True` the precision axis on GPU.
- **Deadline budget.** Not a hardcoded value. After Phase 3 profiling it is set to
  the median end-to-end latency of C1 under nominal compute, so C1 is
  borderline-feasible uncontended but misses under contention while cheaper configs
  meet it. The chosen value and the rule are reported.
- **Compute model has two states only**, {nominal, contended}. Contention is real
  but non-thermal (induced load on commodity hardware we control), so we do not add
  a "throttled" state we cannot faithfully observe.

## Degradation regimes (two tracks, both reported)

- **Track A (headline, real).** Real motion blur, illumination change, and
  occlusion drawn from TartanDrive itself. Fault-segment labels come from a
  documented deterministic threshold-and-smooth rule on the raw degradation signal,
  never hand-labeled. This is the regime the main results report.
- **Track B (validation, controlled).** Controlled persistent fault injection with
  known onset and duration, used only to validate that the belief estimator tracks
  faults against a ground-truth timeline. Every Track B output is labeled a
  validation aid, never the headline.

## Hardware and where reportable numbers come from

Development, unit tests, and full-pipeline correctness runs run locally on whatever
hardware is present (detected and recorded in `REPORT.md`). Reportable latency
numbers come from `run_on_colab.ipynb` on a Colab GPU, because on a laptop CPU fp16
gives no speedup and the large config is intractably slow, so the precision axis
would be meaningless. Latency numbers are environment-specific and the exact
environment is disclosed in the write-up. Contention is induced in a
device-appropriate way: CPU stress for CPU-bound inference, a co-running GPU
workload plus host-to-device pressure for GPU-bound inference.

## Dataset note

The TartanDrive distribution ships a toolkit repository plus dataset files on CMU
AirLab's public store. The actual data is delivered as ROS1 bags (v1 bundles are
~100 GB each; v2 per-trajectory chunks are ~32 GB each). This project extracts the
`/multisense/left/image_rect_color` stream from one v2 trajectory chunk using a
pure-Python bag reader (no ROS install required). The raw bag and extracted frames
are gitignored; see `src/extract_frames.py` to regenerate them.

## Repository layout

```
config.yaml            single source of truth for all parameters
.env.example           environment overrides (copy to .env)
requirements.txt       Python dependencies
src/                   pipeline code (one module per concern)
outputs/               committed plots and tables for review
data/                  raw bag + extracted frames (gitignored)
run_on_colab.ipynb     reportable GPU profiling + latency (Phase 3)
REPORT.md              running self-assessment, per-phase verdicts, diagnoses
```

## Running it (Google Colab and Mac)

This project uses **pretrained YOLO11 and does not train a model from scratch** (a
locked design choice: the contribution is the routing/belief system). "Running" means
fitting the belief estimators, profiling the config frontier, and running the RQ-H /
RQ-A1 / RQ-A2 experiments. Two environments are supported below. The same steps are in
**[RUNNING.md](RUNNING.md)** with extra detail (contention tuning, troubleshooting).

The dataset (~32 GB bag chunks) is not in the repo; both paths download it.

### Option A: Google Colab (NVIDIA GPU) — gives the reportable latency numbers

1. Open **`run_on_colab.ipynb`** in Google Colab.
2. **Runtime -> Change runtime type -> T4 or A100 GPU**, then **Connect**.
3. **Run all cells, top to bottom.** When the "get the code" cell runs it asks for a
   **GitHub personal access token with `repo` scope** (the repo is private) so it can
   clone. Create one at GitHub -> Settings -> Developer settings -> Personal access
   tokens -> Fine-grained or classic, with repo access. (Alternative: set
   `USE_UPLOAD = True` in that cell and upload a zip of `src/` + `config.yaml`.)
4. The notebook then automatically: installs deps, downloads a TartanDrive 2.0 bag
   chunk, extracts frames, and runs Phase 1, Phase 3 (frontier + deadline), and Phases
   4-6 (RQ-H, RQ-A1, RQ-A2) with GPU-appropriate contention.
5. The last cell zips `outputs/` and downloads it. The figures (e.g.
   `outputs/phase5/rqh_centerpiece.png`) and tables are the results.
6. To run the FULL trajectory instead of a subset, set `profiling.n_frames: 0` in
   `config.yaml` before the profiling cell.

### Option B: Mac with Apple Silicon GPU (M1/M2/M3/M4, Metal/MPS)

```bash
git clone https://github.com/VihanAggarwal/belief-space-perception-routing.git
cd belief-space-perception-routing

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision          # IMPORTANT on Mac: default wheel = Metal/MPS,
                                       # do NOT use the CUDA --index-url
pip install -r requirements.txt

python -c "import torch; print('mps available:', torch.backends.mps.is_available())"

python src/download_data.py            # download a ~32 GB bag chunk (resumable)
python src/extract_frames.py           # bag -> PNG frames + manifest
python src/run_pipeline.py --skip-extract   # runs Phases 1-6 on the Apple GPU
```

On Mac the device auto-selects `mps`. The precision (fp16) axis is CUDA-only, so
configs run fp32 on a Mac (resolution still varies); this is disclosed in the write-up.
After Phase 1, check `outputs/phase1/contention_summary.json` -> `p95_shift_ratio`
(want ~2x-3x); if needed, tune `contention.gpu.competitor_sleep_s` in `config.yaml`
(see RUNNING.md, section 3).

### Windows / NVIDIA (development environment used here)

```
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python src/download_data.py && python src/extract_frames.py
python src/run_pipeline.py --skip-extract
```

## Status: complete (local dev slice)

All seven phases ran end-to-end on a real 309-frame TartanDrive 2.0 slice. See
`REPORT.md` for per-phase self-assessments and `WRITEUP.md` for the results and
limitations draft. Headline findings (5 seeds, 95% CIs; latencies dev-only, reportable
numbers via `run_on_colab.ipynb`):

- **RQ-H (headline): small positive, coupling-driven.** The joint policy reduces
  deadline-miss by **1.73 pp [0.48, 2.99]** vs the decoupled baseline in the coupled
  regime, and by **0.0 pp** in the uncoupled control, at matched accuracy. The benefit
  appears only where the coupling is real, so the coupling drives it. Magnitude modest,
  reported as such, not oversold.
- **RQ-A1 (kill-switch): not statistically supported on this slice (underpowered).**
  Persistence halves the switch rate in point estimate but the CI includes 0; resolve
  on the full trajectory.
- **RQ-A2: tradeoff, not dominance.** Model-derived hysteresis is 4x more responsive
  under non-stationary arrival but switches more.

This is a falsification study: negatives and modest effects are reported plainly.
