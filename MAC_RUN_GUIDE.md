# Mac Run Guide (Apple Silicon / Metal GPU)

For running the belief-space-perception-routing experiments on a Mac with an Apple
Silicon GPU (M1/M2/M3/M4/M5). Covers the two built tracks:

- **Track A — TartanDrive** (off-road, normal lens)
- **Track D — RADIATE** (real adverse weather: rain / snow / fog / night)

Track C (WoodScape fisheye) is not built yet and is intentionally out of scope here.

This runs the pipeline exactly as designed (no methodology changes): per-channel
degradation observations, deterministic threshold-and-smooth fault labels, switching-HMM
sensor belief, two-state compute belief, joint vs decoupled routing, both hysteresis
variants, abstention, oracle labeler, and the RQ-H / RQ-A1 / RQ-A2 experiments with CIs
over 5 seeds.

Note on latencies: numbers measured on your Mac's Metal GPU are environment-specific
(disclose the chip). The RQ-H deadline-miss comparison is the result; absolute latencies
are dev-environment.

---

## 0. Prerequisites
- macOS on Apple Silicon (M-series).
- Python 3.11 or 3.12, and `git`.
- Disk: ~40 GB free if you run Track A (TartanDrive downloads a ~32 GB bag). Track D
  needs only a few GB per RADIATE sequence.

## 1. One-time setup
```bash
git clone https://github.com/VihanAggarwal/belief-space-perception-routing.git
cd belief-space-perception-routing

python3.12 -m venv .venv        # or python3.11
source .venv/bin/activate
pip install --upgrade pip

# IMPORTANT: on Mac use the DEFAULT torch wheel (ships Metal/MPS).
# Do NOT use the CUDA --index-url that requirements.txt mentions for NVIDIA.
pip install torch torchvision
pip install -r requirements.txt

# confirm the Metal GPU is visible (expect: mps available: True)
python -c "import torch; print('mps available:', torch.backends.mps.is_available())"
```
`config.yaml` already ships `device.mode: mps`, so the pipeline auto-uses the Metal GPU.
(Force it with `DEVICE=mps` or `DEVICE=cpu` in a `.env` file if ever needed.)

---

## 2. Track A — TartanDrive (off-road, normal lens)
```bash
python src/download_data.py                 # ~32 GB bag chunk (resumable)
python src/extract_frames.py                # bag -> PNG frames + manifest
OUTPUTS_DIR=outputs/trackA python src/run_pipeline.py --skip-extract
```
- Results land in `outputs/trackA/` (RQ-H figure: `outputs/trackA/phase5/rqh_centerpiece.png`).
- This runs the FULL trajectory (~1,198 frames) — more fault onsets than the committed
  309-frame dev slice, so it is a stronger Track A point.
- Profiling all frames is heavy; if you want to bound it, set `profiling.n_frames` in
  `config.yaml` (e.g. 600) or leave it for the full run.

---

## 3. Track D — RADIATE (rain / snow / fog / night)

**Get the data first.** You need the 4 sequence folders (each containing `zed_left/`).
Either get them from Vihan (the 4 zips: `rain_4_0`, `snow_1_0`, `fog_6_0`, `night_1_0`),
or register at https://pro.hw.ac.uk/radiate/downloads/ . Unzip each so you have:
```
data/radiate/rain_4_0/zed_left/000001.png ...
data/radiate/snow_1_0/zed_left/...
data/radiate/fog_6_0/zed_left/...
data/radiate/night_1_0/zed_left/...
```

**Run all four conditions:**
```bash
for SEQ in rain_4_0 snow_1_0 fog_6_0 night_1_0; do
  python src/extract_radiate.py --seq-dir data/radiate/$SEQ
  FRAMES_DIR=data/frames/radiate_$SEQ OUTPUTS_DIR=outputs/trackD_$SEQ \
    python src/run_pipeline.py --skip-extract
done
```
- Each condition writes `outputs/trackD_<seq>/` (its own RQ-H/RQ-A1/RQ-A2).
- The Phase 1 log prints the per-condition fault-onset count — note it next to each result.
- `extract_radiate.py` rectifies the left ZED camera to `camera_left_rect` (matching the
  RADIATE SDK); radar/lidar/annotations are unused, pseudo-GT is YOLO11x, same as every track.

**If a condition is slow** (Phase 1's optical flow is CPU-bound on Mac too), cap frames —
this is the same option used for the committed rain result (1,500 frames, still significant):
```bash
python src/extract_radiate.py --seq-dir data/radiate/$SEQ --max-frames 1500
```

---

## 4. Return the results
If you have push access (collaborator):
```bash
git add outputs/trackA outputs/trackD_*
git commit -m "Track A (full TartanDrive) and Track D (RADIATE rain/snow/fog/night) on Apple MPS"
git push
```
Otherwise zip `outputs/` and send it to Vihan.

---

## 5. Mac / MPS specifics (handled automatically, listed for transparency)
- **Precision axis is CUDA-only.** On MPS the configs run fp32 (resolution still varies),
  because MPS fp16 inference is unreliable for this study. Disclosed in the write-up.
- **Contention runs as a separate process on MPS** (Metal command buffers are not
  thread-safe), so the co-running GPU workload is launched as a child process. Automatic.
- After Phase 1, check `outputs/<track>/phase1/contention_summary.json` -> `p95_shift_ratio`
  (want roughly 2-3x). If it is far off, adjust `contention.gpu.competitor_sleep_s` in
  `config.yaml` (smaller = more contention) and re-run.
- Latencies are your Mac's; the RQ-H comparison is the reportable result, not the absolute ms.

---

## 6. Track C — WoodScape Soiling (fisheye, lens contamination)

Track C is the cross-optics generalization track: real lens-soiling degradation on a
fisheye camera. It is done the rigorous way (Option B): the fisheye image is undistorted
to a rectilinear projection (Valeo's own camera model) before YOLO; **occlusion is real
ground truth from the soiling masks** (not optical flow); **blur is a polar-coordinate
metric** (so radial distortion does not masquerade as blur); illumination is histogram
entropy. Pseudo-GT is YOLO11x on the undistorted frames, same as every track.

### 6a. Get the data (download only the soiling subset, a few GB, < 10 GB)

You do NOT need the full ~29 GB WoodScape. You need only the **soiling subset**: the
public soiling dataset is **5,000 fisheye images (4,000 train + 1,000 test)** with
soiling masks (classes: clear / transparent / semi-transparent / opaque). That is
roughly **2-4 GB**, well under 10 GB.

Where: WoodScape is a public Google Drive folder (from the upstream repo's
`data/download.txt`):
```
https://drive.google.com/drive/folders/1X5JOMEfVlaXfdNy24P8VA-jMs0yzf_HR
```
In that folder, open the **soiling** item (a folder/zip named like `soiling_dataset`)
and download just that. Two ways:

- **Drive web UI (simplest on a Mac):** right-click the soiling folder -> Download,
  then unzip into `data/woodscape_soiling/`.
- **Script:** copy the soiling item's "Get link" URL and run
  ```bash
  pip install gdown
  python src/download_woodscape.py --url "<soiling Drive link>"
  ```

After this you should have, for example:
```
data/woodscape_soiling/train/rgbImages/*.png     (fisheye RGB)
data/woodscape_soiling/train/gtLabels/*.png      (soiling masks)
```
(The exact folder nesting varies by release; the extractor searches for `rgbImages/`
and `gtLabels/` and matches masks to images by filename, so any layout under the
soiling root works.)

### 6b. Run Track C

(Environment is the same one-time setup as Section 1.)
```bash
python src/run_woodscape.py --soiling-dir data/woodscape_soiling/train
# bound it if you like (frame collection, 7-8 frame bursts per scene):
python src/run_woodscape.py --soiling-dir data/woodscape_soiling/train --max-frames 1500
```
This runs extract -> phase1 (fisheye-correct channels + threshold-and-smooth labels) ->
phases 2-6 (RQ-H / RQ-A1 / RQ-A2), all namespaced to `outputs/trackC/`.

### 6c. Track C notes
- **Calibration:** undistortion defaults to Valeo's vendored example calibration. If
  your soiling release ships per-image calibration, pass `--calib <calib.json>` to
  `run_woodscape.py` for the exact soiling-camera model.
- **Temporal axis:** WoodScape soiling is a frame collection (7-8 consecutive frames per
  scene), so the fault timeline is the dataset ordering rather than a continuous video.
  Documented; printed at runtime.
- **Detections:** WoodScape is automotive (cars, pedestrians, cyclists), so YOLO/COCO
  detections are denser than off-road TartanDrive, giving a richer accuracy axis.
- MPS specifics (Section 5) apply unchanged.
- Return results: `git add outputs/trackC` and push, or zip and send.

---

## 7. Add-on analysis sections (run on tracks you ALREADY ran)

These three reuse a track's existing `outputs/<track>/` (and, for the extended frontier,
its frames). You do NOT re-run the whole pipeline. Replace `<track>` with the outputs dir
you used (e.g. `outputs/trackA`, `outputs/trackD_rain_4_0`, `outputs/trackC`) and
`<frames>` with that track's frame dir (e.g. `data/frames/turnpike_afternoon_fall_0`,
`data/frames/radiate_rain_4_0`, `data/frames/woodscape_soiling`).

### 7a. Learned routing baseline (fair MLP on the same belief features) -- CPU, fast
```bash
python src/run_learned_router.py --track <track>
```
Trains a small MLP on the per-frame oracle config labels (split by sequence, not frame),
then compares the joint belief policy vs the learned router on deadline-miss + accuracy
with 95% CIs over 5 seeds. Writes `outputs/<track>/extras/learned_router.{json,png}`.
Both outcomes are valid (joint wins -> structure matters; learned matches/wins -> the
value is in the features).

### 7b. RQ-H phase diagram (coupling x onset-rate sweep) -- CPU, fast
```bash
python src/run_injection_sweep.py --track <track>
```
Sweeps a (coupling-strength x fault-onset-rate) grid of CONTROLLED regimes over the
track's real latency/accuracy profile, recording the joint-vs-decoupled deadline-miss
reduction with CIs and achieved onset counts. Writes
`outputs/<track>/extras/phase_diagram.{json,png}`. The coupling=0 row is the control
(should be ~0); the benefit grows with coupling and onset rate.

### 7c. Extended dense frontier (9 configs) -- GPU, re-profiles
```bash
python src/run_extended_frontier.py --track-outputs <track> --frames <frames> --profile-n 1500
```
Re-profiles a dense 9-config Pareto set (adds yolo11x@960, yolo11m@1280/640, yolo11s@960/640)
reusing the track's labels, and re-runs routing, into `outputs/<track>_extended/`. The
original 4-config results are untouched. This is the only add-on that needs the GPU and
the frames. (On a Mac, profiling 9 configs is heavier; `--profile-n 1500` bounds it.)

Shortcut: run 7a + 7b on several tracks at once (defaults to the strong tracks fog+night):
```bash
python src/run_addons.py                                  # fog + night
python src/run_addons.py --tracks outputs/trackD_rain_4_0 outputs/trackC
```

### 7d. Robustness add-ons (threshold rule, on-time utility, bootstrap, sweeps) -- CPU, fast
```bash
python src/run_robustness.py --track <track>            # threshold + utility + bootstrap
python src/run_robustness.py --track outputs/trackD_fog_6_0 --sweeps   # + deadline & kappa sweeps
```
Reuses the track's `phase1` + `phase3` (no re-profiling) and reproduces the exact RQ-H
substrate (same seeds/draws as Phase 5), then computes on the same draws: the threshold-rule
baseline (`p_cont=max(b_c,1[b_f>0.5])`) vs joint vs decoupled deadline-miss with paired CIs;
on-time utility `U = mean_t a_t * 1[lat_t <= D]` (joint vs decoupled); and a moving-block
bootstrap (block 50, B 2000, pooled across seeds) of the per-frame miss reduction. With
`--sweeps` (auto-on for fog) it also runs the deadline-strictness sweep (0.8-1.2x) and the
kappa sweep (0, 0.5, 0.75, 1.0). Writes `outputs/<track>/extras/robustness.json` and prints a
summary (the printed RQ-H line is a sanity check: it must equal the committed Phase 5 number).

Return results: `git add outputs/<track>/extras outputs/<track>_extended` and push, or zip
and send.
