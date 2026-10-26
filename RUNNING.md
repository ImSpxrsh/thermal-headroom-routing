# Running / Reproducing

**Note on "training".** This project does NOT train a detector from scratch. By
design (the contribution is the belief-space routing system, not model novelty) it
uses **pretrained Ultralytics YOLO11** weights, downloaded automatically on first use.
"Running" the project means: fit the per-channel sensor-fault belief HMMs and the
compute-contention HMM from data, profile the YOLO config frontier, set the deadline,
and run the RQ-H / RQ-A1 / RQ-A2 experiments. The two supported environments below are
**Google Colab (NVIDIA GPU)** and **Mac (Apple Silicon GPU via Metal/MPS)**.

Latency numbers are environment-specific. Colab on a real NVIDIA GPU gives the
reportable numbers; a Mac MPS run is a valid second environment but its latencies are
its own (disclose whichever you use).

---

## 0. Get the dataset (required, not in the repo)

The TartanDrive 2.0 bag chunks are ~32 GB each and live on CMU AirLab's public store,
so they are not committed. Download one chunk with the included script:

```bash
python src/download_data.py                 # default chunk (turnpike_afternoon_fall_0)
# or several chunks of a trajectory:
python src/download_data.py --chunks 0 1 2
```

Then turn the bag into frames:

```bash
python src/extract_frames.py                # full, indexed bag (recommended)
# If your download was interrupted, you can still extract what arrived:
python src/extract_frames_streaming.py      # works on a partial bag (uncompressed chunks)
```

Both write PNG frames + a `manifest.csv` to `data/frames/<trajectory>/` (gitignored).
Cap the number of frames with `EXTRACT_MAX_FRAMES=400 python src/extract_frames.py`.

---

## 1. Google Colab (NVIDIA GPU) — recommended for reportable numbers

Open **`run_on_colab.ipynb`** in Colab and:

1. **Runtime -> Change runtime type -> T4 or A100 GPU**, then Connect.
2. Run the cells top to bottom. The notebook:
   - installs dependencies (torch is preinstalled on Colab GPU runtimes),
   - clones this **private** repo (paste a GitHub token with `repo` scope when
     prompted; create one at GitHub -> Settings -> Developer settings -> Personal
     access tokens. Or set `USE_UPLOAD = True` and upload a zip of `src/` + `config.yaml`),
   - downloads a bag chunk and extracts frames,
   - runs Phase 1 (labels), Phase 3 (frontier + deadline), and Phases 4-6
     (RQ-H, RQ-A1, RQ-A2) with GPU-appropriate contention,
   - zips `outputs/` for download.
3. Report the GPU model printed by the first cell in any write-up.

To profile the FULL trajectory (not a subset), set `profiling.n_frames: 0` in
`config.yaml` before the profiling cell. Send the resulting `outputs/` back and the
reportable latencies + full-data RQ results can be folded into `WRITEUP.md`.

---

## 2. Mac with Apple Silicon GPU (M1/M2/M3/M4/M5, Metal/MPS)

Requires macOS on Apple Silicon and Python 3.11+ (tested on Python 3.12 on an
Apple M5). `config.yaml` ships with `device.mode: mps` so the Metal GPU is used by
default on this machine.

```bash
# 1) clone (you have collaborator access)
git clone https://github.com/VihanAggarwal/belief-space-perception-routing.git
cd belief-space-perception-routing

# 2) environment (NOTE: on Mac use the DEFAULT torch wheel, which ships Metal/MPS;
#    do NOT use the CUDA --index-url that requirements.txt mentions for NVIDIA)
python3.12 -m venv .venv      # or python3.11
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision            # default macOS wheel includes MPS
pip install -r requirements.txt          # the rest (ultralytics, opencv, rosbags, ...)

# 3) confirm MPS is visible
python -c "import torch; print('mps available:', torch.backends.mps.is_available())"

# 4) data
python src/download_data.py
python src/extract_frames.py

# 5) run the whole pipeline (device auto-selects 'mps' on Apple Silicon)
python src/run_pipeline.py --skip-extract
# or phase by phase:
#   python src/phase1_run.py
#   python src/phase2_run.py
#   python src/profile_frontier.py
#   python src/phase4_run.py
#   python src/phase5_run.py
#   python src/phase6_run.py
```

Mac/MPS specifics (handled in code, stated here for transparency):
- `device.mode: auto` in `config.yaml` resolves to `mps` on Apple Silicon. Force it
  with `DEVICE=mps` (or `DEVICE=cpu`) in `.env` if needed.
- The precision (fp16) axis is treated as CUDA-only: configs run **fp32** on MPS (and
  CPU), because MPS fp16 inference is unreliable for this study. So C1/C2 (and C3/C4)
  differ by resolution, not precision, on a Mac. This is disclosed in the write-up.
- Contention is GPU-appropriate on MPS too (a co-running MPS workload + transfer
  pressure, synchronized via `torch.mps.synchronize`).

---

## 3. Tuning the contention generator for YOUR device (important)

The contention intensity is device-specific. After Phase 1, check
`outputs/phase1/contention_summary.json` -> `p95_shift_ratio`. You want roughly
**2x-3x** so the frontier is non-degenerate (the expensive config misses the deadline
under contention while cheaper ones meet it). If the shift is too weak or too strong,
adjust in `config.yaml` under `contention.gpu`:

- `competitor_sleep_s` (smaller = more contention),
- `competitor_matmul_dim` (larger = more contention).

Then re-run Phase 1 and Phase 3. This matters most on fast GPUs (a flat-out competitor
may over- or under-contend depending on the card). Defaults are tuned for a small
laptop GPU and will likely need a smaller `competitor_sleep_s` on a Colab A100.

---

## 4. What to expect

- `outputs/phase1/` contention shift + fault overlays + regimes
- `outputs/phase2/` belief-tracking plots
- `outputs/phase3/frontier_table.csv` + `frontier.png`, deadline written to `config.yaml`
- `outputs/phase5/rqh_centerpiece.png` (the headline RQ-H figure)
- `outputs/phase6/` RQ-A1 and RQ-A2 figures
- `REPORT.md` per-phase verdicts; `WRITEUP.md` results + limitations

---

## 5. Track D: RADIATE (real adverse weather) — getting the data and running it

RADIATE (Heriot-Watt, CC BY-NC-SA 4.0) is real rain/snow/fog/night driving video. It is
the fault-dense real-data regime that powers RQ-H (TartanDrive was too fault-sparse).
Radar/lidar/annotations are unused; we use only the left ZED camera, rectified to
`camera_left_rect`, with YOLO11x pseudo-GT (same as every track).

**The full sequences are gated** (registration + academic email + license). The repo
ships a **runnable public sample** (a short foggy clip) as a GitHub Release asset.

### Quick taste (public sample, no registration)
```bash
# download the sample attached to the repo's release
gh release download radiate-sample -D data/radiate            # -> data/radiate/tiny_foggy.zip
cd data/radiate && unzip -o tiny_foggy.zip && cd ../..
python src/extract_radiate.py --seq-dir data/radiate/tiny_foggy
FRAMES_DIR=data/frames/radiate_tiny_foggy python src/run_pipeline.py --skip-extract
```
(50 frames of uniform fog: this verifies the Track D plumbing; it is too short for a
powered RQ-H result.)

### Full fault-dense sequences (for the powered RQ-H / RQ-A1 / RQ-A2)
You must obtain these yourself — they cannot be auto-downloaded:
1. Go to https://pro.hw.ac.uk/radiate/downloads/ and **fill the registration form**
   with an organizational/academic email; accept the CC BY-NC-SA license.
2. Verify your email; you receive a **Dropbox invitation** with the sequences.
3. Download the fault-dense ones: **Rain (Suburban)**, **Snow (Suburban)**,
   **Night (Motorway)**, and a **Fog** sequence.
4. Unzip each into `data/radiate/<sequence_name>/` (so it contains `zed_left/`).
5. Extract + run per sequence:
   ```bash
   python src/extract_radiate.py --seq-dir data/radiate/<sequence_name>
   FRAMES_DIR=data/frames/radiate_<sequence_name> python src/run_pipeline.py --skip-extract
   ```
6. The pipeline writes RQ-H/RQ-A1/RQ-A2 to `outputs/`. Report the achieved fault-onset
   counts per condition (printed in Phase 1) next to the results.

Notes: RADIATE frames are 672x376; the reference config upscales to its imgsz (1280),
documented. The contention/coupling machinery is reused unchanged (coupling on Track D
is a designed modeling assumption, like the other tracks).

### Disk space: you do NOT need the full 112 GB
The full RADIATE dataset (~112 GB) is radar + lidar + camera over all sequences. This
project uses **camera only** (`zed_left`), and only **one or two fault-dense sequences**.
Practical options if local space is tight:
- **Best:** run Track D on Colab (section 7 of `run_on_colab.ipynb`) so the data lives on
  Colab's disk, not your laptop. Paste a link to one sequence and run.
- Download just the `zed_left/` subfolder of one rain/snow sequence from the Dropbox web
  UI (a few GB, not 112), since radar/lidar are the bulk and are never used.
- Cap frames with `--max-frames N` in `extract_radiate.py` (a few thousand frames still
  has many fault onsets, enough to power RQ-H).
