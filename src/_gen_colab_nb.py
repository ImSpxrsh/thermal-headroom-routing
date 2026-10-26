"""Generate run_on_colab.ipynb: the self-contained notebook that produces the
REPORTABLE GPU profiling and latency numbers (Colab T4/A100). Latency on a laptop
CPU/weak GPU is not reportable; the precision (fp16) axis is only meaningful on a
real GPU, and YOLO11x@1280 needs GPU headroom. The notebook implements
GPU-appropriate contention (co-running GPU workload + host-to-device pressure), the
same as src/contention.py.
"""
import json
from pathlib import Path

MD = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
CODE = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = [
    MD("""# Belief-Space Perception Routing: reportable GPU run (Colab)

**This notebook is for Google Colab (NVIDIA GPU) only.** To run on a **Mac (Apple
Silicon / MPS)** or **Windows/NVIDIA**, do not use this notebook -- follow the
"Running it" section of `README.md`, or `RUNNING.md`, in the repo (per-platform steps).

Run this on a Colab **GPU** runtime (Runtime -> Change runtime type -> T4 or A100).
It produces the reportable profiling, latency, and experiment numbers. Locally
measured laptop latencies are dev-only and are never the reported result.

What it does:
1. installs dependencies and clones the project repo,
2. downloads one TartanDrive 2.0 bag chunk and extracts the camera frames,
3. runs Phase 1 (labels), Phase 3 (frontier + deadline), and Phases 4-6
   (RQ-H, RQ-A1, RQ-A2) with GPU-appropriate contention,
4. zips `outputs/` so you can download the reportable plots and tables.

Disclose the exact environment (GPU model from the first cell) in the write-up.
"""),
    MD("## 1. Environment"),
    CODE("!nvidia-smi"),
    CODE("""
# Dependencies. torch/cuda is preinstalled on Colab GPU runtimes.
!pip -q install "ultralytics>=8.3.0" opencv-python-headless rosbags filterpy python-dotenv pyyaml
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""),
    MD("""## 2. Get the code

The repo is public, so this just clones it (no token needed). If the repo is ever made
private again, the cell will prompt for a GitHub token (repo scope) automatically. Or set
`USE_UPLOAD = True` to upload a `src.zip` of the `src/` folder + `config.yaml` instead.
"""),
    CODE("""
import os, getpass, subprocess
USE_UPLOAD = False
REPO = "VihanAggarwal/belief-space-perception-routing"
if USE_UPLOAD:
    from google.colab import files
    up = files.upload()  # upload src.zip and config.yaml
    os.makedirs("/content/proj", exist_ok=True); os.chdir("/content/proj")
    !unzip -o src.zip -d /content/proj
else:
    # try a public clone first; fall back to a token only if it's private
    r = subprocess.run(["git", "clone", "--depth", "1",
                        f"https://github.com/{REPO}.git", "/content/proj"])
    if r.returncode != 0:
        tok = getpass.getpass("Repo is private -- paste a GitHub token (repo scope): ").strip()
        subprocess.run(["git", "clone", "--depth", "1",
                        f"https://{tok}@github.com/{REPO}.git", "/content/proj"], check=True)
    os.chdir("/content/proj")
print("cwd", os.getcwd()); print(os.listdir("."))
"""),
    MD("""## 3. Download a TartanDrive 2.0 bag chunk and extract frames

~32 GB download; Colab disk is ~100 GB and the link is fast (a few minutes). Set
`EXTRACT_MAX_FRAMES` to cap frames (0 = all ~1198 in the chunk).
"""),
    CODE("""
import os
os.makedirs("data/raw", exist_ok=True)
SWIFT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
BAG = f"{SWIFT}/tartandrive2/bags/2023-10-26-14-42-35_turnpike_afternoon_fall/turnpike_afternoon_fall_0.bag"
DEST = "data/raw/turnpike_afternoon_fall_0.bag"
if not os.path.exists(DEST) or os.path.getsize(DEST) < 32_000_000_000:
    !curl -k -C - --retry 8 --retry-all-errors -o "$DEST" "$BAG"
print("bag GB:", os.path.getsize(DEST)/1e9)
"""),
    CODE("""
os.environ["EXTRACT_MAX_FRAMES"] = "0"   # 0 = all frames in the chunk
!python src/extract_frames.py
"""),
    MD("## 4. Phase 1 labels, Phase 3 frontier + deadline (reportable)"),
    CODE("""
# Phase 1 produces the Track A fault labels used for per-fault accuracy buckets.
!python src/phase1_run.py
# Phase 3 profiles the full frontier on the GPU and sets the deadline. These latency
# numbers ARE reportable (real GPU, real fp16 axis, GPU-appropriate contention).
!python src/profile_frontier.py
"""),
    MD("## 5. Experiments: RQ-H (centerpiece), RQ-A1, RQ-A2 (reportable)"),
    CODE("""
!python src/phase4_run.py
!python src/phase5_run.py
!python src/phase6_run.py
"""),
    MD("## 6. Collect reportable outputs"),
    CODE("""
import shutil
shutil.make_archive("/content/outputs_colab", "zip", "outputs")
from google.colab import files
files.download("/content/outputs_colab.zip")
print("Download started: outputs_colab.zip")
"""),
    MD("""## 7. Track D: RADIATE real adverse weather

The full RADIATE dataset is ~112 GB (radar + lidar + camera over all sequences). You do
NOT need that: this uses the **camera (`zed_left`) of fault-dense sequences** (rain, snow,
fog, night), run on Colab so your laptop never holds it.

Three ways to point at one sequence, in priority order:
1. **Already on Google Drive (recommended if you downloaded there):** run
   `from google.colab import drive; drive.mount('/content/drive')`, then set
   `RADIATE_LOCAL_DIR` to the sequence path, e.g.
   `/content/drive/MyDrive/radiate/rain_4_0` (the folder that contains `zed_left/`).
2. **A direct link:** set `RADIATE_SEQ_URL` to a Dropbox (`?dl=1`) or Google Drive zip link.
3. **Upload:** set `USE_UPLOAD_RADIATE = True` to pick a zip from your machine.

Run section 7 once PER condition (change `SEQ_NAME` + the path/link), then send back each
`outputs_radiate.zip` / `outputs/`."""),
    CODE("""
import os, subprocess, glob, zipfile
RADIATE_LOCAL_DIR = ""           # <-- if already on Drive/Colab disk, set the sequence dir here (skips download)
RADIATE_SEQ_URL   = ""           # or a direct Dropbox(?dl=1)/Google-Drive zip link
USE_UPLOAD_RADIATE = False       # or True to upload a zip via the file picker
SEQ_NAME = "rain_suburban"       # label for this sequence/condition
RADIATE_MAX_FRAMES = 6000        # 0 = all; ~6000 covers many fault onsets and is plenty to power RQ-H

if RADIATE_LOCAL_DIR:
    seq_root = RADIATE_LOCAL_DIR
else:
    os.makedirs("data/radiate", exist_ok=True)
    zip_path = f"data/radiate/{SEQ_NAME}.zip"
    if USE_UPLOAD_RADIATE:
        from google.colab import files
        up = files.upload(); zip_path = "data/radiate/" + list(up.keys())[0]
    elif RADIATE_SEQ_URL:
        if "drive.google" in RADIATE_SEQ_URL:
            subprocess.run(["pip", "-q", "install", "gdown"], check=True)
            import gdown; gdown.download(RADIATE_SEQ_URL, zip_path, quiet=False, fuzzy=True)
        else:
            subprocess.run(["wget", "-q", "-O", zip_path, RADIATE_SEQ_URL.replace("?dl=0", "?dl=1")], check=True)
    else:
        raise SystemExit("Set RADIATE_LOCAL_DIR or RADIATE_SEQ_URL or USE_UPLOAD_RADIATE first.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(f"data/radiate/{SEQ_NAME}")
    seq_root = f"data/radiate/{SEQ_NAME}"

# locate the sequence dir: prefer one containing zed_left/, else a folder of PNGs
zl = glob.glob(f"{seq_root}/**/zed_left", recursive=True)
if zl:
    seq_dir = os.path.dirname(zl[0])
else:
    pngs = glob.glob(f"{seq_root}/**/*.png", recursive=True)
    seq_dir = os.path.dirname(pngs[0]) if pngs else seq_root
print("sequence dir:", seq_dir, "| n PNGs nearby:",
      len(glob.glob(os.path.join(seq_dir, "**", "*.png"), recursive=True)))
"""),
    CODE("""
# Extract rectified left frames, then run the FULL pipeline on Track D.
# (seq_dir and RADIATE_MAX_FRAMES are Python globals from the previous cell; IPython
# interpolates {var} into ! lines. FRAMES_DIR is passed to the subprocess via os.environ.)
import os, shutil
!python src/extract_radiate.py --seq-dir "{seq_dir}" --max-frames {RADIATE_MAX_FRAMES}
frames_dir = "data/frames/radiate_" + os.path.basename(seq_dir.rstrip("/"))
os.environ["FRAMES_DIR"] = frames_dir
os.environ["PROFILE_N"] = "0"     # profile ALL extracted frames so RQ-H sees many fault onsets
print("running pipeline on", frames_dir)
!python src/run_pipeline.py --skip-extract
shutil.make_archive("/content/outputs_radiate", "zip", "outputs")
from google.colab import files; files.download("/content/outputs_radiate.zip")
print("Track D done. Report achieved fault-onset counts (printed in Phase 1) per condition.")
"""),
    MD("""## 8. Track D BATCH: run all RADIATE weather conditions on Colab Pro

Turnkey loop over your RADIATE sequence zips on Google Drive. For each: extract only the
camera (`zed_left`), rectify, run the full pipeline (RQ-H/RQ-A1/RQ-A2) with its own
namespaced `outputs/trackD_<name>/`, then free the space. Finally bundle every
condition's outputs into one download.

First put your `Data UTRC` zips on Google Drive (e.g. `MyDrive/Data UTRC/`). The default
list is ONE sequence per weather condition (the scientific unit); flip `RUN_ALL_ZIPS` to
True to run every sequence (many hours). `MAX_FRAMES_PER_SEQ` bounds each run."""),
    CODE("""
from google.colab import drive; drive.mount('/content/drive')
RADIATE_ZIP_DIR = "/content/drive/MyDrive/Data UTRC"   # folder with rain_4_0.zip, snow_1_0.zip, ...
RUN_ALL_ZIPS = False            # False = one representative sequence per weather condition
MAX_FRAMES_PER_SEQ = 1500       # Phase 1 is CPU-bound on Colab; 1500 is plenty (the sig. rain result used 1500)
DEFAULT_ONE_PER_CONDITION = ["rain_4_0", "snow_1_0", "fog_6_0", "night_1_0"]

import os, glob, zipfile, shutil, subprocess
all_zips = sorted(glob.glob(os.path.join(RADIATE_ZIP_DIR, "*.zip")))
all_zips = [z for z in all_zips if "tiny_foggy" not in os.path.basename(z)]
if RUN_ALL_ZIPS:
    zips = all_zips
else:
    zips = [z for z in all_zips if os.path.splitext(os.path.basename(z))[0] in DEFAULT_ONE_PER_CONDITION]
print("will run:", [os.path.basename(z) for z in zips])

for z in zips:
    name = os.path.splitext(os.path.basename(z))[0]
    dst = f"data/radiate/{name}"
    if not os.path.isdir(os.path.join(dst, "zed_left")):
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(z) as zf:   # camera only, to save Colab disk
            members = [m for m in zf.namelist() if m.startswith("zed_left/") or m == "zed_left.txt"]
            zf.extractall(dst, members or None)
    # stream child output live (no silent gaps); skip the Track B injection pass
    # (validation-only, a 2nd CPU optical-flow pass) to roughly halve Phase 1 on Colab CPUs
    live = dict(os.environ, PYTHONUNBUFFERED="1", SKIP_TRACK_B="1")
    subprocess.run(["python", "-u", "src/extract_radiate.py", "--seq-dir", dst,
                    "--max-frames", str(MAX_FRAMES_PER_SEQ)], check=False, env=live)
    frames = f"data/frames/radiate_{name}"
    env = dict(live, FRAMES_DIR=frames, OUTPUTS_DIR=f"outputs/trackD_{name}", PROFILE_N="0")
    print(f"=== running pipeline: {name} ===", flush=True)
    subprocess.run(["python", "-u", "src/run_pipeline.py", "--skip-extract"], env=env, check=False)
    shutil.rmtree(dst, ignore_errors=True)        # free the extracted camera
    shutil.rmtree(frames, ignore_errors=True)     # free the rectified frames (outputs are kept)

shutil.make_archive("/content/outputs_radiate_all", "zip", "outputs")
from google.colab import files; files.download("/content/outputs_radiate_all.zip")
print("Done. outputs/trackD_<condition>/ for each; send back outputs_radiate_all.zip.")
"""),
    MD("""## Notes for the write-up
- Report the GPU model printed in cell 1 (e.g., Tesla T4 16 GB / A100 40 GB).
- All latency numbers in `outputs/phase3/` and the deadline in `config.yaml`
  (`value_ms`) come from THIS environment, not the laptop.
- Contention here is GPU-appropriate (co-running GPU workload + host-to-device
  pressure), per `src/contention.py::GPUContention`.
- Pseudo-GT note: accuracy is agreement with the C1 reference, not external truth.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
      "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent.parent / "run_on_colab.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
