"""Download a TartanDrive 2.0 bag chunk from CMU AirLab's public store.

The dataset is NOT in the repo (chunks are ~32 GB). This script fetches one (or
more) trajectory bag chunks so the pipeline can run. Chunks are uncompressed ROS1
bags; pair this with `extract_frames.py` (full bag) or `extract_frames_streaming.py`
(works on a partial download).

Cross-platform (macOS / Colab / Windows). Supports HTTP Range resume, so a dropped
or partial download continues where it left off.

    python src/download_data.py                       # default chunk used in the paper
    python src/download_data.py --chunks 0 1 2        # several chunks of the trajectory
    python src/download_data.py --trajectory 2023-10-26-15-32-25_turnpike_warehouse_fall --chunks 0

The CMU endpoint serves a certificate that the default trust store may reject, so we
disable TLS verification for this public, integrity-checked (Content-Length) data
download only. Override the endpoint/bucket via .env (see .env.example).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath

DEFAULT_ENDPOINT = ("https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/"
                    "AUTH_ac8533a83cff4d48bc8c608ad222d330")
DEFAULT_BUCKET = "tartandrive2"


def download(url: str, dest: Path, chunk_bytes: int = 8 * 1024 * 1024) -> bool:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    head = requests.head(url, verify=False, timeout=60)
    total = int(head.headers.get("Content-Length", 0))
    if total and have >= total:
        print(f"  already complete: {dest.name} ({have/1e9:.2f} GB)")
        return True
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    print(f"  downloading {dest.name}: have {have/1e9:.2f} GB of "
          f"{total/1e9:.2f} GB, resuming from byte {have}")
    with requests.get(url, headers=headers, stream=True, verify=False, timeout=120) as r:
        r.raise_for_status()
        with open(dest, mode) as f:
            done = have
            for block in r.iter_content(chunk_size=chunk_bytes):
                if not block:
                    continue
                f.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {done/1e9:.2f}/{total/1e9:.2f} GB ({pct:.1f}%)",
                          end="", flush=True)
        print()
    ok = (not total) or dest.stat().st_size >= total
    print(f"  {'done' if ok else 'INCOMPLETE (re-run to resume)'}: {dest.name}")
    return ok


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default=cfg["dataset"]["trajectory"])
    ap.add_argument("--chunks", nargs="*", type=int, default=[0],
                    help="chunk indices to download (each ~32 GB)")
    ap.add_argument("--out", default=None, help="output dir (default data/raw)")
    args = ap.parse_args()

    endpoint = os.getenv("TARTAN_SWIFT_ENDPOINT", DEFAULT_ENDPOINT)
    bucket = os.getenv("TARTAN_BUCKET", DEFAULT_BUCKET)
    out_dir = abspath(args.out) if args.out else abspath(cfg["paths"]["data_root"]) / "raw"

    # short trajectory name (drop the leading timestamp) for the per-chunk filename
    short = args.trajectory.split("_", 1)[1] if "_" in args.trajectory else args.trajectory
    print(f"Endpoint: {endpoint}\nBucket: {bucket}\nTrajectory: {args.trajectory}")
    print("WARNING: each chunk is ~32 GB. TLS verification is disabled for this public "
          "CMU endpoint; the download is integrity-checked against Content-Length.")
    ok_all = True
    for c in args.chunks:
        key = f"bags/{args.trajectory}/{short}_{c}.bag"
        url = f"{endpoint}/{bucket}/{key}"
        dest = out_dir / f"{short}_{c}.bag"
        ok_all &= download(url, dest)
    print("\nNext: python src/extract_frames.py   (or extract_frames_streaming.py for a partial download)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
