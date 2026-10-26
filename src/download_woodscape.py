"""Download ONLY the WoodScape Soiling subset (Track C). Not the full ~29 GB WoodScape.

WoodScape lives in a public Google Drive folder (see the repo's upstream `data/download.txt`):
  https://drive.google.com/drive/folders/1X5JOMEfVlaXfdNy24P8VA-jMs0yzf_HR
You want the SOILING subset only: the public soiling dataset is 5,000 fisheye images
(4,000 train + 1,000 test) with per-pixel soiling masks (classes: clear / transparent /
semi-transparent / opaque). That is a few GB, well under 10 GB.

Steps:
  1. Open the Drive folder above, find the soiling item (a folder or zip named like
     `soiling_dataset`), right-click -> "Copy link" (or "Get link").
  2. Pass that link here:
       python src/download_woodscape.py --url "<soiling Drive link>"
     (a folder link downloads recursively; a zip link downloads the zip, which you then
     unzip into data/woodscape_soiling/).

On a Mac you can equally just download the soiling folder from the Drive web UI and drop
it under data/woodscape_soiling/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import abspath

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1X5JOMEfVlaXfdNy24P8VA-jMs0yzf_HR"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Google Drive link to the SOILING folder or zip")
    ap.add_argument("--out", default="data/woodscape_soiling")
    args = ap.parse_args()

    if not args.url:
        print("No --url given.\nOpen the WoodScape Drive folder and copy the SOILING link:")
        print(" ", DRIVE_FOLDER)
        print("Then: python src/download_woodscape.py --url \"<soiling link>\"")
        return 1

    try:
        import gdown
    except ImportError:
        print("Install gdown first:  pip install gdown", file=sys.stderr)
        return 2
    # use certifi's CA bundle (fixes 'unable to get local issuer certificate' on some setups)
    try:
        import certifi
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass

    out = abspath(args.out); out.mkdir(parents=True, exist_ok=True)
    url = args.url
    if "/folders/" in url:
        print(f"Downloading soiling FOLDER -> {out}")
        gdown.download_folder(url, output=str(out), quiet=False, use_cookies=False)
    else:
        dest = str(out / "soiling_dataset.zip")
        print(f"Downloading soiling ZIP -> {dest}")
        gdown.download(url, dest, quiet=False, fuzzy=True)
        print(f"Now unzip: (cd {out} && unzip -o soiling_dataset.zip)")
    print("Then run:  python src/run_woodscape.py --soiling-dir data/woodscape_soiling/<...>/train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
