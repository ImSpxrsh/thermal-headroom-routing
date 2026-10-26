"""Frame-iteration harness over the extracted TartanDrive camera stream.

Yields (frame_idx, timestamp_ns, image_bgr). Reads the manifest written by
extract_frames.py so timestamps are preserved for time-aligned latency later.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from config_util import load_config, abspath


@dataclass
class Frame:
    idx: int
    timestamp_ns: int
    image: np.ndarray  # HxWx3 BGR uint8


class FrameSource:
    def __init__(self, frames_dir: Optional[str] = None, cfg: Optional[dict] = None):
        self.cfg = cfg or load_config()
        self.dir = abspath(frames_dir or self.cfg["paths"]["frames_dir"])
        self.manifest = self.dir / "manifest.csv"
        if not self.manifest.exists():
            raise FileNotFoundError(
                f"No manifest at {self.manifest}. Run src/extract_frames.py first."
            )
        self.records = []
        with open(self.manifest, newline="") as f:
            for row in csv.DictReader(f):
                self.records.append(
                    (int(row["frame_idx"]), int(row["timestamp_ns"]), row["filename"])
                )

    def __len__(self) -> int:
        return len(self.records)

    def iter(self, start: int = 0, stop: Optional[int] = None) -> Iterator[Frame]:
        import cv2
        stop = stop if stop is not None else len(self.records)
        for idx, ts, fn in self.records[start:stop]:
            img = cv2.imread(str(self.dir / fn), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Failed to read frame {fn}")
            yield Frame(idx=idx, timestamp_ns=ts, image=img)

    def timestamps_ns(self) -> np.ndarray:
        return np.array([ts for _, ts, _ in self.records], dtype=np.int64)


if __name__ == "__main__":
    src = FrameSource()
    print(f"{len(src)} frames in {src.dir}")
    first = next(src.iter(stop=1))
    print(f"first frame idx={first.idx} ts={first.timestamp_ns} shape={first.image.shape}")
