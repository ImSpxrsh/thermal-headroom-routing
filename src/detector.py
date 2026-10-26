"""YOLO11 detector wrapper for the four frontier configs (model x resolution x
precision). Used by the Phase 0 sanity check and Phase 3 profiling. Weights are
pretrained Ultralytics YOLO11; no training from scratch (the frontier is systems
tradeoffs, not model novelty).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray   # (N,4)
    cls: np.ndarray    # (N,)
    conf: np.ndarray   # (N,)


@dataclass
class Detector:
    name: str
    model_name: str
    imgsz: int
    half: bool
    device: str
    _model: object = field(default=None, repr=False)

    def load(self):
        from ultralytics import YOLO
        weight = f"{self.model_name}.pt"
        self._model = YOLO(weight)
        self._model.to(self.device)
        return self

    def predict(self, image: np.ndarray, conf: float = 0.25) -> Detection:
        use_half = self.half and self.device == "cuda"
        r = self._model.predict(
            image, imgsz=self.imgsz, half=use_half, device=self.device,
            conf=conf, verbose=False,
        )[0]
        b = r.boxes
        if b is None or b.shape[0] == 0:
            return Detection(np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,)))
        return Detection(
            b.xyxy.cpu().numpy(),
            b.cls.cpu().numpy().astype(int),
            b.conf.cpu().numpy(),
        )

    def warmup(self, image: np.ndarray, n: int = 3):
        for _ in range(n):
            self.predict(image)
        return self


def build_detector(cfg: dict, key: str, device: str, force_fp32: bool = False) -> Detector:
    c = cfg["configs"][key]
    half = bool(c.get("half", False)) and (device == "cuda") and (not force_fp32)
    return Detector(name=key, model_name=c["model"], imgsz=int(c["imgsz"]),
                    half=half, device=device)


def all_detectors(cfg: dict, device: str) -> List[Detector]:
    return [build_detector(cfg, k, device) for k in cfg["configs"].keys()]
