"""Track C (WoodScape Soiling, fisheye) — fisheye-correct degradation channels.

LOCAL/UNCOMMITTED build. Option B from the patch, done rigorously:
  - Detector path: undistort the fisheye image to a rectilinear (cylindrical) projection
    using Valeo's own camera model (vendored _woodscape_projection.py), BEFORE YOLO. So
    pseudo-GT (YOLO11x agreement) is computed on processed-but-rectilinear frames.
  - Occlusion channel: GROUND TRUTH from the soiling masks (fraction of soiled pixels),
    NOT optical flow (which is invalid on raw fisheye).
  - Blur channel: variance-of-Laplacian in POLAR coordinates (warpPolar about the image
    center), so radial distortion does not masquerade as blur.
  - Illumination channel: histogram entropy (distortion-robust), reused as-is.

Nothing here is committed; it depends on the gated WoodScape Soiling dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import _woodscape_projection as wp
from scipy.spatial.transform import Rotation as SciRot

HERE = Path(__file__).resolve().parent
DEFAULT_CALIB = HERE / "_woodscape_default_calib.json"


# ---------------------------------------------------------------------------
# Undistortion (fisheye -> centered cylindrical), via Valeo's model
# ---------------------------------------------------------------------------
def _make_cylindrical_cam(cam: "wp.Camera") -> "wp.Camera":
    """Centered-horizon cylindrical target camera (verbatim logic from Valeo example.py)."""
    assert isinstance(cam.lens, wp.RadialPolyCamProjection)
    lens = wp.CylindricalProjection(cam.lens.coefficients[0])
    rot_zxz = SciRot.from_matrix(cam.rotation).as_euler("zxz")
    rot_zxz = np.round(rot_zxz / (np.pi / 2)) * (np.pi / 2)
    rot_zxz[1] = np.pi / 2
    return wp.Camera(
        rotation=SciRot.from_euler(angles=rot_zxz, seq="zxz").as_matrix(),
        translation=cam.translation, lens=lens,
        size=cam.size, principle_point=(cam.cx_offset, cam.cy_offset),
        aspect_ratio=cam.aspect_ratio,
    )


def build_undistort_maps(calib_path: str = None):
    """Return (map1, map2, out_size) for cv2.remap fisheye -> rectilinear(cylindrical)."""
    import cv2
    cam = wp.read_cam_from_json(str(calib_path or DEFAULT_CALIB))
    dest = _make_cylindrical_cam(cam)
    map1, map2 = wp.create_img_projection_maps(cam, dest)
    return map1, map2, (int(dest.width), int(dest.height))


def undistort(img: np.ndarray, map1, map2) -> np.ndarray:
    import cv2
    return cv2.remap(img, map1, map2, cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# Fisheye-correct degradation observations
# ---------------------------------------------------------------------------
def polar_blur(gray: np.ndarray) -> float:
    """Variance of the Laplacian computed in POLAR coordinates about the image center.
    Unwrapping radially means the radial fisheye distortion does not inflate the Laplacian
    response, so this measures genuine blur rather than distortion."""
    import cv2
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    max_r = float(np.hypot(w, h) / 2.0)
    polar = cv2.warpPolar(gray, (w, h), center, max_r, cv2.WARP_POLAR_LINEAR)
    return float(cv2.Laplacian(polar, cv2.CV_64F).var())


def soiled_fraction(mask: np.ndarray) -> float:
    """Fraction of soiled pixels from a WoodScape soiling mask. Clean == 0; any non-zero
    class (transparent / semi-transparent / opaque, or any non-zero RGB) counts as soiled."""
    if mask.ndim == 3:
        soiled = np.any(mask > 0, axis=2)
    else:
        soiled = mask > 0
    return float(soiled.mean())


def observations_for_frame(fisheye_bgr: np.ndarray, mask: np.ndarray) -> dict:
    """Track-C per-frame observations on the RAW fisheye image + soiling mask.
    Returns the same channel names the pipeline expects (blur/illumination/occlusion),
    with occlusion = track-survival-equivalent = 1 - soiled_fraction (high = clean = good,
    matching the 'low is bad' direction used for the other tracks)."""
    import cv2
    import degradation as deg  # reuse illumination_entropy
    gray = cv2.cvtColor(fisheye_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "blur": polar_blur(gray),
        "illumination": deg.illumination_entropy(gray),
        "occlusion": 1.0 - soiled_fraction(mask),
    }
