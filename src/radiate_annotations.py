"""Project RADIATE's real radar-cartesian object annotations onto the rectified left
camera image, replicating the projection method of the RADIATE SDK
(marcelsheeny/radiate_sdk, MIT license -- radiate.py `__get_projected_bbox` /
utils/calibration.py `Calibration.transform`), so we can validate the pseudo-GT
accuracy metric used everywhere else in this paper against REAL annotations rather
than agreement-with-C1.

RADIATE annotations are given as 2D rotated boxes in the radar Cartesian image (pixel
units, 1152x1152, 0.173611 m/px, centered at (576,576)) per OBJECT across the sequence's
RADAR frames -- a different, slower frame rate than the camera. We: (1) sync each camera
frame to its nearest radar annotation frame by timestamp, (2) lift each 2D radar-plane
box to a pseudo-3D box (ground footprint + 2m height, sensor at -1.7m per the SDK), (3)
transform radar->camera extrinsics (Euler-angle composition, exactly as the SDK does) and
project with the LEFT CAMERA'S RAW (distorted) intrinsics -- which is what the SDK itself
does even when drawing onto the rectified image, a known SDK quirk we replicate rather
than "fix", to stay faithful to the accepted community projection. Extrinsic values
(T, R) are the vendored SDK calibration for RADIATE's left ZED camera and radar
(radar is the vehicle-frame origin, T=R=[0,0,0]); the same rows already vendored in
config/radiate-calib.yaml provide the intrinsics.

    from radiate_annotations import RadiateProjector
    proj = RadiateProjector(seq_dir)
    boxes = proj.boxes_for_camera_frame(cam_frame_idx)   # list of (x1,y1,x2,y2,class_name)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

# --- vendored from radiate_sdk config/default-calib.yaml + config/config.yaml
# (MIT license, marcelsheeny/radiate_sdk); intrinsics duplicated here from
# config/radiate-calib.yaml for a self-contained module. ---
LEFT_CAM_T = np.array([0.34001, -0.06988923, 0.287893])       # meters, radar-frame origin
LEFT_CAM_R = np.array([1.278946, -0.530201, 0.000132])        # degrees, Euler XYZ
RADAR_T = np.array([0.0, 0.0, 0.0])
RADAR_R = np.array([0.0, 0.0, 0.0])
RANGE_RES = 0.173611          # m / radar-cartesian pixel
RANGE_CELLS = 576             # radar cartesian image is 2*RANGE_CELLS square, centered
MAX_RANGE_BBOX_CAMERA = 100.0  # meters
OBJ_HEIGHT_M = 2.0
SENSOR_HEIGHT_M = -1.7


def _RX(deg):
    t = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def _RY(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def _RZ(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def _transform(euler_deg, T):
    """Exact replica of radiate_sdk utils.calibration.Calibration.transform()."""
    Rx, Ry, Rz = _RX(euler_deg[0]), _RY(euler_deg[1]), _RZ(euler_deg[2])
    axis_swap = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    R = axis_swap @ Rx @ Ry @ Rz
    M = np.array([[R[0, 0], R[0, 1], R[0, 2], 0.0],
                  [R[1, 0], R[1, 1], R[1, 2], 0.0],
                  [R[2, 0], R[2, 1], R[2, 2], 0.0],
                  [T[0], T[1], T[2], 1.0]]).T
    return M


def radar_to_left_extrinsic() -> np.ndarray:
    euler = RADAR_R - LEFT_CAM_R
    T = RADAR_T - LEFT_CAM_T
    return _transform(euler, T)


def load_left_cam_matrix(calib_yaml: str) -> np.ndarray:
    c = yaml.safe_load(open(calib_yaml))["left_cam_calib"]
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])


def project_bbox_to_camera(bb_xywh, rotation_deg: float, cam_mat: np.ndarray,
                            extrinsic: np.ndarray) -> "np.ndarray | None":
    """Exact replica of radiate_sdk radiate.py `__get_projected_bbox`. bb_xywh is the
    radar-cartesian pixel box [x,y,w,h]; returns the projected 2D image polygon points
    (Nx2) or None if nothing projects in front of the camera within range."""
    rotation = np.deg2rad(-rotation_deg)
    cx, cy = bb_xywh[0] + bb_xywh[2] / 2, bb_xywh[1] + bb_xywh[3] / 2
    T = np.array([[cx], [cy]])
    pc = 0.2
    bb = [bb_xywh[0] + bb_xywh[2] * pc, bb_xywh[1] + bb_xywh[3] * pc,
          bb_xywh[2] - bb_xywh[2] * pc, bb_xywh[3] - bb_xywh[3] * pc]
    R = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    points = np.array([[bb[0], bb[1]], [bb[0] + bb[2], bb[1]],
                        [bb[0] + bb[2], bb[1] + bb[3]], [bb[0], bb[1] + bb[3]]]).T
    points = points - T
    points = (R @ points) + T
    points = points.T
    points[:, 0] = points[:, 0] - RANGE_CELLS
    points[:, 1] = RANGE_CELLS - points[:, 1]
    points = points * RANGE_RES
    points = np.append(points, np.full((points.shape[0], 1), SENSOR_HEIGHT_M), axis=1)
    base = points  # 4 ground-footprint corners
    top = base.copy(); top[:, 2] += OBJ_HEIGHT_M
    pts3d = np.vstack([base, top])

    pts_h = np.append(pts3d, np.ones((pts3d.shape[0], 1)), axis=1)
    cam_pts = pts_h @ extrinsic.T
    cam_pts = (cam_pts / cam_pts[:, 3, None])[:, :3]

    valid = (cam_pts[:, 2] > 0) & (cam_pts[:, 2] < MAX_RANGE_BBOX_CAMERA)
    cam_pts = cam_pts[valid]
    if len(cam_pts) == 0:
        return None
    fx, fy, cx_, cy_ = cam_mat[0, 0], cam_mat[1, 1], cam_mat[0, 2], cam_mat[1, 2]
    xIm = (fx * cam_pts[:, 0] / cam_pts[:, 2]) + cx_
    yIm = (fy * cam_pts[:, 1] / cam_pts[:, 2]) + cy_
    return np.column_stack([xIm, yIm])


def _parse_timestamps(path: Path) -> list:
    ts = []
    if path.exists():
        for line in path.read_text().splitlines():
            if "Time:" in line:
                try:
                    ts.append(float(line.split("Time:")[1].strip()))
                except Exception:
                    pass
    return ts


class RadiateProjector:
    def __init__(self, seq_dir: str, calib_yaml: str = "config/radiate-calib.yaml"):
        self.seq_dir = Path(seq_dir)
        self.cam_mat = load_left_cam_matrix(calib_yaml)
        self.extrinsic = radar_to_left_extrinsic()
        ann_path = self.seq_dir / "annotations" / "annotations.json"
        self.objects = json.load(open(ann_path)) if ann_path.exists() else []
        self.cam_ts = _parse_timestamps(self.seq_dir / "zed_left.txt")
        self.radar_ts = _parse_timestamps(self.seq_dir / "Navtech_Cartesian.txt")

    def camera_to_radar_idx(self, cam_idx: int, tol_s: float = 0.25) -> "int | None":
        if cam_idx >= len(self.cam_ts) or not self.radar_ts:
            return None
        t = self.cam_ts[cam_idx]
        radar_arr = np.asarray(self.radar_ts)
        j = int(np.argmin(np.abs(radar_arr - t)))
        return j if abs(radar_arr[j] - t) <= tol_s else None

    def boxes_for_radar_frame(self, radar_idx: int, img_w=672, img_h=376):
        out = []
        for obj in self.objects:
            if radar_idx >= len(obj["bboxes"]):
                continue
            b = obj["bboxes"][radar_idx]
            if not b or "position" not in b:
                continue
            poly = project_bbox_to_camera(b["position"], b.get("rotation", 0.0),
                                           self.cam_mat, self.extrinsic)
            if poly is None:
                continue
            x1, y1 = poly[:, 0].min(), poly[:, 1].min()
            x2, y2 = poly[:, 0].max(), poly[:, 1].max()
            x1, x2 = np.clip([x1, x2], 0, img_w)
            y1, y2 = np.clip([y1, y2], 0, img_h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append((float(x1), float(y1), float(x2), float(y2), obj["class_name"]))
        return out

    def boxes_for_camera_frame(self, cam_idx: int, img_w=672, img_h=376):
        j = self.camera_to_radar_idx(cam_idx)
        if j is None:
            return []
        return self.boxes_for_radar_frame(j, img_w, img_h)
