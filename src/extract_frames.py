"""Extract the camera stream from a TartanDrive 2.0 ROS1 bag, pure-Python.

Reads /multisense/left/image_rect_color from one v2 trajectory chunk using the
`rosbags` library (no ROS install required) and writes lossless PNG frames plus a
manifest CSV (frame_idx, filename, timestamp_ns). Timestamps are kept because
later phases need a time-aligned latency signal.

Usage:
    python src/extract_frames.py            # uses config.yaml / .env
    EXTRACT_MAX_FRAMES=300 python src/extract_frames.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath  # noqa: E402


def decode_image(msg) -> np.ndarray:
    """sensor_msgs/Image -> HxWx3 BGR uint8 (for cv2 / OpenCV downstream)."""
    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(h, msg.step // 3 if msg.step else w, 3)[:, :w, :]
        if enc == "rgb8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)
    if enc in ("mono8", "8uc1"):
        img = buf.reshape(h, msg.step if msg.step else w)[:, :w]
        return np.ascontiguousarray(np.repeat(img[:, :, None], 3, axis=2))
    if enc.startswith("bayer"):
        import cv2
        raw = buf.reshape(h, msg.step if msg.step else w)[:, :w]
        return cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")


def main() -> int:
    import cv2
    from rosbags.highlevel import AnyReader

    cfg = load_config()
    bag = abspath(cfg["paths"]["raw_bag"])
    out_dir = abspath(cfg["paths"]["frames_dir"])
    topic = cfg["dataset"]["camera_topic"]
    max_frames = int(os.getenv("EXTRACT_MAX_FRAMES", "0"))

    if not bag.exists():
        print(f"ERROR: bag not found at {bag}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    print(f"Opening {bag} ({bag.stat().st_size/1e9:.1f} GB)")
    n = 0
    with AnyReader([bag]) as reader, open(manifest, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(["frame_idx", "filename", "timestamp_ns", "encoding", "width", "height"])
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            print(f"ERROR: topic {topic} not in bag. Available sample:", file=sys.stderr)
            for c in reader.connections[:40]:
                print("   ", c.topic, c.msgtype, file=sys.stderr)
            return 3
        print(f"Found topic {topic} ({conns[0].msgtype}); extracting...")
        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            img = decode_image(msg)
            fn = f"frame_{n:06d}.png"
            cv2.imwrite(str(out_dir / fn), img)
            writer.writerow([n, fn, int(timestamp), msg.encoding, msg.width, msg.height])
            n += 1
            if n % 100 == 0:
                print(f"  {n} frames...")
            if max_frames and n >= max_frames:
                break

    print(f"DONE: extracted {n} frames to {out_dir}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
