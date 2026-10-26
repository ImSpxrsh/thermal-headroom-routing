"""Stream-extract camera frames from a (possibly partial) TartanDrive 2.0 ROS1 bag.

The rosbags library needs the trailing index, so it cannot read a partial download.
But TartanDrive 2.0 chunks are uncompressed (compression=none), so a sequential
parser from the start can recover every image message present in whatever bytes have
downloaded so far. This unblocks the pipeline when the full 32 GB download is slow or
the connection is flaky. Output format matches extract_frames.py (PNG + manifest.csv).

    python src/extract_frames_streaming.py            # uses config paths
    EXTRACT_MAX_FRAMES=400 python src/extract_frames_streaming.py
"""
from __future__ import annotations

import csv
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_util import load_config, abspath

MAGIC = b"#ROSBAG V2.0\n"


def _parse_header(buf: bytes) -> dict:
    fields = {}
    i = 0
    while i < len(buf):
        (flen,) = struct.unpack_from("<I", buf, i); i += 4
        f = buf[i:i + flen]; i += flen
        eq = f.index(b"=")
        fields[f[:eq].decode("latin1")] = f[eq + 1:]
    return fields


def _read_record(f, limit):
    """Read one record (header, data) from file f, not exceeding byte offset limit.
    Returns (fields, data) or None if a full record cannot be read."""
    pos = f.tell()
    head = f.read(4)
    if len(head) < 4 or pos + 4 > limit:
        return None
    (hlen,) = struct.unpack("<I", head)
    hdr = f.read(hlen)
    if len(hdr) < hlen:
        return None
    dl = f.read(4)
    if len(dl) < 4:
        return None
    (dlen,) = struct.unpack("<I", dl)
    if f.tell() + dlen > limit:
        return None
    data = f.read(dlen)
    if len(data) < dlen:
        return None
    return _parse_header(hdr), data


def _iter_chunk_records(payload: bytes):
    """Yield (fields, data) for inner records of an uncompressed chunk payload."""
    i = 0
    n = len(payload)
    while i + 4 <= n:
        (hlen,) = struct.unpack_from("<I", payload, i); i += 4
        if i + hlen > n:
            break
        fields = _parse_header(payload[i:i + hlen]); i += hlen
        if i + 4 > n:
            break
        (dlen,) = struct.unpack_from("<I", payload, i); i += 4
        if i + dlen > n:
            break
        data = payload[i:i + dlen]; i += dlen
        yield fields, data


def _decode_image(data: bytes):
    """Deserialize a ROS1 sensor_msgs/Image from its message bytes -> (BGR, encoding, ts_ns)."""
    o = 0
    seq, sec, nsec = struct.unpack_from("<III", data, o); o += 12
    (fid_len,) = struct.unpack_from("<I", data, o); o += 4 + fid_len
    height, width = struct.unpack_from("<II", data, o); o += 8
    (enc_len,) = struct.unpack_from("<I", data, o); o += 4
    enc = data[o:o + enc_len].decode("latin1"); o += enc_len
    (is_big,) = struct.unpack_from("<B", data, o); o += 1
    (step,) = struct.unpack_from("<I", data, o); o += 4
    (dlen,) = struct.unpack_from("<I", data, o); o += 4
    raw = np.frombuffer(data[o:o + dlen], dtype=np.uint8)
    ts_ns = sec * 1_000_000_000 + nsec
    el = enc.lower()
    if el in ("bgr8", "rgb8"):
        img = raw.reshape(height, step // 3, 3)[:, :width, :]
        if el == "rgb8":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img), enc, ts_ns
    if el in ("mono8", "8uc1"):
        g = raw.reshape(height, step)[:, :width]
        return np.ascontiguousarray(np.repeat(g[:, :, None], 3, axis=2)), enc, ts_ns
    if el.startswith("bayer"):
        import cv2
        g = raw.reshape(height, step)[:, :width]
        return cv2.cvtColor(g, cv2.COLOR_BAYER_BG2BGR), enc, ts_ns
    raise ValueError(f"unsupported encoding {enc!r}")


def main() -> int:
    import cv2
    cfg = load_config()
    bag = abspath(cfg["paths"]["raw_bag"])
    out_dir = abspath(cfg["paths"]["frames_dir"])
    topic = cfg["dataset"]["camera_topic"]
    max_frames = int(os.getenv("EXTRACT_MAX_FRAMES", "0"))
    if not bag.exists():
        print(f"ERROR: bag not found at {bag}", file=sys.stderr); return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    size = bag.stat().st_size
    print(f"Streaming {bag} (readable {size/1e9:.2f} GB), topic {topic}")
    conn_topic = {}
    n = 0
    with open(bag, "rb") as f, open(out_dir / "manifest.csv", "w", newline="") as mf:
        if f.read(len(MAGIC)) != MAGIC:
            print("ERROR: not a ROS V2.0 bag", file=sys.stderr); return 3
        w = csv.writer(mf)
        w.writerow(["frame_idx", "filename", "timestamp_ns", "encoding", "width", "height"])
        while True:
            rec = _read_record(f, size)
            if rec is None:
                break
            fields, data = rec
            op = fields.get("op", b"\x00")[0]
            if op == 7:  # top-level connection
                conn_topic[struct.unpack("<I", fields["conn"])[0]] = fields["topic"].decode("latin1")
            elif op == 5:  # chunk: parse inner connection + message records
                if fields.get("compression", b"none") != b"none":
                    continue  # streaming parser only handles uncompressed chunks
                for ifields, idata in _iter_chunk_records(data):
                    iop = ifields.get("op", b"\x00")[0]
                    if iop == 7:
                        conn_topic[struct.unpack("<I", ifields["conn"])[0]] = ifields["topic"].decode("latin1")
                    elif iop == 2:
                        conn = struct.unpack("<I", ifields["conn"])[0]
                        if conn_topic.get(conn) != topic:
                            continue
                        try:
                            img, enc, ts = _decode_image(idata)
                        except Exception as e:
                            print(f"  skip frame {n}: {e}"); continue
                        fn = f"frame_{n:06d}.png"
                        cv2.imwrite(str(out_dir / fn), img)
                        w.writerow([n, fn, ts, enc, img.shape[1], img.shape[0]])
                        n += 1
                        if n % 50 == 0:
                            print(f"  {n} frames...")
                        if max_frames and n >= max_frames:
                            print(f"DONE (cap): {n} frames -> {out_dir}")
                            return 0
    print(f"DONE: extracted {n} frames -> {out_dir} (from partial/complete bag)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
