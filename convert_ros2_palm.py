#!/usr/bin/env python3
"""Convert a ROS 2 SQLite bag /hand/palm topic to baseline motion_3d.csv.

The expected PointStamped payload is:
    point.x, point.y: normalized color-image coordinates
    point.z: depth in metres

XYZ is reconstructed with the aligned color-camera intrinsics saved by the
expert RealSense extraction. This is valid only when novice and expert were
recorded with the same camera and stream profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


FIELDS = [
    "frame",
    "t",
    "color_frame_number",
    "depth_frame_number",
    "color_timestamp_ms",
    "depth_timestamp_ms",
    "color_clock",
    "depth_clock",
    "depth_offset_ms",
    "palm_x",
    "palm_y",
    "palm_z",
    "palm_X_m",
    "palm_Y_m",
    "palm_Z_m",
    "hand_detected",
    "depth_valid_fraction",
    "depth_iqr_m",
    "quality_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read geometry_msgs/msg/PointStamped from a ROS 2 .db3 bag and "
            "create a baseline-compatible motion_3d.csv."
        )
    )
    parser.add_argument("recording", type=Path, help="ROS 2 SQLite .db3 file")
    parser.add_argument(
        "--intrinsics-summary",
        required=True,
        type=Path,
        help="summary.json produced from the expert RealSense recording",
    )
    parser.add_argument(
        "--topic", default="/hand/palm", help="PointStamped topic (default: /hand/palm)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="parent output directory (default: data/processed)",
    )
    return parser.parse_args()


def load_intrinsics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay summary intrinsics: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        document = json.load(handle)

    intrinsics = document.get("aligned_depth_intrinsics") or document.get("intrinsics")
    if not isinstance(intrinsics, dict):
        raise ValueError(
            "summary.json khong co 'aligned_depth_intrinsics' hoac 'intrinsics'."
        )

    required = ("width", "height", "fx", "fy", "ppx", "ppy")
    missing = [name for name in required if name not in intrinsics]
    if missing:
        raise ValueError(f"Thieu intrinsics: {', '.join(missing)}")

    result = {name: float(intrinsics[name]) for name in required}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Intrinsics chua gia tri khong hop le.")
    if result["width"] <= 0 or result["height"] <= 0:
        raise ValueError("width va height phai lon hon 0.")
    if result["fx"] <= 0 or result["fy"] <= 0:
        raise ValueError("fx va fy phai lon hon 0.")

    coefficients = intrinsics.get("coefficients", [])
    if coefficients and any(abs(float(value)) > 1e-10 for value in coefficients):
        raise ValueError(
            "He so distortion khac 0. Script nay chi dung cong thuc pinhole; "
            "hay deproject bang RealSense SDK voi dung calibration."
        )

    result["width"] = int(result["width"])
    result["height"] = int(result["height"])
    result["distortion_model"] = intrinsics.get("distortion_model")
    result["coefficients"] = coefficients
    return result


def unique_output_dir(root: Path, stem: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / f"{stem}_3d"
    if not candidate.exists():
        candidate.mkdir()
        return candidate
    index = 2
    while True:
        candidate = root / f"{stem}_3d_{index}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        index += 1


def open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay database: {path}")
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def validate_database(connection: sqlite3.Connection, topic: str) -> str:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not {"messages", "topics"}.issubset(tables):
        raise ValueError(
            "Database khong phai ROS 2 bag hop le: can bang 'messages' va 'topics'."
        )

    rows = connection.execute(
        "SELECT type, serialization_format FROM topics WHERE name = ?", (topic,)
    ).fetchall()
    if not rows:
        raise ValueError(f"Khong tim thay topic {topic!r}.")
    if len(rows) != 1:
        raise ValueError(f"Topic {topic!r} xuat hien {len(rows)} lan; khong the chon an toan.")
    msgtype, serialization = rows[0]
    if msgtype != "geometry_msgs/msg/PointStamped":
        raise ValueError(f"Topic {topic!r} co type {msgtype!r}, khong phai PointStamped.")
    if serialization != "cdr":
        raise ValueError(f"Serialization {serialization!r} khong duoc ho tro; can 'cdr'.")
    return msgtype


def iter_messages(
    connection: sqlite3.Connection, topic: str
) -> Iterator[tuple[int, bytes]]:
    query = """
        SELECT messages.timestamp, messages.data
        FROM messages
        JOIN topics ON messages.topic_id = topics.id
        WHERE topics.name = ?
        ORDER BY messages.timestamp, messages.id
    """
    yield from connection.execute(query, (topic,))


def fmt(value: float) -> str:
    return f"{value:.9f}"


def convert(args: argparse.Namespace) -> Path:
    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "Thieu thu vien 'rosbags'. Cai bang: python -m pip install rosbags"
        ) from exc

    intrinsics = load_intrinsics(args.intrinsics_summary)
    output_dir = unique_output_dir(args.output_root, args.recording.stem)
    csv_path = output_dir / "motion_3d.csv"
    summary_path = output_dir / "summary.json"

    flags: Counter[str] = Counter()
    intervals: list[float] = []
    frame_ids: Counter[str] = Counter()
    first_timestamp: int | None = None
    previous_timestamp: int | None = None
    row_count = 0

    typestore = get_typestore(Stores.LATEST)
    with open_read_only(args.recording) as connection:
        msgtype = validate_database(connection, args.topic)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()

            for timestamp, rawdata in iter_messages(connection, args.topic):
                timestamp = int(timestamp)
                if first_timestamp is None:
                    first_timestamp = timestamp
                if previous_timestamp is not None:
                    if timestamp <= previous_timestamp:
                        raise ValueError(
                            "Timestamp /hand/palm khong tang nghiem ngat; "
                            "hay kiem tra rosbag."
                        )
                    intervals.append((timestamp - previous_timestamp) / 1e9)
                previous_timestamp = timestamp

                message = typestore.deserialize_cdr(rawdata, msgtype)
                point = message.point
                x, y, z = float(point.x), float(point.y), float(point.z)
                frame_id = str(getattr(message.header, "frame_id", ""))
                frame_ids[frame_id] += 1
                row_count += 1

                quality_flag = "ok"
                X = Y = Z = None
                if not all(math.isfinite(value) for value in (x, y, z)):
                    quality_flag = "invalid_point"
                elif not (0.0 <= x < 1.0 and 0.0 <= y < 1.0):
                    quality_flag = "palm_outside_image"
                elif z <= 0.0:
                    quality_flag = "depth_missing"
                else:
                    u = x * intrinsics["width"]
                    v = y * intrinsics["height"]
                    X = (u - intrinsics["ppx"]) * z / intrinsics["fx"]
                    Y = (v - intrinsics["ppy"]) * z / intrinsics["fy"]
                    Z = z
                flags[quality_flag] += 1

                t = (timestamp - first_timestamp) / 1e9
                writer.writerow(
                    {
                        "frame": row_count,
                        "t": fmt(t),
                        "color_frame_number": "",
                        "depth_frame_number": "",
                        "color_timestamp_ms": fmt((timestamp - first_timestamp) / 1e6),
                        "depth_timestamp_ms": "",
                        "color_clock": "rosbag_relative",
                        "depth_clock": "",
                        "depth_offset_ms": "",
                        "palm_x": fmt(x),
                        "palm_y": fmt(y),
                        "palm_z": fmt(z),
                        "palm_X_m": "" if X is None else fmt(X),
                        "palm_Y_m": "" if Y is None else fmt(Y),
                        "palm_Z_m": "" if Z is None else fmt(Z),
                        "hand_detected": 1,
                        "depth_valid_fraction": "",
                        "depth_iqr_m": "",
                        "quality_flag": quality_flag,
                    }
                )

    if row_count == 0:
        csv_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise ValueError(f"Topic {args.topic!r} khong co message nao.")

    median_interval = statistics.median(intervals) if intervals else None
    duration = (
        (previous_timestamp - first_timestamp) / 1e9
        if previous_timestamp is not None and first_timestamp is not None
        else 0.0
    )
    summary = {
        "converter_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "finished",
        "input_file": str(args.recording),
        "source_format": "rosbag2_sqlite_cdr_pointstamped",
        "topic": args.topic,
        "message_type": "geometry_msgs/msg/PointStamped",
        "output_csv": csv_path.name,
        "coordinate_system": "aligned_color_camera",
        "xyz_units": "meters",
        "xy_units": "normalized_color_image_coordinates",
        "intrinsics_source": str(args.intrinsics_summary),
        "aligned_depth_intrinsics": intrinsics,
        "rows_written": row_count,
        "quality_flags": dict(flags),
        "valid_xyz_row_fraction": flags["ok"] / row_count,
        "recording_span_seconds": duration,
        "sample_rate_from_median_interval": (
            1.0 / median_interval if median_interval and median_interval > 0 else None
        ),
        "point_frame_ids": dict(frame_ids),
        "notes": [
            "XYZ is reconstructed from normalized x/y, z depth, and expert camera intrinsics.",
            "This conversion assumes novice and expert used the same RealSense camera and 640x480 stream profile.",
            "The bag stores colorized depth rather than raw depth, so local depth-validity and depth-IQR checks cannot be reproduced.",
            "Only detected-hand messages exist; missing-hand video frames are not emitted as CSV rows.",
            "The palm trajectory is a proxy for hand motion, not tool-tip position or applied force.",
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    fps = summary["sample_rate_from_median_interval"]
    fps_text = "n/a" if fps is None else f"{fps:.2f}"
    print(f"Thu muc ket qua: {output_dir.resolve()}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Trang thai: finished | ok={flags['ok']}/{row_count} | fps={fps_text}")
    return csv_path


def main() -> int:
    args = parse_args()
    try:
        convert(args)
    except Exception as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
