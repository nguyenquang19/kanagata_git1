#!/usr/bin/env python3
"""Inspect a local recording and write a small JSON report.

Usage (run in the Python environment used for record.py):
    python inspect_recording.py "data/raw/expert_A_z1_01.db3"

Reads the input without changing, renaming, or copying it. Never opens a live
camera. Reads up to 90 framesets, not the full recording. The resulting report
is a sample inspection, not a full-file integrity or tracking-quality check.
Version 2: RealSense playback is attempted for SQLite .db3 recordings too.
RealSense API: https://realsenseai.github.io/librealsense/python_docs/
"""

import argparse
import importlib.metadata
import json
import platform
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def file_info(path):
    with path.open("rb") as stream:
        header = stream.read(32)
    if header.startswith(b"#ROSBAG V2.0\n"):
        kind = "rosbag_v2"
    elif header.startswith(b"SQLite format 3\x00"):
        kind = "sqlite3"
    else:
        kind = "unknown"
    size = path.stat().st_size
    return {"name": path.name, "size_bytes": size,
            "size_gib": round(size / 1024**3, 3),
            "format_from_header": kind, "header_hex": header.hex()}


def sqlite_info(path):
    # URI mode=ro prevents SQLite from creating or modifying the source file.
    con = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        con.execute("PRAGMA query_only = ON")
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        result = {"tables": tables,
                  "ros2_schema_candidate": "topics" in tables and "messages" in tables}
        if "topics" in tables:
            columns = {r[1] for r in con.execute("PRAGMA table_info(topics)")}
            selected = [c for c in ("id", "name", "type", "serialization_format")
                        if c in columns]
            if selected:
                names = ", ".join('"' + c + '"' for c in selected)
                topics = [dict(zip(selected, row)) for row in
                    con.execute("SELECT " + names + " FROM topics")]
                result["topic_count"] = len(topics)
                result["topics"] = topics
                result["image_topics_declared"] = [t for t in topics
                    if t.get("type") == "sensor_msgs/msg/Image"]
        return result
    finally:
        con.close()


def intrinsics_info(video_profile):
    k = video_profile.get_intrinsics()
    return {"width": k.width, "height": k.height,
            "fx": k.fx, "fy": k.fy, "ppx": k.ppx, "ppy": k.ppy,
            "distortion_model": str(k.model), "coefficients": list(k.coeffs)}


def stream_info(profile):
    result = {"type": str(profile.stream_type()), "index": profile.stream_index(),
              "format": str(profile.format()), "nominal_fps": profile.fps()}
    if profile.is_video_stream_profile():
        video = profile.as_video_stream_profile()
        result.update(width=video.width(), height=video.height(),
                      intrinsics=intrinsics_info(video))
    return result


def timestamp_summary(samples):
    result = {"samples": len(samples),
              "timestamp_domains": dict(Counter(s[2] for s in samples))}
    if not samples:
        return result
    result.update(first_frame_number=samples[0][0], last_frame_number=samples[-1][0],
                  first_timestamp_ms=samples[0][1], last_timestamp_ms=samples[-1][1])
    intervals = [b[1] - a[1] for a, b in zip(samples, samples[1:]) if a[2] == b[2]]
    positive = [v for v in intervals if v > 0]
    result["nonpositive_timestamp_intervals"] = sum(v <= 0 for v in intervals)
    result["nonunit_frame_number_steps"] = sum(
        b[0] - a[0] != 1 for a, b in zip(samples, samples[1:]))
    if positive:
        med = statistics.median(positive)
        result["positive_interval_ms"] = {"min": min(positive), "median": med,
                                          "max": max(positive)}
        result["sample_rate_from_median_interval"] = 1000 / med
    return result


def inspect_realsense(path, limit, report):
    result = report["realsense"] = {"opened": False}
    try:
        import pyrealsense2 as rs
    except (ImportError, OSError) as exc:
        result["error"] = str(exc)
        report["status"] = "sdk_unavailable"
        report["notes"].append(
            "Run this script in the same Python environment as record.py. "
            "pyrealsense2 could not be imported; the recording has not been tested.")
        return
    try:
        result["sdk_package_version"] = importlib.metadata.version("pyrealsense2")
    except importlib.metadata.PackageNotFoundError:
        result["sdk_package_version"] = "unknown"
    pipe = rs.pipeline()
    cfg = rs.config()
    started = False
    try:
        # Selecting a file explicitly prevents fallback to a physical camera.
        cfg.enable_device_from_file(str(path), repeat_playback=False)
        cfg.enable_all_streams()
        profile = pipe.start(cfg)
        started = True
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)
        result["opened"] = True
        result["recording_duration_seconds"] = playback.get_duration().total_seconds()
        if device.supports(rs.camera_info.name):
            result["device_model"] = device.get_info(rs.camera_info.name)
        profiles = list(profile.get_streams())
        result["streams"] = [stream_info(p) for p in profiles]
        color = next((p for p in profiles if p.stream_type() == rs.stream.color), None)
        depth = next((p for p in profiles if p.stream_type() == rs.stream.depth), None)
        result["has_color_stream"] = color is not None
        result["has_depth_stream"] = depth is not None
        if depth is not None:
            result["depth_scale_meters_per_unit"] = device.first_depth_sensor().get_depth_scale()
        if color is not None and depth is not None:
            ext = depth.get_extrinsics_to(color)
            result["depth_to_color_extrinsics"] = {
                "rotation_column_major": list(ext.rotation),
                "translation_meters": list(ext.translation)}
        aligner = rs.align(rs.stream.color) if color is not None and depth is not None else None
        timestamps = {"color": [], "depth": []}
        alignment_tried = False
        received = 0
        stop_reason = "sample_limit"
        for _ in range(limit):
            try:
                frames = pipe.wait_for_frames(5000)
            except RuntimeError as exc:
                stop_reason = "end_of_file" if playback.current_status() == rs.playback_status.stopped else "read_error"
                result["last_read_message"] = str(exc)
                break
            received += 1
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            for name, frame in (("color", c), ("depth", d)):
                if frame:
                    timestamps[name].append((frame.get_frame_number(), frame.get_timestamp(),
                                             str(frame.get_frame_timestamp_domain())))
            if aligner is not None and c and d and not alignment_tried:
                alignment_tried = True
                try:
                    aligned = aligner.process(frames).get_depth_frame()
                    if not aligned:
                        raise RuntimeError("No aligned depth frame was returned.")
                    result["aligned_depth_to_color_intrinsics"] = intrinsics_info(
                        aligned.profile.as_video_stream_profile())
                    try:
                        import numpy as np
                        array = np.asanyarray(aligned.get_data())
                        result["first_aligned_depth_nonzero_fraction"] = float(np.count_nonzero(array) / array.size)
                        result["depth_fraction_note"] = "Whole image, first aligned frame only; not hand tracking quality."
                    except ImportError:
                        pass
                except Exception as exc:
                    result["alignment_error"] = str(exc)
        result["sample"] = {"requested_framesets": limit, "read_framesets": received,
                            "stop_reason": stop_reason,
                            "timestamps": {k: timestamp_summary(v) for k, v in timestamps.items()}}
        if timestamps["color"] and timestamps["depth"]:
            report["status"] = "rgb_depth_sample_read"
        elif received:
            report["status"] = "partial_stream_sample_read"
        else:
            report["status"] = "no_frames_read"
    except Exception as exc:
        result["error"] = str(exc)
        report["status"] = "sdk_read_failed"
    finally:
        if started:
            try:
                pipe.stop()
            except Exception as exc:
                result["stop_error"] = str(exc)


def inspect_input(path, limit, report):
    report["file"] = file_info(path)
    kind = report["file"]["format_from_header"]
    print("Dinh dang theo header:", kind, flush=True)
    if kind == "sqlite3":
        try:
            report["sqlite"] = sqlite_info(path)
        except Exception as exc:
            report["sqlite"] = {"error": str(exc)}
        report["notes"].append(
            "SQLite is a supported container for current RealSense recordings. "
            "Declared image topics alone do not prove that image frames can be decoded.")
    print("Dang doc khung hinh bang RealSense SDK...", flush=True)
    # Do not stop after SQLite metadata. The SDK also handles native .db3 and
    # compressed recordings; renaming .db3 to .bag is not a conversion.
    inspect_realsense(path, limit, report)


def save_report(report, stem):
    # New file each time; avoid overwriting an existing report or input file.
    for suffix in range(10000):
        tail = "" if suffix == 0 else "_" + str(suffix + 1)
        path = Path.cwd() / (stem + "_inspection_v2" + tail + ".json")
        try:
            with path.open("x", encoding="utf-8") as output:
                json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Too many existing reports with the same name.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path, help="Existing .db3 or .bag recording")
    parser.add_argument("--frames", type=int, default=90, help="Framesets to sample, 1-1000 (default: 90)")
    args = parser.parse_args()
    if not 1 <= args.frames <= 1000:
        parser.error("--frames must be between 1 and 1000")
    path = args.recording.expanduser().resolve()
    if not path.is_file():
        parser.error("File not found: " + str(path))
    report = {"report_version": 2, "created_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.system(),
              "status": "started", "notes": [
                  "Input is read without renaming or copying it. No live camera is opened.",
                  "Only metadata and a bounded initial frame sample are inspected.",
                  "This does not verify integrity or tracking quality for the entire recording."]}
    print("Dang kiem tra file. File goc duoc giu nguyen.", flush=True)
    try:
        inspect_input(path, args.frames, report)
    except Exception as exc:
        report["status"] = "inspection_error"
        report["error"] = str(exc)
    try:
        output = save_report(report, path.stem)
    except (OSError, RuntimeError, ValueError) as exc:
        print("Khong luu duoc JSON:", exc)
        return 2
    print("Trang thai:", report["status"])
    print("Bao cao:", output)
    print("Gui file JSON nay de phan tich tiep.")
    return 0 if report["status"] == "rgb_depth_sample_read" else 1


if __name__ == "__main__":
    raise SystemExit(main())
