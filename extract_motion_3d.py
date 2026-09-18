#!/usr/bin/env python3
"""Extract a palm trajectory from a RealSense .db3/.bag recording offline.

Run in the Python environment used for record.py:
    python extract_motion_3d.py "data/raw/expert_A_z1_01.db3" --seconds 30
Omit --seconds to process the full recording. Outputs go to a new directory
under data/processed: motion_3d.csv, summary.json, and up to three RGB previews.

Uses the same 5-landmark palm center as record.py. X/Y/Z are in meters in the
aligned color camera coordinate system, not a coordinate system attached to
the workpiece. Only one hand is tracked; check previews for correct selection.
Missing/uncertain measurements stay blank. No polishing or skill score is
assigned here. The input recording and previous outputs are never overwritten.

API references:
https://realsenseai.github.io/librealsense/python_docs/
https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/
"""

import argparse
import csv
import importlib.metadata
import json
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LANDMARKS = [0, 5, 9, 13, 17]
FIELDS = [
    "frame", "t", "color_frame_number", "depth_frame_number",
    "color_timestamp_ms", "depth_timestamp_ms", "color_clock", "depth_clock",
    "depth_offset_ms", "palm_x", "palm_y", "palm_z",
    "palm_X_m", "palm_Y_m", "palm_Z_m", "hand_detected",
    "depth_valid_fraction", "depth_iqr_m", "quality_flag",
]


def sample_depth(depth, scale, nx, ny, min_fraction=0.5, max_iqr=0.02, radius=5):
    """Median of positive depths near a valid image point, with quality flags."""
    result = {"z": None, "valid_fraction": None, "iqr_m": None, "flag": "ok"}
    if not (math.isfinite(nx) and math.isfinite(ny) and 0 <= nx < 1 and 0 <= ny < 1):
        result["flag"] = "palm_outside_image"
        return result
    h, w = depth.shape
    x, y = int(nx * w), int(ny * h)
    patch = depth[max(0, y-radius):min(h, y+radius+1),
                  max(0, x-radius):min(w, x+radius+1)]
    values = patch[patch > 0].astype(np.float64) * scale
    result["valid_fraction"] = float(values.size / patch.size)
    if values.size == 0:
        result["flag"] = "depth_missing"
        return result
    result["iqr_m"] = float(np.percentile(values, 75) - np.percentile(values, 25))
    if result["valid_fraction"] < min_fraction:
        result["flag"] = "depth_sparse"
    elif result["iqr_m"] > max_iqr:
        result["flag"] = "depth_spread_high"
    else:
        result["z"] = float(np.median(values))
    return result


def new_output_folder(root, stem):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(10000):
        suffix = "" if i == 0 else "_" + str(i+1)
        folder = root / (stem + "_3d" + suffix)
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            pass
    raise RuntimeError("Too many output folders with the same name.")


def intrinsics_dict(k):
    return {"width": k.width, "height": k.height, "fx": k.fx, "fy": k.fy,
            "ppx": k.ppx, "ppy": k.ppy, "distortion_model": str(k.model),
            "coefficients": list(k.coeffs)}


def timestamp_action(current, previous):
    """Reject repeated color samples; stop on backward time or clock changes."""
    if not math.isfinite(current[1]):
        raise ValueError("Nonfinite color timestamp.")
    if previous is None:
        return "accept"
    if current[2] != previous[2]:
        raise ValueError("Color timestamp domain changed; split the recording before analysis.")
    if current[:2] == previous[:2]:
        return "duplicate"
    if current[1] == previous[1]:
        return "same_timestamp"
    if current[1] < previous[1]:
        raise ValueError("Color timestamp moved backwards; partial outputs were retained.")
    return "accept"


def write_preview(cv2, bgr, landmarks, mp, row, path):
    image = bgr.copy()
    if landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(
            image, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    lines = [f"t={row['t']:.3f}s   color frame={row['color_frame_number']}",
             "status=" + row["quality_flag"]]
    if row["hand_detected"]:
        h, w = image.shape[:2]
        x, y = row["palm_x"], row["palm_y"]
        if 0 <= x < 1 and 0 <= y < 1:
            cv2.circle(image, (int(x*w), int(y*h)), 8, (0, 255, 255), 2)
    if row["quality_flag"] == "ok":
        lines.append("XYZ(m): {:.3f}, {:.3f}, {:.3f}".format(
            row["palm_X_m"], row["palm_Y_m"], row["palm_Z_m"]))
    for i, text in enumerate(lines):
        y = 24 + 25*i
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode RGB preview.")
    # pathlib supports Windows paths with Vietnamese characters.
    path.write_bytes(encoded.tobytes())


def run(args):
    folder = new_output_folder(args.output_root, args.recording.stem)
    report = {
        "extractor_version": 1, "input_file": args.recording.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started", "requested_seconds": args.seconds,
        "output_csv": "motion_3d.csv", "coordinate_system": "aligned_color_camera",
        "xyz_units": "meters", "xy_units": "normalized_color_image_coordinates",
        "time_definition": "color frame timestamp minus first accepted color timestamp, in seconds",
        "palm_definition": "2D mean of MediaPipe landmarks 0,5,9,13,17; depth at that image point",
        "settings": {"depth_patch_radius_pixels": 5,
                     "min_depth_valid_fraction": args.min_depth_fraction,
                     "max_depth_iqr_m": args.max_depth_iqr,
                     "max_color_depth_offset_ms": args.max_sync_ms,
                     "max_num_hands": 1, "mediapipe_min_detection_confidence": 0.3,
                     "mediapipe_min_tracking_confidence": 0.3},
        "previews": [], "notes": [
            "Depth checks describe measurement availability and local consistency, not hand tracking accuracy.",
            "The palm trajectory is a proxy for hand motion, not the tool tip or applied force.",
            "Depth is aligned geometrically; frames are not assumed to be simultaneous.",
            "No temporal smoothing, gap interpolation, activity filtering, or scoring is applied.",
            "RGB previews are sparse spot checks, not verification of the entire clip."]}
    counters = Counter()
    flags = Counter()
    intervals = []
    pipe = hands = None
    started = False
    first_ts = previous_color = previous_depth = None
    preview_targets = [2.0, 10.0, 20.0]
    end_reason = "unknown"
    last_progress = time.monotonic()
    print("Thu muc ket qua:", folder.resolve(), flush=True)
    try:
        import cv2
        import mediapipe as mp
        import pyrealsense2 as rs
        for package in ("mediapipe", "pyrealsense2", "numpy", "opencv-python"):
            try:
                report.setdefault("package_versions", {})[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        if not hasattr(mp, "solutions"):
            raise RuntimeError("MediaPipe solutions API is unavailable. Use the Python environment that ran record.py.")
        hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)
        pipe = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(args.recording), repeat_playback=False)
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        profile = pipe.start(config)
        started = True
        device = profile.get_device()
        playback = device.as_playback()
        playback.set_real_time(False)
        report["recording_duration_seconds"] = playback.get_duration().total_seconds()
        if device.supports(rs.camera_info.name):
            report["camera_model"] = device.get_info(rs.camera_info.name)
        scale = device.first_depth_sensor().get_depth_scale()
        report["depth_scale_meters_per_unit"] = scale
        aligner = rs.align(rs.stream.color)
        with (folder / "motion_3d.csv").open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                try:
                    frames = pipe.wait_for_frames(5000)
                except RuntimeError:
                    if playback.current_status() == rs.playback_status.stopped:
                        end_reason = "end_of_file"
                        break
                    raise
                color, raw_depth = frames.get_color_frame(), frames.get_depth_frame()
                counters["framesets_read"] += 1
                if not color:
                    counters["framesets_without_color"] += 1
                    continue
                current = (color.get_frame_number(), color.get_timestamp(),
                           str(color.get_frame_timestamp_domain()))
                action = timestamp_action(current, previous_color)
                if action != "accept":
                    counters["color_" + action + "_skipped"] += 1
                    continue
                if first_ts is None:
                    first_ts = current[1]
                    report["time_origin_color_timestamp_ms"] = first_ts
                    report["color_timestamp_domain"] = current[2]
                elapsed = (current[1] - first_ts) / 1000
                if args.seconds is not None and elapsed >= args.seconds:
                    end_reason = "seconds_limit"
                    break
                if previous_color is not None:
                    intervals.append((current[1] - previous_color[1]) / 1000)
                    if current[0] - previous_color[0] != 1:
                        counters["observed_nonunit_color_frame_steps"] += 1
                previous_color = current
                row = dict.fromkeys(FIELDS, "")
                row.update(frame=counters["rows_written"] + 1, t=elapsed,
                           color_frame_number=current[0], color_timestamp_ms=current[1],
                           color_clock=current[2], hand_detected=0)
                report["last_processed_t_seconds"] = elapsed
                data = np.asanyarray(color.get_data())
                color_format = color.profile.format()
                if color_format == rs.format.bgr8:
                    bgr, rgb = data, cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                elif color_format == rs.format.rgb8:
                    rgb, bgr = data, cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                else:
                    raise RuntimeError("Unsupported color format: " + str(color_format))
                landmarks = None
                detection = hands.process(rgb)
                if detection.multi_hand_landmarks:
                    landmarks = detection.multi_hand_landmarks[0]
                    xy = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y]
                                   for i in LANDMARKS]).mean(axis=0)
                    row.update(palm_x=float(xy[0]), palm_y=float(xy[1]), hand_detected=1)
                depth_issue = "depth_frame_missing"
                if raw_depth:
                    depth_current = (raw_depth.get_frame_number(), raw_depth.get_timestamp(),
                                     str(raw_depth.get_frame_timestamp_domain()))
                    row.update(depth_frame_number=depth_current[0], depth_timestamp_ms=depth_current[1],
                               depth_clock=depth_current[2])
                    if depth_current[2] != current[2]:
                        depth_issue = "different_timestamp_domains"
                    else:
                        offset = depth_current[1] - current[1]
                        row["depth_offset_ms"] = offset
                        if not math.isfinite(offset) or abs(offset) > args.max_sync_ms:
                            depth_issue = "depth_time_offset_high"
                        elif depth_current == previous_depth:
                            depth_issue = "reused_depth_frame"
                        else:
                            depth_issue = None
                    previous_depth = depth_current
                if landmarks is None:
                    row["quality_flag"] = "hand_missing"
                elif not (0 <= row["palm_x"] < 1 and 0 <= row["palm_y"] < 1):
                    row["quality_flag"] = "palm_outside_image"
                elif depth_issue:
                    row["quality_flag"] = depth_issue
                else:
                    aligned = aligner.process(frames).get_depth_frame()
                    if not aligned:
                        row["quality_flag"] = "aligned_depth_missing"
                    else:
                        depth = np.asanyarray(aligned.get_data())
                        k = aligned.profile.as_video_stream_profile().get_intrinsics()
                        report.setdefault("aligned_depth_intrinsics", intrinsics_dict(k))
                        measurement = sample_depth(depth, scale, row["palm_x"], row["palm_y"],
                                                   args.min_depth_fraction, args.max_depth_iqr)
                        row["depth_valid_fraction"] = measurement["valid_fraction"]
                        row["depth_iqr_m"] = measurement["iqr_m"]
                        row["quality_flag"] = measurement["flag"]
                        if measurement["z"] is not None:
                            xyz = rs.rs2_deproject_pixel_to_point(k,
                                [row["palm_x"]*k.width, row["palm_y"]*k.height], measurement["z"])
                            if not all(math.isfinite(v) for v in xyz):
                                row["quality_flag"] = "nonfinite_xyz"
                            else:
                                row.update(palm_z=float(xyz[2]), palm_X_m=float(xyz[0]),
                                           palm_Y_m=float(xyz[1]), palm_Z_m=float(xyz[2]))
                flags[row["quality_flag"]] += 1
                writer.writerow(row)
                counters["rows_written"] += 1
                counters["hand_detected"] += row["hand_detected"]
                if preview_targets and elapsed >= preview_targets[0]:
                    target = preview_targets.pop(0)
                    name = "preview_{:03d}s.jpg".format(int(target))
                    write_preview(cv2, bgr, landmarks, mp, row, folder / name)
                    report["previews"].append({"file": name, "actual_t_seconds": elapsed,
                                               "quality_flag": row["quality_flag"]})
                if time.monotonic() - last_progress >= 2:
                    print("t={:.1f}s | frames={} | XYZ={:.1%}".format(
                        elapsed, counters["rows_written"], flags["ok"]/counters["rows_written"]), flush=True)
                    last_progress = time.monotonic()
        report["status"] = "finished" if counters["rows_written"] else "no_color_frames"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        end_reason = "user_interrupt"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        end_reason = "error"
        print("Loi:", exc, flush=True)
    finally:
        if started:
            try:
                pipe.stop()
            except Exception as exc:
                report["pipeline_stop_error"] = str(exc)
        if hands is not None:
            try:
                hands.close()
            except Exception as exc:
                report["tracking_close_error"] = str(exc)
        report["end_reason"] = end_reason
        report["counters"] = dict(counters)
        report["quality_flags"] = dict(flags)
        count = counters["rows_written"]
        report["valid_xyz_row_fraction"] = flags["ok"]/count if count else None
        report["hand_detection_row_fraction"] = counters["hand_detected"]/count if count else None
        if intervals:
            report["color_intervals_seconds"] = {"median": statistics.median(intervals),
                                                  "max": max(intervals),
                                                  "gaps_over_0_2_seconds": sum(v > 0.2 for v in intervals)}
            report["sample_rate_from_median_interval"] = 1/statistics.median(intervals)
        with (folder / "summary.json").open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    print("Trang thai:", report["status"], "|", end_reason)
    print("Ket qua:", folder.resolve())
    return 0 if report["status"] == "finished" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--seconds", type=float, default=None, help="Process this many seconds from the first color frame")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-depth-fraction", type=float, default=0.5)
    parser.add_argument("--max-depth-iqr", type=float, default=0.02, help="Maximum local depth IQR, meters")
    parser.add_argument("--max-sync-ms", type=float, default=50, help="Maximum absolute color/depth timestamp offset")
    args = parser.parse_args()
    args.recording = args.recording.expanduser().resolve()
    if not args.recording.is_file():
        parser.error("File not found: " + str(args.recording))
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if not 0 < args.min_depth_fraction <= 1:
        parser.error("--min-depth-fraction must be in (0, 1]")
    if not math.isfinite(args.max_depth_iqr) or args.max_depth_iqr <= 0:
        parser.error("--max-depth-iqr must be finite and positive")
    if not math.isfinite(args.max_sync_ms) or args.max_sync_ms <= 0:
        parser.error("--max-sync-ms must be finite and positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
