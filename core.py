"""Offline, reviewable motion measurements. Import this module; run analyze_motion.py.

Inputs are the metric XYZ columns from extract_motion_3d.py, not legacy image XY.
Tracking quality and task labels are independent of the measured skill features.
No automatic polishing labels or expert/novice scores are inferred here.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt


XYZ = ["palm_X_m", "palm_Y_m", "palm_Z_m"]
WEIGHT = {"freq_hz": 0.45, "rhythm_s": 0.35, "amp_cm": 0.20}
LABELS = {"polishing", "not_polishing", "unreviewed"}


@dataclass(frozen=True)
class Config:
    min_segment_s: float = 5.0
    target_window_s: float = 8.0
    max_bridge_s: float = 0.15
    jump_floor_m: float = 0.05
    jump_median_factor: float = 10.0
    lowpass_hz: float = 6.0
    min_turn_gap_s: float = 0.12
    prominence_floor_m: float = 0.0015
    prominence_range_fraction: float = 0.15

    def validate(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("All analysis settings must be finite and positive.")
        if self.target_window_s < self.min_segment_s:
            raise ValueError("target_window_s must be >= min_segment_s.")
        if self.prominence_range_fraction >= 1:
            raise ValueError("prominence_range_fraction must be < 1.")


def load_motion(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"t", "quality_flag", *XYZ}
    missing = required - set(df)
    if missing:
        raise ValueError("Missing 3D columns: " + ", ".join(sorted(missing)))
    if len(df) < 3:
        raise ValueError("Need at least three source rows.")
    for col in ["t", *XYZ]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    t = df.t.to_numpy(float)
    if not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError("Timestamps must be finite and strictly increasing; do not sort away clock errors.")
    if t[0] < 0:
        raise ValueError("Expected nonnegative time relative to the recording.")
    return df


def read_activity(path, t):
    """Optional task labels in source-CSV seconds. Half-open intervals [start,end)."""
    labels = np.full(len(t), "unreviewed", dtype=object)
    if path is None:
        return labels
    table = pd.read_csv(path, encoding="utf-8-sig")
    if not {"start_s", "end_s", "label"} <= set(table):
        raise ValueError("Activity CSV needs start_s,end_s,label.")
    table = table.sort_values("start_s")
    previous_end = -np.inf
    for row in table.itertuples(index=False):
        start, end = float(row.start_s), float(row.end_s)
        if not np.isfinite([start, end]).all() or start < 0 or start >= end:
            raise ValueError("Activity intervals need finite 0 <= start_s < end_s.")
        if start < previous_end:
            raise ValueError("Activity intervals must not overlap.")
        if row.label not in LABELS:
            raise ValueError("Activity label must be polishing, not_polishing, or unreviewed.")
        labels[(t >= start) & (t < end)] = row.label
        previous_end = end
    return labels


def time_cells(t, frame_dt):
    """Duration attributed to each frame, bounded by source span, without filling dropped time."""
    left = np.maximum(t - frame_dt / 2, np.r_[t[0], (t[:-1] + t[1:]) / 2])
    right = np.minimum(t + frame_dt / 2, np.r_[(t[:-1] + t[1:]) / 2, t[-1]])
    return np.maximum(0, right - left)


def measurement_quality(df, labels, cfg):
    t = df.t.to_numpy(float)
    xyz = df[XYZ].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    valid = df.quality_flag.eq("ok").to_numpy() & np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    pairs = valid[:-1] & valid[1:]
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    typical = float(np.median(steps[pairs])) if pairs.any() else 0.0
    jump_limit = max(cfg.jump_floor_m, cfg.jump_median_factor * typical)
    source_idx = np.flatnonzero(valid)
    groups, jumps = [], []
    if len(source_idx):
        start = 0
        for j in range(1, len(source_idx)):
            a, b = source_idx[j - 1], source_idx[j]
            gap = t[b] - t[a]
            distance = float(np.linalg.norm(xyz[b] - xyz[a]))
            jump = gap <= cfg.max_bridge_s and distance > jump_limit
            if jump:
                jumps.append({"before_s": float(t[a]), "after_s": float(t[b]),
                              "distance_cm": distance * 100, "action": "split_no_interpolation"})
            # Never interpolate across a supplied task boundary, even within a short gap.
            changed_label = np.any(labels[a:b + 1] != labels[a])
            if gap > cfg.max_bridge_s or jump or changed_label:
                groups.append(source_idx[start:j])
                start = j
        groups.append(source_idx[start:])
    cells = time_cells(t, dt)
    observed_s = float(cells[valid].sum())
    reviewed = valid & (labels != "unreviewed")
    reviewed_s = float(cells[reviewed].sum())
    active_s = float(cells[valid & (labels == "polishing")].sum())
    inactive_s = float(cells[valid & (labels == "not_polishing")].sum())
    active = {
        "status": "unavailable_no_reviewed_activity" if reviewed_s == 0 else "reviewed_observed_time_only",
        "polishing_s": active_s, "not_polishing_s": inactive_s,
        "reviewed_observed_s": reviewed_s,
        "unreviewed_observed_s": max(0.0, observed_s - reviewed_s),
        "unobserved_s": max(0.0, float(t[-1] - t[0]) - observed_s),
        "reviewed_fraction_of_observed": reviewed_s / observed_s if observed_s else None,
        "active_fraction_of_reviewed_observed": active_s / reviewed_s if reviewed_s else None,
        "included_in_skill_score": False,
    }
    return valid, groups, dt, jump_limit, jumps, active


def measure_cycles(t, xyz, cfg):
    """PCA of linearly detrended, low-pass XYZ; both polarities avoid axis-sign bias.

One cycle is an outward-and-return motion. amp_cm is one-way peak-to-trough
excursion along the segment axis. rhythm_s is the sample SD of full-cycle
intervals pooled over both polarities (correlated observations, not independent
statistical samples). At least three complete cycles in EACH polarity are
needed for the three-metric summary; slower motion is retained as insufficient
cycles, never classified as not-polishing.
"""
    fs = 1 / float(np.median(np.diff(t)))
    cutoff = min(cfg.lowpass_hz, 0.35 * fs)
    filtered = sosfiltfilt(butter(3, cutoff, fs=fs, output="sos"), xyz, axis=0)
    centered = detrend(filtered, axis=0, type="linear")
    _, s, axes = np.linalg.svd(centered, full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    projected = centered @ axis
    prominence = max(cfg.prominence_floor_m,
                     cfg.prominence_range_fraction * float(np.quantile(projected, .95) - np.quantile(projected, .05)))
    distance = max(1, int(np.ceil(cfg.min_turn_gap_s * fs)))
    pos, _ = find_peaks(projected, prominence=prominence, distance=distance)
    neg, _ = find_peaks(-projected, prominence=prominence, distance=distance)
    # Enforce alternating extrema; two maxima without an intervening minimum
    # are represented by the more extreme maximum, likewise for minima.
    turns = []
    for index, polarity in sorted([(int(i), 1) for i in pos] + [(int(i), -1) for i in neg]):
        if turns and turns[-1][1] == polarity:
            if polarity * projected[index] > polarity * projected[turns[-1][0]]:
                turns[-1] = (index, polarity)
        else:
            turns.append((index, polarity))
    pos = np.array([i for i, k in turns if k == 1], dtype=int)
    neg = np.array([i for i, k in turns if k == -1], dtype=int)
    intervals = np.r_[np.diff(t[pos]), np.diff(t[neg])]
    amplitudes = np.array([abs(projected[b] - projected[a]) for (a, _), (b, _) in zip(turns, turns[1:])])
    cycles = max(0, min(len(pos), len(neg)) - 1)
    metrics = {
        "freq_hz": float(1 / intervals.mean()) if len(intervals) else None,
        "rhythm_s": float(np.std(intervals, ddof=1)) if len(intervals) >= 2 else None,
        "amp_cm": float(100 * np.median(amplitudes)) if len(amplitudes) else None,
        "complete_cycles_per_polarity": cycles,
        "metric_status": "measured_review_required" if cycles >= 3 else "insufficient_cycles",
        "axis_variance_fraction": float(s[0] ** 2 / np.sum(s ** 2)) if np.sum(s ** 2) > 0 else None,
        "axis_X": float(axis[0]), "axis_Y": float(axis[1]), "axis_Z": float(axis[2]),
        "peak_prominence_cm": 100 * prominence,
    }
    trace = {"t": t, "projected_cm": projected * 100, "pos": pos, "neg": neg}
    return metrics, trace, filtered


SEGMENT_COLUMNS = ["segment_id", "start_s", "end_s", "duration_s", "activity_label",
                   "source_valid_rows", "source_rows", "source_valid_fraction", "sample_rate_hz",
                   "first_color_frame", "last_color_frame", "freq_hz", "rhythm_s", "amp_cm",
                   "complete_cycles_per_polarity", "metric_status", "axis_variance_fraction",
                   "axis_X", "axis_Y", "axis_Z", "peak_prominence_cm"]


def analyze(df, cfg=None, activity_path=None):
    cfg = cfg or Config()
    cfg.validate()
    t = df.t.to_numpy(float)
    xyz = df[XYZ].to_numpy(float)
    labels = read_activity(activity_path, t)
    valid, groups, dt, jump_limit, jumps, active = measurement_quality(df, labels, cfg)
    segments, traces, cleaned = [], {}, []
    short_blocks = []
    for block_id, idx in enumerate(groups, 1):
        start, end = float(t[idx[0]]), float(t[idx[-1]])
        duration = end - start
        if duration < cfg.min_segment_s:
            short_blocks.append({"start_s": start, "end_s": end, "duration_s": duration})
            continue
        count = max(1, int(np.ceil(duration / cfg.target_window_s)))
        while count > 1 and duration / count < cfg.min_segment_s + dt:
            count -= 1
        for left, right in zip(np.linspace(start, end, count + 1)[:-1], np.linspace(start, end, count + 1)[1:]):
            grid = np.arange(left, right + dt * .1, dt)
            grid = grid[grid <= right]
            if len(grid) < 20:
                continue
            # Only short, bounded gaps within this continuity block can be interpolated.
            values = np.column_stack([np.interp(grid, t[idx], xyz[idx, k]) for k in range(3)])
            mask = (t >= grid[0]) & (t <= grid[-1])
            metrics, trace, filtered = measure_cycles(grid, values, cfg)
            sid = len(segments) + 1
            frame_ids =(
    pd.to_numeric(
        df.loc[mask, "color_frame_number"],
        errors="coerce"
    ).dropna()
    if "color_frame_number" in df
    else None
)
            row = {
                "segment_id": sid, "start_s": float(grid[0]), "end_s": float(grid[-1]),
                "duration_s": float(grid[-1] - grid[0]), "activity_label": str(labels[idx[0]]),
                "source_valid_rows": int((mask & valid).sum()), "source_rows": int(mask.sum()),
                "source_valid_fraction": float(valid[mask].mean()) if mask.any() else None,
                "sample_rate_hz": 1 / dt,
                "first_color_frame": int(frame_ids.iloc[0]) if frame_ids is not None and len(frame_ids) else None,
                "last_color_frame": int(frame_ids.iloc[-1]) if frame_ids is not None and len(frame_ids) else None,
                **metrics,
            }
            segments.append(row)
            traces[sid] = trace
            cleaned.append(pd.DataFrame({"segment_id": sid, "block_id": block_id, "t": grid,
                                         "X_m": filtered[:, 0], "Y_m": filtered[:, 1], "Z_m": filtered[:, 2],
                                         "axis_cm_detrended": trace["projected_cm"]}))
    table = pd.DataFrame(segments, columns=SEGMENT_COLUMNS)
    measured = table.loc[table.metric_status == "measured_review_required"]
    confirmed = measured.loc[measured.activity_label == "polishing"]
    def medians(frame):
        return {k: float(frame[k].median()) if len(frame) else None for k in WEIGHT}
    summary = {
        "analysis_version": 1, "status": "analysis_completed_no_skill_score",
        "rows": len(df), "start_s": float(t[0]), "end_s": float(t[-1]),
        "source_span_s": float(t[-1] - t[0]), "sample_rate_hz": 1 / dt,
        "valid_source_rows": int(valid.sum()), "valid_source_fraction": float(valid.mean()),
        "quality_flag_counts": {str(k): int(v) for k, v in df.quality_flag.value_counts(dropna=False).items()},
        "jump_limit_cm": jump_limit * 100, "jump_boundaries": jumps,
        "continuous_blocks": len(groups), "blocks_below_min_duration": short_blocks,
        "analysis_windows": len(table), "windows_with_enough_cycles": len(measured),
        "confirmed_polishing_windows_with_enough_cycles": len(confirmed),
        "all_measured_window_medians_not_an_expert_template": medians(measured),
        "confirmed_polishing_window_medians": medians(confirmed),
        "active": active, "settings": asdict(cfg), "planned_score_weights": WEIGHT,
        "notes": [
            "Unreviewed motion is not automatically polishing; no expert template or skill score is created.",
            "No frequency, rhythm, or amplitude quality cutoff removes novice behavior.",
            "A full cycle is outward and return. Amplitude is one-way peak-to-trough palm travel along the local PCA axis.",
            "XYZ are in color-camera meters. Position tracks the hand, not the tool contact point or force.",
            "Only bracketed gaps <= max_bridge_s can be interpolated for metric calculation; they remain unobserved in active.",
            "Large coordinate jumps split trajectories; they are suspicious boundaries, not proven tracking errors.",
            "Low-pass and peak prominence settings affect small/fast strokes; review the exported turning points.",
            "activity labels describe visible task state, not whether the operator is skilled.",
        ],
    }
    cleaned_df = pd.concat(cleaned, ignore_index=True) if cleaned else pd.DataFrame(
        columns=["segment_id", "block_id", "t", "X_m", "Y_m", "Z_m", "axis_cm_detrended"])
    return table, summary, traces, cleaned_df
