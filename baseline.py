"""Shared 3D template, scoring, and validation logic. All entry points use this file."""

from dataclasses import asdict
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

import core


VERSION = "1.0.0"
SCHEMA = "kanagata.metric3d.v1"
METRICS = ("freq_hz", "rhythm_s", "amp_cm")
UNITS = {"freq_hz": "Hz", "rhythm_s": "s", "amp_cm": "cm"}
WEIGHT = dict(core.WEIGHT)
ABS_FLOOR = {"freq_hz": 0.10, "rhythm_s": 0.02, "amp_cm": 0.10}
REL_FLOOR = 0.08


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def safe_name(name):
    name = str(name)
    if not name or name in {".", ".."} or re.search(r'[\\/:*?"<>|]', name):
        raise ValueError("Tên phải là tên thư mục/file đơn, không phải đường dẫn.")
    return name


def new_folder(root, name):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    safe_name(name)
    for i in range(10000):
        folder = root / (name if i == 0 else f"{name}_{i+1}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            pass
    raise ValueError("Quá nhiều thư mục kết quả cùng tên.")


def expand_inputs(patterns):
    result = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern), recursive=True))
        if not matches:
            raise ValueError(f"Không tìm thấy file: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                raise ValueError(f"Đầu vào phải là file CSV: {path}")
            if path in result:
                raise ValueError(f"File xuất hiện hai lần trong đầu vào: {path}")
            result.append(path)
    if not result:
        raise ValueError("Cần ít nhất một file CSV.")
    return result


def config_from_file(path=None):
    if path is None:
        return core.Config()
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    cfg = core.Config(**values)
    cfg.validate()
    return cfg


def fingerprint(df):
    # Content identity survives renamed files, BOM/newline changes and column ordering.
    cols = ["t", *core.XYZ]
    text = df[cols].to_csv(index=False, float_format="%.10g", na_rep="NA", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def identify(path, df):
    path = Path(path)
    name = path.parent.name if path.stem.startswith("motion_3d") else path.stem
    result = {"name": name, "csv": str(path), "fingerprint": fingerprint(df)}
    if "color_timestamp_ms" in df:
        stamps = pd.to_numeric(df.color_timestamp_ms, errors="coerce").dropna()
        if len(stamps):
            result["color_timestamp_range_ms"] = [float(stamps.min()), float(stamps.max())]
    return result


def activity_for(path, supplied=None):
    if supplied is not None:
        return Path(supplied)
    candidate = Path(path).with_name("activity.csv")
    return candidate if candidate.is_file() else None


def analyze_source(path, cfg=None, mode="auto", activity=None):
    if mode not in {"auto", "reviewed"}:
        raise ValueError("mode phải là auto hoặc reviewed.")
    cfg = cfg or core.Config()
    df = core.load_motion(path)
    activity = activity_for(path, activity)
    table, quality, traces, cleaned = core.analyze(df, cfg, activity)
    eligible = (table.metric_status == "measured_review_required") & np.isfinite(table[list(METRICS)].to_numpy(dtype=float)).all(axis=1)
    # Broad repeated-motion candidates, NOT a verified activity classifier.
    # Never gate on rhythm, amplitude, closeness to expert, or freq >= 1.2 Hz.
    if mode == "reviewed":
        selected = eligible & table.activity_label.eq("polishing")
    else:
        selected = eligible & ~table.activity_label.eq("not_polishing")
    table = table.copy()
    table["selected"] = selected
    table["selection_reason"] = np.select(
        [~eligible, table.activity_label.eq("not_polishing"), table.activity_label.eq("unreviewed") & (mode == "reviewed")],
        ["insufficient_measurable_cycles", "reviewed_not_polishing", "activity_not_reviewed"],
        default="reviewed_polishing" if mode == "reviewed" else "automatic_motion_candidate")
    table.loc[selected & table.activity_label.eq("polishing"), "selection_reason"] = "reviewed_polishing"
    table["candidate_activity_unverified"] = selected & table.activity_label.eq("unreviewed")
    chosen = table.loc[selected].copy()
    selected_s = float(chosen.duration_s.sum())
    valid = df.quality_flag.eq("ok").to_numpy() & np.isfinite(df[core.XYZ]).all(axis=1).to_numpy()
    valid &= df.palm_Z_m.to_numpy() > 0
    observed_s = float(core.time_cells(df.t.to_numpy(), 1 / quality["sample_rate_hz"])[valid].sum())
    mask = np.zeros(len(df), dtype=bool)
    for row in chosen.itertuples():
        mask |= (df.t.to_numpy() >= row.start_s) & (df.t.to_numpy() <= row.end_s)
    candidate_observed_s = float(core.time_cells(df.t.to_numpy(), 1 / quality["sample_rate_hz"])[mask & valid].sum())
    identity = identify(path, df)
    return {"identity": identity, "table": table, "chosen": chosen, "quality": quality,
            "activity_file": str(activity) if activity else None, "mode": mode,
            "selected_s": selected_s, "candidate_observed_s": candidate_observed_s,
            "coverage_of_recording": selected_s / quality["source_span_s"],
            "candidate_fraction_of_observed": candidate_observed_s / observed_s if observed_s else None,
            "has_unreviewed_candidates": bool(table.candidate_activity_unverified.any())}


def overview_of(run):
    return {**run["identity"], "mode": run["mode"], "activity_file": run["activity_file"],
            "windows_total": len(run["table"]), "windows_used": len(run["chosen"]),
            "selected_s": run["selected_s"], "coverage_of_recording": run["coverage_of_recording"],
            "candidate_fraction_of_observed": run["candidate_fraction_of_observed"],
            "has_unreviewed_candidates": run["has_unreviewed_candidates"],
            "fps": run["quality"]["sample_rate_hz"], "valid_xyz_fraction": run["quality"]["valid_source_fraction"],
            "active": run["quality"]["active"]}


def build_template(files, name, cfg=None, mode="auto", task="unspecified", activities=None):
    cfg = cfg or core.Config()
    runs, segments, weights, seen = [], [], [], set()
    for path in files:
        activity = (activities or {}).get(str(Path(path).resolve()))
        run = analyze_source(path, cfg, mode, activity)
        identity = run["identity"]
        if identity["fingerprint"] in seen:
            raise ValueError(f"Dữ liệu expert bị lặp, dù đổi tên file: {identity['name']}")
        seen.add(identity["fingerprint"])
        if len(run["chosen"]) < 2:
            raise ValueError(f"{identity['name']}: cần >=2 cửa sổ đo được để dựng mẫu; thử cửa sổ dài hơn, kiểm tra tracking/nhãn.")
        chosen = run["chosen"].copy()
        chosen["source_file"] = identity["name"]
        chosen["source_fingerprint"] = identity["fingerprint"]
        segments.append(chosen)
        # Each recording has equal total weight. Within a recording weight by duration.
        weights.extend((chosen.duration_s / chosen.duration_s.sum()).tolist())
        runs.append(run)
    if not runs:
        raise ValueError("Không có file expert để dựng mẫu.")
    merged = pd.concat(segments, ignore_index=True)
    w = np.asarray(weights, float) / len(runs)
    stats = {}
    for key in METRICS:
        v = merged[key].to_numpy(float)
        mean = float(np.sum(w * v))
        empirical = float(np.sqrt(np.sum(w * (v - mean) ** 2)))
        tolerance = max(empirical, REL_FLOOR * abs(mean), ABS_FLOOR[key])
        stats[key] = {"mean": mean, "empirical_std": empirical, "tolerance_std": tolerance,
                      "relative_floor": REL_FLOOR, "absolute_floor": ABS_FLOOR[key], "unit": UNITS[key]}
    distance = sum(WEIGHT[k] * ((merged[k] - stats[k]["mean"]) / stats[k]["tolerance_std"]) ** 2 for k in METRICS)
    representatives = []
    for index in np.argsort(distance.to_numpy())[:5]:
        r = merged.iloc[int(index)]
        representatives.append({"source_file": str(r.source_file), "source_fingerprint": str(r.source_fingerprint),
                                "start_s": float(r.start_s), "end_s": float(r.end_s),
                                "activity_verified": r.activity_label == "polishing",
                                **{k: float(r[k]) for k in METRICS}})
    warnings = []
    if any(r["has_unreviewed_candidates"] for r in runs):
        warnings.append("Mẫu tạm tính có đoạn chuyển động chưa xác nhận là đánh bóng.")
    if len(runs) == 1:
        warnings.append("Chỉ một bản ghi expert: dung sai chủ yếu phản ánh biến thiên trong bản ghi đó.")
    warnings.append("Sàn dung sai là cấu hình kỹ thuật ban đầu, chưa được hiệu chuẩn theo kỹ năng hoặc sai số đo.")
    template = {"schema": SCHEMA, "version": VERSION, "name": safe_name(name), "task": task,
                "created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                "provisional_activity": any(r["has_unreviewed_candidates"] for r in runs),
                "config": asdict(cfg), "weights": WEIGHT, "metrics": stats,
                "n_recordings": len(runs), "n_windows": len(merged),
                "fps_reference": float(np.mean([r["quality"]["sample_rate_hz"] for r in runs])),
                "training_sources": [overview_of(r) for r in runs], "representatives": representatives,
                "warnings": warnings, "aggregation": "equal recording weight; duration weight within recording"}
    return template, merged


def load_template(path):
    path = Path(path)
    if path.is_dir():
        path = path / "template.json"
    if not path.is_file():
        candidates = [Path("templates") / path / "template.json",
                      Path("templates") / (str(path) if str(path).endswith(".json") else str(path) + ".json")]
        path = next((p for p in candidates if p.is_file()), candidates[0])
    template = json.loads(path.read_text(encoding="utf-8-sig"))
    if template.get("schema") != SCHEMA:
        raise ValueError("Template không thuộc baseline 3D này. Dựng lại mẫu; không dùng template 2D cũ.")
    core.Config(**template["config"]).validate()
    for key in METRICS:
        m = template["metrics"][key]
        if not np.isfinite([m["mean"], m["tolerance_std"]]).all() or m["tolerance_std"] <= 0:
            raise ValueError(f"Template có dung sai không hợp lệ: {key}")
    ws = template["weights"]
    if set(ws) != set(METRICS) or any(not np.isfinite(v) or v <= 0 for v in ws.values()) or not np.isclose(sum(ws.values()), 1):
        raise ValueError("Trọng số template không hợp lệ.")
    return template


def feedback(key, value, mean, z):
    """Tạo phản hồi dễ hiểu cho người vận hành."""
    if abs(z) < 1.5:
        return None

    if key == "freq_hz":
        if value < mean:
            return (
                f"Nhịp thao tác đang chậm hơn mẫu chuyên gia: "
                f"trung bình {value:.2f} lượt mỗi giây, trong khi mức tham chiếu "
                f"là {mean:.2f} lượt mỗi giây. "
                "Hãy tăng tốc từ từ và cố gắng duy trì nhịp tay ổn định."
            )

        return (
            f"Nhịp thao tác đang nhanh hơn mẫu chuyên gia: "
            f"trung bình {value:.2f} lượt mỗi giây, trong khi mức tham chiếu "
            f"là {mean:.2f} lượt mỗi giây. "
            "Hãy giảm nhẹ tốc độ để kiểm soát chuyển động tốt hơn."
        )

    if key == "rhythm_s":
        if value > mean:
            return (
                f"Thời gian giữa các lượt đánh chưa đều. "
                f"Mức dao động hiện tại là {value:.3f} giây, cao hơn mức "
                f"{mean:.3f} giây của mẫu chuyên gia. "
                "Hãy giữ tốc độ ổn định và hạn chế thay đổi nhịp đột ngột."
            )

        return (
            f"Nhịp thao tác đang ổn định hơn mức tham chiếu "
            f"({value:.3f} giây so với {mean:.3f} giây). "
            "Đây không nhất thiết là điểm cần điều chỉnh; nên kiểm tra thêm "
            "chất lượng bề mặt sau khi đánh bóng."
        )

    if key == "amp_cm":
        if value < mean:
            return (
                f"Hành trình di chuyển của tay đang ngắn hơn mẫu chuyên gia: "
                f"{value:.2f} cm so với {mean:.2f} cm. "
                "Hãy tăng nhẹ chiều dài mỗi lượt đánh để bao phủ vùng xử lý tốt hơn."
            )

        return (
            f"Hành trình di chuyển của tay đang dài hơn mẫu chuyên gia: "
            f"{value:.2f} cm so với {mean:.2f} cm. "
            "Hãy thu ngắn chuyển động và tập trung tay trong vùng cần đánh bóng."
        )

    return None
def score_source(path, template, activity=None):
    cfg = core.Config(**template["config"])
    run = analyze_source(path, cfg, template["mode"], activity)
    selected = run["chosen"].copy()
    warnings = list(template.get("warnings", []))
    if run["has_unreviewed_candidates"]:
        warnings.append("Điểm tạm tính trên chuyển động lặp lại; hoạt động đánh bóng chưa được xác nhận.")
    trained = any(s["fingerprint"] == run["identity"]["fingerprint"] for s in template["training_sources"])
    if trained:
        warnings.append("File này đã dùng dựng template. Kết quả chỉ là tự kiểm tra, không phải kiểm chứng độc lập.")
    if abs(run["quality"]["sample_rate_hz"] - template["fps_reference"]) > 4:
        warnings.append("FPS lệch quá 4 so với mẫu. Cần kiểm tra độ tương thích phép đo.")
    if run["coverage_of_recording"] < .5:
        warnings.append("Dưới 50% thời gian bản ghi được chấm; điểm chỉ đại diện các đoạn được chọn.")
    base = {"schema": SCHEMA, "version": VERSION, "template": template["name"], "task": template["task"],
            "data_kind": template.get("data_kind", "unspecified"),
            "source": overview_of(run), "self_check_training_source": trained,
            "provisional_activity": template["provisional_activity"] or run["has_unreviewed_candidates"],
            "warnings": list(dict.fromkeys(warnings)), "representatives": template["representatives"],
            "score_meaning": "motion similarity on selected windows, not surface quality or force"}
    if selected.empty:
        return {**base, "status": "no_score", "total_score": None, "metrics": {}, "feedback": [],
                "reason": "Không có đoạn đủ chu kỳ và điều kiện hoạt động. Không gán điểm 0 cho dữ liệu thiếu."}, run["table"]
    total = np.zeros(len(selected))
    metrics, all_feedback = {}, []
    dw = selected.duration_s.to_numpy(float)
    for key in METRICS:
        m = template["metrics"][key]
        z = (selected[key].to_numpy(float) - m["mean"]) / m["tolerance_std"]
        scores = 100 * np.exp(-.5 * np.minimum(z * z, 1500))
        selected[key + "_z"] = z
        selected[key + "_score"] = scores
        total += template["weights"][key] * scores
        value = float(np.average(selected[key], weights=dw))
        metrics[key] = {"value": value, "expert_mean": m["mean"], "unit": UNITS[key],
                        "tolerance_std": m["tolerance_std"], "mean_score": float(np.average(scores, weights=dw))}
    selected["score"] = total
    selected["feedback"] = ""
    for index, row in selected.iterrows():
        items = []
        for key in METRICS:
            msg = feedback(key, row[key], template["metrics"][key]["mean"], row[key + "_z"])
            if msg:
                items.append(msg)
        selected.at[index, "feedback"] = " ".join(items) or "Ba chỉ số gần mức mẫu trong dung sai đang dùng."
        all_feedback.append({"segment_id": int(row.segment_id), "start_s": float(row.start_s),
                             "end_s": float(row.end_s), "score": float(row.score), "messages": items,
                             "values": {key: float(row[key]) for key in METRICS}})
    output = run["table"].merge(selected[["segment_id", "score", "feedback", *[k + suffix for k in METRICS for suffix in ("_z", "_score")]]],
                                on="segment_id", how="left")
    result = {**base, "status": "scored", "total_score": float(np.average(total, weights=dw)),
              "metrics": metrics, "feedback": sorted(all_feedback, key=lambda x: x["score"]),
              "aggregation": "score each window, then duration-weighted average; no cancelling high/low deviations"}
    return result, output


def identities_overlap(a, b):
    if a["fingerprint"] == b["fingerprint"]:
        return True
    # Detect extracted subsets sharing absolute capture times (epoch-like source timestamps).
    x, y = a.get("color_timestamp_range_ms"), b.get("color_timestamp_range_ms")
    if x and y and min(x[0], y[0]) > 1e11:
        return min(x[1], y[1]) - max(x[0], y[0]) > 1000
    return False


def validate_sources(template, experts, novices):
    if not experts or not novices:
        raise ValueError("Kiểm chứng cần ít nhất một file expert giữ riêng và một file novice.")
    identities = []
    for role, paths in [("expert", experts), ("novice", novices)]:
        for path in paths:
            identity = identify(path, core.load_motion(path))
            for training in template["training_sources"]:
                if identities_overlap(identity, training):
                    raise ValueError(f"Rò rỉ dữ liệu: {identity['name']} trùng/chéo bản ghi đã dựng mẫu. Dùng video khác.")
            for prior in identities:
                if identities_overlap(identity, prior):
                    raise ValueError(f"File kiểm chứng bị lặp hoặc chồng thời gian: {identity['name']}")
            identities.append(identity)
    rows = []
    for role, paths in [("expert", experts), ("novice", novices)]:
        for path in paths:
            result, _ = score_source(path, template)
            rows.append({"file": result["source"]["name"], "role": role, "status": result["status"],
                         "score": result["total_score"], "coverage": result["source"]["coverage_of_recording"],
                         "provisional_activity": result["provisional_activity"],
                         "valid_xyz_fraction": result["source"]["valid_xyz_fraction"],
                         **{k: result["metrics"].get(k, {}).get("value") for k in METRICS}})
    table = pd.DataFrame(rows)
    e = table.loc[(table.role == "expert") & table.score.notna(), "score"].to_numpy(float)
    n = table.loc[(table.role == "novice") & table.score.notna(), "score"].to_numpy(float)
    report = {"schema": SCHEMA, "template": template["name"], "mode": template["mode"],
              "data_kind": template.get("data_kind", "unspecified"),
              "n_expert_requested": len(experts), "n_novice_requested": len(novices),
              "n_expert_scored": len(e), "n_novice_scored": len(n),
              "n_no_score": int(table.score.isna().sum()), "status": "insufficient_scored_groups",
              "expert_mean_score": float(e.mean()) if len(e) else None,
              "novice_mean_score": float(n.mean()) if len(n) else None,
              "auc_expert_higher": None, "mean_gap": None, "min_expert_minus_max_novice": None,
              "notes": ["File là đơn vị đánh giá; không dùng từng cửa sổ như một người độc lập.",
                        "Đã chặn file trùng/chéo thời gian huấn luyện. Cần tự giữ riêng phiên quay/người khi chia dữ liệu.",
                        "Không chọn trọng số/ngưỡng bằng chính tập kiểm chứng này.",
                        "AUC mô tả thứ hạng trên các file có điểm. File không chấm được vẫn nằm trong bảng và số lượng thiếu."]}
    if len(e) and len(n):
        comparisons = e[:, None] - n[None, :]
        report.update(status="descriptive_only" if min(len(e), len(n)) < 2 else "held_out_file_comparison",
                      auc_expert_higher=float(np.mean((comparisons > 0) + .5 * (comparisons == 0))),
                      mean_gap=float(e.mean() - n.mean()), min_expert_minus_max_novice=float(e.min() - n.max()))
    if report["data_kind"] == "synthetic":
        report["notes"].insert(0,"DỮ LIỆU GIẢ LẬP: chỉ kiểm tra phần mềm, không phải kết quả trên expert/novice thật.")
    if table.provisional_activity.any():
        report["notes"].append("Có điểm tạm tính trên các đoạn chưa xác nhận hoạt động; xem cột provisional_activity và coverage.")
    return report, table
