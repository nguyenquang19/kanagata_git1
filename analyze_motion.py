#!/usr/bin/env python3
"""Analyze an extracted metric motion CSV, without modifying its source files."""

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core import Config, XYZ, analyze, load_motion


def new_folder(root, name):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(10000):
        path = root / (name if i == 0 else f"{name}_{i + 1}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            pass
    raise RuntimeError("Too many existing result folders.")


def encoded_plot(fig, path):
    fig.savefig(path, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def plots(df, table, summary, traces, folder):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    plotdir = folder / "plots"
    plotdir.mkdir()
    valid = df.quality_flag.eq("ok").to_numpy() & np.isfinite(df[XYZ]).all(axis=1).to_numpy()
    values = df[XYZ].copy() * 100
    values.loc[~valid, :] = np.nan
    # Do not draw connected lines over suspect coordinate-jump boundaries.
    for boundary in summary["jump_boundaries"]:
        values.loc[np.isclose(df.t, boundary["after_s"], atol=1e-7, rtol=0), :] = np.nan
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, .6]})
    for i, (ax, key) in enumerate(zip(axes[:3], XYZ)):
        ax.plot(df.t, values[key], lw=.65, color=["#196eab", "#18856b", "#9c5aaf"][i])
        ax.set_ylabel(key[5] + " (cm)")
        ax.grid(alpha=.2)
    axes[3].fill_between(df.t, 0, valid.astype(int), step="mid", color="#18856b", alpha=.7)
    axes[3].set_yticks([0, 1], ["Thiếu", "Có XYZ"])
    axes[3].set_xlabel("Thời gian theo CSV / bản ghi gốc (giây)")
    axes[0].set_title("Quỹ đạo bàn tay: dữ liệu trích xuất, chưa phân loại thao tác", loc="left", fontweight="bold")
    overview = encoded_plot(fig, plotdir / "overview.png")
    figures = {}
    for row in table.itertuples(index=False):
        trace = traces[row.segment_id]
        fig, ax = plt.subplots(figsize=(10.5, 2.7))
        ax.plot(trace["t"], trace["projected_cm"], color="#196eab", lw=1.2, label="Chuyển động theo trục chính")
        for key, marker, color in [("pos", "^", "#c94b40"), ("neg", "v", "#18856b")]:
            idx = trace[key]
            ax.scatter(trace["t"][idx], trace["projected_cm"][idx], s=22, marker=marker, color=color, zorder=3)
        ax.set_xlabel("Thời gian theo CSV (giây)")
        ax.set_ylabel("Vị trí đã bỏ xu thế (cm)")
        ax.set_title(f"Đoạn {row.segment_id:03d}: {row.start_s:.2f}–{row.end_s:.2f} s | Điểm tam giác: điểm đảo chiều được phát hiện", fontsize=10, loc="left")
        ax.grid(alpha=.2)
        figures[int(row.segment_id)] = encoded_plot(fig, plotdir / f"segment_{row.segment_id:03d}.png")
    return overview, figures


def fmt(value, digits=2):
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def write_report(folder, source, table, summary, overview, figures):
    rows = []
    for row in table.itertuples(index=False):
        choices = "".join(f'<option value="{v}" {"selected" if row.activity_label == v else ""}>{label}</option>'
                          for v, label in [("unreviewed", "Chưa xác nhận"), ("polishing", "Đang đánh bóng"), ("not_polishing", "Không đánh bóng")])
        state = "Đủ chu kỳ để đo" if row.metric_status == "measured_review_required" else "Ít chu kỳ: chưa tổng hợp"
        rows.append(f'<tr data-id="{row.segment_id}" data-start="{row.start_s:.9f}" data-end="{row.end_s:.9f}">'
                    f'<td><button class="segment" onclick="showSegment({row.segment_id},true)">{row.segment_id:03d}</button></td>'
                    f'<td>{row.start_s:.2f}–{row.end_s:.2f}</td><td>{fmt(row.freq_hz)}</td>'
                    f'<td>{fmt(row.rhythm_s,3)}</td><td>{fmt(row.amp_cm)}</td>'
                    f'<td>{row.complete_cycles_per_polarity}</td><td>{state}</td>'
                    f'<td><select>{choices}</select></td></tr>')
    figures_html = "".join(f'<img class="trace" id="trace-{sid}" src="{uri}" alt="Đồ thị đoạn {sid}" hidden>' for sid, uri in figures.items())
    a = summary["active"]
    active_text = "Chưa tính: chưa có nhãn thời gian thao tác."
    if a["active_fraction_of_reviewed_observed"] is not None:
        active_text = (f'{100*a["active_fraction_of_reviewed_observed"]:.1f}% trong phần thời gian đã xem và có dữ liệu. '
                       f'Phần đã xem chiếm {100*a["reviewed_fraction_of_observed"]:.1f}% thời gian có dữ liệu.')
    first = int(table.iloc[0].segment_id) if len(table) else 0
    report = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kanagata — phân tích chuyển động</title><style>
*{box-sizing:border-box}body{font:15px/1.55 system-ui,sans-serif;color:#173044;background:#eef3f7;margin:0}
main{max-width:1200px;margin:30px auto;padding:0 20px 40px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin-top:0}
.card{background:white;padding:22px;border-radius:12px;margin:18px 0;border:1px solid #dce4ec}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.kpi strong{display:block;font-size:27px;color:#126a81}.muted{color:#526573}img{max-width:100%;height:auto}.note{border-left:4px solid #b87a21;padding-left:15px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #e3e9ef}th{background:#edf4f8;white-space:nowrap}select,button{font:inherit;padding:7px;border:1px solid #bdd0df;border-radius:5px;background:white}button{cursor:pointer;color:#126a81}button.primary{background:#126a81;color:white;padding:10px 16px}.tablewrap{overflow:auto}.trace[hidden]{display:none}code{background:#edf4f8;padding:2px 5px}.selected{background:#e5f1f8}@media(max-width:750px){.grid{grid-template-columns:repeat(2,1fr)}main{padding:0 12px}.card{padding:14px}}
</style><main><h1>Kanagata — phân tích chuyển động</h1>
<p class="muted">Nguồn: SOURCE. Đây là kết quả đo chuyển động; chưa phải mẫu expert hay điểm kỹ năng.</p>
<div class="grid"><div class="card kpi"><strong>DURATION s</strong>Thời gian CSV</div><div class="card kpi"><strong>FPS Hz</strong>Tần suất lấy mẫu</div><div class="card kpi"><strong>VALID%</strong>Dòng có XYZ hợp lệ</div><div class="card kpi"><strong>WINDOWS</strong>Đoạn để xem lại</div></div>
<section class="card"><h2>Kết quả kiểm tra dữ liệu</h2><p>ROWS dòng nguồn. JUMPS ranh giới nhảy tọa độ đã được tách, không nối nội suy qua đó. Đoạn thiếu dài được giữ là chưa quan sát.</p>
<p class="note">Tỷ lệ có tọa độ không phải độ chính xác. Dữ liệu chỉ theo một điểm đại diện bàn tay; khi có hai tay trong ảnh cần xem có đổi nhầm tay không.</p><img src="OVERVIEW" alt="Quỹ đạo XYZ và tình trạng dữ liệu"></section>
<section class="card"><h2>Ba chỉ số chuyển động và thời gian thao tác</h2><p><b>freq:</b> chu kỳ đi–về mỗi giây. <b>rhythm:</b> độ lệch chuẩn khoảng thời gian giữa các chu kỳ, đơn vị giây. <b>amp:</b> quãng đi một chiều giữa hai đầu đường đánh theo trục chính, đơn vị cm.</p>
<p><b>active:</b> ACTIVE</p><p>Không có ngưỡng “nhịp phải đều”, “phải nhanh hơn 1,2 Hz” hay “biên độ phải nhỏ” để loại người mới. Cửa sổ có ít chu kỳ vẫn được giữ, nhưng chưa đủ để tổng hợp ba chỉ số.</p></section>
<section class="card"><h2>Xem lại các đoạn</h2><p>Chọn số đoạn để xem đường chuyển động và các điểm đảo chiều. Đối chiếu video gốc rồi chọn nhãn ở cột cuối. Chỉ chọn “Đang đánh bóng” khi cả đoạn thể hiện thao tác đó; nếu đoạn lẫn nhiều hoạt động, giữ “Chưa xác nhận” hoặc chia nhỏ khoảng thời gian trong CSV nhãn.</p>
<p class="note">Thời gian ở đây thuộc CSV / bản ghi .db3. MP4 được record.py ghi ở FPS cố định có thể chạy nhanh hơn, vì vậy không dùng trực tiếp số giây này để cắt MP4 cũ. segments.csv có số khung hình màu gốc để đối chiếu.</p>
<div id="viewer">FIGURES</div><div class="tablewrap"><table><thead><tr><th>Đoạn</th><th>Thời gian (s)</th><th>freq (Hz)</th><th>rhythm (s)</th><th>amp (cm)</th><th>Chu kỳ*</th><th>Trạng thái đo</th><th>Hoạt động</th></tr></thead><tbody>TABLE</tbody></table></div>
<p class="muted">*Số chu kỳ hoàn chỉnh bảo thủ, lấy số nhỏ hơn giữa hai chiều đếm đỉnh. “Đủ chu kỳ để đo” không có nghĩa là đang đánh bóng hay làm đúng.</p>
<button class="primary" onclick="downloadLabels()">Tải nhãn activity.csv</button><p id="saved" class="muted">Nhãn chỉ nằm trong trang đang mở cho đến khi bạn tải CSV. Chạy lại lệnh có --activity để cập nhật kết quả.</p></section>
<section class="card"><h2>Bước tiếp theo</h2><p>Xác nhận các đoạn thao tác rồi chạy cùng bộ xử lý cho những bản ghi expert và novice khác. Khi có dữ liệu hai nhóm, tạo mẫu từ các video expert dành cho huấn luyện và kiểm tra trên video tách riêng. Chưa có cơ sở để chấm kỹ năng từ một bản ghi expert này.</p></section>
</main><script>
function showSegment(id,scroll=false){document.querySelectorAll('.trace').forEach(x=>x.hidden=true);const el=document.getElementById('trace-'+id);if(el)el.hidden=false;document.querySelectorAll('tbody tr').forEach(x=>x.classList.toggle('selected',Number(x.dataset.id)===id));if(scroll)document.getElementById('viewer').scrollIntoView({behavior:'smooth',block:'start'});}
function downloadLabels(){let csv='start_s,end_s,label\\n';document.querySelectorAll('tbody tr').forEach(r=>{csv+=r.dataset.start+','+r.dataset.end+','+r.querySelector('select').value+'\\n'});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['\\ufeff'+csv],{type:'text/csv;charset=utf-8'}));a.download='activity.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);document.getElementById('saved').textContent='Đã yêu cầu tải activity.csv. Giữ file này và dùng --activity khi chạy lại.';}
showSegment(FIRST);
</script></html>'''
    replacements = {
        "SOURCE": html.escape(Path(source).name), "DURATION": fmt(summary["source_span_s"], 1),
        "FPS": fmt(summary["sample_rate_hz"], 2), "VALID": fmt(100 * summary["valid_source_fraction"], 1),
        "WINDOWS": str(len(table)), "ROWS": str(summary["rows"]), "JUMPS": str(len(summary["jump_boundaries"])),
        "OVERVIEW": overview, "ACTIVE": html.escape(active_text), "FIGURES": figures_html,
        "TABLE": "".join(rows), "FIRST": str(first),
    }
    for key, value in replacements.items():
        report = report.replace(key, value)
    (folder / "report.html").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_csv", type=Path)
    parser.add_argument("--name", default=None, help="Recording name, e.g. expert_A_z1_01")
    parser.add_argument("--output-root", type=Path, default=Path("analysis"))
    parser.add_argument("--activity", type=Path, default=None, help="Reviewed start_s,end_s,label CSV")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--min-segment-seconds", type=float, default=5.0)
    parser.add_argument("--max-bridge-seconds", type=float, default=0.15)
    args = parser.parse_args()
    name = args.name or args.motion_csv.parent.name
    if not name or name in {".", ".."} or any(c in name for c in '/\\:*?"<>|'):
        parser.error("--name must be a plain folder name, not a path.")
    cfg = Config(target_window_s=args.window_seconds, min_segment_s=args.min_segment_seconds,
                 max_bridge_s=args.max_bridge_seconds)
    try:
        cfg.validate()
        df = load_motion(args.motion_csv)
        table, summary, traces, cleaned = analyze(df, cfg, args.activity)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Input error: {exc}\n")
    folder = new_folder(args.output_root, name)
    summary["input_file"] = str(args.motion_csv)
    summary["activity_file"] = str(args.activity) if args.activity else None
    table.to_csv(folder / "segments.csv", index=False, encoding="utf-8-sig")
    cleaned.to_csv(folder / "motion_cleaned.csv", index=False, encoding="utf-8-sig")
    table[["start_s", "end_s", "activity_label"]].rename(columns={"activity_label": "label"}).to_csv(
        folder / "activity_review.csv", index=False, encoding="utf-8-sig")
    (folder / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    overview, figures = plots(df, table, summary, traces, folder)
    write_report(folder, args.motion_csv, table, summary, overview, figures)
    print(f"Rows: {len(df)}; span: {summary['source_span_s']:.2f}s; valid XYZ: {100*summary['valid_source_fraction']:.1f}%")
    print(f"Windows: {len(table)}; enough cycles: {summary['windows_with_enough_cycles']}; no skill score created.")
    print(f"Open report: {(folder / 'report.html').resolve()}")
    print(f"Outputs: {folder.resolve()}")


if __name__ == "__main__":
    main()
