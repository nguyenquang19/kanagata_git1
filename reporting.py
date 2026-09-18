"""Portable local HTML reports. No server or remote services required."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline import METRICS


CSS = """
*{box-sizing:border-box}body{margin:0;background:#f0f4f7;color:#193547;font:15px/1.55 system-ui,sans-serif}
main{max-width:1150px;margin:28px auto;padding:0 20px 40px}h1{font-size:28px;margin:0 0 8px}h2{font-size:20px;margin:0 0 12px}
.card{background:white;border:1px solid #dbe4eb;border-radius:10px;padding:20px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.grid.two{grid-template-columns:1fr 1fr}.big{font-size:32px;display:block;color:#176780}.muted{color:#566b78}.notice{background:#fff4de;border-left:4px solid #be8724;padding:12px 16px;margin:12px 0}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:10px 8px;border-bottom:1px solid #e0e8ee;text-align:left}th{background:#edf4f8}.scroll{overflow:auto}button,input,select{font:inherit}button{padding:7px 10px;color:#176780;border:1px solid #b3c9d7;border-radius:5px;background:white;cursor:pointer}video,img{max-width:100%;width:100%}code{background:#edf4f8;padding:2px 4px}input[type=file]{max-width:100%;margin:8px 0}li{margin:5px 0}@media(max-width:750px){.grid,.grid.two{grid-template-columns:1fr}main{padding:0 12px}h1{font-size:24px}}
"""


def esc(value):
    return html.escape(str(value), quote=True)


def num(value, decimals=2):
    return "—" if value is None or pd.isna(value) else f"{value:.{decimals}f}"


def page(title, body, script=""):
    return f'<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style><main><h1>{esc(title)}</h1>{body}</main><script>{script}</script></html>'


def notices(items):
    return "".join(f'<div class="notice">{esc(item)}</div>' for item in items)


def video_panel(references, novice=False):
    # File inputs keep videos local. Only the newly exported, timestamp-aligned videos support seeking.
    choices = "".join(f'<option value="{i}">{esc(r["source_file"])}: {r["start_s"]:.2f}–{r["end_s"]:.2f} s</option>' for i, r in enumerate(references))
    expert = f'<div><b>Video expert</b><p id="expected" class="muted"></p><input type="file" accept="video/*" onchange="loadVideo(this,\'expertVideo\')"><video id="expertVideo" controls preload="metadata"></video><select id="reference" onchange="referenceChanged()">{choices}</select><p><button onclick="playReference()">Xem đoạn mẫu</button></p></div>'
    learner = '<div><b>Video của bản ghi đang chấm</b><p class="muted">Chọn video đồng bộ của đúng bản ghi này.</p><input type="file" accept="video/*" onchange="loadVideo(this,\'noviceVideo\')"><video id="noviceVideo" controls preload="metadata"></video><p id="cueNote" class="muted">Chọn “Xem” ở bảng từng đoạn để chuyển tới thời điểm tương ứng.</p></div>' if novice else ''
    body = f'<section class="card"><h2>Video hướng dẫn và đối chiếu</h2><p>Dùng video tạo bởi <code>export_video.py</code> để thời gian khớp CSV. Video không được gửi ra ngoài máy. Chọn đúng bản ghi theo tên được ghi dưới đây.</p><div class="grid two">{expert}{learner}</div></section>'
    data = json.dumps(references, ensure_ascii=False).replace("<", "\\u003c")
    script = """
const refs=REFERENCES;
function loadVideo(input,id){const v=document.getElementById(id);if(v.dataset.url)URL.revokeObjectURL(v.dataset.url);if(input.files.length){v.src=URL.createObjectURL(input.files[0]);v.dataset.url=v.src;v.load();}}
function getRef(){return refs[Number(document.getElementById('reference').value)||0];}
function referenceChanged(){const r=getRef();if(r)document.getElementById('expected').textContent='Chọn video của '+r.source_file+(r.activity_verified?' — hoạt động đã xác nhận.':' — đoạn mẫu tự chọn, cần đối chiếu hoạt động.');}
function playRange(id,start,end){const v=document.getElementById(id);if(!v.src){alert('Hãy chọn video đồng bộ của đúng bản ghi.');return;}v.currentTime=start;v.dataset.end=end;v.play().catch(()=>{});}
function playReference(){const r=getRef();if(r)playRange('expertVideo',r.start_s,r.end_s);}
function cue(start,end){document.getElementById('cueNote').textContent='Đoạn '+start.toFixed(2)+'–'+end.toFixed(2)+' s';playRange('noviceVideo',start,end);document.getElementById('noviceVideo').scrollIntoView({block:'center',behavior:'smooth'});}
document.querySelectorAll('video').forEach(v=>v.addEventListener('timeupdate',()=>{if(v.dataset.end&&v.currentTime>=Number(v.dataset.end)){v.pause();delete v.dataset.end;}}));referenceChanged();
""".replace("REFERENCES", data)
    return body, script


def plot_data_uri(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def template_report(template, segments, folder):
    folder = Path(folder)
    stats = "".join(f'<tr><td>{k}</td><td>{num(template["metrics"][k]["mean"],3)}</td><td>{num(template["metrics"][k]["empirical_std"],3)}</td><td>{num(template["metrics"][k]["tolerance_std"],3)}</td><td>{template["metrics"][k]["unit"]}</td></tr>' for k in METRICS)
    body = f'<p>Mẫu <b>{esc(template["name"])}</b> · Bối cảnh: {esc(template["task"])} · Chế độ: {esc(template["mode"])}</p>'
    body += notices(template["warnings"])
    body += f'<section class="card"><h2>Mức tham chiếu từ {template["n_recordings"]} bản ghi expert</h2><p>{template["n_windows"]} cửa sổ được dùng. Mỗi bản ghi có tổng trọng số bằng nhau.</p><table><tr><th>Chỉ số</th><th>Trung bình</th><th>Độ lệch chuẩn quan sát</th><th>Dung sai dùng chấm điểm</th><th>Đơn vị</th></tr>{stats}</table></section>'
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, key in zip(axes, METRICS):
        ax.hist(segments[key], bins=min(15, max(3, len(segments) // 2)), color="#4c91ad", edgecolor="white")
        ax.axvline(template["metrics"][key]["mean"], color="#c97724")
        ax.set_title(key)
        ax.set_xlabel(template["metrics"][key]["unit"])
    body += '<section class="card"><h2>Phân bố các cửa sổ</h2><p>Biểu đồ đếm cửa sổ; thống kê template dùng trọng số cân bằng giữa các bản ghi.</p><img alt="Phân bố ba chỉ số" src="' + plot_data_uri(fig, folder / "distribution.png") + '"></section>'
    v, js = video_panel(template["representatives"])
    body += v
    (folder / "training.html").write_text(page("Kanagata — mẫu expert và video hướng dẫn", body, js), encoding="utf-8")


def score_report(result, segments, folder):
    folder = Path(folder)
    src = result["source"]
    mode = "TẠM TÍNH: hoạt động chưa xác nhận" if result["provisional_activity"] else "Hoạt động đã được xác nhận"
    body = f'<p>{esc(src["name"])} so với mẫu {esc(result["template"])}. {esc(mode)}.</p>'
    body += notices(result["warnings"])
    body += f'<div class="grid"><div class="card"><strong class="big">{num(result["total_score"],1)}</strong>Điểm giống mẫu / 100</div><div class="card"><strong class="big">{100*src["coverage_of_recording"]:.1f}%</strong>Thời gian bản ghi được chấm</div><div class="card"><strong class="big">{src["windows_used"]}/{src["windows_total"]}</strong>Cửa sổ được dùng</div></div>'
    body += '<p>Điểm cao nghĩa là gần mẫu ở ba chỉ số chuyển động. Không suy ra lực tay hay chất lượng bề mặt. Điểm thấp không tự chứng minh thao tác sai.</p>'
    if result["status"] == "no_score":
        body += notices([result["reason"]])
    metrics = ''.join(f'<tr><td>{k}</td><td>{num(v["value"],3)}</td><td>{num(v["expert_mean"],3)}</td><td>{v["unit"]}</td><td>{num(v["mean_score"],1)}</td></tr>' for k,v in result["metrics"].items())
    body += f'<section class="card"><h2>So sánh chỉ số</h2><table><tr><th>Chỉ số</th><th>Bản ghi này</th><th>Expert</th><th>Đơn vị</th><th>Điểm trung bình từng đoạn</th></tr>{metrics}</table><p>Điểm tổng được tính từng đoạn rồi lấy trung bình theo thời lượng; các đoạn nhanh và chậm không triệt tiêu nhau.</p></section>'
    a = src["active"]
    text = "Chưa xác định active thực: chưa có nhãn hoạt động."
    if a["active_fraction_of_reviewed_observed"] is not None:
        text = f'active = {100*a["active_fraction_of_reviewed_observed"]:.1f}% trong thời gian đã xem và có dữ liệu. Đã xem {100*a["reviewed_fraction_of_observed"]:.1f}% thời gian quan sát được.'
    body += f'<section class="card"><h2>Thời gian thao tác</h2><p>{text}</p><p>Tỷ lệ thời gian có chuyển động được chọn tự động trong phần quan sát được: {num(None if src["candidate_fraction_of_observed"] is None else 100*src["candidate_fraction_of_observed"],1)}%. Đây không phải active đã xác nhận và không được cộng điểm.</p></section>'
    v, js = video_panel(result["representatives"], novice=True)
    body += v
    feedback_rows = []
    for r in result["feedback"]:
        messages = '<br>'.join(esc(s) for s in r["messages"]) or "Ba chỉ số gần mức mẫu trong dung sai đang dùng."
        feedback_rows.append(f'<tr><td>{r["start_s"]:.2f}–{r["end_s"]:.2f} s</td><td>{num(r["score"],1)}</td><td>{messages}</td><td><button onclick="cue({r["start_s"]},{r["end_s"]})">Xem</button></td></tr>')
    body += '<section class="card"><h2>Phản hồi theo đoạn</h2><p>Xếp từ đoạn có điểm thấp hơn. Xem hoạt động thực tế trước khi áp dụng gợi ý.</p><div class="scroll"><table><tr><th>Thời gian</th><th>Điểm</th><th>Nhận xét</th><th>Video</th></tr>'+''.join(feedback_rows)+'</table></div></section>'
    (folder / "report.html").write_text(page("Kanagata — so sánh chuyển động", body, js), encoding="utf-8")


def validation_report(result, table, folder):
    body = '<p>So sánh trên các bản ghi được giữ riêng, mỗi file là một quan sát.</p>'
    body += notices(result["notes"])
    body += f'<section class="card"><h2>Kết quả mô tả</h2><p>Expert có điểm: {result["n_expert_scored"]}/{result["n_expert_requested"]}. Novice có điểm: {result["n_novice_scored"]}/{result["n_novice_requested"]}. Không chấm được: {result["n_no_score"]}.</p><p>Điểm trung bình expert: {num(result["expert_mean_score"],1)}; novice: {num(result["novice_mean_score"],1)}. AUC với giả thiết expert có điểm cao hơn: <b>{num(result["auc_expert_higher"],3)}</b>.</p><p>AUC = 0,5 tương ứng không có ưu thế thứ hạng trong so sánh cặp; 1 nghĩa là mọi file expert có điểm cao hơn mọi file novice trong tập này. Kết quả không tự chứng minh khả năng tổng quát.</p><p>Trạng thái: {esc(result["status"])}.</p></section>'
    body += '<section class="card"><h2>Từng bản ghi</h2><div class="scroll">'+table.to_html(index=False,escape=True,na_rep="Không có điểm",float_format=lambda v:f"{v:.3f}")+'</div></section>'
    Path(folder,"report.html").write_text(page("Kanagata — kiểm chứng trên video giữ riêng",body),encoding="utf-8")
