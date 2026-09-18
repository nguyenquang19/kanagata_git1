"""One command for template -> scores -> held-out validation -> report index."""
import argparse
from pathlib import Path
import json

import core
from baseline import build_template,new_folder,score_source,validate_sources,write_json,expand_inputs
from reporting import page,esc,notices,template_report,score_report,validation_report


def run_manifest(manifest_path,output_root=None):
    manifest_path=Path(manifest_path).resolve()
    data=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    # Input paths in the manifest are relative to the PROJECT ROOT (this script's
    # folder), so a manifest kept in manifests/ still points at data/ correctly.
    # Absolute paths in the manifest override this (pathlib join keeps them).
    base=Path(__file__).resolve().parent
    def paths(key):
        result=[]
        for item in data.get(key,[]):
            result.extend(expand_inputs([str(base/item)]))
        return result
    train,test,novices=paths("expert_train"),paths("expert_test"),paths("novice_test")
    if not train:raise ValueError("expert_train trống: cần CSV 3D của expert.")
    cfg=core.Config(**data.get("analysis",{}));cfg.validate()
    template,segments=build_template(train,data.get("name","paper600"),cfg,data.get("mode","auto"),data.get("task","unspecified"))
    template["data_kind"]=data.get("data_kind","real")
    if template["data_kind"]=="synthetic":
        template["warnings"].insert(0,"DỮ LIỆU GIẢ LẬP: chỉ kiểm tra phần mềm, không chứng minh hiệu quả trên người thật.")
    folder=new_folder(output_root or base/"outputs",data.get("name","paper600"))
    td=folder/"template";td.mkdir()
    write_json(td/"template.json",template)
    segments.to_csv(td/"segments.csv",index=False,encoding="utf-8-sig")
    template_report(template,segments,td)
    links=['<li><a href="template/training.html">Mẫu expert và video hướng dẫn</a></li>']
    score_summaries=[]
    for i,path in enumerate(novices,1):
        result,table=score_source(path,template)
        out=folder/f"novice_{i:03d}";out.mkdir()
        write_json(out/"score.json",result)
        table.to_csv(out/"segments_scored.csv",index=False,encoding="utf-8-sig")
        score_report(result,table,out)
        links.append(f'<li><a href="novice_{i:03d}/report.html">Novice: {esc(result["source"]["name"])}</a></li>')
        score_summaries.append({"file":result["source"]["name"],"status":result["status"],"score":result["total_score"]})
    validation={"status":"not_run_missing_held_out_data"}
    if test and novices:
        # validate_sources rejects any overlap with training and duplicate held-out data.
        validation,table=validate_sources(template,test,novices)
        vd=folder/"validation";vd.mkdir()
        write_json(vd/"validation.json",validation)
        table.to_csv(vd/"files.csv",index=False,encoding="utf-8-sig")
        validation_report(validation,table,vd)
        links.append('<li><a href="validation/report.html">Kiểm chứng expert–novice giữ riêng</a></li>')
    kind=data.get("data_kind","real")
    notes=list(template["warnings"])
    if kind=="synthetic":notes.insert(0,"DỮ LIỆU GIẢ LẬP: kết quả chỉ kiểm tra phần mềm, không chứng minh hiệu quả trên đánh bóng thật.")
    if not novices:notes.append("Chưa có CSV novice trong manifest: chưa chạy chấm novice.")
    if not test:notes.append("Chưa có expert giữ riêng: chưa kiểm chứng khả năng phân biệt hai nhóm.")
    body=f'<p>Bối cảnh: {esc(template["task"])}. Chế độ: {esc(template["mode"])}. Loại dữ liệu: {esc(kind)}.</p>'+notices(notes)
    body+='<section class="card"><h2>Kết quả</h2><ul>'+''.join(links)+'</ul></section>'
    body+='<section class="card"><h2>Cách xem video</h2><p>Xuất RGB bằng export_video.py để giữ thời gian tương ứng CSV. Mở báo cáo, chọn video ở máy rồi bấm đoạn muốn xem. CSV không chứa hình ảnh video.</p></section>'
    (folder/"index.html").write_text(page("Kanagata — baseline 3D",body),encoding="utf-8")
    status="completed" if test and novices and validation.get("n_no_score",0)==0 else "completed_available_stages"
    summary={"status":status,"data_kind":kind,"mode":template["mode"],"training_files":len(train),
             "novice_scores":score_summaries,"validation":validation,"manifest":str(manifest_path)}
    write_json(folder/"run_summary.json",summary)
    print("Open:",(folder/"index.html").resolve())
    print("Status:",status,"| Data:",kind)
    return folder,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=Path("dataset.json"))
    p.add_argument("--output-root",type=Path)
    a=p.parse_args()
    try:run_manifest(a.manifest,a.output_root)
    except (ValueError,OSError,KeyError,TypeError) as e:p.exit(2,f"ERROR: {e}\n")


if __name__=="__main__":main()
