"""Score each selected 3D motion window, then aggregate; write timed feedback."""
import argparse
from pathlib import Path
from baseline import load_template,new_folder,score_source,write_json
from reporting import score_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("motion_csv",type=Path)
    p.add_argument("--template",required=True,type=Path)
    p.add_argument("--activity",type=Path)
    p.add_argument("--output-root",type=Path,default=Path("scores"))
    a=p.parse_args()
    try:
        result,segments=score_source(a.motion_csv,load_template(a.template),a.activity)
        folder=new_folder(a.output_root,result["source"]["name"])
        write_json(folder/"score.json",result)
        segments.to_csv(folder/"segments_scored.csv",index=False,encoding="utf-8-sig")
        score_report(result,segments,folder)
    except (ValueError,OSError,KeyError,TypeError) as e:p.exit(2,f"ERROR: {e}\n")
    print("Score:","NO SCORE" if result["total_score"] is None else f"{result['total_score']:.1f}/100")
    print(f"Report: {(folder/'report.html').resolve()}")
    for warning in result["warnings"]:print("NOTE:",warning)
    return 0 if result["status"]=="scored" else 3


if __name__=="__main__":raise SystemExit(main())
