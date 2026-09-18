"""Compare held-out expert and novice files; reject training overlap."""
import argparse
from pathlib import Path
from baseline import expand_inputs,load_template,new_folder,validate_sources,write_json
from reporting import validation_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template",required=True,type=Path)
    p.add_argument("--expert",nargs="+",required=True)
    p.add_argument("--novice",nargs="+",required=True)
    p.add_argument("--output-root",type=Path,default=Path("validation"))
    a=p.parse_args()
    try:
        result,table=validate_sources(load_template(a.template),expand_inputs(a.expert),expand_inputs(a.novice))
        folder=new_folder(a.output_root,"validation")
        write_json(folder/"validation.json",result)
        table.to_csv(folder/"files.csv",index=False,encoding="utf-8-sig")
        validation_report(result,table,folder)
    except (ValueError,OSError,KeyError,TypeError) as e:p.exit(2,f"ERROR: {e}\n")
    print(table.to_string(index=False))
    print("AUC (expert higher):",result["auc_expert_higher"])
    print(f"Report: {(folder/'report.html').resolve()}")
    return 0 if result["n_no_score"]==0 else 3


if __name__=="__main__":raise SystemExit(main())
