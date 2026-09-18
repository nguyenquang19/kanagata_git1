"""Build a 3D expert template using the same processing as score.py."""
import argparse
from pathlib import Path
from baseline import build_template, config_from_file, expand_inputs, new_folder, write_json
from reporting import template_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",nargs="+")
    p.add_argument("--name",default="paper600")
    p.add_argument("--task",default="unspecified",help="Same tool, workpiece zone and setup, e.g. paper600_z1")
    p.add_argument("--mode",choices=["auto","reviewed"],default="auto")
    p.add_argument("--config",type=Path)
    p.add_argument("--output-root",type=Path,default=Path("templates"))
    p.add_argument("--plot",action="store_true",help="Accepted for compatibility; plot is always saved")
    a=p.parse_args()
    try:
        template,segments=build_template(expand_inputs(a.inputs),a.name,config_from_file(a.config),a.mode,a.task)
        folder=new_folder(a.output_root,a.name)
        write_json(folder/"template.json",template)
        segments.to_csv(folder/"segments.csv",index=False,encoding="utf-8-sig")
        template_report(template,segments,folder)
    except (ValueError,OSError,KeyError,TypeError) as e:
        p.exit(2,f"ERROR: {e}\n")
    print(f"Template: {(folder/'template.json').resolve()}")
    print(f"Training report: {(folder/'training.html').resolve()}")
    for warning in template["warnings"]:print("NOTE:",warning)


if __name__=="__main__":main()
