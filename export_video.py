"""Export timestamp-aligned RGB video from .db3 for the local training/score reports.

The time origin MUST come from the corresponding motion_3d.csv. Processing speed
does not set playback speed. Video is resampled on a fixed time grid; held frames
and capture gaps are documented. No camera hardware is opened.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from baseline import new_folder,write_json
from extract_motion_3d import timestamp_action


def read_origin(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as file:
        rows=csv.DictReader(file)
        first=next(rows,None)
        if first is None or not first.get("color_timestamp_ms"):
            raise ValueError("Cần CSV trích xuất có color_timestamp_ms để đồng bộ.")
        origin=float(first["color_timestamp_ms"])-1000*float(first["t"])
        domain=first.get("color_clock")
        if not domain:raise ValueError("CSV thiếu color_clock.")
        last=first
        for row in rows:last=row
        end=float(last["t"])
    if not math.isfinite(origin) or not math.isfinite(end) or end<=0:
        raise ValueError("Thời gian CSV không hợp lệ.")
    return origin,domain,end


def slots_before(current_t,next_index,fps):
    """Grid timestamps strictly before current_t are filled using the preceding frame."""
    end=max(next_index,int(math.ceil(current_t*fps-1e-9)))
    return range(next_index,end)


def open_writer(cv2,folder,stem,fps,size):
    # Prefer browser-playable formats. Codec availability depends on the local OpenCV build.
    for extension,codec,browser_expected in [(".webm","VP80",True),(".mp4","avc1",True),(".mp4","mp4v",False)]:
        path=folder/(stem+"_synced"+extension)
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*codec),fps,size)
        if writer.isOpened():return writer,path,codec,browser_expected
        writer.release()
        if path.exists():path.unlink()  # Only the empty failed attempt in our new output folder.
    raise RuntimeError("Không tạo được video. Thử đường dẫn output-root ngắn, không dấu hoặc kiểm tra OpenCV.")


def export(recording,motion_csv,output_root,fps=30):
    import cv2
    import pyrealsense2 as rs
    origin,domain,last_csv_t=read_origin(motion_csv)
    folder=new_folder(output_root,Path(recording).stem)
    destination=None
    pipe=rs.pipeline();cfg=rs.config()
    cfg.enable_device_from_file(str(Path(recording).resolve()),repeat_playback=False)
    cfg.enable_stream(rs.stream.color)
    started=False;writer=None;previous=None;previous_bgr=None;previous_t=None
    next_index=0;captured=0;duplicate=0;held_slots=0;gaps=[];last_t=None
    report={"input_file":Path(recording).name,"motion_csv":str(motion_csv),
            "time_origin_color_timestamp_ms":origin,"fps":fps,"status":"started",
            "time_mapping":"video seconds correspond to CSV t; previous-frame hold on fixed grid"}
    try:
        profile=pipe.start(cfg);started=True
        playback=profile.get_device().as_playback();playback.set_real_time(False)
        while True:
            try:frames=pipe.wait_for_frames(5000)
            except RuntimeError:
                if playback.current_status()==rs.playback_status.stopped:break
                raise
            color=frames.get_color_frame()
            if not color:continue
            current=(color.get_frame_number(),color.get_timestamp(),str(color.get_frame_timestamp_domain()))
            if current[2]!=domain:raise ValueError("Clock domain của .db3 không khớp CSV.")
            if timestamp_action(current,previous)!="accept":duplicate+=1;continue
            previous=current
            elapsed=(current[1]-origin)/1000
            if elapsed < -.1:continue
            if previous_bgr is None and elapsed > .1:
                raise ValueError("Bản ghi không bắt đầu gần mốc CSV. Kiểm tra chọn đúng .db3 và motion_3d.csv.")
            data=np.asanyarray(color.get_data())
            if color.profile.format()==rs.format.bgr8:bgr=data
            elif color.profile.format()==rs.format.rgb8:bgr=cv2.cvtColor(data,cv2.COLOR_RGB2BGR)
            else:raise ValueError("Định dạng ảnh màu chưa được hỗ trợ.")
            if writer is None:
                height,width=bgr.shape[:2]
                writer,destination,codec,browser_expected=open_writer(cv2,folder,Path(recording).stem,fps,(width,height))
                report.update(video=destination.name,codec=codec,browser_playback_expected=browser_expected)
                previous_bgr=bgr.copy();previous_t=elapsed
            if elapsed-previous_t>.2:gaps.append({"start_s":previous_t,"end_s":elapsed})
            if elapsed>last_csv_t+1/fps:break
            indices=slots_before(max(0,elapsed),next_index,fps)
            for _ in indices:writer.write(previous_bgr)
            if len(indices)>1:held_slots+=len(indices)-1
            next_index+=len(indices)
            previous_bgr=bgr.copy();previous_t=elapsed;last_t=elapsed;captured+=1
            if captured%300==0:print(f"Video: {elapsed:.1f}/{last_csv_t:.1f} s",flush=True)
        if last_t is None or abs(last_t-last_csv_t)>.2:
            raise ValueError("Video chưa phủ đủ khoảng thời gian CSV; giữ file một phần và xem metadata lỗi.")
        for _ in slots_before(last_t+1/fps,next_index,fps):writer.write(previous_bgr);next_index+=1
        report["status"]="finished"
    except Exception as error:
        report["status"]="error";report["error"]=str(error)
        raise
    finally:
        if writer is not None:writer.release()
        if started:
            try:pipe.stop()
            except RuntimeError as error:report["pipeline_stop_error"]=str(error)
        report.update(output_frames=next_index,duration_s=next_index/fps,source_frames=captured,
                      duplicates_skipped=duplicate,extra_held_grid_frames=held_slots,capture_gaps_over_0_2s=gaps)
        write_json(folder/"video.json",report)
    print("Video:",destination.resolve())
    if not report["browser_playback_expected"]:
        print("Fallback mp4v: nếu trình duyệt không phát, dùng VLC hoặc lệnh FFmpeg trong README; giữ nguyên thời gian.")
    return destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("recording",type=Path)
    p.add_argument("--motion-csv",required=True,type=Path)
    p.add_argument("--output-root",type=Path,default=Path("videos"))
    p.add_argument("--fps",type=float,default=30)
    a=p.parse_args()
    if not a.recording.is_file():p.error("Không tìm thấy bản ghi.")
    if not math.isfinite(a.fps) or not 1<=a.fps<=120:p.error("FPS phải từ 1 đến 120.")
    try:export(a.recording,a.motion_csv,a.output_root,a.fps)
    except (ValueError,OSError,RuntimeError,ImportError) as error:p.exit(2,f"ERROR: {error}\n")


if __name__=="__main__":main()
