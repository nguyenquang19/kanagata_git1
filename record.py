"""
record.py
Script quay chinh thuc cho buoi lay data.
Moi take ghi ra: .bag (RGB+depth tho) + .mp4 (RGB goc) + _depth.mp4 (xem)
                 + .csv (toa do tay) + 1 dong trong meta.cs
Chay:
    python record.py expert_A_z1          # expert A, vung 1
    python record.py novice_X_z1
    python record.py expert_A_z1 --test   # thu, KHONG ghi file
    python record.py expert_A_z1 --nobag  # bo .bag cho nhe (chi mp4+csv)

Phim:
    r      = BAT DAU / KET THUC mot take
    q      = thoat
    d      = bat/tat khung depth
    m      = bat/tat mask mau
    [ ]    = chinh nguong NEAR
    - =    = chinh nguong FAR

Quy trinh 1 take:
    1. Nguoi lam vao tu the, DUNG YEN TAY
    2. Bam 'r'
    3. Cho chu "DUNG YEN TAY" tat (2.5s), ra hieu bat dau, lam ~30s
    4. Dung lai, de yen tay 2 giay
    5. Bam 'r' -> tu luu, in track%
    6. Track < 70% -> quay lai take do ngay
"""

import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import sys
import os
import csv
import time
from datetime import datetime

# ------------------------------------------------------------------ config
OUT_DIR = "data/raw"
META = "data/meta.csv"

TOOL = "abrasive_stone"     # SUA theo dung cu that (hoi ky su sang ngay 1)
STEP = "arai"               # 粗磨き

RELIABLE = [0, 5, 9, 13, 17]          # co tay + goc ngon, khong bi dung cu che
RED1 = ((0, 120, 70), (10, 255, 255))
RED2 = ((170, 120, 70), (180, 255, 255))

NEAR, FAR = 0.60, 0.85
W, H, FPS = 640, 480, 30


# ------------------------------------------------------------------ helpers

def next_index(prefix):
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(OUT_DIR, f"{prefix}_{n:02d}.mp4")) or \
            os.path.exists(os.path.join(OUT_DIR, f"{prefix}_{n:02d}.db3")):
        n += 1
    return n


def find_marker(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(RED1[0]), np.array(RED1[1])) | \
        cv2.inRange(hsv, np.array(RED2[0]), np.array(RED2[1]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 150:
        return None, mask
    M = cv2.moments(c)
    h, w = mask.shape
    return (M['m10'] / M['m00'] / w, M['m01'] / M['m00'] / h), mask


def depth_3color(depth, scale, near, far):
    d = depth * scale
    out = np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8)
    v = depth > 0
    out[v & (d < near)] = (0, 255, 0)
    out[v & (d >= near) & (d < far)] = (0, 255, 255)
    out[v & (d >= far)] = (0, 0, 255)
    return out


def depth_at(depth, scale, nx, ny, k=5):
    if depth is None:
        return 0.0
    h, w = depth.shape
    x, y = int(nx * w), int(ny * h)
    p = depth[max(0, y-k):min(h, y+k+1), max(0, x-k):min(w, x+k+1)]
    p = p[p > 0]
    return float(np.median(p)) * scale if p.size else 0.0


def put(img, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1)


def append_meta(row):
    os.makedirs(os.path.dirname(META), exist_ok=True)
    new = not os.path.exists(META)
    with open(META, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "person", "role", "zone", "tool", "step",
                        "datetime", "frames", "duration_s", "fps_real",
                        "track_rate", "marker_rate", "depth_rate",
                        "near", "far",
                        "expert_score", "expert_comment", "note"])
        w.writerow(row)


def parse_prefix(prefix):
    """expert_A_z1 -> (person='A', role='expert', zone='z1')"""
    parts = prefix.split("_")
    role = "expert" if parts[0].startswith("expert") else "novice"
    person = parts[1] if len(parts) > 1 else "?"
    zone = parts[2] if len(parts) > 2 else ""
    return person, role, zone


# ------------------------------------------------------------------ main

def main(prefix, test_mode=False, no_bag=False):
    person, role, zone = parse_prefix(prefix)
    near, far = NEAR, FAR

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3)
    draw = mp.solutions.drawing_utils

    show_depth, show_mask = True, False
    recording = False
    pipe = None
    scale = 1.0
    align = rs.align(rs.stream.color)

    n = hit_mp = hit_mk = hit_z = 0
    rows = []
    t0 = 0.0
    cur_name = ""
    wr_rgb = wr_dep = None

    def start_pipe(record_path=None):
        nonlocal pipe, scale
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        if record_path:
            cfg.enable_record_to_file(record_path)
        prof = pipe.start(cfg)
        scale = prof.get_device().first_depth_sensor().get_depth_scale()

    def stop_pipe():
        nonlocal pipe
        if pipe:
            try:
                pipe.stop()
            except RuntimeError:
                pass
            pipe = None

    start_pipe()
    print(f"San sang. prefix={prefix}  person={person} role={role} zone={zone}")
    print("r = bat dau/ket thuc take    q = thoat")
    if test_mode:
        print("*** CHE DO TEST - khong ghi file ***")
    if no_bag:
        print("*** --nobag: chi ghi mp4 + csv, khong ghi .bag ***")

    try:
        while True:
            f = align.process(pipe.wait_for_frames())
            c, dfr = f.get_color_frame(), f.get_depth_frame()
            if not c:
                continue
            bgr = np.asanyarray(c.get_data())
            depth = np.asanyarray(dfr.get_data()) if dfr else None
            frame = bgr.copy()
            h, w = frame.shape[:2]

            # --- track ---
            res = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            palm, pz = None, 0.0
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                pts = np.array([[lm.landmark[i].x, lm.landmark[i].y]
                                for i in RELIABLE])
                palm = pts.mean(axis=0)
                pz = depth_at(depth, scale, palm[0], palm[1])
                draw.draw_landmarks(frame, lm,
                                    mp.solutions.hands.HAND_CONNECTIONS)
                cv2.circle(frame, (int(palm[0]*w), int(palm[1]*h)),
                           10, (0, 255, 0), -1)
                if pz:
                    cv2.putText(frame, f"{pz:.2f}m",
                                (int(palm[0]*w)+14, int(palm[1]*h)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            marker, mask = find_marker(bgr)
            mz = depth_at(depth, scale, *marker) if marker else 0.0
            if marker:
                cv2.circle(frame, (int(marker[0]*w), int(marker[1]*h)),
                           10, (255, 0, 255), -1)

            # --- ghi khi dang record ---
            if recording:
                n += 1
                if palm is not None:
                    hit_mp += 1
                    if pz > 0:
                        hit_z += 1
                if marker:
                    hit_mk += 1
                rows.append([
                    n, round(time.time()-t0, 3),
                    round(palm[0], 5) if palm is not None else "",
                    round(palm[1], 5) if palm is not None else "",
                    round(pz, 4) if pz else "",
                    round(marker[0], 5) if marker else "",
                    round(marker[1], 5) if marker else "",
                    round(mz, 4) if mz else "",
                ])
                # ghi frame GOC (khong co overlay) de con xu ly lai duoc
                if wr_rgb is not None:
                    wr_rgb.write(bgr)
                if wr_dep is not None and depth is not None:
                    wr_dep.write(depth_3color(depth, scale, near, far))

            # --- overlay man hinh ---
            if recording:
                el = time.time() - t0
                cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 4)
                put(frame, f"REC {cur_name}  {el:4.1f}s", 30, (0, 0, 255), 0.7)
                put(frame, f"MediaPipe {hit_mp/max(n,1):6.1%}", 58,
                    (0, 255, 0) if hit_mp/max(n, 1) >= .8 else (0, 0, 255))
                put(frame, f"Marker    {hit_mk/max(n,1):6.1%}", 82,
                    (255, 0, 255))
                put(frame, f"Depth@tay {hit_z/max(hit_mp,1):6.1%}", 106,
                    (0, 255, 255))
                if el < 2.5:
                    put(frame, "DUNG YEN TAY...", 140, (0, 255, 255), 0.8)
            else:
                put(frame, "PREVIEW - bam 'r' de bat dau", 30, (0, 255, 255))
                put(frame, f"tiep theo: {prefix}_{next_index(prefix):02d}", 58)

            panes = [frame]
            if depth is not None and show_depth:
                vd = depth_3color(depth, scale, near, far)
                put(vd, f"near {near:.2f} far {far:.2f}", 30)
                put(vd, "xanh=gan vang=vua do=xa den=mat", 55)
                panes.append(vd)
            if show_mask:
                panes.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            view = np.hstack(panes)
            if view.shape[1] > 1600:
                view = cv2.resize(view, None, fx=0.6, fy=0.6)
            cv2.imshow("RECORD", view)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                if recording:
                    print("Dang ghi - bam 'r' de dung truoc khi thoat.")
                else:
                    break

            elif k == ord('r'):
                if not recording:
                    # ---------------- BAT DAU TAKE ----------------
                    idx = next_index(prefix)
                    cur_name = f"{prefix}_{idx:02d}"
                    n = hit_mp = hit_mk = hit_z = 0
                    rows = []
                    t0 = time.time()
                    if not test_mode:
                        if not no_bag:
                            stop_pipe()
                            try:
                                start_pipe(os.path.join(OUT_DIR,
                                                        cur_name + ".db3"))
                            except RuntimeError as e:
                                print(f"    ! khong ghi duoc bag: {e}")
                                print("    -> chi ghi mp4 + csv")
                                stop_pipe()
                                start_pipe()
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        wr_rgb = cv2.VideoWriter(
                            os.path.join(OUT_DIR, cur_name + ".mp4"),
                            fourcc, FPS, (W, H))
                        wr_dep = cv2.VideoWriter(
                            os.path.join(OUT_DIR, cur_name + "_depth.mp4"),
                            fourcc, FPS, (W, H))
                    recording = True
                    print(f"\n>>> BAT DAU {cur_name}")
                else:
                    # ---------------- KET THUC TAKE ----------------
                    recording = False
                    dur = time.time() - t0
                    if not test_mode:
                        if wr_rgb:
                            wr_rgb.release()
                            wr_rgb = None
                        if wr_dep:
                            wr_dep.release()
                            wr_dep = None
                        if not no_bag:
                            stop_pipe()
                            start_pipe()
                        with open(os.path.join(OUT_DIR, cur_name + ".csv"),
                                  "w", newline="") as cf:
                            wtr = csv.writer(cf)
                            wtr.writerow(["frame", "t", "palm_x", "palm_y",
                                          "palm_z", "mk_x", "mk_y", "mk_z"])
                            wtr.writerows(rows)
                        append_meta([
                            cur_name, person, role, zone, TOOL, STEP,
                            datetime.now().strftime("%Y-%m-%d %H:%M"),
                            n, round(dur, 1), round(n/max(dur, 1e-6), 1),
                            round(hit_mp/max(n, 1), 3),
                            round(hit_mk/max(n, 1), 3),
                            round(hit_z/max(hit_mp, 1), 3),
                            near, far, "", "", "",
                        ])
                    r = hit_mp / max(n, 1)
                    fps_real = n / max(dur, 1e-6)
                    print(f"<<< XONG {cur_name}  {n} frame  {dur:.1f}s  "
                          f"({fps_real:.1f} fps)")
                    print(f"    MediaPipe {r:.1%}  Marker "
                          f"{hit_mk/max(n,1):.1%}  Depth "
                          f"{hit_z/max(hit_mp,1):.1%}")
                    if r < 0.70:
                        print("    !!! TRACK THAP - QUAY LAI TAKE NAY")
                    elif r < 0.80:
                        print("    ! track hoi thap, can nhac quay lai")
                    else:
                        print("    OK")
                    if fps_real < 20:
                        print(f"    ! fps that chi {fps_real:.1f} - "
                              f"cot 't' trong csv van dung duoc, "
                              f"nhung nhip se tho hon")

            elif k == ord('d'):
                show_depth = not show_depth
            elif k == ord('m'):
                show_mask = not show_mask
            elif k == ord('['):
                near = max(0.05, near - 0.02)
            elif k == ord(']'):
                near = min(far - 0.02, near + 0.02)
            elif k == ord('-'):
                far = max(near + 0.02, far - 0.02)
            elif k == ord('='):
                far += 0.02
    finally:
        if wr_rgb:
            wr_rgb.release()
        if wr_dep:
            wr_dep.release()
        stop_pipe()
        cv2.destroyAllWindows()
        print(f"\nKet thuc. Xem {META}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Dung: python record.py <prefix> [--test] [--nobag]")
        print("  vd: python record.py expert_A_z1")
        sys.exit(1)
    main(sys.argv[1], "--test" in sys.argv, "--nobag" in sys.argv)