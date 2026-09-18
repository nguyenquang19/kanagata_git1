# Kanagata Polish AI

Hệ thống đo và chấm kỹ thuật **đánh bóng khuôn (kanagata)** từ video chuyển động tay.
Từ bản ghi chiều sâu Intel RealSense, hệ thống theo dõi bàn tay trong không gian 3D,
tách các đoạn thao tác lặp lại và so sánh ba chỉ số chuyển động của người học với mẫu
chuyên gia, kèm phản hồi theo từng đoạn và báo cáo HTML.

> Điểm số phản ánh **mức giống chuyển động của chuyên gia**, không đo lực tay hay chất
> lượng bề mặt khuôn. Thiếu dữ liệu đo được thì trả `no_score`, **không** ép thành điểm 0.

## Luồng chính

```
record.py           →  extract_motion_3d.py   →  run_baseline.py      →  báo cáo HTML
(quay .db3 + mp4)      (.db3 → motion_3d.csv)     (dataset.json → chấm)   (outputs/…/index.html)
```

Cả quy trình có thể thao tác qua **web dashboard** (khuyến nghị) hoặc bằng dòng lệnh.

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

Chỉ khi cần **quay mới** hoặc **trích xuất** từ `.db3`, cài thêm phần camera
(nên dùng đúng môi trường Python đã trích xuất `.db3` thành công trước đây):

```bash
python -m pip install -r requirements-camera.txt
```

## Cách 1 — Giao diện web (khuyến nghị)

```bash
python webapp/app.py
```

Mở `http://127.0.0.1:5000`. Các trang:

| Trang | Chức năng |
| --- | --- |
| **Bảng điều khiển** | Xem bản ghi đã trích xuất, template, dataset, các lần chạy; mở báo cáo |
| **Dựng dataset** | Chọn CSV cho `expert_train` / `expert_test` / `novice_test`, lưu manifest |
| **Chấm nhanh** | Chấm một `motion_3d.csv` với một template có sẵn |
| **Trích xuất** | Chạy `extract_motion_3d.py` trên một `.db3` để tạo `motion_3d.csv` |
| **Quay** | Khởi động `record.py` (mở cửa sổ camera riêng trên desktop) |

Tác vụ dài chạy nền và hiển thị nhật ký trực tiếp; xong sẽ có nút mở báo cáo.
Báo cáo do engine sinh ra được phục vụ tại `/outputs/…` và mở ngay trong trình duyệt.

## Cách 2 — Dòng lệnh

**1. Trích tọa độ từ bản ghi**

```bash
python extract_motion_3d.py "data/raw/expert_A_z1_01.db3"
```

Tạo thư mục trong `data/processed/` chứa `motion_3d.csv`, `summary.json` và ảnh xem trước.
Dùng đúng đường dẫn CSV mà lệnh in ra. Nếu đã có CSV 3D từ bộ trích xuất này, dùng lại luôn.

**2. Tạo `dataset.json`** (hoặc dùng trang *Dựng dataset* của web UI để sinh vào `manifests/`)

```json
{
  "name": "paper600_z1",
  "task": "Đánh bóng thô bằng giấy nhám 600",
  "data_kind": "real",
  "mode": "auto",
  "expert_train": ["data/processed/expert_A_z1_01_3d/motion_3d.csv"],
  "expert_test": [],
  "novice_test": ["data/processed/novice_A_z1_01_3d/motion_3d.csv"]
}
```

| Nhóm | Điền gì |
| --- | --- |
| `expert_train` | CSV expert dùng dựng mẫu; ≥1 file |
| `expert_test` | CSV expert khác, giữ riêng để kiểm chứng; chưa có thì `[]` |
| `novice_test` | CSV người học cần chấm; chưa có thì `[]` |

**3. Chạy**

```bash
python run_baseline.py --manifest manifests/paper600_z1.json
```

| Có gì trong manifest | Chương trình làm |
| --- | --- |
| Chỉ `expert_train` | Dựng template + báo cáo mẫu expert |
| Thêm `novice_test` | Chấm từng novice + phản hồi theo đoạn |
| Đủ cả ba nhóm | Chạy thêm kiểm chứng expert–novice giữ riêng |

Mở `index.html` mà lệnh in ra; kết quả nằm trong `outputs/`.

**4. (Tùy chọn) Xuất video khớp mốc thời gian để xem trong báo cáo**

```bash
python export_video.py "data/raw/expert_A_z1_01.db3" --motion-csv "data/processed/expert_A_z1_01_3d/motion_3d.csv"
```

## Ba chỉ số chấm điểm

| Chỉ số | Ý nghĩa | Trọng số |
| --- | --- | --- |
| `freq_hz` | Tần số đi–về (nhịp thao tác) | 0.45 |
| `rhythm_s` | Độ lệch chuẩn thời gian chu kỳ (độ đều) | 0.35 |
| `amp_cm` | Biên độ một chiều của bàn tay theo trục PCA | 0.20 |

Chế độ `auto` chọn các đoạn có đủ chuyển động lặp lại để đo; đây **chưa** phải bộ nhận diện
đánh bóng đã kiểm chứng. Khi chưa xác nhận hoạt động, điểm ghi là **tạm tính**. Muốn chắc,
chạy `analyze_motion.py` trên từng CSV, đối chiếu video rồi tạo `activity.csv` và dùng
`"mode": "reviewed"` để chỉ chấm các khoảng đã xác nhận là `polishing`.

## Vai trò các file

| File | Tác dụng |
| --- | --- |
| `core.py` | Lọc dữ liệu, chia đoạn, đo chu kỳ, tính chỉ số (module dùng chung) |
| `baseline.py` | Dựng mẫu, chấm điểm, kiểm chứng, sinh phản hồi (module dùng chung) |
| `reporting.py` | Sinh báo cáo HTML (module dùng chung) |
| `run_baseline.py` | Chạy cả cụm theo `dataset.json` |
| `build_template.py` / `score.py` / `validate.py` | Chạy riêng từng bước qua dòng lệnh |
| `extract_motion_3d.py` | Đọc `.db3`, theo dõi tay, xuất XYZ theo mét |
| `record.py` | Quay bản ghi mới bằng camera RealSense |
| `inspect_recording.py` | Soi nhanh một `.db3` (không mở camera) |
| `analyze_motion.py` | Xem tín hiệu, xác nhận đoạn đánh bóng |
| `export_video.py` | Xuất video RGB khớp mốc thời gian CSV |
| `convert_ros2_palm.py` | Chuyển bản ghi ROS2 sang CSV tọa độ tay |
| `webapp/` | Web dashboard Flask bọc toàn bộ pipeline |

## Cấu trúc thư mục

```
core/baseline/reporting/run_baseline + các CLI   # engine 3D (flat)
webapp/          # giao diện web (app.py, templates/, static/)
data/            # bản ghi thô (.db3/.mp4) và motion_3d.csv đã trích  (không commit)
manifests/       # dataset.json do UI/hoặc bạn tạo
outputs/         # kết quả mỗi lần chạy (template, điểm, báo cáo)      (không commit)
templates/       # nơi lưu template được đặt tên (tùy chọn)
```

## Lưu ý

- Quay novice phải dùng **cùng setup** như khi quay expert: khoảng cách, góc, độ phân giải, fps.
- Khả năng đọc `.db3` và xuất video phụ thuộc SDK RealSense và codec trên máy bạn.
- `data/` và `frames/` đã được `.gitignore`; không commit dữ liệu nặng lên git.
