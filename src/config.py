"""
Cấu hình trung tâm cho dự án NEU Lunch Chatbot:
1. Tập trung toàn bộ đường dẫn dữ liệu mặc định (data/dishes.csv, config/profiles.json, config/crowd_config.json...).
2. Đọc tham số mô hình độ đông & chỗ ngồi từ config/crowd_config.json (không viết cứng hệ số trong code).
3. Hàm suy luận khung giờ đi ăn (cao_diem, binh_thuong, vang) từ đồng hồ hệ thống hoặc chuỗi giờ HH:MM.
"""

import json
import os
from datetime import datetime, time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)

# Đường dẫn tương đối chuẩn (khi chạy từ thư mục gốc)
REL_DISHES_PATH = os.path.join("data", "dishes.csv")
REL_LEGACY_DISHES_V1_PATH = os.path.join("data", "legacy", "dishes_v1.csv")
REL_INTENTS_PATH = os.path.join("data", "intents.csv")
REL_TEST_INTENTS_PATH = os.path.join("data", "test_intents.csv")
REL_FEEDBACK_PATH = os.path.join("data", "feedback.csv")
REL_PROFILES_PATH = os.path.join("config", "profiles.json")
REL_CROWD_CONFIG_PATH = os.path.join("config", "crowd_config.json")
REL_OUTPUTS_DIR = "outputs"
REL_MODEL_PATH = os.path.join("outputs", "classifier_model.joblib")

# Đường dẫn tuyệt đối chuẩn (không phụ thuộc thư mục làm việc hiện tại)
DEFAULT_DISHES_PATH = os.path.join(ROOT_DIR, REL_DISHES_PATH)
LEGACY_DISHES_V1_PATH = os.path.join(ROOT_DIR, REL_LEGACY_DISHES_V1_PATH)
DEFAULT_INTENTS_PATH = os.path.join(ROOT_DIR, REL_INTENTS_PATH)
DEFAULT_TEST_INTENTS_PATH = os.path.join(ROOT_DIR, REL_TEST_INTENTS_PATH)
DEFAULT_FEEDBACK_PATH = os.path.join(ROOT_DIR, REL_FEEDBACK_PATH)
DEFAULT_PROFILES_PATH = os.path.join(ROOT_DIR, REL_PROFILES_PATH)
DEFAULT_CROWD_CONFIG_PATH = os.path.join(ROOT_DIR, REL_CROWD_CONFIG_PATH)
DEFAULT_OUTPUTS_DIR = os.path.join(ROOT_DIR, REL_OUTPUTS_DIR)
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, REL_MODEL_PATH)

# Giá trị mặc định khi đọc file CSV cũ thiếu các cột độ đông / chỗ ngồi
DEFAULT_DO_DONG_CAO_DIEM = 3
DEFAULT_SO_CHO_NGOI = 30
DEFAULT_NGUON_DU_LIEU = "mô phỏng"
ALLOWED_DATA_SOURCES = ("mô phỏng", "quan sát", "khảo sát")

# Các khung giờ hợp lệ
TIME_SLOTS = ("cao_diem", "binh_thuong", "vang")
TIME_SLOT_AUTO = "tu_dong"

TIME_SLOT_LABELS_VI: Dict[str, str] = {
    "cao_diem": "Cao điểm (11h30–13h00)",
    "binh_thuong": "Bình thường (11h00–11h30, 13h00–14h00)",
    "vang": "Vắng (khung giờ khác)",
    "tu_dong": "Tự động theo đồng hồ",
}

TIME_SLOT_SHORT_VI: Dict[str, str] = {
    "cao_diem": "Cao điểm",
    "binh_thuong": "Bình thường",
    "vang": "Vắng",
}


def resolve_project_path(path: str) -> str:
    """Trả về đường dẫn hợp lệ: nếu path tương đối không tồn tại ở cwd thì tìm theo ROOT_DIR."""
    if os.path.exists(path):
        return path
    candidate = os.path.join(ROOT_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def load_crowd_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Đọc cấu hình mô hình độ đông và chỗ ngồi từ config/crowd_config.json.
    Trả về dict chứa: he_so_khung_gio, gio_cao_diem, a, b, he_so_chiem_cho.
    """
    cfg_path = resolve_project_path(path or DEFAULT_CROWD_CONFIG_PATH)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    slots = raw.get("he_so_khung_gio", {})
    return {
        "he_so_khung_gio": {
            "cao_diem": float(slots["cao_diem"]),
            "binh_thuong": float(slots["binh_thuong"]),
            "vang": float(slots["vang"]),
        },
        "gio_cao_diem": list(raw.get("gio_cao_diem", ["11:30", "13:00"])),
        "a": float(raw["a"]),
        "b": float(raw["b"]),
        "he_so_chiem_cho": float(raw["he_so_chiem_cho"]),
        "ngay_hieu_luc": raw.get("ngay_hieu_luc", ""),
    }


def _parse_hhmm_to_minutes(hhmm: Union[str, time, datetime, Tuple[int, int]]) -> int:
    """Chuyển đổi đối tượng thời gian hoặc chuỗi 'HH:MM' sang tổng số phút tính từ 00:00."""
    if isinstance(hhmm, datetime):
        return hhmm.hour * 60 + hhmm.minute
    if isinstance(hhmm, time):
        return hhmm.hour * 60 + hhmm.minute
    if isinstance(hhmm, tuple) and len(hhmm) == 2:
        return int(hhmm[0]) * 60 + int(hhmm[1])
    parts = str(hhmm).strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Định dạng giờ không hợp lệ: '{hhmm}' (cần HH:MM).")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Giờ/phút ngoài khoảng hợp lệ: '{hhmm}'.")
    return h * 60 + m


def infer_time_slot(
    now_or_str: Optional[Union[str, time, datetime, Tuple[int, int]]] = None,
    gio_cao_diem: Sequence[str] = ("11:30", "13:00"),
) -> str:
    """
    Suy ra khung_gio ∈ {'cao_diem', 'binh_thuong', 'vang'} từ thời gian hiện tại hoặc truyền vào:
      - Trong gio_cao_diem (mặc định 11:30 đến 13:00, tính cả hai đầu mút): 'cao_diem'.
      - Từ 11:00 đến trước 11:30 và từ sau 13:00 đến 14:00: 'binh_thuong'.
      - Còn lại: 'vang'.
    """
    target = datetime.now().time() if now_or_str is None else now_or_str
    minutes = _parse_hhmm_to_minutes(target)
    peak_start = _parse_hhmm_to_minutes(gio_cao_diem[0])
    peak_end = _parse_hhmm_to_minutes(gio_cao_diem[1])

    normal_morning_start = 11 * 60  # 11:00
    normal_afternoon_end = 14 * 60  # 14:00

    if peak_start <= minutes <= peak_end:
        return "cao_diem"
    if (normal_morning_start <= minutes < peak_start) or (peak_end < minutes <= normal_afternoon_end):
        return "binh_thuong"
    return "vang"


def crowd_label_vi(do_dong_hieu_dung: float) -> str:
    """Nhãn hiển thị tiếng Việt cho mức độ đông hiệu dụng (thang 0..5)."""
    if do_dong_hieu_dung < 1.5:
        return "Vắng"
    if do_dong_hieu_dung < 2.8:
        return "Vừa"
    if do_dong_hieu_dung < 4.0:
        return "Đông"
    return "Rất đông"
