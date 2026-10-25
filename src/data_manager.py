"""
Module quản lý dữ liệu cho Lunch Chatbot: đọc/ghi/kiểm tra hợp lệ data/dishes.csv và data/intents.csv.
Tách khỏi giao diện để có thể kiểm thử độc lập. Mọi hàm ghi file đều ghi nguyên tử (file tạm rồi thay thế)
để tránh hỏng dữ liệu nếu chương trình bị dừng giữa chừng.
"""

import csv
import os
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

DISH_COLUMNS: List[str] = [
    "id", "ten_mon", "loai", "gia_vnd", "khoang_cach_m", "thoi_gian_phuc_vu_phut",
    "do_no", "dinh_duong", "do_ngon", "cay", "mo_ta",
]

# Nhãn ý định (khớp với data/intents.csv) và tên hiển thị tiếng Việt
INTENT_LABELS: Dict[str, str] = {
    "tiet_kiem": "Tiết kiệm",
    "healthy": "Healthy",
    "nhanh_gon": "Nhanh gọn",
    "no_bung": "No bụng",
    "ngon_doi_gio": "Ngon đổi gió",
    "can_bang": "Cân bằng",
    "chao_hoi": "Chào hỏi",
    "cam_on_tam_biet": "Cảm ơn / Tạm biệt",
}

# Miền giá trị hợp lệ của các cột số: (tối thiểu, tối đa)
_INT_RANGES: Dict[str, Tuple[int, int]] = {
    "gia_vnd": (1000, 1_000_000),
    "khoang_cach_m": (0, 20_000),
    "thoi_gian_phuc_vu_phut": (0, 180),
    "do_no": (1, 5),
    "dinh_duong": (1, 5),
    "do_ngon": (1, 5),
    "cay": (0, 1),
}

_FIELD_VI: Dict[str, str] = {
    "ten_mon": "Tên món",
    "loai": "Loại",
    "gia_vnd": "Giá (VND)",
    "khoang_cach_m": "Khoảng cách (m)",
    "thoi_gian_phuc_vu_phut": "Thời gian phục vụ (phút)",
    "do_no": "Độ no",
    "dinh_duong": "Dinh dưỡng",
    "do_ngon": "Độ ngon",
    "cay": "Cay",
}


def _atomic_write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    """Ghi CSV UTF-8 nguyên tử: ghi ra file tạm cùng thư mục rồi thay thế file đích."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# MÓN ĂN
# ══════════════════════════════════════════════════════════════════════════════
def validate_dish(dish: Dict[str, Any]) -> List[str]:
    """
    Kiểm tra hợp lệ một món ăn (dict chứa các cột trừ id).
    Trả về danh sách lỗi tiếng Việt; danh sách rỗng nghĩa là hợp lệ.
    """
    errors: List[str] = []

    for key in ("ten_mon", "loai"):
        value = str(dish.get(key, "")).strip()
        if not value:
            errors.append(f"{_FIELD_VI[key]} không được để trống.")
        elif len(value) > 120:
            errors.append(f"{_FIELD_VI[key]} quá dài (tối đa 120 ký tự).")

    for key, (low, high) in _INT_RANGES.items():
        raw = str(dish.get(key, "")).strip()
        try:
            number = int(float(raw))
            if float(raw) != number:
                raise ValueError
        except ValueError:
            errors.append(f"{_FIELD_VI[key]} phải là số nguyên (nhận được: '{raw}').")
            continue
        if not low <= number <= high:
            errors.append(f"{_FIELD_VI[key]} phải nằm trong khoảng {low:,} – {high:,} (nhận được: {number:,}).")

    if len(str(dish.get("mo_ta", ""))) > 500:
        errors.append("Mô tả quá dài (tối đa 500 ký tự).")
    return errors


def load_dishes(path: str) -> List[Dict[str, str]]:
    """Đọc data/dishes.csv thành danh sách dict (giá trị giữ dạng chuỗi). Ném FileNotFoundError nếu thiếu file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu món ăn: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in DISH_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"File {path} thiếu cột: {', '.join(missing)}")
        return [{c: (row.get(c) or "").strip() for c in DISH_COLUMNS} for row in reader]


def save_dishes(path: str, dishes: List[Dict[str, Any]]) -> None:
    """Ghi toàn bộ danh sách món ăn vào CSV (đúng thứ tự cột chuẩn)."""
    rows = [[d.get(c, "") for c in DISH_COLUMNS] for d in dishes]
    _atomic_write_csv(path, DISH_COLUMNS, rows)


def next_dish_id(dishes: List[Dict[str, Any]]) -> int:
    """Sinh id mới = id lớn nhất hiện có + 1."""
    ids = [int(d["id"]) for d in dishes if str(d.get("id", "")).isdigit()]
    return (max(ids) if ids else 0) + 1


def is_duplicate_dish(dishes: List[Dict[str, Any]], name: str, ignore_id: Optional[int] = None) -> bool:
    """Kiểm tra trùng tên món (không phân biệt hoa thường), có thể bỏ qua một id khi đang sửa."""
    target = name.strip().lower()
    for d in dishes:
        if ignore_id is not None and str(d.get("id")) == str(ignore_id):
            continue
        if str(d.get("ten_mon", "")).strip().lower() == target:
            return True
    return False


def add_dish(path: str, dish: Dict[str, Any]) -> Dict[str, Any]:
    """Thêm món mới (tự cấp id). Ném ValueError nếu dữ liệu không hợp lệ hoặc trùng tên."""
    errors = validate_dish(dish)
    if errors:
        raise ValueError("\n".join(errors))
    dishes = load_dishes(path)
    if is_duplicate_dish(dishes, dish["ten_mon"]):
        raise ValueError(f"Món '{dish['ten_mon'].strip()}' đã tồn tại.")
    new_dish = _normalize_dish(dish)
    new_dish["id"] = str(next_dish_id(dishes))
    dishes.append(new_dish)
    save_dishes(path, dishes)
    return new_dish


def update_dish(path: str, dish_id: int, dish: Dict[str, Any]) -> None:
    """Cập nhật món theo id. Ném ValueError nếu không hợp lệ/không tìm thấy/trùng tên."""
    errors = validate_dish(dish)
    if errors:
        raise ValueError("\n".join(errors))
    dishes = load_dishes(path)
    if is_duplicate_dish(dishes, dish["ten_mon"], ignore_id=dish_id):
        raise ValueError(f"Món '{dish['ten_mon'].strip()}' đã tồn tại.")
    for idx, d in enumerate(dishes):
        if str(d["id"]) == str(dish_id):
            updated = _normalize_dish(dish)
            updated["id"] = d["id"]
            dishes[idx] = updated
            save_dishes(path, dishes)
            return
    raise ValueError(f"Không tìm thấy món có id = {dish_id}.")


def delete_dish(path: str, dish_id: int) -> None:
    """Xóa món theo id. Ném ValueError nếu không tìm thấy."""
    dishes = load_dishes(path)
    remaining = [d for d in dishes if str(d["id"]) != str(dish_id)]
    if len(remaining) == len(dishes):
        raise ValueError(f"Không tìm thấy món có id = {dish_id}.")
    save_dishes(path, remaining)


def _normalize_dish(dish: Dict[str, Any]) -> Dict[str, str]:
    """Chuẩn hóa dữ liệu món: cắt khoảng trắng, ép số nguyên, bỏ xuống dòng trong mô tả."""
    out: Dict[str, str] = {c: "" for c in DISH_COLUMNS}
    out["ten_mon"] = str(dish["ten_mon"]).strip()
    out["loai"] = str(dish["loai"]).strip()
    for key in _INT_RANGES:
        out[key] = str(int(float(str(dish[key]).strip())))
    out["mo_ta"] = " ".join(str(dish.get("mo_ta", "")).split())
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CÂU MẪU Ý ĐỊNH
# ══════════════════════════════════════════════════════════════════════════════
def validate_intent(text: str, label: str) -> List[str]:
    """Kiểm tra câu mẫu và nhãn. Trả về danh sách lỗi tiếng Việt (rỗng nếu hợp lệ)."""
    errors: List[str] = []
    cleaned = " ".join(str(text).split())
    if not cleaned:
        errors.append("Câu mẫu không được để trống.")
    elif len(cleaned) < 3:
        errors.append("Câu mẫu quá ngắn (tối thiểu 3 ký tự).")
    elif len(cleaned) > 300:
        errors.append("Câu mẫu quá dài (tối đa 300 ký tự).")
    if label not in INTENT_LABELS:
        errors.append(f"Nhãn '{label}' không hợp lệ.")
    return errors


def load_intents(path: str) -> List[Tuple[str, str]]:
    """Đọc data/intents.csv thành danh sách (text, label)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu câu mẫu: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"File {path} phải có hai cột 'text' và 'label'.")
        return [(row["text"], row["label"]) for row in reader]


def add_intent(path: str, text: str, label: str) -> str:
    """
    Thêm một câu mẫu vào intents.csv. Ném ValueError nếu không hợp lệ hoặc đã tồn tại
    (so khớp không phân biệt hoa thường, trong cùng mọi nhãn). Trả về câu đã chuẩn hóa khoảng trắng.
    """
    errors = validate_intent(text, label)
    if errors:
        raise ValueError("\n".join(errors))
    cleaned = " ".join(str(text).split())
    rows = load_intents(path)
    for existing_text, existing_label in rows:
        if existing_text.strip().lower() == cleaned.lower():
            raise ValueError(f"Câu mẫu đã tồn tại (nhãn '{existing_label}').")
    rows.append((cleaned, label))
    _atomic_write_csv(path, ["text", "label"], [list(r) for r in rows])
    return cleaned


def count_intents_by_label(path: str) -> Dict[str, int]:
    """Đếm số câu mẫu theo từng nhãn (đủ cả nhãn có 0 câu)."""
    counts = Counter(label for _, label in load_intents(path))
    return {label: counts.get(label, 0) for label in INTENT_LABELS}
