"""
Lưu và thống kê phản hồi của người dùng về từng câu trả lời của bot (data/feedback.csv).
Phản hồi CHỈ được ghi để tham khảo; KHÔNG tự động đưa vào dữ liệu huấn luyện (data/intents.csv).
Mọi thao tác ghi/đọc CSV đi qua src/io_utils.py (UTF-8 BOM, xuống dòng '\\n').
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from src.io_utils import append_csv_row, read_csv_dicts
except ImportError:  # chạy độc lập
    from io_utils import append_csv_row, read_csv_dicts

FEEDBACK_COLUMNS: List[str] = [
    "thoi_gian", "cau_hoi", "nhan_du_doan", "do_tin_cay", "mon_top1", "danh_gia", "nhan_dung", "ghi_chu",
]

RATING_HELPFUL = "huu_ich"
RATING_WRONG = "chua_dung_y"
_VALID_RATINGS = (RATING_HELPFUL, RATING_WRONG)


def append_feedback(
    path: str,
    question: str,
    predicted_label: str,
    confidence: float,
    top1_dish: str,
    rating: str,
    correct_label: str = "",
    note: str = "",
    now: Optional[datetime] = None,
) -> Dict[str, str]:
    """
    Ghi một phản hồi vào CSV và trả về dòng đã ghi (dict theo FEEDBACK_COLUMNS).
    rating phải là 'huu_ich' hoặc 'chua_dung_y'. Ghi chú được gọn khoảng trắng/xuống dòng.
    """
    if rating not in _VALID_RATINGS:
        raise ValueError(f"Đánh giá '{rating}' không hợp lệ (chỉ nhận {', '.join(_VALID_RATINGS)}).")
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    record = {
        "thoi_gian": stamp,
        "cau_hoi": " ".join(str(question).split()),
        "nhan_du_doan": str(predicted_label or ""),
        "do_tin_cay": f"{float(confidence):.4f}",
        "mon_top1": str(top1_dish or ""),
        "danh_gia": rating,
        "nhan_dung": str(correct_label or ""),
        "ghi_chu": " ".join(str(note or "").split()),
    }
    append_csv_row(path, FEEDBACK_COLUMNS, [record[c] for c in FEEDBACK_COLUMNS])
    return record


def load_feedback(path: str) -> List[Dict[str, str]]:
    """Đọc toàn bộ phản hồi; trả về danh sách rỗng nếu chưa có file."""
    if not os.path.exists(path):
        return []
    return read_csv_dicts(path)


def feedback_summary(path: str) -> Dict[str, Any]:
    """
    Thống kê phản hồi: {'total': n, 'helpful': h, 'rate': h/n hoặc None nếu chưa có phản hồi}.
    """
    rows = load_feedback(path)
    total = len(rows)
    helpful = sum(1 for r in rows if r.get("danh_gia") == RATING_HELPFUL)
    return {"total": total, "helpful": helpful, "rate": (helpful / total) if total else None}


def format_feedback_summary(summary: Dict[str, Any]) -> str:
    """Câu hiển thị ở tab Đánh giá: 'Phản hồi người dùng: tỉ lệ hữu ích x% trên n phản hồi' hoặc 'Chưa có phản hồi'."""
    if not summary.get("total"):
        return "Phản hồi người dùng: Chưa có phản hồi"
    pct = f"{summary['rate'] * 100:.0f}"
    return f"Phản hồi người dùng: tỉ lệ hữu ích {pct}% trên {summary['total']} phản hồi"
