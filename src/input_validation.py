"""
Kiểm tra dữ liệu nhập từ giao diện (tách khỏi PyQt để kiểm thử độc lập):
- Ngân sách mỗi người: số nguyên từ 20 đến 200 (đơn vị nghìn đồng); để trống = không giới hạn.
- Số người đi ăn: số nguyên từ 1 đến 10, mặc định 1.
Mỗi hàm trả về (giá trị, thông báo lỗi); thông báo lỗi là None nếu hợp lệ.
"""

from typing import Optional, Tuple

BUDGET_MIN_K = 20
BUDGET_MAX_K = 200
PEOPLE_MIN = 1
PEOPLE_MAX = 10
PEOPLE_DEFAULT = 1


def parse_budget_k(text: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    Đọc ô "Ngân sách mỗi người (nghìn đồng)".
    - Trống -> (None, None): không giới hạn.
    - Hợp lệ (20..200) -> (số nghìn đồng, None), ví dụ "50" -> 50 nghĩa là 50.000đ.
    - Không phải số nguyên hoặc ngoài khoảng -> (None, thông báo lỗi tiếng Việt).
    """
    raw = (text or "").strip()
    if not raw:
        return None, None
    try:
        value = int(raw)
    except ValueError:
        return None, f"Chỉ nhập số nguyên từ {BUDGET_MIN_K} đến {BUDGET_MAX_K} (nghìn đồng)."
    if not BUDGET_MIN_K <= value <= BUDGET_MAX_K:
        return None, f"Ngân sách phải từ {BUDGET_MIN_K} đến {BUDGET_MAX_K} nghìn đồng."
    return value, None


def parse_num_people(text: Optional[str]) -> Tuple[int, Optional[str]]:
    """
    Đọc ô "Số người đi ăn".
    - Trống -> (1, None): mặc định 1 người.
    - Hợp lệ (1..10) -> (số người, None).
    - Không phải số nguyên hoặc ngoài khoảng -> (1, thông báo lỗi tiếng Việt).
    """
    raw = (text or "").strip()
    if not raw:
        return PEOPLE_DEFAULT, None
    try:
        value = int(raw)
    except ValueError:
        return PEOPLE_DEFAULT, f"Chỉ nhập số nguyên từ {PEOPLE_MIN} đến {PEOPLE_MAX}."
    if not PEOPLE_MIN <= value <= PEOPLE_MAX:
        return PEOPLE_DEFAULT, f"Số người phải từ {PEOPLE_MIN} đến {PEOPLE_MAX}."
    return value, None


def budget_k_to_vnd(value_k: Optional[int]) -> Optional[int]:
    """Đổi nghìn đồng sang đồng (50 -> 50000); None giữ nguyên."""
    return None if value_k is None else int(value_k) * 1000
