"""
Script kiểm tra tính hợp lệ toàn diện của dữ liệu món ăn trong data/dishes_v2.csv.
Kiểm tra các tiêu chí:
1. Đủ 15 cột theo đúng thứ tự chuẩn.
2. Không có ô nào bị bỏ trống.
3. id là số nguyên dương và duy nhất (không trùng lặp).
4. do_dong_cao_diem là số nguyên từ 1 đến 5.
5. so_cho_ngoi là số nguyên dương > 0.
6. cay và chay nhận giá trị 0 hoặc 1.
7. Thống kê phân bố độ đông và số chỗ ngồi.
"""

import os
import sys
from collections import Counter
from typing import Dict, Any, List

# Thiết lập đường dẫn thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.io_utils import read_csv_rows, read_csv_dicts

EXPECTED_COLUMNS = [
    "id",
    "ten_mon",
    "loai",
    "gia_vnd",
    "khoang_cach_m",
    "thoi_gian_phuc_vu_phut",
    "do_no",
    "dinh_duong",
    "do_ngon",
    "cay",
    "chay",
    "do_dong_cao_diem",
    "so_cho_ngoi",
    "mo_ta",
    "nguon_du_lieu"
]


def check_dishes_v2(filepath: str = "data/dishes_v2.csv") -> bool:
    """Kiểm tra toàn diện tính hợp lệ của file dishes_v2.csv."""
    abs_path = os.path.join(parent_dir, filepath) if not os.path.isabs(filepath) else filepath

    print("=" * 95)
    print(f"KIỂM TRA TÍNH HỢP LỆ FILE DỮ LIỆU MÓN ĂN: {filepath}".center(95))
    print("=" * 95)

    if not os.path.exists(abs_path):
        print(f"❌ LỖI: Không tìm thấy file: {abs_path}")
        return False

    # 1. Kiểm tra mã hóa UTF-8 BOM
    with open(abs_path, "rb") as f:
        first_bytes = f.read(3)
        has_bom = (first_bytes == b"\xef\xbb\xbf")
        if has_bom:
            print("  [OK] Mã hóa: UTF-8 có BOM (utf-8-sig) - Tương thích 100% Microsoft Excel.")
        else:
            print("  [CẢNH BÁO] File không chứa UTF-8 BOM.")

    # 2. Đọc header và dòng thô
    raw_rows = read_csv_rows(abs_path)
    if not raw_rows:
        print("❌ LỖI: File CSV rỗng hoàn toàn.")
        return False

    header = raw_rows[0]
    data_rows = raw_rows[1:]

    # Kiểm tra số lượng và thứ tự cột
    if header != EXPECTED_COLUMNS:
        print("❌ LỖI: Danh sách cột không khớp chuẩn!")
        print(f"   Thực tế : {header}")
        print(f"   Kỳ vọng : {EXPECTED_COLUMNS}")
        return False
    else:
        print(f"  [OK] Danh sách và thứ tự 15 cột hoàn toàn CHÍNH XÁC ({len(EXPECTED_COLUMNS)} cột).")

    # 3. Đọc dữ liệu dict
    dishes = read_csv_dicts(abs_path)
    print(f"  [OK] Tổng số dòng dữ liệu đọc được: {len(dishes)} món.")

    errors: List[str] = []
    seen_ids = set()

    for row_idx, d in enumerate(dishes, 2):  # Dòng 2 bắt đầu dữ liệu
        # Kiểm tra ô trống
        for col in EXPECTED_COLUMNS:
            val = d.get(col, "")
            if val is None or str(val).strip() == "":
                errors.append(f"Dòng {row_idx} ({d.get('ten_mon', 'Chưa rõ')}): Cột '{col}' bị để TRỐNG.")

        # Kiểm tra ID
        try:
            dish_id = int(d["id"])
            if dish_id <= 0:
                errors.append(f"Dòng {row_idx}: id={dish_id} phải > 0.")
            if dish_id in seen_ids:
                errors.append(f"Dòng {row_idx}: id={dish_id} bị TRÙNG LẶP.")
            seen_ids.add(dish_id)
        except ValueError:
            errors.append(f"Dòng {row_idx}: id '{d.get('id')}' không phải số nguyên.")

        # Kiểm tra giá, khoảng cách, thời gian
        try:
            gia = int(float(d["gia_vnd"]))
            if gia <= 0:
                errors.append(f"Dòng {row_idx}: gia_vnd={gia} phải > 0.")
        except ValueError:
            errors.append(f"Dòng {row_idx}: gia_vnd không hợp lệ.")

        try:
            dist = int(float(d["khoang_cach_m"]))
            if dist < 0:
                errors.append(f"Dòng {row_idx}: khoang_cach_m={dist} không được âm.")
        except ValueError:
            errors.append(f"Dòng {row_idx}: khoang_cach_m không hợp lệ.")

        try:
            time_srv = int(float(d["thoi_gian_phuc_vu_phut"]))
            if time_srv <= 0:
                errors.append(f"Dòng {row_idx}: thoi_gian_phuc_vu_phut={time_srv} phải > 0.")
        except ValueError:
            errors.append(f"Dòng {row_idx}: thoi_gian_phuc_vu_phut không hợp lệ.")

        # Kiểm tra do_no, dinh_duong, do_ngon (1-5)
        for score_col in ["do_no", "dinh_duong", "do_ngon"]:
            try:
                score = int(float(d[score_col]))
                if score < 1 or score > 5:
                    errors.append(f"Dòng {row_idx}: {score_col}={score} nằm ngoài khoảng 1-5.")
            except ValueError:
                errors.append(f"Dòng {row_idx}: {score_col} không phải số nguyên.")

        # Kiểm tra cay và chay (0 hoặc 1)
        for bin_col in ["cay", "chay"]:
            try:
                bin_val = int(d[bin_col])
                if bin_val not in (0, 1):
                    errors.append(f"Dòng {row_idx}: {bin_col}={bin_val} phải là 0 hoặc 1.")
            except ValueError:
                errors.append(f"Dòng {row_idx}: {bin_col} không phải 0 hoặc 1.")

        # Kiểm tra do_dong_cao_diem (1-5)
        try:
            crowd = int(d["do_dong_cao_diem"])
            if crowd < 1 or crowd > 5:
                errors.append(f"Dòng {row_idx}: do_dong_cao_diem={crowd} phải từ 1 đến 5.")
        except ValueError:
            errors.append(f"Dòng {row_idx}: do_dong_cao_diem không phải số nguyên 1-5.")

        # Kiểm tra so_cho_ngoi (> 0)
        try:
            seats = int(d["so_cho_ngoi"])
            if seats <= 0:
                errors.append(f"Dòng {row_idx}: so_cho_ngoi={seats} phải > 0.")
        except ValueError:
            errors.append(f"Dòng {row_idx}: so_cho_ngoi không phải số nguyên dương.")

    if errors:
        print("\n❌ PHÁT HIỆN CÁC LỖI DỮ LIỆU:")
        for err in errors[:10]:
            print(f"   • {err}")
        if len(errors) > 10:
            print(f"   ... và {len(errors) - 10} lỗi khác.")
        return False

    print("  [OK] Không có ô trống nào, toàn bộ kiểu dữ liệu và ràng buộc hợp lệ 100%!\n")

    # 4. Thống kê phân bố độ đông và số chỗ ngồi
    total = len(dishes)
    crowd_counts = Counter(int(d["do_dong_cao_diem"]) for d in dishes)
    seat_vals = [int(d["so_cho_ngoi"]) for d in dishes]

    print("-" * 95)
    print("THỐNG KÊ PHÂN BỐ ĐỘ ĐÔNG CAO ĐIỂM (1 - 5)")
    print("-" * 95)
    level_names = {
        1: "Rất vắng / Thưa thớt",
        2: "Hơi vắng / Thoáng",
        3: "Bình thường / Ổn định",
        4: "Đông khách",
        5: "Rất đông / Chật kín bàn"
    }
    all_levels_present = True
    no_level_dominant = True
    for lvl in range(1, 6):
        cnt = crowd_counts.get(lvl, 0)
        pct = (cnt / total) * 100
        bar = "█" * int(pct // 2)
        print(f"  • Mức {lvl} ({level_names[lvl]:<24}): {cnt:>2}/{total} món ({pct:>5.1f}%) {bar}")
        if cnt == 0:
            all_levels_present = False
        if pct > 40.0:
            no_level_dominant = False

    if all_levels_present and no_level_dominant:
        print("  -> [ĐẠT] Mỗi mức 1-5 đều xuất hiện và không mức nào vượt quá 40% (phân bố cân đối).\n")
    else:
        print("  -> [CẢNH BÁO] Phân bố chưa đạt yêu cầu (thiếu mức hoặc có mức vượt 40%).\n")

    print("-" * 95)
    print("THỐNG KÊ PHÂN BỐ SỐ CHỖ NGỒI (so_cho_ngoi)")
    print("-" * 95)
    print(f"  • Thấp nhất (Min): {min(seat_vals)} chỗ | Cao nhất (Max): {max(seat_vals)} chỗ | Trung bình: {sum(seat_vals)/total:.1f} chỗ")

    cat_seats = {
        "Gánh hàng / Mang đi (4 - 12 chỗ)": sum(1 for s in seat_vals if 4 <= s <= 12),
        "Quán nhỏ / Vỉa hè (15 - 30 chỗ)": sum(1 for s in seat_vals if 15 <= s <= 30),
        "Quán cơm/bún/phở thường (31 - 60 chỗ)": sum(1 for s in seat_vals if 31 <= s <= 60),
        "Quán lớn / Cantin / Chuỗi (61 - 120 chỗ)": sum(1 for s in seat_vals if 61 <= s <= 120),
    }
    for cat_name, cnt in cat_seats.items():
        pct = (cnt / total) * 100
        bar = "█" * int(pct // 2)
        print(f"  • {cat_name:<40}: {cnt:>2}/{total} món ({pct:>5.1f}%) {bar}")
    print("=" * 95 + "\n")
    return True


if __name__ == "__main__":
    check_dishes_v2()
