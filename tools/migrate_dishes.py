"""
Script di chuyển và nâng cấp dữ liệu món ăn từ data/legacy/dishes_v1.csv sang data/dishes.csv.
Bổ sung hai thuộc tính mới:
- do_dong_cao_diem (1-5): mức độ đông đúc vào khung giờ cao điểm trưa (11:30 - 13:00).
- so_cho_ngoi (nguyên > 0): quy mô chỗ ngồi thực tế của quán.
- nguon_du_lieu: nguồn gốc dữ liệu ("mô phỏng").

Bổ sung thêm các món ăn quen thuộc của sinh viên NEU: xôi, nem nướng, bánh giò (IDs 46 - 50).
Toàn bộ file CSV được đọc và ghi chuẩn UTF-8 có BOM (utf-8-sig) và xuống dòng '\\n'.
"""

import os
import sys
from typing import Any, Dict, List, Tuple

# Thiết lập đường dẫn thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.io_utils import read_csv_dicts, write_csv_rows

# Thứ tự 15 cột chuẩn của dishes.csv theo yêu cầu
COLUMNS_V2 = [
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

# Danh sách món ăn mới bổ sung (xôi, nem nướng, bánh giò)
NEW_DISHES = [
    {
        "id": 46,
        "ten_mon": "Xôi xéo gà xé ruốc hành phi ngõ Tự Do",
        "loai": "xôi",
        "gia_vnd": 25000,
        "khoang_cach_m": 120,
        "thoi_gian_phuc_vu_phut": 5,
        "do_no": 4,
        "dinh_duong": 3,
        "do_ngon": 4,
        "cay": 0,
        "chay": 0,
        "mo_ta": "Xôi nếp cái hoa vàng dẻo thơm đỗ xanh xéo bùi béo gà xé xào nấm ruốc thịt và hành phi giòn rụm",
    },
    {
        "id": 47,
        "ten_mon": "Nem nướng Nha Trang mẹt đầy đủ Trần Đại Nghĩa",
        "loai": "nem nướng",
        "gia_vnd": 40000,
        "khoang_cach_m": 320,
        "thoi_gian_phuc_vu_phut": 8,
        "do_no": 4,
        "dinh_duong": 4,
        "do_ngon": 5,
        "cay": 1,
        "chay": 0,
        "mo_ta": "Mẹt nem nướng thơm lừng cuộn bánh tráng ram giòn rau sống xoài xanh dưa chuột chấm sốt tương nếp ấm béo ngậy",
    },
    {
        "id": 48,
        "ten_mon": "Bánh giò nóng thịt băm mộc nhĩ kèm chả cốm KTX NEU",
        "loai": "bánh giò",
        "gia_vnd": 25000,
        "khoang_cach_m": 90,
        "thoi_gian_phuc_vu_phut": 5,
        "do_no": 3,
        "dinh_duong": 3,
        "do_ngon": 4,
        "cay": 1,
        "chay": 0,
        "mo_ta": "Bánh giò bột mịn núng nính nhân thịt băm mộc nhĩ trứng cút đậm đà ăn kèm chả cốm giò tai và dưa góp chống ngấy",
    },
    {
        "id": 49,
        "ten_mon": "Xôi sườn cay kim chi Lê Thanh Nghị",
        "loai": "xôi",
        "gia_vnd": 35000,
        "khoang_cach_m": 400,
        "thoi_gian_phuc_vu_phut": 6,
        "do_no": 4,
        "dinh_duong": 3,
        "do_ngon": 4,
        "cay": 1,
        "chay": 0,
        "mo_ta": "Xôi trắng dẻo rưới sốt sườn cay đượm vị sườn non ninh nhừ róc xương kèm kim chi chua cay giòn tan",
    },
    {
        "id": 50,
        "ten_mon": "Bánh giò chay nấm hương mộc nhĩ ngõ Tự Do",
        "loai": "bánh giò",
        "gia_vnd": 20000,
        "khoang_cach_m": 160,
        "thoi_gian_phuc_vu_phut": 5,
        "do_no": 3,
        "dinh_duong": 3,
        "do_ngon": 3,
        "cay": 0,
        "chay": 1,
        "mo_ta": "Bánh giò chay vỏ mềm mướt nhân đậu xanh nấm hương mộc nhĩ xào tiêu thơm lừng ăn kèm dưa chua ngọt thanh mát",
    }
]


def calculate_crowd_level(dish: Dict[str, Any]) -> Tuple[int, str]:
    """
    Quy tắc tính toán mức độ đông đúc giờ cao điểm (do_dong_cao_diem: 1 đến 5).
    Giải thích logic:
    - Giá rẻ (<=30k: sinh viên tập trung rất đông) -> tăng độ đông.
    - Gần trường (<=200m: ngõ Tự Do, KTX, Trần Đại Nghĩa đi bộ nhanh) -> tăng độ đông.
    - Thời gian phục vụ ngắn (<=6 phút) -> tăng độ đông vì sinh viên tranh thủ giờ nghỉ trưa.
    - Món ăn quen thuộc bữa trưa (cơm, bún, phở, bánh mì, xôi) -> tăng độ đông.
    - Quán đắt tiền (>60k), xa (>600m), kiểu ăn nhẹ kiêng khem (salad, poke bowl, cơm văn phòng cao cấp) -> vắng hơn.
    
    Trả về:
        Tuple[int, str]: (Mức độ đông 1-5, Giải thích ngắn gọn)
    """
    gia = int(float(dish["gia_vnd"]))
    dist = int(float(dish["khoang_cach_m"]))
    time_min = int(float(dish["thoi_gian_phuc_vu_phut"]))
    loai = str(dish["loai"]).lower()
    name = str(dish["ten_mon"]).lower()
    taste = int(float(dish.get("do_ngon", 3)))

    score = 0.0
    reasons = []

    # 1. Điểm giá tiền
    if gia <= 25000:
        score += 2.0
        reasons.append("giá siêu rẻ sinh viên")
    elif gia <= 35000:
        score += 1.0
        reasons.append("giá vừa túi tiền")
    elif gia <= 45000:
        score += 0.0
    elif gia <= 60000:
        score -= 1.0
        reasons.append("giá hơi cao")
    else:
        score -= 2.0
        reasons.append("giá cao kén sinh viên")

    # 2. Điểm khoảng cách đi bộ từ cổng NEU
    if dist <= 150:
        score += 2.0
        reasons.append("sát cổng/KTX")
    elif dist <= 300:
        score += 1.0
        reasons.append("gần trường")
    elif dist <= 550:
        score += 0.0
    elif dist <= 750:
        score -= 1.0
        reasons.append("hơi xa")
    else:
        score -= 2.0
        reasons.append("quá xa cổng NEU")

    # 3. Thời gian phục vụ
    if time_min <= 5:
        score += 1.0
        reasons.append("lên món cực nhanh")
    elif time_min <= 8:
        score += 0.0
    elif time_min <= 11:
        score -= 1.0
        reasons.append("chờ tương đối lâu")
    else:
        score -= 2.0
        reasons.append("chế biến lâu")

    # 4. Loại món và món quen thuộc
    popular_keywords = ["bình dân", "tự chọn", "mắm tôm", "xôi xéo", "pate chả", "que hải phòng", "nem nướng"]
    if any(k in name for k in popular_keywords):
        score += 1.0
        reasons.append("món tủ sinh viên")
    elif loai in ["cơm", "bún", "phở", "bánh mì", "xôi", "nem nướng"]:
        score += 0.5
    elif loai in ["salad"]:
        score -= 1.5
        reasons.append("món ăn kiêng ít khách trưa")
    elif loai in ["cơm văn phòng"]:
        score -= 1.5
        reasons.append("khách công sở ít sinh viên")
    elif loai in ["đồ ăn nhanh"]:
        score -= 0.5

    # 5. Độ ngon xuất sắc kéo khách
    if taste >= 5:
        score += 0.5
        reasons.append("độ ngon đánh giá 5/5")

    # Ánh xạ tổng điểm (-6.0 đến +6.5) sang thang đo 1 - 5
    if score >= 3.0:
        crowd = 5
    elif score >= 1.0:
        crowd = 4
    elif score >= -1.0:
        crowd = 3
    elif score >= -3.0:
        crowd = 2
    else:
        crowd = 1

    reason_str = ", ".join(reasons)
    return crowd, reason_str


def calculate_seats(dish: Dict[str, Any]) -> Tuple[int, str]:
    """
    Quy tắc tính toán số chỗ ngồi (so_cho_ngoi: số nguyên):
    - Gánh hàng / quán bánh mì vỉa hè mang đi: 4 - 12 chỗ
    - Quán nhỏ / bánh cuốn, cháo sườn, bánh giò, salad nhỏ: 15 - 30 chỗ
    - Quán cơm / bún / phở / mì thông thường: 30 - 60 chỗ
    - Quán lớn / nhà hàng cơm niêu, cơm văn phòng, chuỗi đồ ăn nhanh: 60 - 120 chỗ
    """
    loai = str(dish["loai"]).lower()
    name = str(dish["ten_mon"]).lower()
    dish_id = int(dish["id"])

    # 1. Gánh hàng / bánh mì, xôi mang đi (4 - 12 chỗ)
    if "doner kebab" in name:
        return 6, "xe bánh mì mua mang đi vỉa hè"
    elif "que hải phòng" in name or "thịt xiên nướng" in name:
        return 8, "quầy ăn vặt mang đi có 2-3 ghế nhựa"
    elif "pate chả ngõ tự do" in name or "bánh mì chay chả nấm" in name:
        return 10, "quán bánh mì nhỏ vỉa hè"
    elif "xôi xéo" in name or "bánh giò chay" in name:
        return 12, "gánh xôi / bánh giò sáng trưa vỉa hè"

    # 2. Bánh mì có quán ngồi bàn đàng hoàng (15 - 30 chỗ)
    elif loai == "bánh mì":
        if "chảo" in name:
            return 28, "quán bánh mì chảo 1 tầng có bàn ghế"
        elif "sốt vang" in name:
            return 24, "quán bánh mì sốt vang vỉa hè trong nhà"
        return 20, "tiệm bánh mì có chỗ ngồi"

    # 3. Quán nhỏ, salad, bánh cuốn, cháo sườn, xôi có quán (15 - 30 chỗ)
    elif loai in ["salad"]:
        seats = 18 if dish_id % 2 == 0 else 22
        return seats, "quầy salad eat clean nhỏ xinh"
    elif loai in ["khác", "bánh giò"]:
        if "cháo" in name:
            return 16, "quán cháo sườn vỉa hè ít bàn"
        elif "bánh cuốn" in name:
            return 22, "quán bánh cuốn nóng 1 gian"
        elif "bánh giò" in name:
            return 18, "quán bánh giò nóng bàn nhựa"
        return 20, "quán ăn vặt nhỏ"
    elif loai in ["xôi"]:
        return 26, "quán xôi có phòng ăn khép kín"
    elif loai in ["nem nướng"]:
        return 42, "quán nem nướng Nha Trang bàn bệt/ghế gỗ"

    # 4. Quán lớn / nhà hàng / chuỗi đồ ăn nhanh (60 - 120 chỗ)
    elif "lotteria" in name:
        return 85, "chuỗi thức ăn nhanh 2 tầng rộng rãi"
    elif "niêu singapore" in name:
        return 75, "nhà hàng cơm niêu 3 tầng máy lạnh"
    elif "phở cuốn hương mai" in name:
        return 65, "nhà hàng phở cuốn cơ sở lớn"
    elif "mì cay 7 cấp độ" in name:
        return 70, "quán mì cay Hàn Quốc 3 tầng bàn bệt"
    elif loai == "cơm văn phòng":
        return 65, "nhà hàng cơm văn phòng nhiều tầng"
    elif loai == "đồ ăn nhanh":
        if "pizza" in name:
            return 35, "tiệm pizza mini"
        return 40, "tiệm gà rán đồ ăn nhanh"

    # 5. Quán cơm bình dân, bún, phở, mì thông thường (30 - 60 chỗ)
    elif loai in ["cơm", "bún", "phở", "mì"]:
        # Phân bổ phong phú giữa 32 và 56 dựa trên ID cố định (deterministic)
        base_seats = 36 + (dish_id * 7) % 22
        return base_seats, "quán ăn trưa sinh viên tiêu chuẩn"

    return 30, "quán ăn thông thường"


def migrate():
    """Thực hiện di chuyển dữ liệu từ data/legacy/dishes_v1.csv sang data/dishes.csv."""
    v1_path = os.path.join(parent_dir, "data", "legacy", "dishes_v1.csv")
    v2_path = os.path.join(parent_dir, "data", "dishes.csv")

    if not os.path.exists(v1_path):
        raise FileNotFoundError(f"Không tìm thấy file nguồn: {v1_path}")

    print("=" * 80)
    print("BẮT ĐẦU CHUYỂN ĐỔI DỮ LIỆU: dishes_v1.csv -> dishes.csv")
    print("=" * 80)

    legacy_dishes = read_csv_dicts(v1_path)
    print(f"1. Đã đọc {len(legacy_dishes)} món từ data/legacy/dishes_v1.csv")

    migrated_rows: List[List[Any]] = []

    # Xử lý 45 món gốc (giữ nguyên id và toàn bộ giá trị cũ)
    for d in legacy_dishes:
        crowd, _ = calculate_crowd_level(d)
        seats, _ = calculate_seats(d)
        row = [
            int(d["id"]),
            d["ten_mon"],
            d["loai"],
            int(float(d["gia_vnd"])),
            int(float(d["khoang_cach_m"])),
            int(float(d["thoi_gian_phuc_vu_phut"])),
            int(float(d["do_no"])),
            int(float(d["dinh_duong"])),
            int(float(d["do_ngon"])),
            int(float(d["cay"])),
            int(float(d.get("chay", 0))),
            crowd,
            seats,
            d["mo_ta"],
            "mô phỏng"
        ]
        migrated_rows.append(row)

    # Thêm các món ăn mới (IDs 46 - 50: xôi, nem nướng, bánh giò)
    print(f"2. Bổ sung {len(NEW_DISHES)} món ăn mới (xôi, nem nướng, bánh giò)...")
    for d in NEW_DISHES:
        crowd, _ = calculate_crowd_level(d)
        seats, _ = calculate_seats(d)
        row = [
            int(d["id"]),
            d["ten_mon"],
            d["loai"],
            int(d["gia_vnd"]),
            int(d["khoang_cach_m"]),
            int(d["thoi_gian_phuc_vu_phut"]),
            int(d["do_no"]),
            int(d["dinh_duong"]),
            int(d["do_ngon"]),
            int(d["cay"]),
            int(d["chay"]),
            crowd,
            seats,
            d["mo_ta"],
            "mô phỏng"
        ]
        migrated_rows.append(row)

    # Ghi dữ liệu ra file data/dishes.csv bằng hàm tiện ích dùng chung chuẩn UTF-8 BOM
    write_csv_rows(v2_path, COLUMNS_V2, migrated_rows)
    print(f"3. Đã ghi thành công {len(migrated_rows)} món vào: {v2_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    migrate()
