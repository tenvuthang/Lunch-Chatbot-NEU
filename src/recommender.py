"""
Module hệ thống gợi ý món ăn đa tiêu chí (Multi-Criteria Dish Recommender).
Thực hiện:
1. Đọc dữ liệu món ăn từ data/dishes_v2.csv, hồ sơ trọng số 7 tiêu chí từ config/profiles.json,
   và tham số độ đông từ config/crowd_config.json (hỗ trợ tương thích ngược với file CSV cũ thiếu cột mới).
2. Tính toán các đại lượng phụ thuộc độ đông và số người đi ăn:
   - do_dong_hieu_dung = do_dong_cao_diem * he_so_khung_gio[khung_gio]
   - thoi_gian_cho = thoi_gian_phuc_vu_phut * (1 + a * do_dong_hieu_dung / 5) * (1 + b * (so_nguoi - 1))
   - cho_trong = so_cho_ngoi * (1 - he_so_chiem_cho * do_dong_hieu_dung / 5)
   - cho_ngoi_nhom = min(1, cho_trong / so_nguoi) ∈ [0, 1]
3. Chuẩn hóa giá trị 7 tiêu chí về đoạn [0, 1] bằng Min-Max Scaling:
   - Tiêu chí chi phí (gia, khoang_cach, thoi_gian dùng thoi_gian_cho): đảo chiều vì càng thấp càng tốt.
   - Tiêu chí lợi ích (do_no, dinh_duong, do_ngon, cho_ngoi dùng cho_ngoi_nhom): giữ chiều thuận.
4. Lọc danh sách món ăn theo các ràng buộc (ngân sách/người, không cay, chỉ món chay, loại trừ món, đủ chỗ ngồi).
5. Tính tổng điểm trọng số: Điểm = Tổng(trọng số_i × giá trị_chuẩn_hóa_i) và trả về Top-K kèm giải thích/cảnh báo.
"""

import json
import os
from typing import Dict, Any, List, Optional, Sequence, Tuple

try:
    from src.io_utils import read_csv_dicts
    from src.config import (
        DEFAULT_DISHES_PATH,
        DEFAULT_PROFILES_PATH,
        DEFAULT_CROWD_CONFIG_PATH,
        DEFAULT_DO_DONG_CAO_DIEM,
        DEFAULT_SO_CHO_NGOI,
        DEFAULT_NGUON_DU_LIEU,
        TIME_SLOTS,
        TIME_SLOT_AUTO,
        TIME_SLOT_LABELS_VI,
        resolve_project_path,
        load_crowd_config,
        infer_time_slot,
        crowd_label_vi,
    )
except ImportError:
    from io_utils import read_csv_dicts
    from config import (
        DEFAULT_DISHES_PATH,
        DEFAULT_PROFILES_PATH,
        DEFAULT_CROWD_CONFIG_PATH,
        DEFAULT_DO_DONG_CAO_DIEM,
        DEFAULT_SO_CHO_NGOI,
        DEFAULT_NGUON_DU_LIEU,
        TIME_SLOTS,
        TIME_SLOT_AUTO,
        TIME_SLOT_LABELS_VI,
        resolve_project_path,
        load_crowd_config,
        infer_time_slot,
        crowd_label_vi,
    )


# Ánh xạ tên 7 tiêu chí trong profiles.json sang trường dữ liệu của dish
CRITERIA_COLUMN_MAP: Dict[str, str] = {
    "gia": "gia_vnd",
    "khoang_cach": "khoang_cach_m",
    "thoi_gian": "thoi_gian_cho",
    "do_no": "do_no",
    "dinh_duong": "dinh_duong",
    "do_ngon": "do_ngon",
    "cho_ngoi": "cho_ngoi_nhom",
}

# Tiêu chí chi phí (càng thấp càng tốt -> đảo chiều chuẩn hóa)
COST_CRITERIA = {"gia", "khoang_cach", "thoi_gian"}

# Tiêu chí lợi ích (càng cao càng tốt -> chuẩn hóa thuận)
BENEFIT_CRITERIA = {"do_no", "dinh_duong", "do_ngon", "cho_ngoi"}

# Giới hạn hợp lệ của các tham số lọc truyền từ giao diện
BUDGET_MIN_VND = 20_000
BUDGET_MAX_VND = 200_000
PEOPLE_MIN = 1
PEOPLE_MAX = 10

SOURCE_TEXT = "từ câu nói"
SOURCE_UI = "từ ô nhập"
SOURCE_UI_CHECK = "từ ô tích"
SOURCE_CLOCK = "tự động theo đồng hồ"

SEAT_WARNING_TEXT = "Có thể phải chờ chỗ"

# Thứ tự áp dụng các bước lọc và tên hiển thị
FILTER_STEP_LABELS = [
    ("budget", "Sau lọc ngân sách"),
    ("spicy", "Sau lọc cay"),
    ("vegetarian", "Sau lọc chay"),
    ("exclude", "Sau lọc loại trừ món"),
    ("seats", "Sau lọc chỗ ngồi"),
]


def compute_crowd_metrics(
    dish: Dict[str, Any],
    so_nguoi: int = 1,
    khung_gio: str = "cao_diem",
    crowd_cfg: Optional[Dict[str, Any]] = None,
    prefer_quiet: bool = False,
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số độ đông, thời gian chờ và chỗ ngồi nhóm cho 1 món ăn theo đúng công thức mục B:
      - do_dong_hieu_dung = do_dong_cao_diem * he_so_khung_gio[khung_gio]
      - thoi_gian_cho = thoi_gian_phuc_vu_phut * (1 + a_eff * do_dong_hieu_dung / 5) * (1 + b * (so_nguoi - 1))
        (trong đó a_eff = 2 * a nếu prefer_quiet / uu_tien_vang = True, ngược lại a_eff = a)
      - cho_trong = so_cho_ngoi * (1 - he_so_chiem_cho * do_dong_hieu_dung / 5)
      - cho_ngoi_nhom = min(1, cho_trong / so_nguoi) ∈ [0, 1]
    """
    cfg = crowd_cfg or load_crowd_config()
    slots_map = cfg["he_so_khung_gio"]
    if khung_gio not in slots_map:
        khung_gio = "cao_diem"

    he_so_kg = float(slots_map[khung_gio])
    a_base = float(cfg["a"])
    a_eff = a_base * 2.0 if prefer_quiet else a_base
    b_val = float(cfg["b"])
    he_so_chiem_cho = float(cfg["he_so_chiem_cho"])

    do_dong_cao_diem = float(dish.get("do_dong_cao_diem", DEFAULT_DO_DONG_CAO_DIEM))
    so_cho_ngoi = int(dish.get("so_cho_ngoi", DEFAULT_SO_CHO_NGOI))
    tg_phuc_vu = float(dish["thoi_gian_phuc_vu_phut"])
    n_people = max(1, int(so_nguoi))

    do_dong_hieu_dung = do_dong_cao_diem * he_so_kg
    thoi_gian_cho = (
        tg_phuc_vu
        * (1.0 + a_eff * do_dong_hieu_dung / 5.0)
        * (1.0 + b_val * (n_people - 1))
    )
    cho_trong = so_cho_ngoi * (1.0 - he_so_chiem_cho * do_dong_hieu_dung / 5.0)
    cho_trong = max(0.0, cho_trong)
    cho_ngoi_nhom = min(1.0, max(0.0, cho_trong / n_people))
    seat_warning = SEAT_WARNING_TEXT if cho_trong < n_people else None

    return {
        "do_dong_cao_diem": int(do_dong_cao_diem),
        "so_cho_ngoi": so_cho_ngoi,
        "khung_gio": khung_gio,
        "he_so_khung_gio": he_so_kg,
        "a_base": a_base,
        "a_effective": a_eff,
        "b": b_val,
        "he_so_chiem_cho": he_so_chiem_cho,
        "do_dong_hieu_dung": round(do_dong_hieu_dung, 4),
        "crowd_label": crowd_label_vi(do_dong_hieu_dung),
        "thoi_gian_cho": round(thoi_gian_cho, 4),
        "cho_trong": round(cho_trong, 4),
        "cho_trong_int": max(0, int(round(cho_trong))),
        "cho_ngoi_nhom": round(cho_ngoi_nhom, 4),
        "seat_warning": seat_warning,
    }


def merge_constraints(
    constraints: Optional[Dict[str, Any]] = None,
    max_price_per_person: Optional[float] = None,
    num_people: int = 1,
    vegetarian_only: bool = False,
    exclude_spicy: bool = False,
    time_slot: Optional[str] = None,
    require_enough_seats: bool = False,
    current_time: Optional[Any] = None,
    gio_cao_diem: Sequence[str] = ("11:30", "13:00"),
) -> Dict[str, Any]:
    """
    Gộp ràng buộc trích từ câu nói (constraints) với giá trị nhập tay trên giao diện.

    Quy tắc ưu tiên:
    - Ngân sách, số người, khung giờ trích từ câu nói được ƯU TIÊN hơn giá trị trên giao diện.
    - Bộ lọc cay, chay, đủ chỗ ngồi là bộ lọc cứng: bật nếu câu nói yêu cầu HOẶC ô tích được bật.
    - Khung giờ: nếu câu nói không nêu và giao diện để 'tu_dong'/None thì suy từ đồng hồ hệ thống.

    Ném ValueError nếu num_people ngoài 1-10 hoặc max_price_per_person ngoài 20.000-200.000đ.
    """
    base = constraints or {}

    if isinstance(num_people, bool) or int(num_people) != num_people or not PEOPLE_MIN <= int(num_people) <= PEOPLE_MAX:
        raise ValueError(f"Số người phải là số nguyên từ {PEOPLE_MIN} đến {PEOPLE_MAX} (nhận được: {num_people}).")
    ui_people = int(num_people)

    if max_price_per_person is not None and not BUDGET_MIN_VND <= max_price_per_person <= BUDGET_MAX_VND:
        raise ValueError(
            f"Ngân sách mỗi người phải từ {BUDGET_MIN_VND:,}đ đến {BUDGET_MAX_VND:,}đ (nhận được: {max_price_per_person:,.0f}đ)."
        )

    # 1. Ngân sách: ưu tiên câu nói
    if base.get("max_price") is not None:
        max_price, budget_source = float(base["max_price"]), SOURCE_TEXT
    elif max_price_per_person is not None:
        max_price, budget_source = float(max_price_per_person), SOURCE_UI
    else:
        max_price, budget_source = None, None

    # 2. Số người: ưu tiên câu nói (nếu hợp lệ 1..10)
    text_people = base.get("num_people") if base.get("num_people") is not None else base.get("so_nguoi")
    if text_people is not None and PEOPLE_MIN <= int(text_people) <= PEOPLE_MAX:
        eff_people, people_source = int(text_people), SOURCE_TEXT
    else:
        eff_people, people_source = ui_people, SOURCE_UI

    # 3. Khung giờ: ưu tiên câu nói -> ô chọn UI -> tự động theo đồng hồ
    text_slot = base.get("time_slot") or base.get("khung_gio")
    parsed_time = base.get("parsed_time") or (base.get("intermediate") or {}).get("parsed_time")
    if text_slot in TIME_SLOTS:
        eff_slot = text_slot
        slot_source = f"{SOURCE_TEXT} ({parsed_time})" if parsed_time else SOURCE_TEXT
    elif time_slot in TIME_SLOTS:
        eff_slot = time_slot
        slot_source = SOURCE_UI
    else:
        eff_slot = infer_time_slot(current_time, gio_cao_diem=gio_cao_diem)
        slot_source = SOURCE_CLOCK

    # 4. Ưu tiên quán vắng (uu_tien_vang)
    prefer_quiet = bool(base.get("prefer_quiet") or base.get("uu_tien_vang"))
    quiet_source = SOURCE_TEXT if prefer_quiet else None

    # 5. Cay
    if base.get("exclude_spicy"):
        spicy, spicy_source = True, SOURCE_TEXT
    elif exclude_spicy:
        spicy, spicy_source = True, SOURCE_UI_CHECK
    else:
        spicy, spicy_source = False, None

    # 6. Chay
    if base.get("vegetarian_only"):
        veg, veg_source = True, SOURCE_TEXT
    elif vegetarian_only:
        veg, veg_source = True, SOURCE_UI_CHECK
    else:
        veg, veg_source = False, None

    # 7. Chỉ quán đủ chỗ cho cả nhóm
    if base.get("require_enough_seats"):
        enough_seats, seats_source = True, SOURCE_TEXT
    elif require_enough_seats:
        enough_seats, seats_source = True, SOURCE_UI_CHECK
    else:
        enough_seats, seats_source = False, None

    return {
        "max_price": max_price,
        "exclude_spicy": spicy,
        "vegetarian_only": veg,
        "exclude_types": list(base.get("exclude_types") or []),
        "num_people": eff_people,
        "so_nguoi": eff_people,
        "time_slot": eff_slot,
        "khung_gio": eff_slot,
        "parsed_time": parsed_time,
        "prefer_quiet": prefer_quiet,
        "uu_tien_vang": prefer_quiet,
        "require_enough_seats": enough_seats,
        "group_budget": (max_price * eff_people) if max_price is not None else None,
        "sources": {
            "budget": budget_source,
            "num_people": people_source,
            "time_slot": slot_source,
            "prefer_quiet": quiet_source,
            "spicy": spicy_source,
            "vegetarian": veg_source,
            "seats": seats_source,
        },
    }


class DishRecommender:
    """Hệ thống chấm điểm và gợi ý món ăn theo phương pháp đa tiêu chí MCDM/SAW (7 tiêu chí)."""

    def __init__(
        self,
        dishes_path: str = DEFAULT_DISHES_PATH,
        profiles_path: str = DEFAULT_PROFILES_PATH,
        crowd_config_path: str = DEFAULT_CROWD_CONFIG_PATH,
    ):
        self.dishes_path = resolve_project_path(dishes_path)
        self.profiles_path = resolve_project_path(profiles_path)
        self.crowd_config_path = resolve_project_path(crowd_config_path)
        self.dishes: List[Dict[str, Any]] = []
        self.profiles: Dict[str, Dict[str, float]] = {}
        self.crowd_config: Dict[str, Any] = {}
        self.stats: Dict[str, Dict[str, float]] = {}

        self.load_data()

    def load_data(self) -> None:
        """Đọc danh sách món ăn từ CSV, hồ sơ trọng số từ profiles.json và hệ số độ đông từ crowd_config.json."""
        # 1. Đọc crowd_config.json
        self.crowd_config = load_crowd_config(self.crowd_config_path)

        # 2. Đọc profiles.json
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.profiles = {k: v for k, v in data.items() if not k.startswith("_")}
        else:
            self.profiles = {}

        # 3. Đọc dishes CSV (tương thích ngược với file thiếu cột chay, do_dong_cao_diem, so_cho_ngoi, nguon_du_lieu)
        self.dishes = []
        if os.path.exists(self.dishes_path):
            raw_rows = read_csv_dicts(self.dishes_path)
            for row in raw_rows:
                do_dong_raw = (row.get("do_dong_cao_diem") or "").strip()
                so_cho_raw = (row.get("so_cho_ngoi") or "").strip()
                nguon_raw = (row.get("nguon_du_lieu") or "").strip()
                dish = {
                    "id": int(row["id"]),
                    "ten_mon": row["ten_mon"].strip(),
                    "loai": row["loai"].strip(),
                    "gia_vnd": float(row["gia_vnd"]),
                    "khoang_cach_m": float(row["khoang_cach_m"]),
                    "thoi_gian_phuc_vu_phut": float(row["thoi_gian_phuc_vu_phut"]),
                    "do_no": float(row["do_no"]),
                    "dinh_duong": float(row["dinh_duong"]),
                    "do_ngon": float(row["do_ngon"]),
                    "cay": int(row["cay"]),
                    "chay": int(row.get("chay") or 0),
                    "do_dong_cao_diem": int(float(do_dong_raw)) if do_dong_raw else DEFAULT_DO_DONG_CAO_DIEM,
                    "so_cho_ngoi": int(float(so_cho_raw)) if so_cho_raw else DEFAULT_SO_CHO_NGOI,
                    "mo_ta": row["mo_ta"].strip(),
                    "nguon_du_lieu": nguon_raw if nguon_raw else DEFAULT_NGUON_DU_LIEU,
                }
                self.dishes.append(dish)

        # 4. Tính toán min, max ban đầu cho từng tiêu chí
        self._calculate_stats()

    def _calculate_stats(self, dishes: Optional[List[Dict[str, Any]]] = None) -> None:
        """Tính giá trị min và max cho từng tiêu chí trên tập dishes hiện có."""
        target = self.dishes if dishes is None else dishes
        self.stats = {}
        if not target:
            for crit in CRITERIA_COLUMN_MAP:
                self.stats[crit] = {"min": 0.0, "max": 1.0}
            return

        for crit, col in CRITERIA_COLUMN_MAP.items():
            if crit == "thoi_gian":
                vals = [float(d.get("thoi_gian_cho", d["thoi_gian_phuc_vu_phut"])) for d in target]
            elif crit == "cho_ngoi":
                vals = [float(d.get("cho_ngoi_nhom", 1.0)) for d in target]
            else:
                vals = [float(d[col]) for d in target]
            self.stats[crit] = {
                "min": min(vals),
                "max": max(vals),
            }

    def _enrich_dishes_with_crowd(
        self,
        dishes: List[Dict[str, Any]],
        so_nguoi: int,
        khung_gio: str,
        crowd_cfg: Dict[str, Any],
        prefer_quiet: bool = False,
    ) -> List[Dict[str, Any]]:
        """Gắn thêm các trường độ đông, thời gian chờ và chỗ ngồi nhóm cho từng món ăn."""
        enriched: List[Dict[str, Any]] = []
        for d in dishes:
            metrics = compute_crowd_metrics(
                d,
                so_nguoi=so_nguoi,
                khung_gio=khung_gio,
                crowd_cfg=crowd_cfg,
                prefer_quiet=prefer_quiet,
            )
            merged = dict(d)
            merged.update(metrics)
            enriched.append(merged)
        return enriched

    def _resolve_crowd_config(self, crowd_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Gộp cấu hình độ đông gốc từ file JSON với tham số tùy chỉnh từ giao diện (nếu có)."""
        base = dict(self.crowd_config) if self.crowd_config else load_crowd_config(self.crowd_config_path)
        slots = dict(base.get("he_so_khung_gio", {"cao_diem": 1.0, "binh_thuong": 0.6, "vang": 0.3}))
        if crowd_params:
            if "a" in crowd_params and crowd_params["a"] is not None:
                base["a"] = float(crowd_params["a"])
            if "b" in crowd_params and crowd_params["b"] is not None:
                base["b"] = float(crowd_params["b"])
            if "he_so_chiem_cho" in crowd_params and crowd_params["he_so_chiem_cho"] is not None:
                base["he_so_chiem_cho"] = float(crowd_params["he_so_chiem_cho"])
            if "he_so_khung_gio" in crowd_params and isinstance(crowd_params["he_so_khung_gio"], dict):
                for k, v in crowd_params["he_so_khung_gio"].items():
                    if k in slots and v is not None:
                        slots[k] = float(v)
        base["he_so_khung_gio"] = slots
        return base

    def normalize_criterion(self, criterion: str, raw_value: float) -> float:
        """
        Chuẩn hóa giá trị tiêu chí về đoạn [0, 1]:
        - Nếu là tiêu chí chi phí (gia, khoang_cach, thoi_gian):
          norm = (max - val) / (max - min) (càng thấp càng gần 1.0).
        - Nếu là tiêu chí lợi ích (do_no, dinh_duong, do_ngon, cho_ngoi):
          norm = (val - min) / (max - min) (càng cao càng gần 1.0).
        - Nếu max == min thì trả về 1.0.
        """
        stat = self.stats.get(criterion, {"min": 0.0, "max": 1.0})
        c_min = stat["min"]
        c_max = stat["max"]

        if abs(c_max - c_min) < 1e-12:
            return 1.0

        if criterion in COST_CRITERIA:
            norm = (c_max - raw_value) / (c_max - c_min)
        else:
            norm = (raw_value - c_min) / (c_max - c_min)

        return max(0.0, min(1.0, norm))

    @staticmethod
    def _dish_failures(dish: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, str]:
        """
        Kiểm tra một món với từng ràng buộc ĐỘC LẬP. Trả về dict {khóa_bộ_lọc: lý do} cho các bộ lọc mà món
        không đạt (khóa: 'budget', 'spicy', 'vegetarian', 'exclude', 'seats'); rỗng nghĩa là đạt mọi điều kiện.
        """
        failures: Dict[str, str] = {}
        max_price = constraints.get("max_price")
        exclude_types = constraints.get("exclude_types") or []
        so_nguoi = int(constraints.get("num_people") or 1)

        # 1. Ràng buộc giá (ngân sách tính cho TỪNG NGƯỜI, giá món là giá một suất)
        if max_price is not None and dish["gia_vnd"] > max_price:
            failures["budget"] = f"Giá {dish['gia_vnd']:,.0f}đ vượt ngân sách {max_price:,.0f}đ/người"

        # 2. Ràng buộc cay
        if constraints.get("exclude_spicy", False) and dish["cay"] == 1:
            failures["spicy"] = "Món có vị cay"

        # 3. Ràng buộc chay (bộ lọc cứng: chỉ giữ món chay = 1)
        if constraints.get("vegetarian_only", False) and int(dish.get("chay", 0)) != 1:
            failures["vegetarian"] = "Món mặn (không phải món chay)"

        # 4. Ràng buộc loại trừ
        if exclude_types:
            dish_text = f"{dish['loai']} {dish['ten_mon']}".lower()
            for exc in exclude_types:
                exc_norm = exc.strip().lower()
                if exc_norm and exc_norm in dish_text:
                    failures["exclude"] = f"Nằm trong danh sách loại trừ '{exc}'"
                    break

        # 5. Ràng buộc chỉ quán đủ chỗ cho cả nhóm (cho_trong >= so_nguoi)
        if constraints.get("require_enough_seats", False):
            cho_trong = float(dish.get("cho_trong", dish.get("so_cho_ngoi", DEFAULT_SO_CHO_NGOI)))
            if cho_trong < so_nguoi:
                failures["seats"] = (
                    f"Chỗ trống ước tính (~{cho_trong:.1f} chỗ) không đủ cho nhóm {so_nguoi} người"
                )

        return failures

    def filter_dishes_detailed(
        self,
        dishes: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Lọc món theo các ràng buộc và trả thêm SỐ MÓN CÒN LẠI SAU TỪNG BƯỚC LỌC (để hiển thị ở tab Các bước xử lý).
        Thứ tự bước lọc: ngân sách -> cay -> chay -> loại trừ món -> chỗ ngồi.
        """
        constraints = constraints or {}
        active = {
            "budget": constraints.get("max_price") is not None,
            "spicy": bool(constraints.get("exclude_spicy", False)),
            "vegetarian": bool(constraints.get("vegetarian_only", False)),
            "exclude": bool(constraints.get("exclude_types")),
            "seats": bool(constraints.get("require_enough_seats", False)),
        }

        failures_by_dish = [(dish, self._dish_failures(dish, constraints)) for dish in dishes]

        steps: List[Dict[str, Any]] = [
            {"key": "start", "label": "Ban đầu", "count": len(dishes), "active": True}
        ]
        remaining = list(failures_by_dish)
        for key, label in FILTER_STEP_LABELS:
            if key == "seats" and not active["seats"]:
                continue
            if active[key]:
                remaining = [(d, f) for d, f in remaining if key not in f]
            steps.append({"key": key, "label": label, "count": len(remaining), "active": active[key]})

        passed: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for dish, failures in failures_by_dish:
            if not failures:
                passed.append(dish)
            else:
                rejected.append({
                    "dish_id": dish["id"],
                    "ten_mon": dish["ten_mon"],
                    "reasons": list(failures.values()),
                })
        return passed, rejected, steps

    def filter_dishes(
        self,
        dishes: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Lọc danh sách món ăn theo các ràng buộc."""
        if not constraints:
            return list(dishes), []
        passed, rejected, _ = self.filter_dishes_detailed(dishes, constraints)
        return passed, rejected

    def score_dish(
        self,
        dish: Dict[str, Any],
        weights: Dict[str, float],
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Tính điểm số cho một món ăn dựa trên bộ trọng số:
        Điểm = Tổng(trọng số_i × giá trị_chuẩn_hóa_i).
        Tiêu chí 'thoi_gian' dùng thoi_gian_cho (chi phí, đảo chiều);
        tiêu chí 'cho_ngoi' dùng cho_ngoi_nhom (lợi ích, thuận chiều).
        """
        normalized_criteria: Dict[str, float] = {}
        contributions: Dict[str, float] = {}
        total_score = 0.0

        for crit, col in CRITERIA_COLUMN_MAP.items():
            if crit == "thoi_gian":
                raw_val = float(dish.get("thoi_gian_cho", dish["thoi_gian_phuc_vu_phut"]))
            elif crit == "cho_ngoi":
                raw_val = float(dish.get("cho_ngoi_nhom", 1.0))
            else:
                raw_val = float(dish[col])

            norm_val = self.normalize_criterion(crit, raw_val)
            normalized_criteria[crit] = round(norm_val, 4)

            w = float(weights.get(crit, 0.0))
            contrib = w * norm_val
            contributions[crit] = round(contrib, 4)
            total_score += contrib

        return round(total_score, 4), normalized_criteria, contributions

    def _score_items(
        self,
        dishes: List[Dict[str, Any]],
        weights: Dict[str, float],
        num_people: int,
    ) -> List[Dict[str, Any]]:
        """Chấm điểm danh sách món; mỗi mục kèm thông tin chi phí nhóm, độ đông, thời gian chờ và chỗ ngồi."""
        items: List[Dict[str, Any]] = []
        for dish in dishes:
            score, norm_crit, contribs = self.score_dish(dish, weights)
            items.append({
                "dish": dish,
                "score": score,
                "normalized_criteria": norm_crit,
                "contributions": contribs,
                "group_price": dish["gia_vnd"] * num_people,
                "do_dong_hieu_dung": dish.get("do_dong_hieu_dung", float(dish.get("do_dong_cao_diem", DEFAULT_DO_DONG_CAO_DIEM))),
                "crowd_label": dish.get("crowd_label", ""),
                "thoi_gian_cho": dish.get("thoi_gian_cho", dish["thoi_gian_phuc_vu_phut"]),
                "cho_trong": dish.get("cho_trong", float(dish.get("so_cho_ngoi", DEFAULT_SO_CHO_NGOI))),
                "cho_trong_int": dish.get("cho_trong_int", int(dish.get("so_cho_ngoi", DEFAULT_SO_CHO_NGOI))),
                "cho_ngoi_nhom": dish.get("cho_ngoi_nhom", 1.0),
                "seat_warning": dish.get("seat_warning"),
            })
        return items

    def analyze_no_match(
        self,
        constraints: Dict[str, Any],
        weights: Dict[str, float],
        num_people: int,
        k: int = 3,
        dishes_pool: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Khi KHÔNG món nào thỏa mọi điều kiện lọc: phân tích điều kiện nào đang quá chặt và gợi ý k món gần điều kiện nhất.
        """
        source_dishes = dishes_pool if dishes_pool is not None else self.dishes
        labels = {
            "budget": "ngân sách",
            "seats": "đủ chỗ ngồi cho nhóm",
            "spicy": "không cay",
            "vegetarian": "chỉ món chay",
            "exclude": "loại trừ món",
        }
        active_keys = [
            key for key, on in (
                ("budget", constraints.get("max_price") is not None),
                ("seats", bool(constraints.get("require_enough_seats"))),
                ("spicy", bool(constraints.get("exclude_spicy"))),
                ("vegetarian", bool(constraints.get("vegetarian_only"))),
                ("exclude", bool(constraints.get("exclude_types"))),
            ) if on
        ]
        failures = [(d, self._dish_failures(d, constraints)) for d in source_dishes]

        def passing_without(skip: List[str]) -> List[Dict[str, Any]]:
            return [d for d, f in failures if all(k in skip or k not in f for k in active_keys)]

        culprits = []
        for key in active_keys:
            pool = passing_without([key])
            if pool:
                entry: Dict[str, Any] = {"key": key, "label": labels[key], "count_if_relaxed": len(pool)}
                if key == "budget":
                    entry["cheapest_price"] = min(d["gia_vnd"] for d in pool)
                elif key == "seats":
                    entry["max_available_seats"] = max(float(d.get("cho_trong", 0.0)) for d in pool)
                culprits.append(entry)

        # Nới dần theo thứ tự ưu tiên: ngân sách -> đủ chỗ ngồi -> loại trừ món -> cay -> chay
        relaxed: List[str] = []
        selected_ids = set()
        ordered_pool: List[Dict[str, Any]] = []
        target_count = min(k, len(source_dishes))

        for key in ("budget", "seats", "exclude", "spicy", "vegetarian"):
            if key not in active_keys:
                continue
            relaxed.append(key)
            step_pool = passing_without(relaxed)
            step_items = self._score_items(
                [d for d in step_pool if d["id"] not in selected_ids], weights, num_people
            )
            if "budget" in relaxed and "seats" not in relaxed:
                step_items.sort(key=lambda it: (it["dish"]["gia_vnd"], -it["score"]))
            elif "seats" in relaxed and "budget" not in relaxed:
                step_items.sort(key=lambda it: (-float(it.get("cho_trong", 0.0)), -it["score"]))
            else:
                step_items.sort(key=lambda it: it["score"], reverse=True)

            for it in step_items:
                ordered_pool.append(it)
                selected_ids.add(it["dish"]["id"])
            if len(ordered_pool) >= target_count:
                break

        if len(ordered_pool) < target_count:
            extra_items = self._score_items(
                [d for d in source_dishes if d["id"] not in selected_ids], weights, num_people
            )
            extra_items.sort(key=lambda it: it["score"], reverse=True)
            ordered_pool.extend(extra_items)

        return {
            "active": [labels[k] for k in active_keys],
            "culprits": culprits,
            "relaxed": relaxed,
            "relaxed_labels": [labels[k] for k in relaxed],
            "kept_labels": [labels[k] for k in active_keys if k not in relaxed],
            "near_items": ordered_pool[:k],
        }

    def recommend(
        self,
        intent: str,
        constraints: Optional[Dict[str, Any]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        top_k: int = 3,
        max_price_per_person: Optional[float] = None,
        num_people: int = 1,
        vegetarian_only: bool = False,
        exclude_spicy: bool = False,
        time_slot: Optional[str] = None,
        require_enough_seats: bool = False,
        crowd_params: Optional[Dict[str, Any]] = None,
        current_time: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Gợi ý Top-K món ăn trưa phù hợp nhất, tích hợp độ đông, thời gian chờ và chỗ ngồi theo số người.
        """
        crowd_cfg = self._resolve_crowd_config(crowd_params)
        gio_cao_diem = crowd_cfg.get("gio_cao_diem", ["11:30", "13:00"])

        effective = merge_constraints(
            constraints,
            max_price_per_person=max_price_per_person,
            num_people=num_people,
            vegetarian_only=vegetarian_only,
            exclude_spicy=exclude_spicy,
            time_slot=time_slot,
            require_enough_seats=require_enough_seats,
            current_time=current_time,
            gio_cao_diem=gio_cao_diem,
        )
        eff_people = effective["num_people"]
        eff_slot = effective["time_slot"]
        prefer_quiet = effective["prefer_quiet"]

        # 1. Xác định trọng số 7 tiêu chí
        if custom_weights:
            profile_name = "custom"
            active_weights = dict(custom_weights)
            total_w = sum(active_weights.values())
            if total_w > 0 and abs(total_w - 1.0) > 1e-4:
                active_weights = {k: v / total_w for k, v in active_weights.items()}
        elif intent in self.profiles:
            profile_name = intent
            active_weights = dict(self.profiles[intent])
        else:
            profile_name = "can_bang"
            active_weights = dict(self.profiles.get("can_bang", {
                "gia": 0.18, "khoang_cach": 0.14, "thoi_gian": 0.14,
                "do_no": 0.14, "dinh_duong": 0.14, "do_ngon": 0.16, "cho_ngoi": 0.10,
            }))

        # 2. Tính các chỉ số độ đông & chỗ ngồi trên toàn bộ tập món, sau đó cập nhật min-max
        enriched_dishes = self._enrich_dishes_with_crowd(
            self.dishes,
            so_nguoi=eff_people,
            khung_gio=eff_slot,
            crowd_cfg=crowd_cfg,
            prefer_quiet=prefer_quiet,
        )
        self._calculate_stats(enriched_dishes)

        # 3. Lọc món ăn theo ràng buộc (kèm số món còn lại sau từng bước)
        filtered_dishes, rejected_dishes, filter_steps = self.filter_dishes_detailed(enriched_dishes, effective)

        # 4. Chấm điểm từng món
        scored_dishes = self._score_items(filtered_dishes, active_weights, eff_people)

        # 5. Sắp xếp điểm giảm dần
        scored_dishes.sort(key=lambda item: item["score"], reverse=True)

        # 6. Lấy Top-K
        top_recommendations = scored_dishes[:top_k]

        # 7. Không có món nào thỏa: phân tích điều kiện quá chặt và gợi ý 3 món gần nhất
        no_match = None
        near_matches: List[Dict[str, Any]] = []
        if not scored_dishes and enriched_dishes:
            no_match = self.analyze_no_match(
                effective, active_weights, eff_people, k=3, dishes_pool=enriched_dishes
            )
            near_matches = no_match["near_items"]

        he_so_kg = float(crowd_cfg["he_so_khung_gio"][eff_slot])
        a_base = float(crowd_cfg["a"])
        a_eff = a_base * 2.0 if prefer_quiet else a_base
        crowd_info = {
            "so_nguoi": eff_people,
            "so_nguoi_source": effective["sources"].get("num_people"),
            "khung_gio": eff_slot,
            "khung_gio_label": TIME_SLOT_LABELS_VI.get(eff_slot, eff_slot),
            "khung_gio_source": effective["sources"].get("time_slot"),
            "he_so_khung_gio": he_so_kg,
            "a_base": a_base,
            "a_effective": a_eff,
            "b": float(crowd_cfg["b"]),
            "he_so_chiem_cho": float(crowd_cfg["he_so_chiem_cho"]),
            "uu_tien_vang": prefer_quiet,
            "require_enough_seats": bool(effective.get("require_enough_seats")),
        }

        return {
            "profile_used": profile_name,
            "weights_used": active_weights,
            "total_dishes": len(enriched_dishes),
            "filtered_dishes_count": len(filtered_dishes),
            "rejected_dishes": rejected_dishes,
            "top_k": top_recommendations,
            "all_ranked": scored_dishes,
            "constraints_used": effective,
            "filter_steps": filter_steps,
            "no_match": no_match,
            "near_matches": near_matches,
            "num_people": eff_people,
            "group_budget": effective["group_budget"],
            "crowd_info": crowd_info,
        }
