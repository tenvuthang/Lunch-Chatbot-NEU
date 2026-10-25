"""
Module hệ thống gợi ý món ăn đa tiêu chí (Multi-Criteria Dish Recommender).
Thực hiện:
1. Đọc dữ liệu món ăn từ data/dishes.csv và hồ sơ trọng số từ config/profiles.json.
2. Chuẩn hóa giá trị các tiêu chí về đoạn [0, 1] bằng Min-Max Scaling:
   - Các tiêu chí chi phí (giá, khoảng cách, thời gian phục vụ): đảo chiều vì càng thấp càng tốt.
   - Các tiêu chí lợi ích (độ no, dinh dưỡng, độ ngon): giữ chiều thuận vì càng cao càng tốt.
3. Lọc danh sách món ăn theo các ràng buộc (ngân sách tối đa, không cay, loại trừ món/loại).
4. Tính tổng điểm trọng số: Điểm = Tổng(trọng số_i × giá trị_chuẩn_hóa_i).
5. Trả về Top-K món ăn phù hợp nhất kèm bảng đóng góp chi tiết của từng tiêu chí để hiển thị giao diện.
6. Cho phép truyền trọng số tùy chỉnh (custom_weights) để ghi đè cấu hình profile.
"""

import csv
import json
import os
from typing import Dict, Any, List, Optional, Tuple


# Ánh xạ tên tiêu chí trong profiles.json sang tên cột trong dishes.csv
CRITERIA_COLUMN_MAP: Dict[str, str] = {
    "gia": "gia_vnd",
    "khoang_cach": "khoang_cach_m",
    "thoi_gian": "thoi_gian_phuc_vu_phut",
    "do_no": "do_no",
    "dinh_duong": "dinh_duong",
    "do_ngon": "do_ngon"
}

# Tiêu chí chi phí (càng thấp càng tốt -> đảo chiều chuẩn hóa)
COST_CRITERIA = {"gia", "khoang_cach", "thoi_gian"}

# Tiêu chí lợi ích (càng cao càng tốt -> chuẩn hóa thuận)
BENEFIT_CRITERIA = {"do_no", "dinh_duong", "do_ngon"}


class DishRecommender:
    """Hệ thống chấm điểm và gợi ý món ăn theo phương pháp đa tiêu chí MCDM/SAW."""

    def __init__(
        self,
        dishes_path: str = "data/dishes.csv",
        profiles_path: str = "config/profiles.json"
    ):
        self.dishes_path = dishes_path
        self.profiles_path = profiles_path
        self.dishes: List[Dict[str, Any]] = []
        self.profiles: Dict[str, Dict[str, float]] = {}
        self.stats: Dict[str, Dict[str, float]] = {}

        self.load_data()

    def load_data(self) -> None:
        """Đọc danh sách món ăn từ CSV và hồ sơ trọng số từ JSON."""
        # 1. Đọc profiles.json
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Loại bỏ các key chú thích như _comment
                self.profiles = {k: v for k, v in data.items() if not k.startswith("_")}
        else:
            self.profiles = {}

        # 2. Đọc dishes.csv
        self.dishes = []
        if os.path.exists(self.dishes_path):
            with open(self.dishes_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
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
                        "mo_ta": row["mo_ta"].strip()
                    }
                    self.dishes.append(dish)

        # 3. Tính toán min, max cho từng tiêu chí trên toàn bộ tập dữ liệu
        self._calculate_stats()

    def _calculate_stats(self) -> None:
        """Tính giá trị min và max cho từng tiêu chí trên tập dishes hiện có."""
        self.stats = {}
        if not self.dishes:
            for crit in CRITERIA_COLUMN_MAP:
                self.stats[crit] = {"min": 0.0, "max": 1.0}
            return

        for crit, col in CRITERIA_COLUMN_MAP.items():
            vals = [d[col] for d in self.dishes]
            self.stats[crit] = {
                "min": min(vals),
                "max": max(vals)
            }

    def normalize_criterion(self, criterion: str, raw_value: float) -> float:
        """
        Chuẩn hóa giá trị tiêu chí về đoạn [0, 1]:
        - Nếu là tiêu chí chi phí (giá, khoảng cách, thời gian):
          norm = (max - val) / (max - min) (càng thấp càng gần 1.0).
        - Nếu là tiêu chí lợi ích (độ no, dinh dưỡng, độ ngon):
          norm = (val - min) / (max - min) (càng cao càng gần 1.0).
        - Nếu max == min thì trả về 1.0.
        """
        stat = self.stats.get(criterion, {"min": 0.0, "max": 1.0})
        c_min = stat["min"]
        c_max = stat["max"]

        if c_max == c_min:
            return 1.0

        if criterion in COST_CRITERIA:
            # Đảo chiều
            norm = (c_max - raw_value) / (c_max - c_min)
        else:
            # Thuận chiều
            norm = (raw_value - c_min) / (c_max - c_min)

        # Giữ trong khoảng [0, 1] phòng ngừa giá trị vượt ngưỡng
        return max(0.0, min(1.0, norm))

    def filter_dishes(
        self,
        dishes: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Lọc danh sách món ăn theo các ràng buộc:
        - max_price: bỏ món có giá vượt mức cho phép.
        - exclude_spicy: bỏ món cay (cay == 1).
        - exclude_types: bỏ món có tên hoặc loại chứa từ khóa cần loại trừ.
        
        Trả về:
            Tuple[List[Dict], List[Dict]]: (Danh sách món đạt điều kiện, Danh sách món bị loại kèm lý do)
        """
        if not constraints:
            return list(dishes), []

        max_price = constraints.get("max_price")
        exclude_spicy = constraints.get("exclude_spicy", False)
        exclude_types = constraints.get("exclude_types", [])

        passed = []
        rejected = []

        for dish in dishes:
            reasons = []

            # 1. Ràng buộc giá
            if max_price is not None and dish["gia_vnd"] > max_price:
                reasons.append(f"Giá {dish['gia_vnd']:,.0f}đ vượt ngân sách {max_price:,.0f}đ")

            # 2. Ràng buộc cay
            if exclude_spicy and dish["cay"] == 1:
                reasons.append("Món có vị cay")

            # 3. Ràng buộc loại trừ
            if exclude_types:
                dish_text = f"{dish['loai']} {dish['ten_mon']}".lower()
                for exc in exclude_types:
                    exc_norm = exc.strip().lower()
                    if exc_norm and exc_norm in dish_text:
                        reasons.append(f"Nằm trong danh sách loại trừ '{exc}'")
                        break

            if not reasons:
                passed.append(dish)
            else:
                rejected.append({
                    "dish_id": dish["id"],
                    "ten_mon": dish["ten_mon"],
                    "reasons": reasons
                })

        return passed, rejected

    def score_dish(
        self,
        dish: Dict[str, Any],
        weights: Dict[str, float]
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Tính điểm số cho một món ăn dựa trên bộ trọng số:
        Điểm = Tổng(trọng số_i × giá trị_chuẩn_hóa_i).
        
        Trả về:
            Tuple[float, Dict[str, float], Dict[str, float]]:
                - Tổng điểm (0.0 đến 1.0)
                - Bảng giá trị chuẩn hóa từng tiêu chí
                - Bảng điểm đóng góp của từng tiêu chí (trọng số × chuẩn hóa)
        """
        normalized_criteria = {}
        contributions = {}
        total_score = 0.0

        for crit, col in CRITERIA_COLUMN_MAP.items():
            raw_val = dish[col]
            norm_val = self.normalize_criterion(crit, raw_val)
            normalized_criteria[crit] = round(norm_val, 4)

            w = weights.get(crit, 0.0)
            contrib = w * norm_val
            contributions[crit] = round(contrib, 4)
            total_score += contrib

        return round(total_score, 4), normalized_criteria, contributions

    def recommend(
        self,
        intent: str,
        constraints: Optional[Dict[str, Any]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Gợi ý Top-K món ăn trưa phù hợp nhất.
        
        Tham số:
            intent (str): Nhãn ý định (vd: 'tiet_kiem', 'healthy', 'nhanh_gon', 'no_bung', 'ngon_doi_gio', 'can_bang').
            constraints (Optional[Dict]): Ràng buộc giá, cay, món loại trừ.
            custom_weights (Optional[Dict]): Bộ trọng số tùy chỉnh ghi đè profile.
            top_k (int): Số lượng món gợi ý hàng đầu (mặc định 3).

        Trả về:
            Dict[str, Any]: Đầy đủ thông tin và kết quả trung gian:
                - 'profile_used': Tên profile được áp dụng.
                - 'weights_used': Bảng trọng số 6 tiêu chí thực tế được dùng.
                - 'total_dishes': Tổng số món có trong CSDL.
                - 'filtered_dishes_count': Số món còn lại sau khi lọc ràng buộc.
                - 'rejected_dishes': Danh sách món bị lọc bỏ kèm lý do.
                - 'top_k': Top K món ăn có điểm cao nhất kèm bảng đóng góp tiêu chí.
                - 'all_ranked': Danh sách toàn bộ các món đạt chuẩn xếp theo điểm giảm dần.
        """
        # 1. Xác định trọng số
        if custom_weights:
            profile_name = "custom"
            active_weights = dict(custom_weights)
            # Chuẩn hóa tổng trọng số về 1.0 nếu cần
            total_w = sum(active_weights.values())
            if total_w > 0 and abs(total_w - 1.0) > 1e-4:
                active_weights = {k: v / total_w for k, v in active_weights.items()}
        elif intent in self.profiles:
            profile_name = intent
            active_weights = dict(self.profiles[intent])
        else:
            # Fallback về cân bằng nếu không tìm thấy profile tương ứng
            profile_name = "can_bang"
            active_weights = dict(self.profiles.get("can_bang", {
                "gia": 0.2, "khoang_cach": 0.15, "thoi_gian": 0.15,
                "do_no": 0.15, "dinh_duong": 0.15, "do_ngon": 0.2
            }))

        # 2. Lọc món ăn theo ràng buộc
        filtered_dishes, rejected_dishes = self.filter_dishes(self.dishes, constraints)

        # 3. Chấm điểm từng món
        scored_dishes = []
        for dish in filtered_dishes:
            score, norm_crit, contribs = self.score_dish(dish, active_weights)
            scored_dishes.append({
                "dish": dish,
                "score": score,
                "normalized_criteria": norm_crit,
                "contributions": contribs
            })

        # 4. Sắp xếp điểm giảm dần
        scored_dishes.sort(key=lambda item: item["score"], reverse=True)

        # 5. Lấy Top-K
        top_recommendations = scored_dishes[:top_k]

        return {
            "profile_used": profile_name,
            "weights_used": active_weights,
            "total_dishes": len(self.dishes),
            "filtered_dishes_count": len(filtered_dishes),
            "rejected_dishes": rejected_dishes,
            "top_k": top_recommendations,
            "all_ranked": scored_dishes
        }


if __name__ == "__main__":
    recommender = DishRecommender()
    print("Tổng số món đọc được:", len(recommender.dishes))
    print("Khoảng giá trị các tiêu chí:", recommender.stats)

    # Thử nghiệm gợi ý tiết kiệm với ràng buộc dưới 35k và không cay
    test_constraints = {"max_price": 35000, "exclude_spicy": True, "exclude_types": ["bún"]}
    rec_result = recommender.recommend("tiet_kiem", constraints=test_constraints, top_k=3)

    print(f"\nKết quả gợi ý profile '{rec_result['profile_used']}':")
    print(f"Số món đạt điều kiện: {rec_result['filtered_dishes_count']}/{rec_result['total_dishes']}")
    for idx, item in enumerate(rec_result["top_k"], 1):
        d = item["dish"]
        print(f"{idx}. {d['ten_mon']} - Giá: {d['gia_vnd']:,.0f}đ - Điểm: {item['score']}")
        print(f"   Đóng góp tiêu chí: {item['contributions']}")
