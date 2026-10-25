"""
Kiểm thử cho module src/recommender.py
Sử dụng ví dụ nhỏ tính toán thủ công bằng tay (manual calculation) để đối chiếu kết quả:
1. Chuẩn hóa Min-Max đảo chiều cho các tiêu chí chi phí (giá, khoảng cách, thời gian)
2. Chuẩn hóa Min-Max thuận chiều cho các tiêu chí lợi ích (độ no, dinh dưỡng, độ ngon)
3. Điểm tổng hợp trọng số = tổng(trọng số * giá trị chuẩn hóa)
4. Đóng góp từng tiêu chí
5. Bộ lọc ràng buộc (giá tối đa, không cay, loại trừ món/loại)
"""

import pytest
from src.recommender import DishRecommender


@pytest.fixture
def mini_recommender():
    """Tạo bộ gợi ý với tập dữ liệu nhỏ 3 món mẫu để tính toán bằng tay."""
    recommender = DishRecommender(dishes_path="data/dishes.csv", profiles_path="config/profiles.json")

    # Tập 3 món giả lập dùng để tính tay
    mock_dishes = [
        {
            "id": 1,
            "ten_mon": "Cơm bình dân",
            "loai": "cơm",
            "gia_vnd": 20000.0,
            "khoang_cach_m": 100.0,
            "thoi_gian_phuc_vu_phut": 5.0,
            "do_no": 4.0,
            "dinh_duong": 3.0,
            "do_ngon": 3.0,
            "cay": 0,
            "mo_ta": "rẻ ngon"
        },
        {
            "id": 2,
            "ten_mon": "Phở bò tái lăn",
            "loai": "phở",
            "gia_vnd": 50000.0,
            "khoang_cach_m": 500.0,
            "thoi_gian_phuc_vu_phut": 10.0,
            "do_no": 3.0,
            "dinh_duong": 4.0,
            "do_ngon": 5.0,
            "cay": 0,
            "mo_ta": "đậm đà"
        },
        {
            "id": 3,
            "ten_mon": "Mì cay 7 cấp độ",
            "loai": "mì",
            "gia_vnd": 60000.0,
            "khoang_cach_m": 300.0,
            "thoi_gian_phuc_vu_phut": 15.0,
            "do_no": 3.0,
            "dinh_duong": 2.0,
            "do_ngon": 4.0,
            "cay": 1,
            "mo_ta": "cay xé lưỡi"
        }
    ]

    recommender.dishes = mock_dishes
    recommender._calculate_stats()
    return recommender


def test_min_max_bounds(mini_recommender):
    """Kiểm tra cận min và max của các tiêu chí trong tập mini."""
    stats = mini_recommender.stats
    assert stats["gia"]["min"] == 20000.0 and stats["gia"]["max"] == 60000.0
    assert stats["khoang_cach"]["min"] == 100.0 and stats["khoang_cach"]["max"] == 500.0
    assert stats["thoi_gian"]["min"] == 5.0 and stats["thoi_gian"]["max"] == 15.0
    assert stats["do_no"]["min"] == 3.0 and stats["do_no"]["max"] == 4.0
    assert stats["dinh_duong"]["min"] == 2.0 and stats["dinh_duong"]["max"] == 4.0
    assert stats["do_ngon"]["min"] == 3.0 and stats["do_ngon"]["max"] == 5.0


def test_manual_criteria_normalization(mini_recommender):
    """
    Kiểm tra giá trị chuẩn hóa từng tiêu chí tính tay:
    - Giá (chi phí: min=20k, max=60k):
      + 20k -> (60k - 20k)/(60k - 20k) = 1.0 (tốt nhất)
      + 50k -> (60k - 50k)/(60k - 20k) = 10k/40k = 0.25
      + 60k -> (60k - 60k)/(60k - 20k) = 0.0 (kém nhất)
    - Khoảng cách (chi phí: min=100, max=500):
      + 100m -> (500 - 100)/400 = 1.0
      + 500m -> (500 - 500)/400 = 0.0
      + 300m -> (500 - 300)/400 = 0.5
    - Dinh dưỡng (lợi ích: min=2, max=4):
      + 3.0 -> (3 - 2)/(4 - 2) = 0.5
      + 4.0 -> (4 - 2)/(4 - 2) = 1.0
      + 2.0 -> (2 - 2)/(4 - 2) = 0.0
    """
    # 1. Giá
    assert mini_recommender.normalize_criterion("gia", 20000.0) == pytest.approx(1.0)
    assert mini_recommender.normalize_criterion("gia", 50000.0) == pytest.approx(0.25)
    assert mini_recommender.normalize_criterion("gia", 60000.0) == pytest.approx(0.0)

    # 2. Khoảng cách
    assert mini_recommender.normalize_criterion("khoang_cach", 100.0) == pytest.approx(1.0)
    assert mini_recommender.normalize_criterion("khoang_cach", 500.0) == pytest.approx(0.0)
    assert mini_recommender.normalize_criterion("khoang_cach", 300.0) == pytest.approx(0.5)

    # 3. Dinh dưỡng
    assert mini_recommender.normalize_criterion("dinh_duong", 3.0) == pytest.approx(0.5)
    assert mini_recommender.normalize_criterion("dinh_duong", 4.0) == pytest.approx(1.0)
    assert mini_recommender.normalize_criterion("dinh_duong", 2.0) == pytest.approx(0.0)


def test_manual_score_calculation(mini_recommender):
    """
    Kiểm tra điểm tổng hợp theo trọng số tùy chỉnh tính bằng tay:
    Trọng số:
      gia: 0.4
      khoang_cach: 0.2
      thoi_gian: 0.1
      do_no: 0.1
      dinh_duong: 0.1
      do_ngon: 0.1
      Tổng = 1.0

    Điểm Món 1 (Cơm bình dân):
      gia: 0.4 * 1.0 = 0.40
      khoang_cach: 0.2 * 1.0 = 0.20
      thoi_gian: 0.1 * 1.0 = 0.10
      do_no: 0.1 * 1.0 = 0.10
      dinh_duong: 0.1 * 0.5 = 0.05
      do_ngon: 0.1 * 0.0 = 0.00
      -> Tổng Món 1 = 0.40 + 0.20 + 0.10 + 0.10 + 0.05 + 0.00 = 0.8500!

    Điểm Món 2 (Phở bò):
      gia: 0.4 * 0.25 = 0.10
      khoang_cach: 0.2 * 0.0 = 0.00
      thoi_gian: 0.1 * 0.5 = 0.05
      do_no: 0.1 * 0.0 = 0.00
      dinh_duong: 0.1 * 1.0 = 0.10
      do_ngon: 0.1 * 1.0 = 0.10
      -> Tổng Món 2 = 0.10 + 0.00 + 0.05 + 0.00 + 0.10 + 0.10 = 0.3500!

    Điểm Món 3 (Mì cay):
      gia: 0.4 * 0.0 = 0.00
      khoang_cach: 0.2 * 0.5 = 0.10
      thoi_gian: 0.1 * 0.0 = 0.00
      do_no: 0.1 * 0.0 = 0.00
      dinh_duong: 0.1 * 0.0 = 0.00
      do_ngon: 0.1 * 0.5 = 0.05
      -> Tổng Món 3 = 0.1500!
    """
    weights = {
        "gia": 0.4,
        "khoang_cach": 0.2,
        "thoi_gian": 0.1,
        "do_no": 0.1,
        "dinh_duong": 0.1,
        "do_ngon": 0.1
    }

    dish1 = mini_recommender.dishes[0]
    dish2 = mini_recommender.dishes[1]
    dish3 = mini_recommender.dishes[2]

    score1, norm1, contrib1 = mini_recommender.score_dish(dish1, weights)
    score2, norm2, contrib2 = mini_recommender.score_dish(dish2, weights)
    score3, norm3, contrib3 = mini_recommender.score_dish(dish3, weights)

    assert score1 == pytest.approx(0.85, abs=1e-4)
    assert score2 == pytest.approx(0.35, abs=1e-4)
    assert score3 == pytest.approx(0.15, abs=1e-4)

    # Kiểm tra đóng góp từng tiêu chí của món 1
    assert contrib1["gia"] == pytest.approx(0.40)
    assert contrib1["khoang_cach"] == pytest.approx(0.20)
    assert contrib1["dinh_duong"] == pytest.approx(0.05)


def test_constraint_filtering(mini_recommender):
    """Kiểm tra lọc theo ràng buộc ngân sách, không cay và loại trừ."""
    # 1. Ràng buộc giá <= 45.000: Món 2 (50k) và Món 3 (60k) bị loại
    passed, rejected = mini_recommender.filter_dishes(mini_recommender.dishes, {"max_price": 45000})
    assert len(passed) == 1
    assert passed[0]["id"] == 1
    assert len(rejected) == 2

    # 2. Ràng buộc không cay (exclude_spicy): Món 3 (cay=1) bị loại
    passed, rejected = mini_recommender.filter_dishes(mini_recommender.dishes, {"exclude_spicy": True})
    assert len(passed) == 2
    assert {d["id"] for d in passed} == {1, 2}

    # 3. Ràng buộc loại trừ phở: Món 2 bị loại
    passed, rejected = mini_recommender.filter_dishes(mini_recommender.dishes, {"exclude_types": ["phở"]})
    assert len(passed) == 2
    assert {d["id"] for d in passed} == {1, 3}


def test_recommend_top_k(mini_recommender):
    """Kiểm tra hàm recommend trả về kết quả xếp hạng đúng thứ tự Top-K."""
    weights = {
        "gia": 0.4, "khoang_cach": 0.2, "thoi_gian": 0.1,
        "do_no": 0.1, "dinh_duong": 0.1, "do_ngon": 0.1
    }
    result = mini_recommender.recommend(
        intent="can_bang",
        custom_weights=weights,
        top_k=2
    )

    assert result["profile_used"] == "custom"
    assert len(result["top_k"]) == 2
    # Món 1 điểm 0.85 đứng đầu, Món 2 điểm 0.35 đứng nhì
    assert result["top_k"][0]["dish"]["id"] == 1
    assert result["top_k"][1]["dish"]["id"] == 2
    assert "contributions" in result["top_k"][0]
    assert "normalized_criteria" in result["top_k"][0]
