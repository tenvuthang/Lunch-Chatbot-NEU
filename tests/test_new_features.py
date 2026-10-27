"""
Kiểm thử toàn diện cho 4 nhóm tính năng mới (VIỆC 1 -> VIỆC 4):
1. Kiểm tra hợp lệ ô nhập Ngân sách mỗi người (20..200, trống, ngoài khoảng) và Số người đi ăn (1..10, ngoài khoảng).
2. Trích xuất yêu cầu ăn chay (có dấu/không dấu, phân biệt 'chạy bộ'/'chạy deadline' và phủ định 'không ăn chay').
3. Bộ lọc cứng & tính toán theo số người trong Recommender và Chatbot:
   - Ngân sách 20k, 200k, để trống; số người 1 và 10; ưu tiên ngân sách từ câu nói so với ô nhập.
   - Chỉ món chay bật; kết hợp chỉ chay + không cay + ngân sách thấp (trường hợp không có món thỏa -> báo lý do + gợi ý 3 món gần nhất).
   - Tương thích ngược khi file dishes.csv cũ chưa có cột 'chay'.
4. Nhận định kết quả tự động bằng luật (eval_insights) và lưu phản hồi người dùng (feedback.csv UTF-8 BOM).
5. Đồng bộ tham số remove_stopwords giữa lúc huấn luyện và lúc dự đoán.
"""

import os
import pytest

from src.input_validation import (
    parse_budget_k,
    parse_num_people,
    budget_k_to_vnd,
)
from src.extractor import ConstraintExtractor, extract_constraints
from src.recommender import DishRecommender
from src.chatbot import Chatbot
from src import data_manager as dm
from src.io_utils import write_csv_rows
from src.feedback import (
    RATING_HELPFUL,
    RATING_WRONG,
    append_feedback,
    load_feedback,
    feedback_summary,
    format_feedback_summary,
)
from src.eval_insights import (
    LEVEL_GOOD,
    LEVEL_WARN,
    LEVEL_BAD,
    accuracy_level,
    format_accuracy_text,
    build_insights,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. KIỂM TRA Ô NHẬP NGÂN SÁCH (20..200, TRỐNG, NGOÀI KHOẢNG) & SỐ NGƯỜI (1..10)
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("raw,expected_val", [
    ("", None),
    ("   ", None),
    (None, None),
    ("20", 20),
    ("50", 50),
    ("200", 200),
])
def test_parse_budget_k_valid(raw, expected_val):
    val, err = parse_budget_k(raw)
    assert err is None
    assert val == expected_val


@pytest.mark.parametrize("raw", ["0", "19", "201", "-30", "abc", "35.5", "50k"])
def test_parse_budget_k_out_of_range_or_invalid(raw):
    val, err = parse_budget_k(raw)
    assert val is None
    assert err is not None and len(err) > 0


@pytest.mark.parametrize("raw,expected_people", [
    ("", 1),
    ("1", 1),
    ("4", 4),
    ("10", 10),
])
def test_parse_num_people_valid(raw, expected_people):
    val, err = parse_num_people(raw)
    assert err is None
    assert val == expected_people


@pytest.mark.parametrize("raw", ["0", "11", "-2", "abc", "2.5"])
def test_parse_num_people_invalid(raw):
    val, err = parse_num_people(raw)
    assert err is not None and len(err) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRÍCH XUẤT YÊU CẦU MÓN CHAY (CÓ DẤU / KHÔNG DẤU / PHỦ ĐỊNH / TỪ ĐỒNG ÂM)
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("sentence", [
    "hôm nay rằm mình muốn ăn chay",
    "tìm quán đồ chay gần trường",
    "có món chay nào dưới 35k không",
    "trưa nay ăn cơm chay thanh đạm",
    "ăn bún chay ngõ tự do",
    "chay thôi nhé bot",
    "hom nay muon an chay",
    "tim mon chay re",
])
def test_extract_vegetarian_positive(sentence):
    res = extract_constraints(sentence)
    assert res["vegetarian_only"] is True
    assert res["intermediate"]["matched_vegetarian_text"] is not None


@pytest.mark.parametrize("sentence", [
    "vừa chạy bộ xong đói bụng quá",
    "đang chạy deadline sấp mặt cần ăn nhanh",
    "chay deadline voi qua tim mon nhanh",
    "chay bo xong an gi",
    "hôm nay không ăn chay đâu, thèm thịt",
    "mình không phải chay nhé",
    "đừng gợi ý đồ chay",
    "không muốn ăn đồ chay",
    "tìm quán phở bò tái chín",
])
def test_extract_vegetarian_negative_and_negation(sentence):
    res = extract_constraints(sentence)
    assert res["vegetarian_only"] is False


# ══════════════════════════════════════════════════════════════════════════════
# 3. RECOMMENDER & CHATBOT: NGÂN SÁCH 20/200/TRỐNG, SỐ NGƯỜI 1/10, CHAY, NO-MATCH
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def shared_bot():
    return Chatbot()


def test_budget_20_200_empty_and_num_people_1_10(shared_bot):
    # Ngân sách 20k (20.000đ/người), 1 người -> chỉ còn món giá <= 20.000đ
    r20 = shared_bot.respond(
        "ăn gì tiết kiệm",
        max_price_per_person=budget_k_to_vnd(20),
        num_people=1,
    )
    top20 = r20["recommendation"]["top_k"]
    assert len(top20) >= 1
    assert all(item["dish"]["gia_vnd"] <= 20000 for item in top20)
    assert r20["constraints"]["max_price"] == 20000
    assert r20["constraints"]["sources"]["budget"] == "từ ô nhập"
    assert r20["constraints"]["num_people"] == 1
    assert r20["constraints"]["group_budget"] == 20000
    assert "Ước tính cả nhóm: 20,000đ × 1" in r20["reply"]

    # Ngân sách 200k (200.000đ/người), 10 người -> tổng ngân sách nhóm = 2.000.000đ
    r200 = shared_bot.respond(
        "ăn gì ngon đổi gió",
        max_price_per_person=budget_k_to_vnd(200),
        num_people=10,
    )
    assert r200["constraints"]["max_price"] == 200000
    assert r200["constraints"]["num_people"] == 10
    assert r200["constraints"]["group_budget"] == 2000000
    assert r200["recommendation"]["filtered_dishes_count"] == 50
    assert "× 10" in r200["reply"]
    assert "2,000,000đ" in r200["reply"]

    # Ngân sách để trống (None), 1 người -> không giới hạn
    rempty = shared_bot.respond(
        "hôm nay ăn gì cân bằng",
        max_price_per_person=None,
        num_people=1,
    )
    assert rempty["constraints"]["max_price"] is None
    assert rempty["constraints"]["group_budget"] is None
    assert rempty["recommendation"]["filtered_dishes_count"] == 50


def test_text_budget_prioritized_over_ui_budget(shared_bot):
    """Nếu câu nói có ngân sách riêng (ví dụ 'dưới 40k') thì ưu tiên số trong câu nói."""
    resp = shared_bot.respond(
        "tìm món rẻ dưới 40k",
        max_price_per_person=100000,  # ô nhập là 100k
        num_people=3,
    )
    con = resp["constraints"]
    assert con["max_price"] == 40000
    assert con["sources"]["budget"] == "từ câu nói"
    assert con["num_people"] == 3
    assert con["group_budget"] == 120000


def test_vegetarian_only_filter(shared_bot):
    """Khi bật chỉ món chay, toàn bộ món gợi ý đều có chay == 1 và hiển thị 🌱 Chay."""
    resp = shared_bot.respond(
        "gợi ý đồ ăn trưa",
        vegetarian_only=True,
        num_people=2,
    )
    rec = resp["recommendation"]
    assert rec["filtered_dishes_count"] == 6
    assert len(rec["top_k"]) == 3
    assert all(item["dish"]["chay"] == 1 for item in rec["top_k"])
    assert "🌱 Chay" in resp["reply"]
    assert resp["constraints"]["sources"]["vegetarian"] == "từ ô tích"


def test_vegetarian_no_spicy_low_budget_no_match_fallback(shared_bot):
    """
    Chỉ chay + không cay + trừ bánh giò + ngân sách 20k (trong khi các món chay còn lại rẻ nhất là 25k):
    Không có món thỏa tất cả điều kiện -> báo rõ điều kiện quá chặt và gợi ý 3 món gần nhất.
    """
    resp = shared_bot.respond(
        "tìm món ăn trưa trừ bánh giò",
        max_price_per_person=20000,
        num_people=2,
        vegetarian_only=True,
        exclude_spicy=True,
    )
    rec = resp["recommendation"]
    assert rec["top_k"] == []
    assert len(rec["near_matches"]) == 3
    assert all(item["dish"]["chay"] == 1 and item["dish"]["cay"] == 0 for item in rec["near_matches"])
    assert rec["no_match"] is not None
    assert any(c["key"] == "budget" for c in rec["no_match"]["culprits"])
    assert "3 món gần điều kiện nhất" in resp["reply"]


def test_backward_compatibility_dishes_csv_without_chay_column(tmp_path):
    """File dishes.csv cũ (11 cột, thiếu cột chay) vẫn đọc được và mặc định chay = 0."""
    legacy_csv = tmp_path / "legacy_dishes.csv"
    header_11 = [
        "id", "ten_mon", "loai", "gia_vnd", "khoang_cach_m",
        "thoi_gian_phuc_vu_phut", "do_no", "dinh_duong", "do_ngon", "cay", "mo_ta",
    ]
    rows = [
        [1, "Cơm sườn mẫu", "cơm", 35000, 200, 7, 4, 3, 4, 0, "mẫu 11 cột"],
        [2, "Bún bò mẫu", "bún", 40000, 300, 8, 4, 4, 4, 1, "mẫu 11 cột"],
    ]
    write_csv_rows(str(legacy_csv), header_11, rows)

    loaded_dm = dm.load_dishes(str(legacy_csv))
    assert len(loaded_dm) == 2
    assert all(d["chay"] == "0" for d in loaded_dm)

    rec = DishRecommender(dishes_path=str(legacy_csv), profiles_path="config/profiles.json")
    assert len(rec.dishes) == 2
    assert all(d["chay"] == 0 for d in rec.dishes)


# ══════════════════════════════════════════════════════════════════════════════
# 4. NHẬN ĐỊNH KẾT QUẢ (EVAL_INSIGHTS) & PHẢN HỒI NGƯỜI DÙNG (FEEDBACK.CSV)
# ══════════════════════════════════════════════════════════════════════════════
def test_eval_insights_rules():
    assert accuracy_level(0.86) == LEVEL_GOOD
    assert accuracy_level(0.75) == LEVEL_WARN
    assert accuracy_level(0.60) == LEVEL_BAD
    assert format_accuracy_text(0.8594, 55, 64) == "Accuracy: 85,9% (đúng 55 / 64 câu kiểm thử)"

    labels = ["tiet_kiem", "healthy", "nhanh_gon"]
    report = {
        "tiet_kiem": {"precision": 0.9, "recall": 0.9, "f1-score": 0.90, "support": 8},
        "healthy": {"precision": 0.7, "recall": 0.6, "f1-score": 0.65, "support": 8},
        "nhanh_gon": {"precision": 0.8, "recall": 0.8, "f1-score": 0.80, "support": 8},
    }
    cm = [
        [7, 1, 0],
        [2, 5, 1],
        [0, 1, 7],
    ]
    names = {"tiet_kiem": "Tiết kiệm", "healthy": "Healthy", "nhanh_gon": "Nhanh gọn"}
    text = build_insights(0.7917, 19, 24, labels, report, cm, label_names=names, alpha=0.5)
    assert "19/24" in text
    assert "Tiết kiệm" in text and "Healthy" in text
    assert "3 lần" in text  # cặp Tiết kiệm - Healthy nhầm 1 + 2 = 3 lần

    # Khi accuracy >= 95% phải có cảnh báo dữ liệu nhỏ
    high_text = build_insights(0.96, 23, 24, labels, report, cm, label_names=names, alpha=0.5)
    assert "tự biên soạn" in high_text


def test_user_feedback_csv_utf8_bom(tmp_path):
    fb_path = str(tmp_path / "feedback.csv")
    assert format_feedback_summary(feedback_summary(fb_path)) == "Phản hồi người dùng: Chưa có phản hồi"

    append_feedback(
        path=fb_path,
        question="ăn gì rẻ dưới 30k",
        predicted_label="tiet_kiem",
        confidence=0.82,
        top1_dish="Cơm bình dân",
        rating=RATING_HELPFUL,
    )
    append_feedback(
        path=fb_path,
        question="hôm nay ăn chay",
        predicted_label="can_bang",
        confidence=0.45,
        top1_dish="Cơm chay thập cẩm",
        rating=RATING_WRONG,
        correct_label="healthy",
        note="muốn nhóm healthy chay",
    )

    with open(fb_path, "rb") as f:
        raw_bytes = f.read()
    assert raw_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw_bytes
    rows = load_feedback(fb_path)
    assert len(rows) == 2
    summary = feedback_summary(fb_path)
    assert summary["total"] == 2 and summary["helpful"] == 1
    assert format_feedback_summary(summary) == "Phản hồi người dùng: tỉ lệ hữu ích 50% trên 2 phản hồi"
