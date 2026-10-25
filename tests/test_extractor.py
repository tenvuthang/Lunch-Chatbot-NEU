"""
Kiểm thử cho module src/extractor.py
Kiểm tra:
1. Trích xuất ngân sách (các định dạng 40k, 35 nghìn, 50.000, 30-40k)
2. Trích xuất yêu cầu cay (không cay, k cay, đừng cay)
3. Trích xuất danh sách loại trừ (trừ phở, không ăn bún)
4. Trả về kết quả trung gian
"""

import pytest
from src.extractor import ConstraintExtractor, extract_constraints


def test_extract_budget_formats():
    """Kiểm tra trích xuất chính xác các định dạng giá tiền tiếng Việt."""
    extractor = ConstraintExtractor()

    # Dạng 'k': 40k, 35k
    val, text = extractor.extract_budget("tìm món dưới 40k nhé")
    assert val == 40000

    # Dạng 'nghìn' / 'ngàn'
    val, text = extractor.extract_budget("tầm 35 nghìn có món nào ngon")
    assert val == 35000

    # Dạng số chấm: 50.000đ
    val, text = extractor.extract_budget("ngân sách khoảng 50.000đ")
    assert val == 50000

    # Dạng dải giá: 25-30k -> lấy cận trên 30000
    val, text = extractor.extract_budget("ăn trưa tầm 25-30k quanh NEU")
    assert val == 30000

    # Không có số tiền
    val, text = extractor.extract_budget("ăn gì ngon ngon gần trường")
    assert val is None


def test_extract_spicy_requirement():
    """Kiểm tra nhận diện yêu cầu không cay."""
    extractor = ConstraintExtractor()

    # Các biến thể không cay
    assert extractor.extract_spicy("món nào không ăn cay được")[0] is True
    assert extractor.extract_spicy("tìm cơm 25k k cay ngõ tự do")[0] is True
    assert extractor.extract_spicy("đừng cay quá nhé bot")[0] is True
    assert extractor.extract_spicy("bỏ ớt ra giúp mình")[0] is True

    # Không yêu cầu không cay (thậm chí thích ăn cay)
    assert extractor.extract_spicy("mình muốn ăn cay xé lưỡi")[0] is False
    assert extractor.extract_spicy("tìm quán phở bò gần trường")[0] is False


def test_extract_exclusions():
    """Kiểm tra trích xuất danh sách loại trừ món/loại."""
    extractor = ConstraintExtractor()

    # Trừ phở
    exclusions, matched = extractor.extract_exclusions("trưa nay chán phở rồi, trừ phở ra nhé")
    assert "phở" in exclusions

    # Không ăn bún cá
    exclusions, matched = extractor.extract_exclusions("món nào rẻ nhưng không ăn bún cá nha")
    assert "bún cá" in exclusions or "bún" in exclusions

    # Đảm bảo 'không ăn cay' không bị nhầm thành loại trừ món 'cay'
    exclusions, matched = extractor.extract_exclusions("trưa nay ăn gì rẻ dưới 30k không ăn cay")
    assert "cay" not in exclusions


def test_extract_all_intermediate():
    """Kiểm tra hàm extract tổng hợp trả về đầy đủ các trường và kết quả trung gian."""
    sentence = "Trưa nay ăn gì dưới 35k không ăn cay trừ bún cá ra"
    res = extract_constraints(sentence)

    assert isinstance(res, dict)
    assert res["max_price"] == 35000
    assert res["exclude_spicy"] is True
    assert any("bún" in exc for exc in res["exclude_types"])

    # Kiểm tra kết quả trung gian
    assert "intermediate" in res
    inter = res["intermediate"]
    assert "matched_price_text" in inter
    assert "matched_spicy_text" in inter
    assert "matched_exclusion_texts" in inter
    assert inter["matched_price_text"] is not None
    assert inter["matched_spicy_text"] is not None
