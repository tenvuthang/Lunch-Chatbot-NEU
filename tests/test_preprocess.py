"""
Kiểm thử cho module src/preprocess.py
Kiểm tra:
1. Chuẩn hóa chuỗi (chữ thường, unicode NFC, ký tự đặc biệt, viết tắt)
2. Tách từ tiếng Việt bằng PyVi
3. Lọc bỏ stopword nhỏ và bảo toàn các từ quan trọng (không, rẻ, gần, nhanh, cay, no, ngon, dưới)
4. Cấu trúc kết quả trung gian trả về
"""

import pytest
from src.preprocess import normalize_text, tokenize_text, filter_stopwords, preprocess


def test_normalize_text():
    """Kiểm tra hàm normalize_text chuẩn hóa tiếng Việt, chữ thường và từ viết tắt."""
    raw = "Hnay Ăn Gì RẺ RẺ?! ktx NEU ạ..."
    normalized = normalize_text(raw)

    # Viết tắt 'hnay' thành 'hôm nay', 'ktx' thành 'ký túc xá', 'neu' thành 'kinh tế quốc dân'
    assert "hôm nay" in normalized
    assert "ký túc xá" in normalized
    assert "kinh tế quốc dân" in normalized
    # Không còn dấu chấm than, hỏi, chấm
    assert "?" not in normalized
    assert "!" not in normalized
    assert "." not in normalized
    # Đã lowercase
    assert normalized == normalized.lower()


def test_tokenize_text():
    """Kiểm tra hàm tokenize_text tách từ tiếng Việt ghép."""
    text = "kinh tế quốc dân ký túc xá"
    tokens = tokenize_text(text)
    assert isinstance(tokens, list)
    # PyVi thường ghép từ phức thành: kinh_tế, quốc_dân, ký_túc_xá
    assert "kinh_tế" in tokens or "kinh" in tokens
    assert len(tokens) > 0


def test_stopwords_preservation():
    """Kiểm tra không bị xóa các từ mang ý nghĩa ngữ nghĩa trọng yếu."""
    crucial_words = ["không", "rẻ", "gần", "nhanh", "cay", "no", "ngon", "dưới"]
    filtered = filter_stopwords(crucial_words)
    for word in crucial_words:
        assert word in filtered, f"Từ quan trọng '{word}' đã bị lọc nhầm thành stopword!"


def test_stopwords_removal():
    """Kiểm tra các hư từ thông thường được lọc bỏ chính xác."""
    filler_words = ["thì", "là", "mà", "ở", "của", "ạ", "ơi", "nha", "nhé"]
    filtered = filter_stopwords(filler_words)
    assert len(filtered) == 0, "Các từ hư từ chưa được lọc hết!"


def test_preprocess_intermediate_structure():
    """Kiểm tra cấu trúc dict kết quả trung gian từ hàm preprocess."""
    input_text = "Hnay ăn gì rẻ dưới 30k k cay ở ngõ Tự Do ạ"
    result = preprocess(input_text, remove_stopwords=True)

    assert isinstance(result, dict)
    assert "original" in result
    assert "normalized" in result
    assert "tokens" in result
    assert "tokens_no_stopwords" in result

    assert result["original"] == input_text
    assert "ạ" not in result["tokens_no_stopwords"]
    # Các từ quan trọng vẫn có mặt
    tokens_no_sw = result["tokens_no_stopwords"]
    assert any("không" in t or "cay" in t for t in tokens_no_sw)
    assert any("rẻ" in t for t in tokens_no_sw)
