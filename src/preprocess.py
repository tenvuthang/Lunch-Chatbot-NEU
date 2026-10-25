"""
Module tiền xử lý văn bản tiếng Việt cho Lunch Chatbot.
Bao gồm:
- Chuẩn hóa Unicode NFC, chuyển chữ thường, loại bỏ ký tự đặc biệt/ký tự thừa.
- Tách từ tiếng Việt sử dụng thư viện PyVi (ViTokenizer).
- Lọc bỏ stopword (tập stopword tiếng Việt nhỏ gọn, bảo toàn các từ quan trọng như: 'không', 'rẻ', 'gần', 'nhanh', 'cay', 'no', 'ngon', 'dưới').
- Trả về đầy đủ kết quả trung gian để hiển thị trên giao diện người dùng.
"""

import re
import unicodedata
from typing import Dict, Any, List, Set, Optional

try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

# Danh sách stopword tiếng Việt nhỏ gọn (chủ yếu là hư từ, trợ từ, đại từ nhân xưng, từ cảm thán)
# TUYỆT ĐỐI KHÔNG chứa: không, rẻ, gần, nhanh, cay, no, ngon, dưới, ít, nhiều, cơm, bún, phở,...
DEFAULT_STOPWORDS: Set[str] = {
    "thì", "là", "mà", "ở", "của", "và", "những", "các", "cho", "để", "với",
    "được", "bị", "bởi", "lại", "này", "đó", "kia", "ạ", "ơi", "nha", "nhé",
    "vậy", "nghen", "thôi", "nhỉ", "hả", "cái", "chiếc", "bạn", "mình",
    "em", "anh", "chị", "ad", "admin", "bot"
}

# Bảng ánh xạ viết tắt / tiếng lóng sinh viên phổ biến
SLANG_MAP: Dict[str, str] = {
    "hnay": "hôm nay",
    "hqua": "hôm qua",
    "bme": "bố mẹ",
    "sv": "sinh viên",
    "ktx": "ký túc xá",
    "neu": "kinh tế quốc dân",
    "dc": "được",
    "đc": "được",
    "k": "không",
    "ko": "không",
    "khong": "không",
    "tks": "cảm ơn",
    "thx": "cảm ơn",
    "thanks": "cảm ơn",
    "cam on": "cảm ơn",
    "chao": "chào",
    "tam biet": "tạm biệt"
}


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa chuỗi văn bản đầu vào:
    1. Chuyển về chuẩn Unicode NFC.
    2. Chuyển thành chữ thường (lowercase).
    3. Chuẩn hóa từ viết tắt / tiếng lóng cơ bản.
    4. Loại bỏ các ký tự đặc biệt thừa, giữ lại chữ cái tiếng Việt, chữ số và khoảng trắng.
    5. Xóa khoảng trắng thừa.
    """
    if not text:
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Lowercase
    text = text.lower().strip()

    # 3. Chuẩn hóa tiếng lóng / viết tắt theo từ riêng biệt
    words = text.split()
    expanded_words = [SLANG_MAP.get(w, w) for w in words]
    text = " ".join(expanded_words)

    # 4. Loại bỏ ký tự đặc biệt thừa (giữ lại chữ cái Unicode tiếng Việt, số, dấu gạch nối trong từ ghép)
    # Loại bỏ các dấu câu như: ! ? . , ; : " ' ( ) [ ] { } ~ ` @ # $ % ^ & * + = < > / \ |
    text = re.sub(r"[^\w\s\d_]", " ", text, flags=re.UNICODE)

    # 5. Xóa nhiều khoảng trắng liên tiếp
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tách từ tiếng Việt bằng PyVi ViTokenizer.
    Trả về danh sách các token (từ đơn hoặc từ ghép nối bằng dấu gạch dưới '_').
    Nếu chưa có PyVi, sử dụng tách theo khoảng trắng làm fallback.
    """
    if not text:
        return []

    if HAS_PYVI:
        segmented = ViTokenizer.tokenize(text)
        return segmented.split()
    else:
        # Fallback tách theo khoảng trắng
        return text.split()


def filter_stopwords(tokens: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Lọc bỏ các stopword không quan trọng khỏi danh sách token.
    Bảo toàn toàn bộ các từ khóa ngữ nghĩa quan trọng.
    """
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    return [token for token in tokens if token.lower() not in stopwords]


def preprocess(text: str, remove_stopwords: bool = True) -> Dict[str, Any]:
    """
    Hàm tiền xử lý toàn diện cho câu đầu vào.
    
    Tham số:
        text (str): Câu văn bản gốc của người dùng.
        remove_stopwords (bool): Có loại bỏ stopword hay không (mặc định True).

    Trả về:
        Dict[str, Any]: Chứa đầy đủ các bước trung gian:
            - 'original': Câu văn bản ban đầu.
            - 'normalized': Câu sau khi chuẩn hóa Unicode, viết thường và xóa ký tự thừa.
            - 'tokens': Danh sách token sau khi tách từ bằng PyVi.
            - 'tokens_no_stopwords': Danh sách token sau khi lọc bỏ stopword.
    """
    original = text if text is not None else ""
    normalized = normalize_text(original)
    tokens = tokenize_text(normalized)

    if remove_stopwords:
        tokens_no_stopwords = filter_stopwords(tokens, DEFAULT_STOPWORDS)
    else:
        tokens_no_stopwords = list(tokens)

    return {
        "original": original,
        "normalized": normalized,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stopwords
    }


if __name__ == "__main__":
    test_input = "Hnay ăn gì rẻ rẻ dưới 30k ktx NEU ạ?!"
    result = preprocess(test_input)
    print("Kết quả tiền xử lý:")
    for k, v in result.items():
        print(f"  {k}: {v}")
