"""
Module trích xuất ràng buộc từ văn bản (Constraint Extractor).
Sử dụng biểu thức chính quy (Regex) và hệ luật ngôn ngữ để trích xuất:
1. Ngân sách tối đa (max_price): hiểu các định dạng '40k', '40 nghìn', '40.000', 'dưới 50k', 'tầm 35k đổ lại'.
2. Ràng buộc không cay (exclude_spicy): nhận diện 'không cay', 'k cay', 'ko cay', 'đừng cay', 'không ăn cay'.
3. Danh sách loại trừ (exclude_types): nhận diện 'không ăn bún', 'trừ phở', 'ko muốn ăn cơm', 'trừ bún cá'.

Tất cả các hàm đều có docstring tiếng Việt và trả về kết quả trung gian phục vụ hiển thị GUI.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class ConstraintExtractor:
    """Lớp trích xuất các ràng buộc về giá, độ cay và món loại trừ từ câu của người dùng."""

    def __init__(self):
        # Từ khóa các loại món hoặc nguyên liệu phổ biến để nhận diện loại trừ
        self.known_food_keywords = [
            "cơm tấm", "cơm rang", "cơm niêu", "cơm bình dân", "cơm",
            "bún chả", "bún đậu", "bún bò", "bún cá", "bún riêu", "bún ốc", "bún",
            "phở bò", "phở gà", "phở cuốn", "phở",
            "bánh mì chảo", "bánh mì",
            "mì cay", "mì vằn thắn", "mì xào", "mì",
            "miến lươn", "miến gà", "miến",
            "salad", "đồ ăn nhanh", "burger", "pizza", "gà rán",
            "bánh cuốn", "cháo", "hải sản", "cá", "thịt bò", "thịt gà"
        ]

    def extract_budget(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Trích xuất ngân sách tối đa từ văn bản tiếng Việt.
        
        Các mẫu hỗ trợ:
        - 'dưới 40k', 'tầm 35k', 'khoảng 30k', '40k quay đầu', '25k đổ lại'
        - '40 nghìn', '40 ngàn', '40 ngìn', '40k'
        - '40.000', '40,000', '40000', '40000đ', '40000 vnd'
        - Dải giá: '30-40k' -> lấy cận trên 40000
        
        Trả về:
            Tuple[Optional[int], Optional[str]]: (Số tiền VND, chuỗi khớp được)
        """
        if not text:
            return None, None

        norm_text = text.lower()

        # 1. Bắt dải giá, ví dụ: '30-40k', '25 - 30 nghìn'
        range_pattern = r"(\d{1,3})\s*[-–]\s*(\d{1,3})\s*(k|nghìn|ngàn|ngin|ng|k\b)"
        range_match = re.search(range_pattern, norm_text)
        if range_match:
            high_val = int(range_match.group(2))
            return high_val * 1000, range_match.group(0)

        # 2. Bắt dạng số kèm nghìn/ngàn/k: 'dưới 40k', '40k', '35 nghìn', '50 ngàn'
        k_pattern = r"(?:dưới|duoi|tầm|tam|khoảng|khoang|tối đa|toi da|chỉ|chi|budget|ngân sách|ngan sach)?\s*(\d{1,3}(?:\.\d{1,3})?)\s*(?:k|nghìn|ngàn|ngin|ng)\b"
        k_match = re.search(k_pattern, norm_text)
        if k_match:
            try:
                val_float = float(k_match.group(1))
                price = int(val_float * 1000)
                return price, k_match.group(0).strip()
            except ValueError:
                pass

        # 3. Bắt dạng số đầy đủ: '40.000', '40,000', '40000' (từ 15.000 đến 200.000)
        full_pattern = r"(?:dưới|duoi|tầm|tam|khoảng|khoang|tối đa|toi da|chỉ|chi)?\s*(\b\d{2}[.,]\d{3}\b|\b[1-9]\d{4,5}\b)\s*(?:đ|d|vnd|đồng|dong)?"
        full_match = re.search(full_pattern, norm_text)
        if full_match:
            raw_num = full_match.group(1).replace(".", "").replace(",", "")
            try:
                price = int(raw_num)
                if 10000 <= price <= 500000:
                    return price, full_match.group(0).strip()
            except ValueError:
                pass

        # 4. Bắt dạng ngữ cảnh: 'dưới 40', 'tam 30' (nếu số trong khoảng 20-100)
        implicit_pattern = r"(?:dưới|duoi|tầm|tam|khoảng|khoang|dưới tầm|tối đa|toi da|chỉ có|chi co)\s*(\d{2})\b"
        implicit_match = re.search(implicit_pattern, norm_text)
        if implicit_match:
            val = int(implicit_match.group(1))
            if 15 <= val <= 100:
                return val * 1000, implicit_match.group(0).strip()

        return None, None

    def extract_spicy(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Trích xuất yêu cầu không ăn cay.
        
        Các mẫu hỗ trợ:
        - 'không cay', 'khong cay', 'k cay', 'ko cay'
        - 'đừng cay', 'dung cay', 'không ăn cay', 'ko an cay', 'k ăn cay'
        - 'bỏ ớt', 'không ớt', 'ko ớt', 'k ớt'
        
        Trả về:
            Tuple[bool, Optional[str]]: (True nếu yêu cầu không cay, chuỗi khớp được)
        """
        if not text:
            return False, None

        norm_text = text.lower()
        no_spicy_pattern = r"\b(?:không|khong|ko|k|đừng|dung|chớ)\s*(?:ăn\s*)?(?:được\s*)?(?:cay|quá cay|nhiều ớt|ớt)\b|\b(?:bỏ ớt|không ớt|ko ớt|k ớt)\b"
        match = re.search(no_spicy_pattern, norm_text)
        if match:
            return True, match.group(0).strip()

        return False, None

    def extract_exclusions(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Trích xuất danh sách các món hoặc loại đồ ăn cần loại trừ.
        
        Các mẫu hỗ trợ:
        - 'không ăn bún', 'ko ăn cơm', 'k ăn phở'
        - 'trừ bún cá', 'ngoại trừ phở', 'trừ đồ ngọt ra', 'trừ bún cá ra'
        - 'không muốn ăn cơm', 'ko thèm mì'
        
        Trả về:
            Tuple[List[str], List[str]]: (Danh sách từ khóa loại trừ, danh sách cụm từ khớp được)
        """
        if not text:
            return [], []

        norm_text = text.lower()
        exclusions = []
        matched_phrases = []

        # Các mẫu phát hiện mệnh đề loại trừ
        exclusion_patterns = [
            r"\b(?:không ăn|khong an|ko ăn|ko an|k ăn|k an|không muốn ăn|ko muon an)\s+([a-zA-Zà-ỹÀ-Ỹ\s]+?)(?:nhé|nha|ạ|ạ\?|\.|\,|$|\bnhưng\b)",
            r"\b(?:trừ|tru|ngoại trừ|ngoai tru)\s+([a-zA-Zà-ỹÀ-Ỹ\s]+?)(?:ra|nhé|nha|ạ|ạ\?|\.|\,|$|\bnhưng\b)",
            r"\b(?:ko|k|không|khong)\s+(?:thèm|thích)\s+([a-zA-Zà-ỹÀ-Ỹ\s]+?)(?:nhé|nha|ạ|ạ\?|\.|\,|$)"
        ]

        for pattern in exclusion_patterns:
            for match in re.finditer(pattern, norm_text):
                phrase = match.group(1).strip()
                # Bỏ qua nếu cụm từ này là yêu cầu về cay (đã được extract_spicy xử lý)
                if any(w in phrase for w in ["cay", "quá cay", "ớt", "ot"]):
                    continue

                matched_phrases.append(match.group(0).strip())
                # Đối chiếu với danh sách các loại món / món đã biết
                found_specific = False
                for kw in self.known_food_keywords:
                    if kw in phrase:
                        exclusions.append(kw)
                        found_specific = True

                # Nếu cụm từ ngắn gọn không nằm trong danh sách nhưng hợp lý, lưu luôn cụm từ đó
                if not found_specific and 1 <= len(phrase.split()) <= 3:
                    exclusions.append(phrase)

        # Loại bỏ trùng lặp và giữ nguyên thứ tự
        unique_exclusions = list(dict.fromkeys(exclusions))
        return unique_exclusions, matched_phrases

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Hàm chính trích xuất toàn bộ ràng buộc từ câu người dùng.
        
        Trả về:
            Dict[str, Any]:
                - 'max_price': Ngân sách tối đa (int VND hoặc None)
                - 'exclude_spicy': Có loại trừ đồ cay hay không (bool)
                - 'exclude_types': Danh sách món/loại đồ ăn loại trừ (List[str])
                - 'intermediate': Chi tiết các chuỗi khớp được trong câu để phục vụ hiển thị.
        """
        max_price, price_match = self.extract_budget(text)
        exclude_spicy, spicy_match = self.extract_spicy(text)
        exclude_types, exclusion_matches = self.extract_exclusions(text)

        return {
            "max_price": max_price,
            "exclude_spicy": exclude_spicy,
            "exclude_types": exclude_types,
            "intermediate": {
                "matched_price_text": price_match,
                "matched_spicy_text": spicy_match,
                "matched_exclusion_texts": exclusion_matches
            }
        }


def extract_constraints(text: str) -> Dict[str, Any]:
    """Hàm tiện ích bọc nhanh lớp ConstraintExtractor."""
    extractor = ConstraintExtractor()
    return extractor.extract(text)


if __name__ == "__main__":
    samples = [
        "Trưa nay ăn gì dưới 40k không ăn cay ở ngõ Tự Do",
        "Có món nào tầm 35 nghìn, trừ bún cá ra nhé bot",
        "Tìm quán cơm bình dân 25k k cay gần trường",
        "Budget 50.000đ đổ lại, không ăn bún và không cay"
    ]
    extractor = ConstraintExtractor()
    for s in samples:
        print(f"Câu: '{s}'")
        res = extractor.extract(s)
        print(f"  -> Ràng buộc: max_price={res['max_price']}, exclude_spicy={res['exclude_spicy']}, exclude_types={res['exclude_types']}")
        print(f"     Khớp: {res['intermediate']}")
