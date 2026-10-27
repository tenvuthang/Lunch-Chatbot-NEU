"""
Module trích xuất ràng buộc từ văn bản (Constraint Extractor).
Sử dụng biểu thức chính quy (Regex) và hệ luật ngôn ngữ để trích xuất:
1. Ngân sách tối đa (max_price): hiểu các định dạng '40k', '40 nghìn', '40.000', 'dưới 50k', 'tầm 35k đổ lại'.
2. Ràng buộc không cay (exclude_spicy): nhận diện 'không cay', 'k cay', 'ko cay', 'đừng cay', 'không ăn cay'.
3. Danh sách loại trừ (exclude_types): nhận diện 'không ăn bún', 'trừ phở', 'ko muốn ăn cơm', 'trừ bún cá'.
4. Yêu cầu ăn chay (vegetarian_only): nhận diện 'ăn chay', 'đồ chay', 'món chay', 'cơm chay', 'chay thôi', 'an chay';
   không nhận nhầm 'chạy' (chạy bộ, chạy deadline) và phủ định ('không ăn chay', 'không phải chay').
5. Số người đi ăn (num_people): nhận diện 'đi 4 người', '4 đứa', 'cả nhóm 6 bạn', 'một mình' = 1, 'hai đứa' = 2
   (cả chữ số và chữ tiếng Việt 1-10, có dấu và không dấu). Ngoài khoảng 1-10 thì bỏ qua và ghi chú.
6. Ưu tiên quán vắng (prefer_quiet / uu_tien_vang): nhận diện 'chỗ nào vắng', 'đỡ đông', 'không đông', 'yên tĩnh'
   (nhân đôi hệ số phạt độ đông a). Không nhận nhầm 'đông vui'.
7. Khung giờ đi ăn (time_slot / khung_gio): nhận diện '12h', '12h30', '11 giờ rưỡi', '13h15'... và suy ra
   khung_gio ∈ {'cao_diem', 'binh_thuong', 'vang'}. Không nhầm với số tiền ('40k') hay số người ('4 người').

Tất cả các hàm đều có docstring tiếng Việt và trả về kết quả trung gian phục vụ hiển thị GUI.
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional, Sequence, Tuple

try:
    from src.config import infer_time_slot
except ImportError:
    from config import infer_time_slot

PEOPLE_MIN_EXTRACTOR = 1
PEOPLE_MAX_EXTRACTOR = 10


class ConstraintExtractor:
    """Lớp trích xuất các ràng buộc về giá, độ cay, ăn chay, loại trừ, số người, độ vắng và khung giờ từ câu nói."""

    # ── Mẫu nhận diện ăn chay ────────────────────────────────────────────────
    _VEG_PREFIXED = re.compile(
        r"(?<!\w)(?:ăn|an|đồ|món|mon|cơm|com|bún|bun|phở|pho|bánh mì|banh mi|mì|mi|miến|mien|"
        r"quán|quan|tiệm|tiem|nhà hàng|nha hang|bữa|bua|suất|suat|thực đơn|thuc don|"
        r"thuần|thuan|toàn|toan)\s+chay(?!\w)"
    )
    _VEG_STANDALONE = re.compile(r"(?<!\w)chay(?!\w)")
    _VEG_RUNNING_AFTER = re.compile(
        r"\s+(?:bo|bộ|deadline|dead|nhanh|kip|kịp|ve|về|di|đi|xe|lop|lớp|ra|vao|vào|den|đến|toi|tới|"
        r"dua|đua|marathon|viec|việc|kpi|task|sprint|show|mat|mất|tron|trốn)(?!\w)"
    )
    _VEG_NEGATION_BEFORE = re.compile(
        r"(?<!\w)(?:không|khong|ko|k|chẳng|chang|chả|cha|đừng|dung|hổng|hong|hông)\s+"
        r"(?:(?:phải|phai|cần|can|muốn|muon|thích|thich|ưa|ua|nên|nen|được|duoc|hẳn|han|là|la|ăn|an|gợi ý|goi y|chọn|chon|lấy|lay)\s+)*$"
    )

    # ── Bảng số đếm tiếng Việt (có dấu & không dấu) ──────────────────────────
    _VI_NUMBER_WORDS: Dict[str, int] = {
        "một": 1, "mot": 1,
        "hai": 2, "đôi": 2, "doi": 2,
        "ba": 3,
        "bốn": 4, "bon": 4, "tư": 4,
        "năm": 5, "nam": 5,
        "sáu": 6, "sau": 6,
        "bảy": 7, "bay": 7, "bẩy": 7,
        "tám": 8, "tam": 8,
        "chín": 9, "chin": 9,
        "mười": 10, "muoi": 10, "chục": 10, "chuc": 10,
    }

    # Cụm chỉ đi 1 người
    _ALONE_PATTERN = re.compile(
        r"(?<!\w)(?:đi\s+|di\s+|ăn\s+|an\s+|ngồi\s+|ngoi\s+)?(?:một|mot|1)\s+(?:mình|minh)(?!\w)"
    )

    # Cụm số chữ số kèm danh từ chỉ người: "đi 4 người", "4 đứa", "cả nhóm 6 bạn", "15 người"
    _DIGIT_PEOPLE_PATTERN = re.compile(
        r"(?<!\w)(?:(?:đi|di|ăn|an|cả\s+nhóm|ca\s+nhom|nhóm|nhom|team|bọn\s+mình|bon\s+minh|tụi\s+mình|tui\s+minh|cỡ|co|tầm|tam|khoảng|khoang)\s+)*"
        r"(\d{1,3})\s*(?:người|nguoi|đứa|dua|bạn|ban|anh\s+em|chị\s+em|chi\s+em|thành\s+viên|thanh\s+vien|mạng|mang|mem)(?!\w)"
    )

    # Cụm chữ số tiếng Việt kèm danh từ chỉ người: "hai đứa", "bốn người", "sáu bạn"
    # Lưu ý: từ "nam" không dấu dễ trùng "bún bò nam bộ" nên chỉ khớp khi đi kèm danh từ chỉ người rõ ràng.
    _WORD_PEOPLE_PATTERN = re.compile(
        r"(?<!\w)(?:(?:đi|di|ăn|an|cả\s+nhóm|ca\s+nhom|nhóm|nhom|bọn\s+mình|bon\s+minh|tụi\s+mình|tui\s+minh)\s+)*"
        r"(một|mot|hai|đôi|doi|ba|bốn|bon|tư|năm|nam|sáu|sau|bảy|bay|bẩy|tám|tam|chín|chin|mười|muoi|chục|chuc)\s+"
        r"(người|nguoi|đứa|dua|bạn|ban|anh\s+em|chị\s+em|chi\s+em|thành\s+viên|thanh\s+vien|mình|minh)(?!\w)"
    )

    # ── Mẫu nhận diện ưu tiên quán vắng / yên tĩnh (uu_tien_vang) ────────────
    # Loại trừ các cụm mang nghĩa thích đông vui trước
    _CROWD_POSITIVE_PATTERN = re.compile(
        r"(?<!\w)(?:đông\s+vui|dong\s+vui|càng\s+đông\s+càng\s+vui|cang\s+dong\s+cang\s+vui|"
        r"thích\s+đông|thich\s+dong|quán\s+đông\s+vui|nhộn\s+nhịp|nhon\s+nhip)(?!\w)"
    )
    _PREFER_QUIET_PATTERN = re.compile(
        r"(?<!\w)(?:"
        r"(?:chỗ|cho|quán|quan|tiệm|tiem|nơi|noi)\s+(?:nào\s+|nao\s+)?(?:vắng|vang|ít\s+người|it\s+nguoi|ít\s+khách|it\s+khach|thoáng|thoang|yên\s+tĩnh|yen\s+tinh)|"
        r"(?:vắng\s+vẻ|vang\s+ve|vắng\s+người|vang\s+nguoi|vắng\s+khách|vang\s+khach|vắng\s+chút|vang\s+chut)|"
        r"(?:đỡ\s+đông|do\s+dong|ít\s+đông|it\s+dong|không\s+đông|khong\s+dong|ko\s+đông|ko\s+dong|k\s+đông|k\s+dong|"
        r"đừng\s+đông|dung\s+dong|sợ\s+đông|so\s+dong|ngại\s+đông|ngai\s+dong|tránh\s+đông|tranh\s+dong|né\s+chỗ\s+đông|ne\s+cho\s+dong)|"
        r"(?:yên\s+tĩnh|yen\s+tinh|thoáng\s+chỗ|thoang\s+cho|rộng\s+rãi\s+đỡ\s+đông)"
        r")(?!\w)"
    )

    # ── Mẫu nhận diện khung giờ đi ăn (khung_gio) ────────────────────────────
    # 1. Giờ cụ thể: '12h', '12h30', '11 giờ rưỡi', '13h15', '11g30', '12:30'
    _TIME_HM_PATTERN = re.compile(
        r"(?<!\w)(?:lúc\s+|luc\s+|tầm\s+|tam\s+|khoảng\s+|khoang\s+|vào\s+|vao\s+|cỡ\s+|co\s+)?"
        r"(\d{1,2})\s*(?:h|g|giờ|gio)\s*(?:(rưỡi|ruoi)|(\d{1,2})\s*(?:phút|phut|p)?)?(?:\s*(trưa|trua|chiều|chieu|sáng|sang))?(?!\w)"
    )
    _TIME_COLON_PATTERN = re.compile(
        r"(?<!\w)(?:lúc\s+|luc\s+|tầm\s+|tam\s+|khoảng\s+|khoang\s+|vào\s+|vao\s+)?"
        r"(\d{1,2}):(\d{2})(?:\s*(trưa|trua|chiều|chieu|sáng|sang))?(?!\w)"
    )
    # 2. Cụm từ chỉ trực tiếp khung giờ
    _TIME_DIRECT_PEAK = re.compile(
        r"(?<!\w)(?:giờ\s+cao\s+điểm|gio\s+cao\s+diem|cao\s+điểm\s+trưa|cao\s+diem\s+trua|giữa\s+trưa|giua\s+trua)(?!\w)"
    )
    _TIME_DIRECT_OFFPEAK = re.compile(
        r"(?<!\w)(?:ngoài\s+giờ\s+cao\s+điểm|ngoai\s+gio\s+cao\s+diem|giờ\s+thấp\s+điểm|gio\s+thap\s+diem|giờ\s+vắng|gio\s+vang)(?!\w)"
    )

    def __init__(self):
        self.known_food_keywords = [
            "cơm tấm", "cơm rang", "cơm niêu", "cơm bình dân", "cơm",
            "bún chả", "bún đậu", "bún bò", "bún cá", "bún riêu", "bún ốc", "bún",
            "phở bò", "phở gà", "phở cuốn", "phở",
            "bánh mì chảo", "bánh mì",
            "mì cay", "mì vằn thắn", "mì xào", "mì",
            "miến lươn", "miến gà", "miến",
            "salad", "đồ ăn nhanh", "burger", "pizza", "gà rán",
            "bánh cuốn", "cháo", "hải sản", "cá", "thịt bò", "thịt gà",
            "xôi", "nem nướng", "bánh giò",
        ]

    def extract_budget(self, text: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Trích xuất ngân sách tối đa từ văn bản tiếng Việt.
        Trả về: Tuple[Optional[int], Optional[str]]: (Số tiền VND, chuỗi khớp được)
        """
        if not text:
            return None, None

        norm_text = unicodedata.normalize("NFC", text).lower()

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

        # 4. Bắt dạng ngữ cảnh: 'dưới 40', 'tam 30' (nếu số trong khoảng 15-100, không theo sau bởi h/giờ/người)
        implicit_pattern = r"(?:dưới|duoi|tầm|tam|khoảng|khoang|dưới tầm|tối đa|toi da|chỉ có|chi co)\s*(\d{2})\b(?!\s*(?:h|g|giờ|gio|phút|phut|người|nguoi|đứa|dua|bạn|ban|:))"
        implicit_match = re.search(implicit_pattern, norm_text)
        if implicit_match:
            val = int(implicit_match.group(1))
            if 15 <= val <= 100:
                return val * 1000, implicit_match.group(0).strip()

        return None, None

    def extract_spicy(self, text: str) -> Tuple[bool, Optional[str]]:
        """Trích xuất yêu cầu không ăn cay."""
        if not text:
            return False, None

        norm_text = unicodedata.normalize("NFC", text).lower()
        no_spicy_pattern = r"\b(?:không|khong|ko|k|đừng|dung|chớ)\s*(?:ăn\s*)?(?:được\s*)?(?:cay|quá cay|nhiều ớt|ớt)\b|\b(?:bỏ ớt|không ớt|ko ớt|k ớt)\b"
        match = re.search(no_spicy_pattern, norm_text)
        if match:
            return True, match.group(0).strip()

        return False, None

    def extract_exclusions(self, text: str) -> Tuple[List[str], List[str]]:
        """Trích xuất danh sách các món hoặc loại đồ ăn cần loại trừ."""
        if not text:
            return [], []

        norm_text = unicodedata.normalize("NFC", text).lower()
        exclusions = []
        matched_phrases = []

        exclusion_patterns = [
            r"\b(?:không ăn|khong an|ko ăn|ko an|k ăn|k an|không muốn ăn|ko muon an)\s+([a-zA-Zà-ỹÀ-Ỹ\s]+?)(?:nhé|nha|ạ|ạ\?|\.|\,|$|\bnhưng\b)",
            r"\b(?:trừ|tru|ngoại trừ|ngoai tru)\s+([a-zA-Zà-ỹÀ-Ỹ\s]+?)(?:ra|nhé|nha|ạ|ạ\?|\.|\,|$|\bnhưng\b)",
            r"\b(?:ko|k|không|khong)\s+(?:thèm|thích)\s+([a-zA-Zà-ỹÀ-Ỹ\s]+?)(?:nhé|nha|ạ|ạ\?|\.|\,|$)",
        ]

        for pattern in exclusion_patterns:
            for match in re.finditer(pattern, norm_text):
                phrase = match.group(1).strip()
                if any(w in phrase for w in ["cay", "quá cay", "ớt", "ot"]):
                    continue
                # Không coi 'ăn một mình' / 'chay' là loại trừ món
                if phrase in ("một mình", "mot minh", "chay", "đồ chay", "món chay"):
                    continue

                matched_phrases.append(match.group(0).strip())
                found_specific = False
                for kw in self.known_food_keywords:
                    if kw in phrase:
                        exclusions.append(kw)
                        found_specific = True

                if not found_specific and 1 <= len(phrase.split()) <= 3:
                    exclusions.append(phrase)

        unique_exclusions = list(dict.fromkeys(exclusions))
        return unique_exclusions, matched_phrases

    def extract_vegetarian(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Nhận diện yêu cầu ăn chay trong câu nói.
        Không nhận nhầm 'chạy' (chạy bộ, chạy deadline) và phủ định ('không ăn chay', 'không phải chay').
        """
        if not text:
            return False, None

        norm = unicodedata.normalize("NFC", text).lower()
        has_accent = any(ord(ch) > 127 for ch in norm)

        candidates = list(self._VEG_PREFIXED.finditer(norm))
        if has_accent:
            for m in self._VEG_STANDALONE.finditer(norm):
                inside_prefixed = any(p.start() <= m.start() < p.end() for p in candidates)
                if not inside_prefixed and not self._VEG_RUNNING_AFTER.match(norm, m.end()):
                    candidates.append(m)

        for m in sorted(candidates, key=lambda x: x.start()):
            before = norm[: m.start()]
            if self._VEG_NEGATION_BEFORE.search(before):
                continue
            return True, m.group(0).strip()
        return False, None

    def extract_num_people(self, text: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
        """
        Trích xuất số người đi ăn từ câu nói (chữ số hoặc chữ tiếng Việt 1–10, có dấu và không dấu).
        - Hỗ trợ: 'đi 4 người', '4 đứa', 'cả nhóm 6 bạn', 'một mình' = 1, 'hai đứa' = 2...
        - Nếu số người nằm ngoài khoảng 1–10 (ví dụ 'đi 15 người'): bỏ qua (trả về None) và trả kèm ghi chú.

        Trả về:
            Tuple[Optional[int], Optional[str], Optional[str]]:
                (so_nguoi hợp lệ 1..10 hoặc None, chuỗi khớp được, ghi chú nếu ngoài khoảng)
        """
        if not text:
            return None, None, None

        norm = unicodedata.normalize("NFC", text).lower()

        # 1. Kiểm tra 'một mình' / '1 mình'
        m_alone = self._ALONE_PATTERN.search(norm)
        if m_alone:
            return 1, m_alone.group(0).strip(), None

        # 2. Kiểm tra dạng chữ số + danh từ chỉ người ('đi 4 người', '4 đứa', 'cả nhóm 6 bạn', '15 người')
        m_digit = self._DIGIT_PEOPLE_PATTERN.search(norm)
        if m_digit:
            n = int(m_digit.group(1))
            matched = m_digit.group(0).strip()
            if PEOPLE_MIN_EXTRACTOR <= n <= PEOPLE_MAX_EXTRACTOR:
                return n, matched, None
            note = (
                f"Phát hiện '{matched}' ({n} người) ngoài khoảng hỗ trợ "
                f"{PEOPLE_MIN_EXTRACTOR}–{PEOPLE_MAX_EXTRACTOR} người nên đã bỏ qua."
            )
            return None, matched, note

        # 3. Kiểm tra dạng chữ số tiếng Việt ('hai đứa', 'bốn người', 'sáu bạn', 'mười người')
        for m_word in self._WORD_PEOPLE_PATTERN.finditer(norm):
            word = m_word.group(1)
            noun = m_word.group(2)
            # Tránh nhận nhầm 'nam bạn' / 'bạn nam' hoặc 'phía nam' nếu gõ không dấu 'nam'
            if word == "nam" and noun not in ("người", "nguoi", "đứa", "dua", "anh em"):
                continue
            n = self._VI_NUMBER_WORDS.get(word)
            if n is not None:
                matched = m_word.group(0).strip()
                if PEOPLE_MIN_EXTRACTOR <= n <= PEOPLE_MAX_EXTRACTOR:
                    return n, matched, None

        return None, None, None

    def extract_prefer_quiet(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Nhận diện yêu cầu ưu tiên quán vắng / đỡ đông / yên tĩnh (uu_tien_vang).
        - Nhận: 'chỗ nào vắng', 'đỡ đông', 'không đông', 'yên tĩnh', 'quán vắng', 'ít đông', 'sợ đông'...
        - Không nhận nhầm: 'đông vui', 'càng đông càng vui', 'thích đông'.

        Trả về:
            Tuple[bool, Optional[str]]: (True nếu ưu tiên vắng, chuỗi khớp được)
        """
        if not text:
            return False, None

        norm = unicodedata.normalize("NFC", text).lower()

        # Loại bỏ các cụm 'đông vui' trước khi kiểm tra để không bao giờ nhận nhầm
        cleaned = self._CROWD_POSITIVE_PATTERN.sub(" ", norm)
        match = self._PREFER_QUIET_PATTERN.search(cleaned)
        if match:
            return True, match.group(0).strip()
        return False, None

    def extract_time_slot(
        self,
        text: str,
        gio_cao_diem: Sequence[str] = ("11:30", "13:00"),
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Nhận diện thời gian đi ăn trong câu nói ('12h', '12h30', '11 giờ rưỡi', '13h15', '12:30'...)
        và suy ra khung_gio ∈ {'cao_diem', 'binh_thuong', 'vang'}.
        Không nhận nhầm số tiền ('40k') hay số người ('4 người').

        Trả về:
            Tuple[Optional[str], Optional[str], Optional[str]]:
                (khung_gio, chuỗi khớp được trong câu, chuỗi giờ chuẩn hóa 'HH:MM' nếu có)
        """
        if not text:
            return None, None, None

        norm = unicodedata.normalize("NFC", text).lower()

        # 1. Dạng giờ:phút hoặc giờ + h/g/giờ ('12h', '12h30', '11 giờ rưỡi', '13h15')
        for m in self._TIME_HM_PATTERN.finditer(norm):
            hour = int(m.group(1))
            is_half = m.group(2) is not None
            min_str = m.group(3)
            period = m.group(4) or ""
            minute = 30 if is_half else (int(min_str) if min_str is not None else 0)

            if period in ("trưa", "trua", "chiều", "chieu") and 1 <= hour <= 5:
                hour += 12
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                continue

            hhmm = f"{hour:02d}:{minute:02d}"
            slot = infer_time_slot(hhmm, gio_cao_diem=gio_cao_diem)
            return slot, m.group(0).strip(), hhmm

        # 2. Dạng HH:MM ('12:30', '11:15')
        for m in self._TIME_COLON_PATTERN.finditer(norm):
            hour = int(m.group(1))
            minute = int(m.group(2))
            period = m.group(3) or ""
            if period in ("trưa", "trua", "chiều", "chieu") and 1 <= hour <= 5:
                hour += 12
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                continue

            hhmm = f"{hour:02d}:{minute:02d}"
            slot = infer_time_slot(hhmm, gio_cao_diem=gio_cao_diem)
            return slot, m.group(0).strip(), hhmm

        # 3. Cụm từ chỉ khung giờ trực tiếp ('giờ cao điểm', 'ngoài giờ cao điểm')
        m_off = self._TIME_DIRECT_OFFPEAK.search(norm)
        if m_off:
            return "vang", m_off.group(0).strip(), None
        m_peak = self._TIME_DIRECT_PEAK.search(norm)
        if m_peak:
            return "cao_diem", m_peak.group(0).strip(), None

        return None, None, None

    def extract(
        self,
        text: str,
        gio_cao_diem: Sequence[str] = ("11:30", "13:00"),
    ) -> Dict[str, Any]:
        """
        Hàm chính trích xuất toàn bộ ràng buộc từ câu người dùng.

        Trả về:
            Dict[str, Any]:
                - 'max_price': Ngân sách tối đa (int VND hoặc None)
                - 'exclude_spicy': Có loại trừ đồ cay hay không (bool)
                - 'vegetarian_only': Có yêu cầu chỉ ăn món chay hay không (bool)
                - 'exclude_types': Danh sách món/loại đồ ăn loại trừ (List[str])
                - 'num_people': Số người đi ăn trích từ câu nói (int 1..10 hoặc None)
                - 'prefer_quiet' / 'uu_tien_vang': Có ưu tiên quán vắng hay không (bool)
                - 'time_slot' / 'khung_gio': Khung giờ suy từ câu nói ('cao_diem', 'binh_thuong', 'vang' hoặc None)
                - 'parsed_time': Giờ 'HH:MM' trích từ câu nói (hoặc None)
                - 'intermediate': Chi tiết các chuỗi khớp được và ghi chú để phục vụ hiển thị GUI.
        """
        max_price, price_match = self.extract_budget(text)
        exclude_spicy, spicy_match = self.extract_spicy(text)
        vegetarian_only, veg_match = self.extract_vegetarian(text)
        exclude_types, exclusion_matches = self.extract_exclusions(text)
        num_people, people_match, people_note = self.extract_num_people(text)
        prefer_quiet, quiet_match = self.extract_prefer_quiet(text)
        time_slot, time_match, parsed_time = self.extract_time_slot(text, gio_cao_diem=gio_cao_diem)

        return {
            "max_price": max_price,
            "exclude_spicy": exclude_spicy,
            "vegetarian_only": vegetarian_only,
            "exclude_types": exclude_types,
            "num_people": num_people,
            "so_nguoi": num_people,
            "prefer_quiet": prefer_quiet,
            "uu_tien_vang": prefer_quiet,
            "time_slot": time_slot,
            "khung_gio": time_slot,
            "parsed_time": parsed_time,
            "intermediate": {
                "matched_price_text": price_match,
                "matched_spicy_text": spicy_match,
                "matched_vegetarian_text": veg_match,
                "matched_exclusion_texts": exclusion_matches,
                "matched_people_text": people_match,
                "num_people_note": people_note,
                "matched_quiet_text": quiet_match,
                "matched_time_text": time_match,
                "parsed_time": parsed_time,
            },
        }


def extract_constraints(text: str) -> Dict[str, Any]:
    """Hàm tiện ích bọc nhanh lớp ConstraintExtractor."""
    extractor = ConstraintExtractor()
    return extractor.extract(text)
