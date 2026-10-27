"""
Module Chatbot trung tâm (Lunch Chatbot Pipeline).
Tích hợp:
1. Tiền xử lý văn bản (preprocess)
2. Phân loại ý định người dùng (NeedClassifier — 8 nhãn không đổi)
3. Trích xuất ràng buộc ngữ nghĩa (ConstraintExtractor: ngân sách, cay, chay, loại trừ, số người, ưu tiên vắng, khung giờ)
4. Hệ thống gợi ý món ăn đa tiêu chí 7 tiêu chí (DishRecommender: tích hợp độ đông & chỗ ngồi theo nhóm)
5. Sinh phản hồi tự nhiên tiếng Việt kèm giải thích lý do cụ thể và trả về toàn bộ kết quả trung gian.
"""

import os
import sys
import random
from typing import Dict, Any, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from src.config import (
        DEFAULT_MODEL_PATH,
        DEFAULT_DISHES_PATH,
        DEFAULT_PROFILES_PATH,
        DEFAULT_CROWD_CONFIG_PATH,
        DEFAULT_INTENTS_PATH,
        TIME_SLOT_SHORT_VI,
        resolve_project_path,
    )
    from src.preprocess import preprocess
    from src.classifier import NeedClassifier
    from src.extractor import ConstraintExtractor
    from src.recommender import DishRecommender, merge_constraints
except ImportError:
    from config import (
        DEFAULT_MODEL_PATH,
        DEFAULT_DISHES_PATH,
        DEFAULT_PROFILES_PATH,
        DEFAULT_CROWD_CONFIG_PATH,
        DEFAULT_INTENTS_PATH,
        TIME_SLOT_SHORT_VI,
        resolve_project_path,
    )
    from preprocess import preprocess
    from classifier import NeedClassifier
    from extractor import ConstraintExtractor
    from recommender import DishRecommender, merge_constraints


class Chatbot:
    """Lớp điều phối trung tâm tiếp nhận yêu cầu và tạo phản hồi thông minh."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        dishes_path: str = DEFAULT_DISHES_PATH,
        profiles_path: str = DEFAULT_PROFILES_PATH,
        crowd_config_path: str = DEFAULT_CROWD_CONFIG_PATH,
        confidence_threshold: float = 0.30,
        default_top_k: int = 3,
    ):
        self.model_path = resolve_project_path(model_path)
        self.dishes_path = resolve_project_path(dishes_path)
        self.profiles_path = resolve_project_path(profiles_path)
        self.crowd_config_path = resolve_project_path(crowd_config_path)
        self.confidence_threshold = confidence_threshold
        self.default_top_k = default_top_k

        # Khởi tạo các module thành phần
        self.extractor = ConstraintExtractor()
        self.recommender = DishRecommender(
            dishes_path=self.dishes_path,
            profiles_path=self.profiles_path,
            crowd_config_path=self.crowd_config_path,
        )
        self.classifier = NeedClassifier()

        self._init_classifier()

    def _init_classifier(self) -> None:
        """Tải mô hình đã lưu hoặc tự động huấn luyện nếu chưa có."""
        if os.path.exists(self.model_path):
            try:
                self.classifier.load(self.model_path)
                return
            except Exception:
                pass

        intents_csv = resolve_project_path(DEFAULT_INTENTS_PATH)
        if os.path.exists(intents_csv):
            self.classifier.train(intents_csv)
            self.classifier.save(self.model_path)

    def _generate_greeting_reply(self) -> str:
        """Sinh câu trả lời chào hỏi xã giao thân thiện."""
        greetings = [
            "Chào bạn sinh viên NEU! Trưa nay bạn muốn ăn gì nè? Cứ nói ngân sách, số người đi ăn hoặc khẩu vị (tiết kiệm, healthy, ăn no, đổi gió...), mình gợi ý ngay nhé!",
            "Xin chào bạn! Bụng đã réo chưa? Hãy cho mình biết bạn muốn ăn món gì, đi mấy người, ngân sách bao nhiêu quanh Kinh tế Quốc dân để mình tư vấn nhé!",
            "Helo bạn! Hôm nay ăn trưa ở ngõ Tự Do, Trần Đại Nghĩa hay KTX NEU? Chia sẻ nhu cầu để mình tìm quán ngon và đủ chỗ ngồi cho bạn nha!",
        ]
        return random.choice(greetings)

    def _generate_farewell_reply(self) -> str:
        """Sinh câu trả lời cảm ơn / tạm biệt."""
        farewells = [
            "Không có gì đâu nè! Chúc bạn có một bữa trưa thật ngon miệng và học tập hiệu quả tại NEU nhé. Cần gì cứ hỏi mình!",
            "Rất vui được hỗ trợ bạn! Chúc bạn nạp đầy năng lượng cho buổi học chiều. Hẹn gặp lại vào bữa trưa mai nha!",
            "Cảm ơn bạn! Đi ăn ngon miệng nhé, cần đổi gió quán ăn quanh Bách - Kinh - Xây thì lại ghé tìm mình nha!",
        ]
        return random.choice(farewells)

    @staticmethod
    def describe_constraints(constraints: Dict[str, Any]) -> List[str]:
        """
        Mô tả ngắn gọn các ràng buộc đang áp dụng (kèm nguồn), ví dụ:
        ['ngân sách tối đa 40,000đ/người (từ câu nói)', 'đi 4 người (từ câu nói)', 'không ăn cay'].
        """
        sources = constraints.get("sources", {}) or {}
        parts: List[str] = []
        if constraints.get("max_price") is not None:
            src = f" ({sources['budget']})" if sources.get("budget") else ""
            parts.append(f"ngân sách tối đa {constraints['max_price']:,.0f}đ/người{src}")
        if constraints.get("num_people", 1) > 1:
            src = f" ({sources['num_people']})" if sources.get("num_people") == "từ câu nói" else ""
            parts.append(f"đi {constraints['num_people']} người{src}")
        if constraints.get("prefer_quiet"):
            parts.append("ưu tiên quán vắng/đỡ đông")
        if constraints.get("require_enough_seats"):
            parts.append("chỉ quán đủ chỗ cho cả nhóm")
        if constraints.get("exclude_spicy"):
            src = f" ({sources['spicy']})" if sources.get("spicy") else ""
            parts.append(f"không ăn cay{src}")
        if constraints.get("vegetarian_only"):
            src = f" ({sources['vegetarian']})" if sources.get("vegetarian") else ""
            parts.append(f"chỉ món chay{src}")
        if constraints.get("exclude_types"):
            parts.append(f"không chọn các món liên quan đến '{', '.join(constraints['exclude_types'])}'")
        return parts

    def _generate_explanation(
        self,
        intent: str,
        constraints: Dict[str, Any],
        is_fallback: bool,
    ) -> str:
        """Tạo chuỗi giải thích lý do gợi ý cho người dùng, nêu rõ ảnh hưởng của nhóm và độ đông."""
        intent_explanations = {
            "tiet_kiem": "ưu tiên tối đa tiêu chí giá hạt dẻ, tiết kiệm chi phí cho sinh viên",
            "healthy": "ưu tiên độ dinh dưỡng cao, ít dầu mỡ, nhiều rau củ thanh đạm",
            "nhanh_gon": "ưu tiên các quán gần cổng NEU và thời gian chờ phục vụ siêu nhanh",
            "no_bung": "ưu tiên các món khẩu phần lớn, chắc dạ để no lâu",
            "ngon_doi_gio": "ưu tiên món có điểm đánh giá hương vị cao, lạ miệng để bạn đổi khẩu vị",
            "can_bang": "cân đối hài hòa giữa mức giá sinh viên, dinh dưỡng, thời gian chờ và khoảng cách",
        }

        reasons: List[str] = []
        if is_fallback:
            reasons.append("Hệ thống nhận thấy câu hỏi đa dạng nên chọn chế độ 'Cân bằng'")
        else:
            exp = intent_explanations.get(intent, "cân bằng các tiêu chí")
            reasons.append(f"Mình {exp}")

        c_parts = self.describe_constraints(constraints)
        if c_parts:
            reasons.append(f"kết hợp lọc đúng theo yêu cầu của bạn: {', '.join(c_parts)}")

        base_sentence = " ".join(reasons) + "."

        # Bổ sung câu giải thích về số người và độ đông khung giờ
        extra_notes: List[str] = []
        n = int(constraints.get("num_people") or 1)
        slot = str(constraints.get("time_slot") or "cao_diem")
        slot_vi = TIME_SLOT_SHORT_VI.get(slot, slot).lower()
        if n > 1:
            extra_notes.append(
                f"Với nhóm {n} người vào giờ {slot_vi}, mình ưu tiên quán rộng chỗ còn đủ ghế trống và thời gian chờ hợp lý."
            )
        if constraints.get("prefer_quiet"):
            extra_notes.append(
                "Vì bạn muốn tìm chỗ vắng/đỡ đông nên hệ số phạt thời gian chờ theo độ đông (a) đã được nhân đôi để ưu tiên quán thoáng."
            )

        if extra_notes:
            return base_sentence + " " + " ".join(extra_notes)
        return base_sentence

    @staticmethod
    def _format_dish_lines(
        items: List[Dict[str, Any]],
        num_people: int,
        max_price: Optional[float] = None,
    ) -> List[str]:
        """
        Định dạng danh sách món (markdown nhẹ, GUI đổi **...** thành chữ đậm).
        Mỗi món hiển thị:
        - Giá một suất, khoảng cách, "Chờ ước tính ~x phút"
        - Nhãn độ đông (Vắng/Vừa/Đông/Rất đông), số chỗ trống ước tính, và cảnh báo nếu cho_trong < so_nguoi
        - Ước tính chi phí cả nhóm: giá × số người
        - Điểm tổng hợp, nhãn cay, nhãn Chay/Mặn và mô tả.
        """
        lines: List[str] = []
        for idx, item in enumerate(items, 1):
            d = item["dish"]
            cay_icon = "🌶️ Có cay" if d["cay"] == 1 else "🟢 Không cay"
            chay_icon = "🌱 Chay" if int(d.get("chay", 0)) == 1 else "🍖 Mặn"
            over = ""
            if max_price is not None and d["gia_vnd"] > max_price:
                over = f" (vượt ngân sách {d['gia_vnd'] - max_price:,.0f}đ/người)"

            tg_cho = float(item.get("thoi_gian_cho", d.get("thoi_gian_cho", d["thoi_gian_phuc_vu_phut"])))
            do_dong_hd = float(item.get("do_dong_hieu_dung", d.get("do_dong_hieu_dung", 3.0)))
            crowd_lbl = item.get("crowd_label") or d.get("crowd_label") or "Vừa"
            cho_trong = float(item.get("cho_trong", d.get("cho_trong", 30.0)))
            cho_trong_int = int(item.get("cho_trong_int", round(cho_trong)))
            so_cho_ngoi = int(d.get("so_cho_ngoi", 30))
            warning = item.get("seat_warning") or d.get("seat_warning")

            warn_str = f" | ⚠️ **{warning}**" if (warning or cho_trong < num_people) else ""
            lines.append(
                f"{idx}. 🍽️ **{d['ten_mon']}** ({d['loai'].capitalize()})\n"
                f"   - 💰 Giá: **{d['gia_vnd']:,.0f}đ**/suất{over} | 📍 Cách NEU: **{d['khoang_cach_m']:.0f}m** | "
                f"⏱️ Chờ ước tính **~{tg_cho:.1f} phút** (phục vụ chuẩn {d['thoi_gian_phuc_vu_phut']:.0f}p)\n"
                f"   - 🚦 Độ đông: **{crowd_lbl}** ({do_dong_hd:.1f}/5) | "
                f"🪑 Chỗ trống: **~{cho_trong_int}/{so_cho_ngoi} chỗ**{warn_str}\n"
                f"   - 👥 Ước tính cả nhóm: {d['gia_vnd']:,.0f}đ × {num_people} = **{item.get('group_price', d['gia_vnd'] * num_people):,.0f}đ**\n"
                f"   - ⭐ Điểm tổng hợp: **{item['score']:.2f}** | Vị: {cay_icon} | {chay_icon}\n"
                f"   - 📝 {d['mo_ta']}\n"
            )
        return lines

    @staticmethod
    def _group_budget_line(constraints: Dict[str, Any]) -> Optional[str]:
        """Dòng tóm tắt ngân sách nhóm và khung giờ."""
        n = constraints.get("num_people", 1)
        mp = constraints.get("max_price")
        slot = constraints.get("time_slot", "cao_diem")
        slot_vi = TIME_SLOT_SHORT_VI.get(slot, slot)
        if mp is not None:
            return f"👥 Ngân sách nhóm ({n} người, khung giờ {slot_vi}): {mp:,.0f}đ × {n} người = **{mp * n:,.0f}đ**\n"
        if n > 1:
            return f"👥 Nhóm **{n} người** đi ăn (khung giờ: **{slot_vi}**).\n"
        return None

    def _build_no_match_reply(
        self,
        effective_intent: str,
        constraints: Dict[str, Any],
        no_match: Dict[str, Any],
    ) -> str:
        """Câu trả lời khi không món nào thỏa mọi điều kiện: nêu điều kiện quá chặt + 3 món gần điều kiện nhất."""
        n = constraints.get("num_people", 1)
        cond = ", ".join(self.describe_constraints(constraints)) or "các điều kiện lọc hiện tại"
        lines = [
            f"Mình hiểu bạn đang tìm món theo phong cách '{effective_intent}'. "
            f"Tuy nhiên hiện không có món nào thỏa đồng thời tất cả điều kiện: {cond}.\n"
        ]

        if no_match["culprits"]:
            lines.append("Điều kiện đang quá chặt:")
            for c in no_match["culprits"]:
                if c["key"] == "budget":
                    lines.append(
                        f"   • Ngân sách: nếu bỏ giới hạn giá sẽ có {c['count_if_relaxed']} món thỏa các điều kiện còn lại, "
                        f"rẻ nhất là {c['cheapest_price']:,.0f}đ/người."
                    )
                elif c["key"] == "seats":
                    lines.append(
                        f"   • Chỗ ngồi nhóm ({n} người): nếu tắt lọc 'Chỉ quán đủ chỗ cho cả nhóm' sẽ có "
                        f"{c['count_if_relaxed']} món (quán rộng nhất còn ~{c.get('max_available_seats', 0):.0f} chỗ trống)."
                    )
                else:
                    lines.append(
                        f"   • Điều kiện '{c['label']}': nếu bỏ riêng điều kiện này sẽ có {c['count_if_relaxed']} món."
                    )
        else:
            lines.append(
                "Không có điều kiện đơn lẻ nào là nguyên nhân; việc kết hợp nhiều điều kiện cùng lúc khiến không còn món nào."
            )

        kept = ", ".join(no_match["kept_labels"]) or "không có"
        relaxed = ", ".join(no_match["relaxed_labels"]) or "không có"
        lines.append(
            f"\nMình gợi ý {len(no_match['near_items'])} món gần điều kiện nhất "
            f"(giữ nguyên: {kept}; đã nới: {relaxed}):\n"
        )
        max_price = constraints.get("max_price") if "budget" in no_match["relaxed"] else None
        lines.extend(self._format_dish_lines(no_match["near_items"], n, max_price))
        lines.append("Bạn thử nới lỏng điều kiện đã nêu ở trên để có thêm lựa chọn nhé!")
        return "\n".join(lines)

    def respond(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        remove_stopwords: Optional[bool] = None,
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
        Xử lý toàn diện yêu cầu của người dùng qua toàn bộ pipeline:
        1. Tiền xử lý (preprocess)
        2. Phân loại ý định (classifier - 8 nhãn)
        3. Trích xuất ràng buộc (extractor: giá, cay, chay, loại trừ, số người, ưu tiên vắng, khung giờ)
        4. Gợi ý món ăn 7 tiêu chí (recommender: tích hợp độ đông & chỗ ngồi)
        5. Sinh câu trả lời tự nhiên có giải thích lý do
        """
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        k = top_k if top_k is not None else self.default_top_k
        sw = self.classifier.remove_stopwords if remove_stopwords is None else remove_stopwords

        # Kiểm tra tham số lọc từ UI ngay từ đầu (ném ValueError nếu ngoài khoảng)
        merge_constraints(
            None,
            max_price_per_person=max_price_per_person,
            num_people=num_people,
            vegetarian_only=vegetarian_only,
            exclude_spicy=exclude_spicy,
            time_slot=time_slot,
            require_enough_seats=require_enough_seats,
            current_time=current_time,
        )

        # 1. Tiền xử lý
        prep_result = preprocess(text, remove_stopwords=sw)

        # 2. Phân loại ý định
        clf_result = self.classifier.predict(text, remove_stopwords=sw)
        predicted_label = clf_result["label"]
        confidence = clf_result["confidence"]
        model_params = {
            "model_type": self.classifier.model_type,
            "alpha": self.classifier.alpha,
            "ngram_range": tuple(self.classifier.ngram_range),
            "remove_stopwords": sw,
        }

        # 3. Trích xuất ràng buộc từ câu nói, rồi gộp với bộ lọc nhập tay (câu nói được ưu tiên)
        gio_cao_diem = self.recommender.crowd_config.get("gio_cao_diem", ["11:30", "13:00"])
        text_constraints = self.extractor.extract(text, gio_cao_diem=gio_cao_diem)
        constraint_result = merge_constraints(
            text_constraints,
            max_price_per_person=max_price_per_person,
            num_people=num_people,
            vegetarian_only=vegetarian_only,
            exclude_spicy=exclude_spicy,
            time_slot=time_slot,
            require_enough_seats=require_enough_seats,
            current_time=current_time,
            gio_cao_diem=gio_cao_diem,
        )
        constraint_result["intermediate"] = text_constraints.get("intermediate", {})
        constraint_result["from_text"] = {
            "max_price": text_constraints.get("max_price"),
            "exclude_spicy": text_constraints.get("exclude_spicy", False),
            "vegetarian_only": text_constraints.get("vegetarian_only", False),
            "exclude_types": text_constraints.get("exclude_types", []),
            "num_people": text_constraints.get("num_people"),
            "prefer_quiet": text_constraints.get("prefer_quiet", False),
            "time_slot": text_constraints.get("time_slot"),
            "parsed_time": text_constraints.get("parsed_time"),
        }

        # Xử lý các nhãn xã giao
        if predicted_label == "chao_hoi":
            reply_text = self._generate_greeting_reply()
            return {
                "text": text,
                "preprocessing": prep_result,
                "classification": clf_result,
                "model_params": model_params,
                "effective_intent": "chao_hoi",
                "is_fallback": False,
                "fallback_reason": None,
                "constraints": constraint_result,
                "recommendation": None,
                "explanation": "Người dùng gửi lời chào hỏi thân mật.",
                "reply": reply_text,
            }

        if predicted_label == "cam_on_tam_biet":
            reply_text = self._generate_farewell_reply()
            return {
                "text": text,
                "preprocessing": prep_result,
                "classification": clf_result,
                "model_params": model_params,
                "effective_intent": "cam_on_tam_biet",
                "is_fallback": False,
                "fallback_reason": None,
                "constraints": constraint_result,
                "recommendation": None,
                "explanation": "Người dùng nói lời cảm ơn hoặc chào tạm biệt.",
                "reply": reply_text,
            }

        # 4. Kiểm tra độ tin cậy để quyết định fallback
        is_fallback = False
        fallback_reason = None
        if confidence < conf_thresh:
            effective_intent = "can_bang"
            is_fallback = True
            fallback_reason = (
                f"Độ tin cậy của nhãn '{predicted_label}' đạt {confidence:.2f} (< {conf_thresh:.2f}). "
                f"Chuyển sang profile an toàn 'can_bang'."
            )
        else:
            effective_intent = predicted_label

        # 5. Gợi ý món ăn
        rec_result = self.recommender.recommend(
            intent=effective_intent,
            constraints=text_constraints,
            custom_weights=custom_weights,
            top_k=k,
            max_price_per_person=max_price_per_person,
            num_people=num_people,
            vegetarian_only=vegetarian_only,
            exclude_spicy=exclude_spicy,
            time_slot=time_slot,
            require_enough_seats=require_enough_seats,
            crowd_params=crowd_params,
            current_time=current_time,
        )

        # 6. Sinh giải thích và câu trả lời tự nhiên
        explanation = self._generate_explanation(effective_intent, constraint_result, is_fallback)

        top_dishes = rec_result["top_k"]
        n = constraint_result["num_people"]
        if not top_dishes:
            if rec_result.get("no_match"):
                reply_text = self._build_no_match_reply(effective_intent, constraint_result, rec_result["no_match"])
            else:
                reply_text = (
                    f"Mình hiểu bạn đang tìm món theo phong cách '{effective_intent}'. "
                    "Tuy nhiên hiện chưa có món ăn nào trong dữ liệu. "
                    "Bạn hãy thêm món ở tab Quản lý dữ liệu nhé!"
                )
        else:
            lines = [f"Dạ, {explanation} Dưới đây là {len(top_dishes)} gợi ý tốt nhất dành cho bạn:\n"]
            group_line = self._group_budget_line(constraint_result)
            if group_line:
                lines.append(group_line)
            lines.extend(self._format_dish_lines(top_dishes, n))
            lines.append("Bạn thấy món nào hợp gu nhất? Chúc bạn có bữa trưa ngon miệng nhé!")
            reply_text = "\n".join(lines)

        return {
            "text": text,
            "preprocessing": prep_result,
            "classification": clf_result,
            "model_params": model_params,
            "effective_intent": effective_intent,
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "constraints": constraint_result,
            "recommendation": rec_result,
            "explanation": explanation,
            "reply": reply_text,
        }


# Tương thích tên lớp theo yêu cầu
LunchChatbot = Chatbot
