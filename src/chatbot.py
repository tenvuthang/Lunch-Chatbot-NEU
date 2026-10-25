"""
Module Chatbot trung tâm (Lunch Chatbot Pipeline).
Tích hợp:
1. Tiền xử lý văn bản (preprocess)
2. Phân loại ý định người dùng (NeedClassifier)
3. Trích xuất ràng buộc ngữ nghĩa (ConstraintExtractor)
4. Hệ thống gợi ý món ăn đa tiêu chí (DishRecommender)
5. Sinh phản hồi tự nhiên tiếng Việt kèm giải thích lý do cụ thể và trả về toàn bộ kết quả trung gian.
"""

import os
import sys
import random
from typing import Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from src.preprocess import preprocess
    from src.classifier import NeedClassifier
    from src.extractor import ConstraintExtractor
    from src.recommender import DishRecommender
except ImportError:
    from preprocess import preprocess
    from classifier import NeedClassifier
    from extractor import ConstraintExtractor
    from recommender import DishRecommender


class Chatbot:
    """Lớp điều phối trung tâm tiếp nhận yêu cầu và tạo phản hồi thông minh."""

    def __init__(
        self,
        model_path: str = "outputs/classifier_model.joblib",
        dishes_path: str = "data/dishes.csv",
        profiles_path: str = "config/profiles.json",
        confidence_threshold: float = 0.30,
        default_top_k: int = 3
    ):
        self.model_path = model_path
        self.dishes_path = dishes_path
        self.profiles_path = profiles_path
        self.confidence_threshold = confidence_threshold
        self.default_top_k = default_top_k

        # Khởi tạo các module thành phần
        self.extractor = ConstraintExtractor()
        self.recommender = DishRecommender(dishes_path=dishes_path, profiles_path=profiles_path)
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

        # Huấn luyện mới từ data/intents.csv
        intents_csv = "data/intents.csv"
        if os.path.exists(intents_csv):
            self.classifier.train(intents_csv)
            self.classifier.save(self.model_path)

    def _generate_greeting_reply(self) -> str:
        """Sinh câu trả lời chào hỏi xã giao thân thiện."""
        greetings = [
            "Chào bạn sinh viên NEU! Trưa nay bạn muốn ăn gì nè? Cứ nói ngân sách hoặc khẩu vị (tiết kiệm, healthy, ăn no, đổi gió...), mình gợi ý ngay nhé!",
            "Xin chào bạn! Bụng đã réo chưa? Hãy cho mình biết bạn muốn ăn món gì, ngân sách bao nhiêu quanh Kinh tế Quốc dân để mình tư vấn nhé!",
            "Helo bạn! Hôm nay ăn trưa ở ngõ Tự Do, Trần Đại Nghĩa hay KTX NEU? Chia sẻ nhu cầu để mình tìm quán ngon nhất cho bạn nha!"
        ]
        return random.choice(greetings)

    def _generate_farewell_reply(self) -> str:
        """Sinh câu trả lời cảm ơn / tạm biệt."""
        farewells = [
            "Không có gì đâu nè! Chúc bạn có một bữa trưa thật ngon miệng và học tập hiệu quả tại NEU nhé. Cần gì cứ hỏi mình!",
            "Rất vui được hỗ trợ bạn! Chúc bạn nạp đầy năng lượng cho buổi học chiều. Hẹn gặp lại vào bữa trưa mai nha!",
            "Cảm ơn bạn! Đi ăn ngon miệng nhé, cần đổi gió quán ăn quanh Bách - Kinh - Xây thì lại ghé tìm mình nha!"
        ]
        return random.choice(farewells)

    def _generate_explanation(
        self,
        intent: str,
        constraints: Dict[str, Any],
        is_fallback: bool
    ) -> str:
        """Tạo chuỗi giải thích lý do gợi ý cho người dùng."""
        intent_explanations = {
            "tiet_kiem": "ưu tiên tối đa tiêu chí giá hạt dẻ, tiết kiệm chi phí cho sinh viên",
            "healthy": "ưu tiên độ dinh dưỡng cao, ít dầu mỡ, nhiều rau củ thanh đạm",
            "nhanh_gon": "ưu tiên các quán gần cổng NEU và thời gian phục vụ siêu nhanh",
            "no_bung": "ưu tiên các món khẩu phần lớn, chắc dạ để no lâu",
            "ngon_doi_gio": "ưu tiên món có điểm đánh giá hương vị cao, lạ miệng để bạn đổi khẩu vị",
            "can_bang": "cân đối hài hòa giữa mức giá sinh viên, dinh dưỡng và khoảng cách đi lại"
        }

        reasons = []
        if is_fallback:
            reasons.append("Hệ thống nhận thấy câu hỏi đa dạng nên chọn chế độ 'Cân bằng'")
        else:
            exp = intent_explanations.get(intent, "cân bằng các tiêu chí")
            reasons.append(f"Mình {exp}")

        # Ràng buộc
        c_parts = []
        if constraints.get("max_price") is not None:
            c_parts.append(f"ngân sách dưới {constraints['max_price']:,.0f}đ")
        if constraints.get("exclude_spicy"):
            c_parts.append("không ăn cay")
        if constraints.get("exclude_types"):
            c_parts.append(f"không chọn các món liên quan đến '{', '.join(constraints['exclude_types'])}'")

        if c_parts:
            reasons.append(f"kết hợp lọc đúng theo yêu cầu của bạn: {', '.join(c_parts)}")

        return " ".join(reasons) + "."

    def respond(
        self,
        text: str,
        confidence_threshold: Optional[float] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
        remove_stopwords: bool = True
    ) -> Dict[str, Any]:
        """
        Xử lý toàn diện yêu cầu của người dùng qua toàn bộ pipeline:
        1. Tiền xử lý (preprocess)
        2. Phân loại ý định (classifier)
        3. Trích xuất ràng buộc (extractor)
        4. Gợi ý món ăn (recommender)
        5. Sinh câu trả lời tự nhiên có giải thích lý do
        
        Trả về:
            Dict[str, Any]: Toàn bộ kết quả trung gian và câu trả lời hoàn chỉnh.
        """
        conf_thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        k = top_k if top_k is not None else self.default_top_k

        # 1. Tiền xử lý
        prep_result = preprocess(text, remove_stopwords=remove_stopwords)

        # 2. Phân loại ý định
        clf_result = self.classifier.predict(text, remove_stopwords=remove_stopwords)
        predicted_label = clf_result["label"]
        confidence = clf_result["confidence"]

        # 3. Trích xuất ràng buộc
        constraint_result = self.extractor.extract(text)

        # Xử lý các nhãn xã giao
        if predicted_label == "chao_hoi":
            reply_text = self._generate_greeting_reply()
            return {
                "text": text,
                "preprocessing": prep_result,
                "classification": clf_result,
                "effective_intent": "chao_hoi",
                "is_fallback": False,
                "fallback_reason": None,
                "constraints": constraint_result,
                "recommendation": None,
                "explanation": "Người dùng gửi lời chào hỏi thân mật.",
                "reply": reply_text
            }

        if predicted_label == "cam_on_tam_biet":
            reply_text = self._generate_farewell_reply()
            return {
                "text": text,
                "preprocessing": prep_result,
                "classification": clf_result,
                "effective_intent": "cam_on_tam_biet",
                "is_fallback": False,
                "fallback_reason": None,
                "constraints": constraint_result,
                "recommendation": None,
                "explanation": "Người dùng nói lời cảm ơn hoặc chào tạm biệt.",
                "reply": reply_text
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
            constraints=constraint_result,
            custom_weights=custom_weights,
            top_k=k
        )

        # 6. Sinh giải thích và câu trả lời tự nhiên
        explanation = self._generate_explanation(effective_intent, constraint_result, is_fallback)

        top_dishes = rec_result["top_k"]
        if not top_dishes:
            reply_text = (
                f"Mình hiểu bạn đang tìm món theo phong cách '{effective_intent}'. "
                f"Tuy nhiên, sau khi lọc theo ràng buộc ({explanation}), hiện tại không tìm thấy quán nào quanh NEU thỏa mãn đồng thời toàn bộ yêu cầu. "
                "Bạn thử nới lỏng ngân sách hoặc bỏ bớt điều kiện không cay / loại trừ xem sao nhé!"
            )
        else:
            lines = [f"Dạ, {explanation} Dưới đây là {len(top_dishes)} gợi ý tốt nhất dành cho bạn:\n"]
            for idx, item in enumerate(top_dishes, 1):
                d = item["dish"]
                cay_icon = "🌶️ (Có cay)" if d["cay"] == 1 else "🟢 (Không cay)"
                lines.append(
                    f"{idx}. 🍽️ **{d['ten_mon']}** ({d['loai'].capitalize()})\n"
                    f"   - 💰 Giá: **{d['gia_vnd']:,.0f}đ** | 📍 Cách NEU: **{d['khoang_cach_m']:.0f}m** | ⏱️ Chờ: ~{d['thoi_gian_phuc_vu_phut']:.0f} phút\n"
                    f"   - ⭐ Điểm tổng hợp: **{item['score']:.2f}** | Vị: {cay_icon}\n"
                    f"   - 📝 {d['mo_ta']}\n"
                )
            lines.append("Bạn thấy món nào hợp gu nhất? Chúc bạn có bữa trưa ngon miệng nhé!")
            reply_text = "\n".join(lines)

        return {
            "text": text,
            "preprocessing": prep_result,
            "classification": clf_result,
            "effective_intent": effective_intent,
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "constraints": constraint_result,
            "recommendation": rec_result,
            "explanation": explanation,
            "reply": reply_text
        }


# Tương thích tên lớp theo yêu cầu
LunchChatbot = Chatbot


if __name__ == "__main__":
    bot = Chatbot()
    test_queries = [
        "Xin chào bot nhé",
        "Trưa nay ăn gì rẻ rẻ dưới 35k không ăn cay ở ngõ Tự Do",
        "Có món nào đổi gió lạ miệng quanh trường NEU không",
        "Cảm ơn bạn nhiều nha"
    ]
    for q in test_queries:
        print("=" * 80)
        print(f"HỎI: {q}")
        resp = bot.respond(q)
        print(f"Nhãn: {resp['classification']['label']} (Độ tin cậy: {resp['classification']['confidence']:.2f})")
        print(f"Nhãn áp dụng: {resp['effective_intent']} (Fallback: {resp['is_fallback']})")
        print(f"Ràng buộc: {resp['constraints']}")
        print(f"Trả lời:\n{resp['reply']}")
