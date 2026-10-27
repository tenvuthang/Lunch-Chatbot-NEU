"""
Tiện ích cho tab "Đánh giá" (không phụ thuộc PyQt để kiểm thử độc lập):
- accuracy_level(): phân mức Accuracy (xanh/vàng/đỏ).
- build_insights(): sinh khung "Nhận định kết quả" 4-6 câu tiếng Việt BẰNG LUẬT từ số liệu thật
  (không viết cứng số): mức accuracy, nhãn tốt nhất/yếu nhất theo F1, cặp nhãn nhầm nhiều nhất,
  cảnh báo dữ liệu nhỏ khi accuracy >= 95%, gợi ý hành động.
- Lưu/đọc cache kết quả đánh giá (outputs/eval_result.json) kèm "chữ ký" tham số + dữ liệu để biết
  kết quả đã lưu còn khớp với mô hình/dữ liệu hiện tại hay không.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

LEVEL_GOOD = "good"   # >= 85%
LEVEL_WARN = "warn"   # 70% - dưới 85%
LEVEL_BAD = "bad"     # < 70%

ACC_GOOD_THRESHOLD = 0.85
ACC_WARN_THRESHOLD = 0.70
ACC_SUSPICIOUS_THRESHOLD = 0.95


def accuracy_level(accuracy: float) -> str:
    """Phân mức accuracy: từ 85% trở lên 'good' (xanh), 70-85% 'warn' (vàng), dưới 70% 'bad' (đỏ)."""
    if accuracy >= ACC_GOOD_THRESHOLD:
        return LEVEL_GOOD
    if accuracy >= ACC_WARN_THRESHOLD:
        return LEVEL_WARN
    return LEVEL_BAD


def format_percent_vi(value: float, digits: int = 1) -> str:
    """Định dạng phần trăm kiểu Việt Nam (dấu phẩy thập phân): 0.8594 -> '85,9%'."""
    return f"{value * 100:.{digits}f}".replace(".", ",") + "%"


def format_accuracy_text(accuracy: float, n_correct: int, n_total: int) -> str:
    """Chuỗi hiển thị: 'Accuracy: xx,x% (đúng a / b câu kiểm thử)'."""
    return f"Accuracy: {format_percent_vi(accuracy)} (đúng {n_correct} / {n_total} câu kiểm thử)"


def most_confused_pair(
    cm: Sequence[Sequence[int]], labels: Sequence[str]
) -> Optional[Tuple[str, str, int, int]]:
    """
    Tìm cặp nhãn bị nhầm lẫn nhiều nhất (cộng hai chiều). Trả về (nhãn_a, nhãn_b, a→b, b→a)
    hoặc None nếu không có nhầm lẫn nào. Hàng = nhãn thật, cột = nhãn dự đoán.
    """
    best: Optional[Tuple[str, str, int, int]] = None
    best_total = 0
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            a_to_b = int(cm[i][j])
            b_to_a = int(cm[j][i])
            total = a_to_b + b_to_a
            if total > best_total:
                best_total = total
                best = (labels[i], labels[j], a_to_b, b_to_a)
    return best


def build_insights(
    accuracy: float,
    n_correct: int,
    n_total: int,
    labels: Sequence[str],
    report_dict: Dict[str, Dict[str, float]],
    confusion: Sequence[Sequence[int]],
    label_names: Optional[Dict[str, str]] = None,
    alpha: Optional[float] = None,
) -> str:
    """
    Sinh đoạn "Nhận định kết quả" (4-6 câu) hoàn toàn từ số liệu truyền vào.
    Tham số:
        accuracy, n_correct, n_total: kết quả tổng thể trên tập kiểm thử.
        labels: danh sách nhãn theo thứ tự của ma trận nhầm lẫn.
        report_dict: classification_report(output_dict=True) (precision/recall/f1-score/support từng nhãn).
        confusion: ma trận nhầm lẫn (hàng = thật, cột = dự đoán).
        label_names: ánh xạ nhãn -> tên tiếng Việt để hiển thị.
        alpha: giá trị alpha đang dùng (để gợi ý điều chỉnh), có thể None.
    """
    names = label_names or {}
    nm = lambda key: names.get(key, key)
    sentences: List[str] = []

    # 1. Mức accuracy tổng thể
    level = accuracy_level(accuracy)
    pct = format_percent_vi(accuracy)
    if level == LEVEL_GOOD:
        verdict = "ở mức tốt"
    elif level == LEVEL_WARN:
        verdict = "ở mức khá, vẫn còn chỗ để cải thiện"
    else:
        verdict = "còn thấp, mô hình chưa đủ tin cậy"
    sentences.append(
        f"Mô hình đoán đúng {n_correct}/{n_total} câu kiểm thử (Accuracy {pct}), {verdict}."
    )

    # 2. Nhãn tốt nhất / yếu nhất theo F1 (chỉ xét nhãn có mẫu kiểm thử)
    scored: List[Tuple[str, float]] = [
        (lbl, float(report_dict[lbl].get("f1-score", 0.0)))
        for lbl in labels
        if lbl in report_dict and report_dict[lbl].get("support", 0) > 0
    ]
    best = worst = None
    if scored:
        best = max(scored, key=lambda x: x[1])
        worst = min(scored, key=lambda x: x[1])
        if best[0] == worst[0] or abs(best[1] - worst[1]) < 1e-9:
            sentences.append(
                f"Các nhãn có F1 gần như đồng đều (F1 = {best[1]:.2f})."
            )
        else:
            sentences.append(
                f"Nhãn nhận diện tốt nhất là “{nm(best[0])}” (F1 = {best[1]:.2f}), "
                f"yếu nhất là “{nm(worst[0])}” (F1 = {worst[1]:.2f})."
            )

    # 3. Cặp nhãn bị nhầm nhiều nhất
    pair = most_confused_pair(confusion, labels)
    if pair is None:
        sentences.append("Không có cặp nhãn nào bị nhầm lẫn trên tập kiểm thử này.")
    else:
        a, b, a_to_b, b_to_a = pair
        sentences.append(
            f"Cặp nhãn bị nhầm nhiều nhất là “{nm(a)}” và “{nm(b)}” "
            f"({a_to_b + b_to_a} lần: {a_to_b} lần {nm(a)}→{nm(b)}, {b_to_a} lần ngược lại)."
        )

    # 4. Cảnh báo dữ liệu nhỏ khi accuracy quá cao
    if accuracy >= ACC_SUSPICIOUS_THRESHOLD:
        sentences.append(
            "Lưu ý: dữ liệu huấn luyện và kiểm thử khá nhỏ và do tự biên soạn nên Accuracy "
            "có thể cao hơn hiệu năng thực tế khi gặp câu nói mới của sinh viên."
        )

    # 5. Gợi ý hành động cho nhãn yếu
    if worst is not None and worst[1] < 0.9 and not (best[0] == worst[0]):
        sentences.append(
            f"Gợi ý: hãy thêm câu mẫu đa dạng cho nhãn “{nm(worst[0])}” ở tab Quản lý dữ liệu "
            f"rồi bấm Huấn luyện lại."
        )
    else:
        sentences.append(
            "Gợi ý: các nhãn đều ổn, bạn có thể bổ sung thêm câu kiểm thử mới để kiểm chứng chắc chắn hơn."
        )

    # 6. Gợi ý đổi alpha (khi chưa quá cao)
    if accuracy < ACC_SUSPICIOUS_THRESHOLD:
        if alpha is not None:
            sentences.append(
                f"Bạn cũng có thể thử đổi alpha (đang là {alpha:g}) ở tab Tham số, "
                f"ví dụ tăng hoặc giảm nhẹ rồi so sánh lại Accuracy."
            )
        else:
            sentences.append(
                "Bạn cũng có thể thử đổi alpha ở tab Tham số rồi so sánh lại Accuracy."
            )

    return " ".join(sentences)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE KẾT QUẢ ĐÁNH GIÁ
# ══════════════════════════════════════════════════════════════════════════════
def file_fingerprint(path: str) -> str:
    """Mã băm SHA1 rút gọn của nội dung file (chuỗi rỗng nếu không có file)."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def make_signature(
    alpha: float,
    ngram_range: Sequence[int],
    remove_stopwords: bool,
    train_path: str,
    test_path: str,
    model_type: str = "nb",
) -> Dict[str, Any]:
    """Chữ ký xác định một lần đánh giá: tham số mô hình + nội dung dữ liệu huấn luyện/kiểm thử."""
    return {
        "model_type": model_type,
        "alpha": round(float(alpha), 6),
        "ngram_range": [int(x) for x in ngram_range],
        "remove_stopwords": bool(remove_stopwords),
        "train_hash": file_fingerprint(train_path),
        "test_hash": file_fingerprint(test_path),
    }


def result_to_payload(result: Dict[str, Any], labels: Sequence[str], signature: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyển kết quả ModelEvaluator.evaluate_on_test thành dict JSON-hóa được để lưu cache."""
    y_true = list(result["y_true"])
    y_pred = list(result["y_pred"])
    n_total = len(y_true)
    n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    cm = [[int(v) for v in row] for row in result["confusion_matrix"]]
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signature": dict(signature),
        "accuracy": float(result["accuracy"]),
        "n_correct": n_correct,
        "n_total": n_total,
        "labels": list(labels),
        "report_dict": result["report_dict"],
        "confusion_matrix": cm,
    }


def save_eval_cache(path: str, payload: Dict[str, Any]) -> None:
    """Lưu cache đánh giá ra JSON (UTF-8)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_eval_cache(path: str, signature: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Đọc cache đánh giá. Nếu truyền signature thì chỉ trả về khi cache khớp chữ ký (cùng tham số và dữ liệu);
    ngược lại hoặc khi file hỏng/thiếu trả về None.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for key in ("accuracy", "n_correct", "n_total", "labels", "report_dict", "confusion_matrix", "signature"):
            if key not in payload:
                return None
    except (OSError, ValueError):
        return None
    if signature is not None and payload["signature"] != signature:
        return None
    return payload
