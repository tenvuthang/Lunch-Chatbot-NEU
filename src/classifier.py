"""
Module phân loại ý định người dùng (Need/Intent Classifier) cho Lunch Chatbot.
Sử dụng scikit-learn Pipeline kết hợp TfidfVectorizer và thuật toán phân loại (MultinomialNB, LogisticRegression, LinearSVC).
Tất cả các hàm đều có docstring tiếng Việt và trả về kết quả trung gian đầy đủ.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Đảm bảo đường dẫn import hoạt động cả khi chạy độc lập
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from src.preprocess import preprocess
except ImportError:
    from preprocess import preprocess


def build_model(
    name: str = "nb",
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    alpha: float = 0.5,
    C: float = 1.0,
    **kwargs
) -> Pipeline:
    """
    Xây dựng Pipeline học máy gồm TfidfVectorizer và mô hình phân loại.
    
    Tham số:
        name (str): Tên mô hình ('nb' cho Naive Bayes, 'logreg' cho Logistic Regression, 'svm' cho Calibrated LinearSVC).
        ngram_range (Tuple[int, int]): Dải n-gram cho TfidfVectorizer (mặc định (1, 2)).
        sublinear_tf (bool): Sử dụng scaling sublinear cho TF (mặc định True).
        alpha (float): Hệ số smoothing cho MultinomialNB (mặc định 0.5).
        C (float): Hệ số điều chuẩn cho LogisticRegression / SVM (mặc định 1.0).

    Trả về:
        Pipeline: Sklearn Pipeline đã sẵn sàng fit.
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=1,
        **kwargs
    )

    name_lower = name.lower()
    if name_lower in ("nb", "naive_bayes"):
        clf = MultinomialNB(alpha=alpha)
    elif name_lower in ("logreg", "logistic"):
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    elif name_lower in ("svm", "linearsvc"):
        # Dùng CalibratedClassifierCV để LinearSVC có hàm predict_proba
        base_svc = LinearSVC(C=C, max_iter=2000, random_state=42)
        clf = CalibratedClassifierCV(estimator=base_svc)
    else:
        raise ValueError(f"Mô hình '{name}' không được hỗ trợ. Hãy chọn 'nb', 'logreg' hoặc 'svm'.")

    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])


class NeedClassifier:
    """
    Lớp phân loại nhu cầu/ý định ăn uống của sinh viên.
    Hỗ trợ huấn luyện, dự đoán, giải thích trọng số TF-IDF và lưu/tải mô hình.
    """

    def __init__(
        self,
        model_type: str = "nb",
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        alpha: float = 0.5,
        C: float = 1.0,
        remove_stopwords: bool = True
    ):
        self.model_type = model_type
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.alpha = alpha
        self.C = C
        self.remove_stopwords = remove_stopwords
        self.pipeline: Optional[Pipeline] = None
        self.classes_: List[str] = []

    def _prepare_text(self, text: str) -> str:
        """Tiền xử lý chuỗi và nối các token thành câu văn bản chuẩn."""
        prep = preprocess(text, remove_stopwords=self.remove_stopwords)
        tokens_list = prep["tokens_no_stopwords"] if self.remove_stopwords else prep["tokens"]
        return " ".join(tokens_list)

    def train(self, csv_path: str = "data/intents.csv") -> Dict[str, Any]:
        """
        Huấn luyện mô hình từ file CSV (gồm 2 cột: 'text' và 'label').
        
        Trả về:
            Dict[str, Any]: Thông tin tóm tắt quá trình huấn luyện:
                - 'num_samples': Tổng số mẫu huấn luyện.
                - 'num_classes': Số lượng nhãn.
                - 'classes': Danh sách các nhãn.
                - 'model_type': Tên thuật toán sử dụng.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu huấn luyện: {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8")
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"File {csv_path} phải chứa đủ 2 cột 'text' và 'label'.")

        X_raw = df["text"].astype(str).tolist()
        y = df["label"].astype(str).tolist()

        # Tiền xử lý dữ liệu
        X_processed = [self._prepare_text(t) for t in X_raw]

        # Xây dựng và fit mô hình
        self.pipeline = build_model(
            name=self.model_type,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            alpha=self.alpha,
            C=self.C
        )
        self.pipeline.fit(X_processed, y)

        clf_step = self.pipeline.named_steps["clf"]
        self.classes_ = [str(c) for c in clf_step.classes_]

        return {
            "num_samples": len(X_processed),
            "num_classes": len(self.classes_),
            "classes": self.classes_,
            "model_type": self.model_type
        }

    def predict(self, text: str, remove_stopwords: Optional[bool] = None) -> Dict[str, Any]:
        """
        Dự đoán nhãn ý định từ câu văn bản của người dùng.
        
        Tham số:
            text (str): Câu văn bản người dùng.
            remove_stopwords (Optional[bool]): Có loại bỏ stopword hay không. Nếu None thì dùng cấu hình khi khởi tạo.
        
        Trả về:
            Dict[str, Any]:
                - 'label': Nhãn có xác suất cao nhất (str).
                - 'confidence': Xác suất của nhãn cao nhất (0.0 đến 1.0).
                - 'probabilities': Danh sách toàn bộ nhãn cùng xác suất, sắp xếp giảm dần.
                - 'tfidf_terms': Danh sách các từ/cụm từ có trọng số TF-IDF > 0 trong câu, sắp xếp giảm dần.
                - 'preprocessed': Kết quả tiền xử lý trung gian.
        """
        if self.pipeline is None:
            raise RuntimeError("Mô hình chưa được huấn luyện hoặc tải từ file! Hãy gọi .train() hoặc .load() trước.")

        sw = self.remove_stopwords if remove_stopwords is None else remove_stopwords
        prep = preprocess(text, remove_stopwords=sw)
        tokens_list = prep["tokens_no_stopwords"] if sw else prep["tokens"]
        processed_text = " ".join(tokens_list)

        tfidf = self.pipeline.named_steps["tfidf"]
        clf = self.pipeline.named_steps["clf"]

        # 1. Trích xuất vector TF-IDF và danh sách từ khóa có trọng số > 0
        tfidf_vec = tfidf.transform([processed_text])
        feature_names = tfidf.get_feature_names_out()
        coo = tfidf_vec.tocoo()

        tfidf_terms = []
        for col, val in zip(coo.col, coo.data):
            tfidf_terms.append({
                "term": str(feature_names[col]),
                "tfidf": round(float(val), 4)
            })
        tfidf_terms.sort(key=lambda item: item["tfidf"], reverse=True)

        # 2. Tính xác suất cho từng nhãn
        probs = clf.predict_proba(tfidf_vec)[0]
        prob_pairs = sorted(zip(clf.classes_, probs), key=lambda item: item[1], reverse=True)

        probabilities = [
            {"label": str(label), "prob": round(float(p), 4)}
            for label, p in prob_pairs
        ]

        top_label = str(probabilities[0]["label"])
        top_prob = float(probabilities[0]["prob"])

        return {
            "label": top_label,
            "confidence": top_prob,
            "probabilities": probabilities,
            "tfidf_terms": tfidf_terms,
            "preprocessed": prep
        }

    def save(self, filepath: str) -> None:
        """Lưu toàn bộ mô hình và thông số vào file bằng joblib."""
        if self.pipeline is None:
            raise RuntimeError("Không có mô hình để lưu! Hãy huấn luyện trước.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "classes_": self.classes_,
            "model_type": self.model_type,
            "ngram_range": self.ngram_range,
            "sublinear_tf": self.sublinear_tf,
            "alpha": self.alpha,
            "C": self.C
        }
        joblib.dump(payload, filepath)

    def load(self, filepath: str) -> "NeedClassifier":
        """Tải mô hình và thông số từ file joblib."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")
        payload = joblib.load(filepath)
        self.pipeline = payload["pipeline"]
        self.classes_ = payload["classes_"]
        self.model_type = payload.get("model_type", "nb")
        self.ngram_range = payload.get("ngram_range", (1, 2))
        self.sublinear_tf = payload.get("sublinear_tf", True)
        self.alpha = payload.get("alpha", 0.5)
        self.C = payload.get("C", 1.0)
        return self


if __name__ == "__main__":
    clf = NeedClassifier(model_type="nb")
    print("Bắt đầu huấn luyện mô hình với data/intents.csv...")
    summary = clf.train("data/intents.csv")
    print("Huấn luyện xong:", summary)

    test_sentence = "trưa nay muốn ăn gì rẻ rẻ dưới 30k ktx neu"
    res = clf.predict(test_sentence)
    print(f"\nDự đoán cho câu: '{test_sentence}'")
    print(f"Nhãn: {res['label']} (độ tin cậy: {res['confidence']:.4f})")
    print("Xác suất các nhãn:", res["probabilities"][:3])
    print("Các từ khóa TF-IDF nổi bật:", res["tfidf_terms"][:5])
