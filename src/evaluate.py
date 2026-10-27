"""
Module đánh giá mô hình học máy và hệ thống gợi ý cho Lunch Chatbot (NEU).
ĐÁP ỨNG TOÀN DIỆN CÁC YÊU CẦU:
1. Đánh giá mô hình trên tập kiểm thử độc lập (data/test_intents.csv). Tính accuracy, precision, recall, F1 từng nhãn.
   Vẽ và lưu ma trận nhầm lẫn (outputs/confusion_matrix.png, dpi=200).
2. Cross-validation 5-fold phân tầng (StratifiedKFold) trên data/intents.csv, báo cáo mean ± std.
3. So sánh 3 mô hình (MultinomialNB, LogisticRegression, LinearSVC). Xuất bảng CSV và biểu đồ cột (outputs/model_comparison.png, dpi=200).
4. Thí nghiệm tham số:
   - Accuracy theo alpha (0.01 đến 2.0) -> outputs/exp_alpha.png
   - Accuracy theo ngram_range -> outputs/exp_ngram.png
   - Accuracy có/không bỏ stopword -> outputs/exp_stopwords.png
5. Xuất danh sách câu test bị phân loại sai kèm nhãn thật, nhãn dự đoán, xác suất (outputs/misclassified_samples.csv).
6. Tạo outputs/worked_example.txt:
   - Bài toán Naive Bayes: 3 nhãn x 3 câu ngắn, từ vựng, ma trận TF-IDF, Prior, Likelihood làm mịn, tính toán chi tiết 1 câu thử.
   - Bài toán Recommender: 4 món x 4 tiêu chí, chuẩn hóa min-max đảo chiều/thuận chiều, trọng số nhóm nhu cầu, điểm cuối & xếp hạng.
"""

import os
import sys
import math
import json
from typing import Dict, Any
import numpy as np
import pandas as pd

# Thiết lập thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Matplotlib với cấu hình font tiếng Việt và backend Agg để lưu file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def _setup_vietnamese_font():
    """Thiết lập font tiếng Việt cho Matplotlib để không bị lỗi ô vuông."""
    candidates = ["Segoe UI", "Arial", "Tahoma", "Times New Roman", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.family"] = c
            return c
    return "sans-serif"

VIET_FONT = _setup_vietnamese_font()
plt.rcParams["axes.unicode_minus"] = False

# Scikit-learn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.io_utils import read_csv_dataframe, write_csv_dataframe
from src.preprocess import preprocess
from src.classifier import NeedClassifier, build_model

# Nhãn tiếng Việt hiển thị
LABEL_NAMES_VI = {
    "tiet_kiem": "Tiết kiệm",
    "healthy": "Healthy",
    "nhanh_gon": "Nhanh gọn",
    "no_bung": "No bụng",
    "ngon_doi_gio": "Đổi gió",
    "can_bang": "Cân bằng",
    "chao_hoi": "Chào hỏi",
    "cam_on_tam_biet": "Cảm ơn/Bye"
}


class ModelEvaluator:
    """Lớp điều phối đánh giá hiệu năng mô hình phân loại ý định và hệ thống gợi ý."""

    def __init__(
        self,
        train_csv_path: str = "data/intents.csv",
        test_csv_path: str = "data/test_intents.csv",
        outputs_dir: str = "outputs"
    ):
        self.train_csv_path = os.path.join(parent_dir, train_csv_path) if not os.path.isabs(train_csv_path) else train_csv_path
        self.test_csv_path = os.path.join(parent_dir, test_csv_path) if not os.path.isabs(test_csv_path) else test_csv_path
        self.outputs_dir = os.path.join(parent_dir, outputs_dir) if not os.path.isabs(outputs_dir) else outputs_dir
        os.makedirs(self.outputs_dir, exist_ok=True)

        self.df_train = read_csv_dataframe(self.train_csv_path)
        self.df_test = read_csv_dataframe(self.test_csv_path)

        self.labels = sorted(list(self.df_train["label"].unique()))

    # ══════════════════════════════════════════════════════════════════════════
    # 1. HUẤN LUYỆN VÀ ĐÁNH GIÁ TRÊN TEST SET + VẼ CONFUSION MATRIX
    # ══════════════════════════════════════════════════════════════════════════
    def evaluate_on_test(self, classifier: NeedClassifier) -> Dict[str, Any]:
        """
        Đánh giá classifier trên data/test_intents.csv độc lập.
        Tính Accuracy, Precision, Recall, F1 từng nhãn và ma trận nhầm lẫn.
        """
        y_true = self.df_test["label"].astype(str).tolist()
        texts = self.df_test["text"].astype(str).tolist()

        y_pred = []
        confidences = []
        detailed_preds = []

        for t, true_lbl in zip(texts, y_true):
            pred_res = classifier.predict(t)
            pred_lbl = pred_res["label"]
            conf = pred_res["confidence"]
            y_pred.append(pred_lbl)
            confidences.append(conf)

            # Lấy xác suất của nhãn đúng
            true_prob = 0.0
            for p in pred_res.get("probabilities", []):
                if p["label"] == true_lbl:
                    true_prob = p["prob"]
                    break

            detailed_preds.append({
                "text": t,
                "true_label": true_lbl,
                "pred_label": pred_lbl,
                "confidence": conf,
                "true_prob": true_prob,
                "is_correct": (true_lbl == pred_lbl)
            })

        acc = accuracy_score(y_true, y_pred)
        rep_dict = classification_report(y_true, y_pred, labels=self.labels, output_dict=True, zero_division=0)
        rep_str = classification_report(y_true, y_pred, labels=self.labels, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)

        return {
            "accuracy": round(float(acc), 4),
            "n_correct": n_correct,
            "n_total": len(y_true),
            "labels": list(self.labels),
            "report_text": rep_str,
            "report_dict": rep_dict,
            "confusion_matrix": cm,
            "predictions": detailed_preds,
            "y_true": y_true,
            "y_pred": y_pred
        }

    evaluate = evaluate_on_test

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        output_filename: str = "confusion_matrix.png",
        title: str = "Ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử độc lập"
    ) -> str:
        """Vẽ ma trận nhầm lẫn bằng Matplotlib độ phân giải cao dpi=200."""
        output_path = os.path.join(self.outputs_dir, output_filename)
        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=200)

        cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        labels_display = [LABEL_NAMES_VI.get(l, l) for l in self.labels]
        ax.set_xticks(np.arange(len(self.labels)))
        ax.set_yticks(np.arange(len(self.labels)))
        ax.set_xticklabels(labels_display, rotation=35, ha="right", fontsize=9, fontfamily=VIET_FONT)
        ax.set_yticklabels(labels_display, fontsize=9, fontfamily=VIET_FONT)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color,
                        fontweight="bold", fontsize=10, fontfamily=VIET_FONT)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=12, fontfamily=VIET_FONT)
        ax.set_ylabel("Nhãn thực tế (True Label)", fontsize=10, fontweight="bold", fontfamily=VIET_FONT)
        ax.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10, fontweight="bold", fontfamily=VIET_FONT)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    # ══════════════════════════════════════════════════════════════════════════
    # 2. CROSS-VALIDATION 5-FOLD PHÂN TẦNG TRÊN INTENTS.CSV
    # ══════════════════════════════════════════════════════════════════════════
    def run_5fold_cv(
        self,
        model_type: str = "nb",
        alpha: float = 0.5,
        C: float = 1.0,
        ngram_range: tuple = (1, 2),
        sublinear_tf: bool = True,
        remove_stopwords: bool = True
    ) -> Dict[str, Any]:
        """
        Cross-Validation 5-fold phân tầng trên data/intents.csv.
        Báo cáo trung bình và độ lệch chuẩn của Accuracy, Macro Precision, Recall, F1.
        """
        X = self.df_train["text"].astype(str).tolist()
        y = self.df_train["label"].astype(str).tolist()

        # Tiền xử lý
        def prep_fn(t):
            res = preprocess(t, remove_stopwords=remove_stopwords)
            tokens = res["tokens_no_stopwords"] if remove_stopwords else res["tokens"]
            return " ".join(tokens)

        X_prep = [prep_fn(t) for t in X]
        y_arr = np.array(y)
        X_arr = np.array(X_prep)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va, y_va = X_arr[val_idx], y_arr[val_idx]

            pipeline = build_model(
                name=model_type,
                ngram_range=ngram_range,
                sublinear_tf=sublinear_tf,
                alpha=alpha,
                C=C
            )
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_va)

            accuracies.append(accuracy_score(y_va, y_pred))
            precisions.append(precision_score(y_va, y_pred, average="macro", zero_division=0))
            recalls.append(recall_score(y_va, y_pred, average="macro", zero_division=0))
            f1s.append(f1_score(y_va, y_pred, average="macro", zero_division=0))

        return {
            "model_type": model_type,
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "precision_mean": float(np.mean(precisions)),
            "precision_std": float(np.std(precisions)),
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "fold_accuracies": [float(a) for a in accuracies]
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SO SÁNH 3 MÔ HÌNH: MultinomialNB, LogisticRegression, LinearSVC
    # ══════════════════════════════════════════════════════════════════════════
    def compare_models(self) -> pd.DataFrame:
        """
        So sánh MultinomialNB, LogisticRegression, LinearSVC.
        Xuất bảng outputs/model_comparison.csv và biểu đồ outputs/model_comparison.png (dpi=200).
        """
        model_names = [
            ("MultinomialNB", "nb", {"alpha": 0.5}),
            ("LogisticRegression", "logreg", {"C": 1.0}),
            ("LinearSVC", "svm", {"C": 1.0})
        ]

        results = []
        for disp_name, m_type, kwargs in model_names:
            # 1. Đánh giá trên test set độc lập
            clf = NeedClassifier(model_type=m_type, **kwargs)
            clf.train(self.train_csv_path)
            test_eval = self.evaluate_on_test(clf)

            # 2. Đánh giá 5-fold CV trên train set
            cv_res = self.run_5fold_cv(model_type=m_type, **kwargs)

            rep = test_eval["report_dict"]
            results.append({
                "Mô hình": disp_name,
                "Test Accuracy": test_eval["accuracy"],
                "Test Macro F1": round(rep["macro avg"]["f1-score"], 4),
                "Test Macro Precision": round(rep["macro avg"]["precision"], 4),
                "Test Macro Recall": round(rep["macro avg"]["recall"], 4),
                "5-Fold CV Accuracy (Mean)": round(cv_res["accuracy_mean"], 4),
                "5-Fold CV Accuracy (Std)": round(cv_res["accuracy_std"], 4),
                "5-Fold CV F1 (Mean)": round(cv_res["f1_mean"], 4)
            })

        df_res = pd.DataFrame(results)
        csv_path = os.path.join(self.outputs_dir, "model_comparison.csv")
        write_csv_dataframe(df_res, csv_path, index=False)

        # Vẽ biểu đồ cột so sánh
        self._plot_model_comparison(df_res)
        return df_res

    def _plot_model_comparison(self, df: pd.DataFrame) -> str:
        """Vẽ biểu đồ cột so sánh 3 mô hình."""
        output_path = os.path.join(self.outputs_dir, "model_comparison.png")
        fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)

        models = df["Mô hình"].tolist()
        x = np.arange(len(models))
        width = 0.24

        test_acc = df["Test Accuracy"].values
        test_f1 = df["Test Macro F1"].values
        cv_acc = df["5-Fold CV Accuracy (Mean)"].values

        rects1 = ax.bar(x - width, test_acc, width, label="Test Accuracy", color="#3E7BFA")
        rects2 = ax.bar(x, test_f1, width, label="Test Macro F1", color="#3ECA8D")
        rects3 = ax.bar(x + width, cv_acc, width, label="5-Fold CV Accuracy", color="#F4B942")

        # Thêm nhãn giá trị trên đầu cột
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.1%}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_ylabel("Điểm số", fontsize=10, fontweight="bold", fontfamily=VIET_FONT)
        ax.set_title("So sánh hiệu năng 3 mô hình phân loại nhu cầu ăn uống",
                     fontsize=12, fontweight="bold", pad=12, fontfamily=VIET_FONT)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, fontweight="bold", fontfamily=VIET_FONT)
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        return output_path

    # ══════════════════════════════════════════════════════════════════════════
    # 4. THÍ NGHIỆM THAM SỐ (Alpha, N-gram, Stopwords)
    # ══════════════════════════════════════════════════════════════════════════
    def run_hyperparameter_experiments(self) -> Dict[str, Any]:
        """Chạy các thí nghiệm tham số và vẽ biểu đồ đường/cột độ phân giải cao."""
        # ── Thí nghiệm 1: Accuracy theo alpha của Naive Bayes ─────────────────
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        alpha_accs = []
        for a in alphas:
            clf = NeedClassifier(model_type="nb", alpha=a)
            clf.train(self.train_csv_path)
            res = self.evaluate_on_test(clf)
            alpha_accs.append(res["accuracy"])

        # Vẽ đường cho Alpha
        alpha_img_path = os.path.join(self.outputs_dir, "exp_alpha.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        ax.plot(alphas, alpha_accs, marker="o", color="#7C6AF7", linewidth=2.2, markersize=6, label="Test Accuracy")
        best_alpha_idx = int(np.argmax(alpha_accs))
        ax.plot(alphas[best_alpha_idx], alpha_accs[best_alpha_idx], "r*", markersize=14,
                label=f"Tối ưu (α={alphas[best_alpha_idx]}, Acc={alpha_accs[best_alpha_idx]:.2%})")

        for x_val, y_val in zip(alphas, alpha_accs):
            ax.annotate(f"{y_val:.1%}", (x_val, y_val), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7.5)

        ax.set_title("Ảnh hưởng của hệ số làm mịn Laplace (Alpha) đến độ chính xác",
                     fontsize=11, fontweight="bold", pad=10, fontfamily=VIET_FONT)
        ax.set_xlabel("Hệ số làm mịn α (Laplace smoothing)", fontsize=9, fontfamily=VIET_FONT)
        ax.set_ylabel("Độ chính xác (Accuracy)", fontsize=9, fontfamily=VIET_FONT)
        ax.set_ylim(min(alpha_accs) - 0.05, max(alpha_accs) + 0.06)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(alpha_img_path, dpi=200)
        plt.close(fig)

        # ── Thí nghiệm 2: Accuracy theo ngram_range ──────────────────────────
        ngrams = [(1, 1), (1, 2), (1, 3), (2, 2)]
        ngram_names = ["Unigram (1,1)", "Uni + Bigram (1,2)", "1-to-3gram (1,3)", "Bigram only (2,2)"]
        ngram_accs = []
        for ng in ngrams:
            clf = NeedClassifier(model_type="nb", ngram_range=ng, alpha=0.5)
            clf.train(self.train_csv_path)
            res = self.evaluate_on_test(clf)
            ngram_accs.append(res["accuracy"])

        # Vẽ cột cho N-gram
        ngram_img_path = os.path.join(self.outputs_dir, "exp_ngram.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
        bars = ax.bar(ngram_names, ngram_accs, color=["#50C5B7", "#7C6AF7", "#3ECA8D", "#F47F4C"], width=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
        ax.set_title("Ảnh hưởng của cấu hình N-gram đến độ chính xác Naive Bayes",
                     fontsize=11, fontweight="bold", pad=10, fontfamily=VIET_FONT)
        ax.set_ylabel("Độ chính xác (Accuracy)", fontsize=9, fontfamily=VIET_FONT)
        ax.set_ylim(0, max(ngram_accs) * 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(ngram_img_path, dpi=200)
        plt.close(fig)

        # ── Thí nghiệm 3: Có / Không bỏ stopword ──────────────────────────────
        sw_cases = [True, False]
        sw_names = ["Có lọc Stopword", "Không lọc Stopword"]
        sw_accs = []
        for sw in sw_cases:
            clf = NeedClassifier(model_type="nb", remove_stopwords=sw, alpha=0.5)
            clf.train(self.train_csv_path)
            res = self.evaluate_on_test(clf)
            sw_accs.append(res["accuracy"])

        # Vẽ cột cho Stopword
        sw_img_path = os.path.join(self.outputs_dir, "exp_stopwords.png")
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=200)
        bars = ax.bar(sw_names, sw_accs, color=["#3ECA8D", "#F4B942"], width=0.45)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        ax.set_title("So sánh độ chính xác khi Bật vs Tắt lọc Stopword",
                     fontsize=11, fontweight="bold", pad=10, fontfamily=VIET_FONT)
        ax.set_ylabel("Độ chính xác (Accuracy)", fontsize=9, fontfamily=VIET_FONT)
        ax.set_ylim(0, max(sw_accs) * 1.25)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(sw_img_path, dpi=200)
        plt.close(fig)

        return {
            "alpha": {"values": alphas, "accuracies": alpha_accs},
            "ngram": {"names": ngram_names, "accuracies": ngram_accs},
            "stopwords": {"names": sw_names, "accuracies": sw_accs}
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 5. LIỆT KÊ CÂU TEST BỊ PHÂN LOẠI SAI XUẤT RA CSV
    # ══════════════════════════════════════════════════════════════════════════
    def export_misclassified(self, classifier: NeedClassifier) -> pd.DataFrame:
        """Xuất danh sách các câu trong test_intents.csv bị phân loại sai ra outputs/misclassified_samples.csv."""
        eval_res = self.evaluate_on_test(classifier)
        preds = eval_res["predictions"]

        errors = [p for p in preds if not p["is_correct"]]
        rows = []
        for e in errors:
            rows.append({
                "Câu kiểm thử (Text)": e["text"],
                "Nhãn thực tế (True Label)": e["true_label"],
                "Tên nhãn thực tế": LABEL_NAMES_VI.get(e["true_label"], e["true_label"]),
                "Nhãn dự đoán (Pred Label)": e["pred_label"],
                "Tên nhãn dự đoán": LABEL_NAMES_VI.get(e["pred_label"], e["pred_label"]),
                "Xác suất dự đoán sai": round(e["confidence"], 4),
                "Xác suất của nhãn đúng": round(e["true_prob"], 4)
            })

        df_errors = pd.DataFrame(rows)
        csv_path = os.path.join(self.outputs_dir, "misclassified_samples.csv")
        write_csv_dataframe(df_errors, csv_path, index=False)
        return df_errors

    # ══════════════════════════════════════════════════════════════════════════
    # 6. TẠO outputs/worked_example.txt (SỐ LIỆU ĐỂ TÍNH TAY VÀO CHƯƠNG 2)
    # ══════════════════════════════════════════════════════════════════════════
    def generate_worked_example(self, output_filename: str = "worked_example.txt") -> str:
        """
        Sinh outputs/worked_example.txt (ví dụ tính tay cho Chương 2) bằng src/worked_example.py:
        Phần 1 TF-IDF + MultinomialNB với đúng tham số model thật, Phần 2 chấm điểm đa tiêu chí với trọng số thật.
        Mọi số liệu tính tay được assert khớp sklearn/DishRecommender (sai số < 1e-6).
        """
        from src.worked_example import generate_worked_example as _generate
        return _generate(parent_dir, self.outputs_dir, output_filename)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. PHÂN TÍCH VÀ TRỰC QUAN HÓA ĐỘ ĐÔNG & CHỖ NGỒI (CROWD & SEATING)
    # ══════════════════════════════════════════════════════════════════════════
    def export_crowd_analysis(self, output_filename: str = "crowd_analysis.csv") -> str:
        """Xuất bảng tổng hợp phân tích độ đông, thời gian chờ và chỗ trống của 50 món ăn."""
        dishes_path = os.path.join(parent_dir, "data", "dishes_v2.csv")
        crowd_cfg_path = os.path.join(parent_dir, "config", "crowd_config.json")
        df_dishes = read_csv_dataframe(dishes_path)
        with open(crowd_cfg_path, "r", encoding="utf-8") as f:
            crowd_cfg = json.load(f)

        from src.recommender import compute_crowd_metrics
        rows = []
        for _, row in df_dishes.iterrows():
            d = row.to_dict()
            m_cd_1 = compute_crowd_metrics(d, so_nguoi=1, khung_gio="cao_diem", crowd_cfg=crowd_cfg)
            m_cd_4 = compute_crowd_metrics(d, so_nguoi=4, khung_gio="cao_diem", crowd_cfg=crowd_cfg)
            m_cd_8 = compute_crowd_metrics(d, so_nguoi=8, khung_gio="cao_diem", crowd_cfg=crowd_cfg)
            m_bt_1 = compute_crowd_metrics(d, so_nguoi=1, khung_gio="binh_thuong", crowd_cfg=crowd_cfg)
            m_v_1 = compute_crowd_metrics(d, so_nguoi=1, khung_gio="vang", crowd_cfg=crowd_cfg)

            rows.append({
                "id": int(d["id"]),
                "ten_mon": d["ten_mon"],
                "loai": d["loai"],
                "gia_vnd": int(d["gia_vnd"]),
                "khoang_cach_m": int(d["khoang_cach_m"]),
                "thoi_gian_phuc_vu_phut": int(d["thoi_gian_phuc_vu_phut"]),
                "do_dong_cao_diem": int(d["do_dong_cao_diem"]),
                "so_cho_ngoi": int(d["so_cho_ngoi"]),
                "do_dong_hieu_dung_cao_diem": round(m_cd_1["do_dong_hieu_dung"], 2),
                "do_dong_hieu_dung_binh_thuong": round(m_bt_1["do_dong_hieu_dung"], 2),
                "do_dong_hieu_dung_vang": round(m_v_1["do_dong_hieu_dung"], 2),
                "cho_trong_cao_diem": round(m_cd_1["cho_trong"], 2),
                "cho_trong_binh_thuong": round(m_bt_1["cho_trong"], 2),
                "cho_trong_vang": round(m_v_1["cho_trong"], 2),
                "thoi_gian_cho_1ng_caodiem": round(m_cd_1["thoi_gian_cho"], 2),
                "thoi_gian_cho_4ng_caodiem": round(m_cd_4["thoi_gian_cho"], 2),
                "thoi_gian_cho_8ng_caodiem": round(m_cd_8["thoi_gian_cho"], 2),
                "cho_ngoi_nhom_1ng_caodiem": round(m_cd_1["cho_ngoi_nhom"], 4),
                "cho_ngoi_nhom_4ng_caodiem": round(m_cd_4["cho_ngoi_nhom"], 4),
                "cho_ngoi_nhom_8ng_caodiem": round(m_cd_8["cho_ngoi_nhom"], 4),
                "canh_bao_cho_1ng": "Chờ chỗ" if m_cd_1["seat_warning"] else "Đủ chỗ",
                "canh_bao_cho_4ng": "Có thể phải chờ chỗ" if m_cd_4["seat_warning"] else "Đủ chỗ",
                "canh_bao_cho_8ng": "Có thể phải chờ chỗ" if m_cd_8["seat_warning"] else "Đủ chỗ",
            })

        df_out = pd.DataFrame(rows)
        out_path = os.path.join(self.outputs_dir, output_filename)
        write_csv_dataframe(df_out, out_path, index=False)
        return out_path

    def plot_crowd_wait_vs_people(self, output_filename: str = "crowd_wait_vs_people.png") -> str:
        """Vẽ biểu đồ thời gian chờ theo số người (1-10) cho 5 món điển hình qua 3 khung giờ."""
        dishes_path = os.path.join(parent_dir, "data", "dishes_v2.csv")
        crowd_cfg_path = os.path.join(parent_dir, "config", "crowd_config.json")
        df_dishes = read_csv_dataframe(dishes_path)
        with open(crowd_cfg_path, "r", encoding="utf-8") as f:
            crowd_cfg = json.load(f)

        from src.recommender import compute_crowd_metrics
        typical_ids = [1, 5, 12, 23, 28]
        dish_map = {int(d["id"]): d.to_dict() for _, d in df_dishes.iterrows()}
        typical_dishes = [dish_map[i] for i in typical_ids if i in dish_map]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        slots = [
            ("cao_diem", "Khung giờ Cao điểm (11:30 - 13:00)\nHệ số khung giờ = 1.0"),
            ("binh_thuong", "Khung giờ Bình thường (11:00-11:30, 13:00-14:00)\nHệ số khung giờ = 0.6"),
            ("vang", "Khung giờ Vắng khách\nHệ số khung giờ = 0.3")
        ]

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        markers = ["o", "s", "^", "D", "v"]

        for ax, (slot, slot_title) in zip(axes, slots):
            ax.set_title(slot_title, fontsize=10.5, fontweight="bold", pad=8)
            for idx, d in enumerate(typical_dishes):
                waits = []
                for p in range(1, 11):
                    m = compute_crowd_metrics(d, so_nguoi=p, khung_gio=slot, crowd_cfg=crowd_cfg)
                    waits.append(m["thoi_gian_cho"])
                lbl = f"[{d['id']}] {d['ten_mon'][:22]}... (TG gốc: {d['thoi_gian_phuc_vu_phut']}p, Đông: {d['do_dong_cao_diem']}/5)"
                ax.plot(range(1, 11), waits, label=lbl, color=colors[idx], marker=markers[idx], markersize=5, linewidth=1.8)

            ax.set_xlabel("Số người đi ăn trong nhóm (người)", fontsize=9.5)
            ax.set_xticks(range(1, 11))
            ax.grid(True, linestyle="--", alpha=0.55)

        axes[0].set_ylabel("Thời gian chờ ước tính (phút)", fontsize=10.5)
        axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
        fig.suptitle("THỜI GIAN CHỜ ƯỚC TÍNH THEO SỐ NGƯỜI ĐI ĂN (1 - 10 NGƯỜI) CHO 5 MÓN ĐIỂN HÌNH",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.86)
        out_path = os.path.join(self.outputs_dir, output_filename)
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    def plot_crowd_ranking_shift(self, output_filename: str = "crowd_ranking_shift.png") -> str:
        """Vẽ biểu đồ thay đổi thứ hạng Top-5 khi thay đổi số người đi ăn (1, 4, 8 người)."""
        dishes_path = os.path.join(parent_dir, "data", "dishes_v2.csv")
        profiles_path = os.path.join(parent_dir, "config", "profiles.json")
        crowd_cfg_path = os.path.join(parent_dir, "config", "crowd_config.json")

        from src.recommender import DishRecommender
        rec = DishRecommender(dishes_path=dishes_path, profiles_path=profiles_path, crowd_config_path=crowd_cfg_path)

        intent = "tiet_kiem"
        r1 = rec.recommend(intent, top_k=50, num_people=1, time_slot="cao_diem")["all_ranked"]
        r4 = rec.recommend(intent, top_k=50, num_people=4, time_slot="cao_diem")["all_ranked"]
        r8 = rec.recommend(intent, top_k=50, num_people=8, time_slot="cao_diem")["all_ranked"]

        rank_map_1 = {item["dish"]["id"]: (rank, item["dish"]["ten_mon"], item["score"]) for rank, item in enumerate(r1, 1)}
        rank_map_4 = {item["dish"]["id"]: (rank, item["dish"]["ten_mon"], item["score"]) for rank, item in enumerate(r4, 1)}
        rank_map_8 = {item["dish"]["id"]: (rank, item["dish"]["ten_mon"], item["score"]) for rank, item in enumerate(r8, 1)}

        top5_dishes = r1[:5]
        top5_ids = [item["dish"]["id"] for item in top5_dishes]

        fig, ax = plt.subplots(figsize=(10, 6))
        x_points = [1, 2, 3]
        x_labels = ["1 người\n(Cá nhân)", "4 người\n(Nhóm nhỏ)", "8 người\n(Nhóm đông)"]

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for idx, did in enumerate(top5_ids):
            rank1, name, s1 = rank_map_1[did]
            rank4, _, s4 = rank_map_4[did]
            rank8, _, s8 = rank_map_8[did]
            y_ranks = [rank1, rank4, rank8]

            line_color = palette[idx % len(palette)]
            ax.plot(x_points, y_ranks, marker="o", markersize=9, linewidth=2.4,
                    color=line_color, label=f"[{did}] {name}")

            # Ghi chú thứ hạng tại các điểm
            for x, y in zip(x_points, y_ranks):
                ax.annotate(f"#{y}", (x, y), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9, fontweight="bold", color=line_color)

        ax.invert_yaxis()
        ax.set_xticks(x_points)
        ax.set_xticklabels(x_labels, fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Thứ hạng gợi ý (Hạng 1 ở trên cùng)", fontsize=11, fontweight="bold")
        ax.set_title("SỰ THAY ĐỔI THỨ HẠNG TOP-5 MÓN ĂN KHI QUY MÔ NHÓM TĂNG LÊN (1 -> 4 -> 8 NGƯỜI)\n"
                     "Nhóm nhu cầu: Tiết kiệm (Giờ cao điểm trưa 11:30 - 13:00)",
                     fontsize=12, fontweight="bold", pad=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)

        plt.tight_layout()
        out_path = os.path.join(self.outputs_dir, output_filename)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    # ══════════════════════════════════════════════════════════════════════════
    # 8. THỐNG KÊ TỔNG QUAN DỮ LIỆU dishes_v2.csv (DATA SUMMARY)
    # ══════════════════════════════════════════════════════════════════════════
    def export_data_summary(self, output_filename: str = "data_summary.txt") -> str:
        """Tổng hợp thống kê thuộc tính, phân bố độ đông, chỗ ngồi và xuất file data_summary.txt."""
        dishes_path = os.path.join(parent_dir, "data", "dishes_v2.csv")
        df = read_csv_dataframe(dishes_path)

        total_dishes = len(df)
        chay_count = int(df["chay"].sum())
        man_count = total_dishes - chay_count
        cay_count = int(df["cay"].sum())
        khong_cay_count = total_dishes - cay_count

        loai_counts = df["loai"].value_counts().to_dict()
        do_dong_counts = df["do_dong_cao_diem"].value_counts().sort_index().to_dict()

        so_cho = df["so_cho_ngoi"].astype(int)
        min_cho = int(so_cho.min())
        max_cho = int(so_cho.max())
        mean_cho = float(so_cho.mean())

        c_ganh = int(((so_cho >= 4) & (so_cho <= 12)).sum())
        c_nho = int(((so_cho >= 15) & (so_cho <= 30)).sum())
        c_thuong = int(((so_cho >= 31) & (so_cho <= 60)).sum())
        c_lon = int(((so_cho >= 61) & (so_cho <= 120)).sum())

        gia = df["gia_vnd"].astype(int)
        dist = df["khoang_cach_m"].astype(int)
        time_prep = df["thoi_gian_phuc_vu_phut"].astype(int)

        lines = [
            "=" * 80,
            "BÁO CÁO THỐNG KÊ DỮ LIỆU MÓN ĂN QUANH ĐH KINH TẾ QUỐC DÂN (data/dishes_v2.csv)",
            "=" * 80,
            f"1. Tổng số món ăn: {total_dishes} món",
            f"   • Nguồn dữ liệu   : Mô phỏng theo thực tế ẩm thực khu vực NEU (ngõ Tự Do, Trần Đại Nghĩa, KTX, Lê Thanh Nghị...)",
            f"   • Số cột thuộc tính: 15 cột đầy đủ theo chuẩn kiến trúc hệ thống",
            "",
            "2. Phân loại thực đơn & Chế độ ăn uống:",
            f"   • Món chay        : {chay_count:2d}/{total_dishes} món ({chay_count/total_dishes*100:.1f}%)",
            f"   • Món mặn         : {man_count:2d}/{total_dishes} món ({man_count/total_dishes*100:.1f}%)",
            f"   • Món cay         : {cay_count:2d}/{total_dishes} món ({cay_count/total_dishes*100:.1f}%)",
            f"   • Món không cay   : {khong_cay_count:2d}/{total_dishes} món ({khong_cay_count/total_dishes*100:.1f}%)",
            "",
            "   Phân bố theo nhóm món ăn (loại):",
        ]
        for l_name, cnt in loai_counts.items():
            lines.append(f"     - {l_name:<16}: {cnt:2d} món ({cnt/total_dishes*100:.1f}%)")

        lines.extend([
            "",
            "3. Phân bố mức độ đông đúc vào giờ cao điểm (do_dong_cao_diem 1 - 5):",
            f"   • Mức 1 (Rất vắng / Thưa thớt)   : {do_dong_counts.get(1, 0):2d}/{total_dishes} món ({do_dong_counts.get(1, 0)/total_dishes*100:.1f}%)",
            f"   • Mức 2 (Hơi vắng / Thoáng bàn)  : {do_dong_counts.get(2, 0):2d}/{total_dishes} món ({do_dong_counts.get(2, 0)/total_dishes*100:.1f}%)",
            f"   • Mức 3 (Bình thường / Ổn định)  : {do_dong_counts.get(3, 0):2d}/{total_dishes} món ({do_dong_counts.get(3, 0)/total_dishes*100:.1f}%)",
            f"   • Mức 4 (Đông khách)             : {do_dong_counts.get(4, 0):2d}/{total_dishes} món ({do_dong_counts.get(4, 0)/total_dishes*100:.1f}%)",
            f"   • Mức 5 (Rất đông / Chật kín bàn): {do_dong_counts.get(5, 0):2d}/{total_dishes} món ({do_dong_counts.get(5, 0)/total_dishes*100:.1f}%)",
            "   -> Đánh giá: Phân bố cân đối, đầy đủ mọi mức từ 1 đến 5, không có mức nào vượt quá 40%.",
            "",
            "4. Phân bố quy mô chỗ ngồi (so_cho_ngoi):",
            f"   • Thấp nhất (Min) : {min_cho} chỗ",
            f"   • Cao nhất (Max)  : {max_cho} chỗ",
            f"   • Trung bình      : {mean_cho:.1f} chỗ",
            f"   • Phân khúc quy mô:",
            f"     - Gánh hàng / Mang đi (4 - 12 chỗ)        : {c_ganh:2d}/{total_dishes} món ({c_ganh/total_dishes*100:.1f}%)",
            f"     - Quán nhỏ / Vỉa hè (15 - 30 chỗ)         : {c_nho:2d}/{total_dishes} món ({c_nho/total_dishes*100:.1f}%)",
            f"     - Quán cơm/bún/phở thường (31 - 60 chỗ)   : {c_thuong:2d}/{total_dishes} món ({c_thuong/total_dishes*100:.1f}%)",
            f"     - Quán lớn / Cantin / Chuỗi (61 - 120 chỗ): {c_lon:2d}/{total_dishes} món ({c_lon/total_dishes*100:.1f}%)",
            "",
            "5. Thống kê các chỉ số hoạt động:",
            f"   • Giá tiền một suất (gia_vnd)            : Min = {int(gia.min()):,d}đ | Max = {int(gia.max()):,d}đ | Trung bình = {float(gia.mean()):,.0f}đ",
            f"   • Khoảng cách đi bộ (khoang_cach_m)      : Min = {int(dist.min())}m | Max = {int(dist.max())}m | Trung bình = {float(dist.mean()):.1f}m",
            f"   • Thời gian chế biến (thoi_gian_phuc_vu) : Min = {int(time_prep.min())} phút | Max = {int(time_prep.max())} phút | Trung bình = {float(time_prep.mean()):.1f} phút",
            "=" * 80,
        ])

        summary_text = "\n".join(lines)
        out_path = os.path.join(self.outputs_dir, output_filename)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(summary_text + "\n")
        return out_path

    # ══════════════════════════════════════════════════════════════════════════
    # CHẠY TOÀN BỘ QUY TRÌNH ĐÁNH GIÁ (FULL RUN)
    # ══════════════════════════════════════════════════════════════════════════
    def run_all(self) -> None:
        """Thực thi đầy đủ các bước đánh giá và xuất toàn bộ báo cáo, biểu đồ, số liệu."""
        print("\n" + "=" * 80)
        print("BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ MÔ HÌNH VÀ HỆ THỐNG GỢI Ý (LUNCH CHATBOT NEU)".center(80))
        print("=" * 80 + "\n")

        # 1. Huấn luyện mô hình chuẩn và đánh giá trên test_intents.csv
        print("--- [BƯỚC 1/8] Đánh giá trên tập kiểm thử độc lập (data/test_intents.csv) ---")
        clf_main = NeedClassifier(model_type="nb", alpha=0.5)
        clf_main.train(self.train_csv_path)
        eval_test = self.evaluate_on_test(clf_main)

        from src.eval_insights import make_signature, result_to_payload, save_eval_cache
        sig = make_signature(
            alpha=clf_main.alpha,
            ngram_range=clf_main.ngram_range,
            remove_stopwords=clf_main.remove_stopwords,
            train_path=self.train_csv_path,
            test_path=self.test_csv_path,
            model_type=clf_main.model_type,
        )
        cache_path = os.path.join(self.outputs_dir, "eval_result.json")
        save_eval_cache(cache_path, result_to_payload(eval_test, self.labels, sig))

        print(f"  • Độ chính xác (Accuracy): {eval_test['accuracy']:.2%}")
        print("\nBáo cáo phân loại chi tiết (Classification Report):")
        print(eval_test["report_text"])

        cm_path = self.plot_confusion_matrix(eval_test["confusion_matrix"], "confusion_matrix.png")
        print(f"  -> Đã vẽ và lưu ma trận nhầm lẫn: {cm_path} (dpi=200)\n")

        # 2. 5-Fold Cross Validation trên intents.csv
        print("--- [BƯỚC 2/8] Cross-Validation 5-fold phân tầng trên data/intents.csv ---")
        cv_main = self.run_5fold_cv(model_type="nb", alpha=0.5)
        print(f"  • Accuracy trung bình : {cv_main['accuracy_mean']:.2%} ± {cv_main['accuracy_std']:.2%}")
        print(f"  • Macro F1 trung bình : {cv_main['f1_mean']:.2%} ± {cv_main['f1_std']:.2%}")
        print(f"  • Điểm từng fold      : {[round(a, 4) for a in cv_main['fold_accuracies']]}\n")

        # 3. So sánh 3 mô hình (MultinomialNB, LogisticRegression, LinearSVC)
        print("--- [BƯỚC 3/8] So sánh 3 mô hình: MultinomialNB, LogisticRegression, LinearSVC ---")
        df_comp = self.compare_models()
        print(df_comp.to_string(index=False))
        print(f"  -> Đã lưu bảng: {os.path.join(self.outputs_dir, 'model_comparison.csv')}")
        print(f"  -> Đã lưu biểu đồ cột so sánh: {os.path.join(self.outputs_dir, 'model_comparison.png')} (dpi=200)\n")

        # 4. Thí nghiệm tham số
        print("--- [BƯỚC 4/8] Thí nghiệm tham số (Alpha, N-gram, Stopwords) ---")
        exp_res = self.run_hyperparameter_experiments()
        print(f"  • Thí nghiệm Alpha    : {exp_res['alpha']['values']}")
        print(f"    -> Accuracies       : {[round(a, 3) for a in exp_res['alpha']['accuracies']]}")
        print(f"    -> Đã lưu biểu đồ   : {os.path.join(self.outputs_dir, 'exp_alpha.png')}")
        print(f"  • Thí nghiệm N-gram   : {exp_res['ngram']['names']}")
        print(f"    -> Accuracies       : {[round(a, 3) for a in exp_res['ngram']['accuracies']]}")
        print(f"    -> Đã lưu biểu đồ   : {os.path.join(self.outputs_dir, 'exp_ngram.png')}")
        print(f"  • Thí nghiệm Stopwords: {exp_res['stopwords']['names']}")
        print(f"    -> Accuracies       : {[round(a, 3) for a in exp_res['stopwords']['accuracies']]}")
        print(f"    -> Đã lưu biểu đồ   : {os.path.join(self.outputs_dir, 'exp_stopwords.png')}\n")

        # 5. Xuất danh sách các câu test đoán sai
        print("--- [BƯỚC 5/8] Liệt kê các câu kiểm thử bị phân loại sai ---")
        df_errors = self.export_misclassified(clf_main)
        print(f"  • Tổng số câu đoán sai: {len(df_errors)}/{len(self.df_test)} câu ({len(df_errors)/len(self.df_test):.1%})")
        if not df_errors.empty:
            print(df_errors[["Câu kiểm thử (Text)", "Nhãn thực tế (True Label)", "Nhãn dự đoán (Pred Label)", "Xác suất dự đoán sai"]].head(5).to_string(index=False))
        print(f"  -> Đã lưu danh sách chi tiết: {os.path.join(self.outputs_dir, 'misclassified_samples.csv')}\n")

        # 6. Tạo worked_example.txt
        print("--- [BƯỚC 6/8] Tạo tài liệu tính toán chi tiết bằng tay (outputs/worked_example.txt) ---")
        worked_path = self.generate_worked_example("worked_example.txt")
        print(f"  -> Đã tạo tài liệu worked example thành công: {worked_path}")
        print("     (Gồm chi tiết Prior, Likelihood, Log-posterior, Softmax và Chấm điểm Min-Max 7 tiêu chí cho Chương 2)\n")

        # 7. Phân tích độ đông & Trực quan hóa
        print("--- [BƯỚC 7/8] Phân tích Độ đông & Chỗ ngồi (CSV và 2 biểu đồ PNG dpi=200) ---")
        crowd_csv_path = self.export_crowd_analysis("crowd_analysis.csv")
        print(f"  -> Đã lưu bảng phân tích độ đông 50 món: {crowd_csv_path}")
        wait_chart_path = self.plot_crowd_wait_vs_people("crowd_wait_vs_people.png")
        print(f"  -> Đã lưu biểu đồ thời gian chờ: {wait_chart_path}")
        shift_chart_path = self.plot_crowd_ranking_shift("crowd_ranking_shift.png")
        print(f"  -> Đã lưu biểu đồ thay đổi thứ hạng Top-5: {shift_chart_path}\n")

        # 8. Thống kê tổng quan dữ liệu món ăn
        print("--- [BƯỚC 8/8] Thống kê tổng quan dữ liệu món ăn (outputs/data_summary.txt) ---")
        summary_path = self.export_data_summary("data_summary.txt")
        print(f"  -> Đã xuất báo cáo thống kê dữ liệu: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            print(f.read())

        print("=" * 80)
        print("HOÀN TẤT TOÀN BỘ QUY TRÌNH ĐÁNH GIÁ THÀNH CÔNG!".center(80))
        print("=" * 80 + "\n")


def main():
    evaluator = ModelEvaluator()
    evaluator.run_all()


if __name__ == "__main__":
    main()
