"""
Giao diện desktop PyQt6 cho NEU Lunch Chatbot.
Bố cục: Trái - khung chat & bộ lọc (ngân sách, số người, khung giờ, không cay, chỉ món chay, đủ chỗ ngồi);
Phải - QTabWidget (Bước xử lý | Biểu đồ | Tham số | Đánh giá | Quản lý dữ liệu).
Huấn luyện và đánh giá mô hình chạy trong QThread để không đơ giao diện.
Hỗ trợ tiếng Việt có dấu đầy đủ trên biểu đồ matplotlib và toàn bộ CSV đọc/ghi qua utf-8-sig.
"""

import os
import sys
import re
from typing import Dict, Any, Optional, List, Tuple

# ── Đường dẫn project ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── PyQt6 ──────────────────────────────────────────────────────────────────────
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QLineEdit, QGroupBox,
    QTabWidget, QScrollArea, QTableWidget, QTableWidgetItem,
    QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QHeaderView,
    QSizePolicy, QFrame, QMessageBox, QProgressBar, QFormLayout,
    QGridLayout, QAbstractItemView,
)

# ── Matplotlib nhúng trong Qt ──────────────────────────────────────────────────
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as fm
import numpy as np

# ── Tìm font tiếng Việt ────────────────────────────────────────────────────────
def _find_viet_font() -> str:
    """Tìm font hỗ trợ tiếng Việt trong hệ thống."""
    candidates = [
        "Segoe UI", "Arial", "Tahoma", "DejaVu Sans",
        "Times New Roman", "Verdana",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "DejaVu Sans"

VIET_FONT = _find_viet_font()
plt.rcParams["font.family"] = VIET_FONT
plt.rcParams["axes.unicode_minus"] = False

# ── Src modules ────────────────────────────────────────────────────────────────
from src.config import (
    DEFAULT_DISHES_PATH,
    DEFAULT_INTENTS_PATH,
    DEFAULT_TEST_INTENTS_PATH,
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_PROFILES_PATH,
    DEFAULT_CROWD_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUTS_DIR,
    TIME_SLOT_LABELS_VI,
    TIME_SLOT_SHORT_VI,
    load_crowd_config,
)
from src.chatbot import Chatbot
from src.classifier import NeedClassifier
from src.recommender import compute_crowd_metrics
from src.evaluate import ModelEvaluator
from src.input_validation import (
    parse_budget_k, parse_num_people, budget_k_to_vnd,
)
from src.feedback import (
    RATING_HELPFUL, RATING_WRONG,
    append_feedback, feedback_summary, format_feedback_summary,
)
from src.eval_insights import (
    LEVEL_GOOD, LEVEL_WARN,
    accuracy_level, format_accuracy_text, build_insights,
    make_signature, result_to_payload, save_eval_cache, load_eval_cache,
)

# ══════════════════════════════════════════════════════════════════════════════
# MÀU SẮC VÀ STYLE
# ══════════════════════════════════════════════════════════════════════════════
C_BG        = "#1E1E2E"
C_SURFACE   = "#2A2A3E"
C_CARD      = "#313145"
C_ACCENT    = "#7C6AF7"
C_ACCENT2   = "#50C5B7"
C_USER_BUB  = "#7C6AF7"
C_BOT_BUB   = "#313145"
C_TEXT      = "#E0DEF4"
C_TEXT_DIM  = "#908CAA"
C_SUCCESS   = "#3ECA8D"
C_WARNING   = "#F4B942"
C_ERROR     = "#F47F4C"
C_BORDER    = "#403E5C"

LABEL_NAMES = {
    "tiet_kiem":       "Tiết kiệm",
    "healthy":         "Healthy",
    "nhanh_gon":       "Nhanh gọn",
    "no_bung":         "No bụng",
    "ngon_doi_gio":    "Ngon đổi gió",
    "can_bang":        "Cân bằng",
    "chao_hoi":        "Chào hỏi",
    "cam_on_tam_biet": "Cảm ơn/Bye",
}

CRITERIA_ORDER = ["gia", "khoang_cach", "thoi_gian", "do_no", "dinh_duong", "do_ngon", "cho_ngoi"]

CRITERIA_NAMES = {
    "gia":         "Giá",
    "khoang_cach": "Khoảng cách",
    "thoi_gian":   "Thời gian",
    "do_no":       "Độ no",
    "dinh_duong":  "Dinh dưỡng",
    "do_ngon":     "Độ ngon",
    "cho_ngoi":    "Chỗ ngồi",
}

APP_STYLE = f"""
QWidget {{
    background: {C_BG};
    color: {C_TEXT};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: {C_ACCENT2};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}}
QTabWidget::pane {{
    border: 1px solid {C_BORDER};
    background: {C_SURFACE};
    border-radius: 6px;
}}
QTabBar::tab {{
    background: {C_CARD};
    color: {C_TEXT_DIM};
    padding: 8px 16px;
    margin: 2px 1px;
    border-radius: 6px 6px 0 0;
    font-size: 12px;
}}
QTabBar::tab:selected {{
    background: {C_ACCENT};
    color: #fff;
    font-weight: bold;
}}
QPushButton {{
    background: {C_ACCENT};
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: bold;
}}
QPushButton:hover {{ background: #6A5ADF; }}
QPushButton:pressed {{ background: #5A4ACF; }}
QPushButton:disabled {{ background: {C_BORDER}; color: {C_TEXT_DIM}; }}
QLineEdit, QTextEdit, QComboBox {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 5px 10px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
}}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border-color: {C_ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {C_BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {C_ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {C_ACCENT}; border-radius: 2px; }}
QScrollBar:vertical {{
    background: {C_SURFACE};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {C_BORDER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {C_ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTableWidget {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    gridline-color: {C_BORDER};
    alternate-background-color: {C_SURFACE};
}}
QHeaderView::section {{
    background: {C_ACCENT};
    color: #fff;
    padding: 6px;
    border: none;
    font-weight: bold;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid {C_BORDER};
}}
QCheckBox::indicator:checked {{
    background: {C_ACCENT};
    border-color: {C_ACCENT};
}}
QDoubleSpinBox, QSpinBox {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    color: {C_TEXT};
}}
QProgressBar {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    text-align: center;
    color: {C_TEXT};
}}
QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 5px; }}
"""


# ══════════════════════════════════════════════════════════════════════════════
# QTHREAD HUẤN LUYỆN & ĐÁNH GIÁ
# ══════════════════════════════════════════════════════════════════════════════
class TrainingThread(QThread):
    """Huấn luyện mô hình trong luồng nền để GUI không bị đơ."""
    finished = pyqtSignal(dict)   # {success, message, classifier}
    progress = pyqtSignal(str)

    def __init__(
        self,
        alpha: float,
        model_type: str,
        intents_path: str,
        model_save_path: str,
        ngram_range: Tuple[int, int] = (1, 2),
        remove_stopwords: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.alpha = alpha
        self.model_type = model_type
        self.intents_path = intents_path
        self.model_save_path = model_save_path
        self.ngram_range = ngram_range
        self.remove_stopwords = remove_stopwords

    def run(self):
        try:
            self.progress.emit("⏳ Đang tiền xử lý dữ liệu...")
            clf = NeedClassifier(
                model_type=self.model_type,
                alpha=self.alpha,
                ngram_range=self.ngram_range,
                remove_stopwords=self.remove_stopwords,
            )
            self.progress.emit("⏳ Đang fit TF-IDF và mô hình Naive Bayes...")
            summary = clf.train(self.intents_path)
            self.progress.emit("💾 Đang lưu mô hình vào outputs/...")
            clf.save(self.model_save_path)
            self.progress.emit("✅ Hoàn tất!")
            self.finished.emit({
                "success": True,
                "message": (
                    f"Huấn luyện xong! {summary['num_samples']} mẫu, "
                    f"{summary['num_classes']} nhãn, α={self.alpha:g}, "
                    f"ngram={self.ngram_range}, bỏ SW={'Bật' if self.remove_stopwords else 'Tắt'}"
                ),
                "classifier": clf,
            })
        except Exception as e:
            self.finished.emit({"success": False, "message": str(e), "classifier": None})


class EvaluationThread(QThread):
    """Chạy đánh giá mô hình trên data/test_intents.csv trong luồng nền."""
    finished = pyqtSignal(dict)   # {success, payload, message}

    def __init__(
        self,
        train_path: str,
        test_path: str,
        outputs_dir: str,
        alpha: float,
        ngram_range: Tuple[int, int],
        remove_stopwords: bool,
        classifier: Optional[NeedClassifier] = None,
        use_custom_params: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.train_path = train_path
        self.test_path = test_path
        self.outputs_dir = outputs_dir
        self.alpha = alpha
        self.ngram_range = ngram_range
        self.remove_stopwords = remove_stopwords
        self.classifier = classifier
        self.use_custom_params = use_custom_params

    def run(self):
        try:
            evaluator = ModelEvaluator(
                train_csv_path=self.train_path,
                test_csv_path=self.test_path,
                outputs_dir=self.outputs_dir,
            )
            clf = self.classifier
            if (
                clf is None
                or clf.pipeline is None
                or self.use_custom_params
                or abs(float(clf.alpha) - float(self.alpha)) > 1e-9
                or tuple(clf.ngram_range) != tuple(self.ngram_range)
                or bool(clf.remove_stopwords) != bool(self.remove_stopwords)
            ):
                clf = NeedClassifier(
                    model_type="nb",
                    alpha=self.alpha,
                    ngram_range=self.ngram_range,
                    remove_stopwords=self.remove_stopwords,
                )
                clf.train(self.train_path)

            result = evaluator.evaluate_on_test(clf)
            sig = make_signature(
                alpha=clf.alpha,
                ngram_range=clf.ngram_range,
                remove_stopwords=clf.remove_stopwords,
                train_path=self.train_path,
                test_path=self.test_path,
                model_type=clf.model_type,
            )
            payload = result_to_payload(result, evaluator.labels, sig)
            cache_path = os.path.join(self.outputs_dir, "eval_result.json")
            save_eval_cache(cache_path, payload)
            self.finished.emit({"success": True, "payload": payload, "message": ""})
        except Exception as e:
            self.finished.emit({"success": False, "payload": None, "message": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# WIDGET BONG BÓNG CHAT (KÈM PHẢN HỒI NGƯỜI DÙNG CHO TIN NHẮN BOT)
# ══════════════════════════════════════════════════════════════════════════════
class ChatBubble(QWidget):
    """Widget hiển thị một tin nhắn trong khung chat dưới dạng bong bóng, kèm nút phản hồi cho bot."""

    feedback_submitted = pyqtSignal(dict)

    def __init__(
        self,
        text: str,
        is_user: bool,
        feedback_meta: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.is_user = is_user
        self.feedback_meta = feedback_meta
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 3, 6, 3)

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setOpenExternalLinks(False)
        lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        lbl.setMaximumWidth(560)

        bub_color = C_USER_BUB if is_user else C_BOT_BUB
        txt_color = "#FFFFFF"
        radius_tl = "4px" if is_user else "18px"
        radius_tr = "18px" if is_user else "4px"

        lbl.setStyleSheet(f"""
            QLabel {{
                background: {bub_color};
                color: {txt_color};
                border-radius: 16px;
                border-top-left-radius: {radius_tl};
                border-top-right-radius: {radius_tr};
                padding: 10px 14px;
                font-size: 13px;
                line-height: 1.5;
            }}
        """)

        if is_user:
            outer.addStretch()
            outer.addWidget(lbl)
        else:
            av = QLabel("🤖")
            av.setFixedSize(32, 32)
            av.setAlignment(Qt.AlignmentFlag.AlignCenter)
            av.setStyleSheet(f"background:{C_CARD}; border-radius:16px; font-size:18px;")
            outer.addWidget(av, 0, Qt.AlignmentFlag.AlignTop)

            col = QVBoxLayout()
            col.setSpacing(4)
            col.setContentsMargins(0, 0, 0, 0)
            col.addWidget(lbl)

            if self.feedback_meta:
                self._build_feedback_controls(col)

            outer.addLayout(col)
            outer.addStretch()

    def _build_feedback_controls(self, parent_layout: QVBoxLayout) -> None:
        """Tạo hàng nút 👍 Hữu ích / 👎 Chưa đúng ý và form góp ý ngắn ngay dưới bong bóng bot."""
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.setContentsMargins(4, 0, 0, 0)

        btn_style = (
            f"QPushButton {{ background:{C_SURFACE}; color:{C_TEXT_DIM}; border:1px solid {C_BORDER}; "
            f"border-radius:6px; padding:3px 10px; font-size:11px; font-weight:normal; }}"
            f"QPushButton:hover {{ color:{C_TEXT}; border-color:{C_ACCENT}; }}"
        )
        self.btn_helpful = QPushButton("👍 Hữu ích")
        self.btn_helpful.setStyleSheet(btn_style)
        self.btn_wrong = QPushButton("👎 Chưa đúng ý")
        self.btn_wrong.setStyleSheet(btn_style)
        self.lbl_fb_done = QLabel("")
        self.lbl_fb_done.setStyleSheet(f"color:{C_SUCCESS}; font-size:11px;")
        self.lbl_fb_done.setVisible(False)

        btn_row.addWidget(self.btn_helpful)
        btn_row.addWidget(self.btn_wrong)
        btn_row.addWidget(self.lbl_fb_done)
        btn_row.addStretch()
        parent_layout.addLayout(btn_row)

        self.wrong_panel = QFrame()
        self.wrong_panel.setVisible(False)
        self.wrong_panel.setStyleSheet(
            f"QFrame {{ background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:8px; }}"
        )
        wp_lay = QVBoxLayout(self.wrong_panel)
        wp_lay.setContentsMargins(8, 6, 8, 6)
        wp_lay.setSpacing(5)

        r1 = QHBoxLayout()
        r1.setSpacing(6)
        lbl_true = QLabel("Nhãn đúng:")
        lbl_true.setStyleSheet(f"color:{C_TEXT_DIM}; font-size:11px; border:none;")
        self.cmb_correct_label = QComboBox()
        self.cmb_correct_label.setStyleSheet("font-size:11px; padding:2px 6px;")
        for key, name in LABEL_NAMES.items():
            self.cmb_correct_label.addItem(f"{name} ({key})", key)
        pred_lbl = (self.feedback_meta or {}).get("predicted_label", "")
        idx = self.cmb_correct_label.findData(pred_lbl)
        if idx >= 0:
            self.cmb_correct_label.setCurrentIndex(idx)
        r1.addWidget(lbl_true)
        r1.addWidget(self.cmb_correct_label, 1)
        wp_lay.addLayout(r1)

        r2 = QHBoxLayout()
        r2.setSpacing(6)
        self.in_fb_note = QLineEdit()
        self.in_fb_note.setPlaceholderText("Ghi chú ngắn (tùy chọn, ví dụ: muốn tìm món chay rẻ)...")
        self.in_fb_note.setStyleSheet("font-size:11px; padding:3px 6px;")
        self.btn_submit_wrong = QPushButton("Gửi")
        self.btn_submit_wrong.setStyleSheet(
            f"background:{C_ACCENT}; color:#fff; border-radius:6px; padding:3px 10px; font-size:11px;"
        )
        r2.addWidget(self.in_fb_note, 1)
        r2.addWidget(self.btn_submit_wrong)
        wp_lay.addLayout(r2)

        parent_layout.addWidget(self.wrong_panel)

        self.btn_helpful.clicked.connect(self._on_helpful_clicked)
        self.btn_wrong.clicked.connect(self._on_wrong_clicked)
        self.btn_submit_wrong.clicked.connect(self._on_submit_wrong)

    def _on_helpful_clicked(self) -> None:
        self._emit_feedback(RATING_HELPFUL, correct_label="", note="")

    def _on_wrong_clicked(self) -> None:
        self.wrong_panel.setVisible(not self.wrong_panel.isVisible())

    def _on_submit_wrong(self) -> None:
        corr = str(self.cmb_correct_label.currentData() or "")
        note = self.in_fb_note.text().strip()
        self.wrong_panel.setVisible(False)
        self._emit_feedback(RATING_WRONG, correct_label=corr, note=note)

    def _emit_feedback(self, rating: str, correct_label: str, note: str) -> None:
        meta = dict(self.feedback_meta or {})
        meta["rating"] = rating
        meta["correct_label"] = correct_label
        meta["note"] = note
        self.btn_helpful.setEnabled(False)
        self.btn_wrong.setEnabled(False)
        self.lbl_fb_done.setText("✅ Đã ghi nhận phản hồi")
        self.lbl_fb_done.setVisible(True)
        self.feedback_submitted.emit(meta)


# ══════════════════════════════════════════════════════════════════════════════
# CANVAS MATPLOTLIB
# ══════════════════════════════════════════════════════════════════════════════
class MplCanvas(FigureCanvas):
    """Matplotlib canvas với nền tối."""
    def __init__(self, figsize=(7, 4), parent=None):
        self.fig = Figure(figsize=figsize, facecolor=C_CARD)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def clear(self):
        self.fig.clf()
        self.draw()


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(C_BORDER)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_TEXT_DIM, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_TEXT_DIM, fontsize=9)
    ax.title.set_fontfamily(VIET_FONT)


# ══════════════════════════════════════════════════════════════════════════════
# CỬA SỔ CHÍNH
# ══════════════════════════════════════════════════════════════════════════════
class LunchChatbotGUI(QMainWindow):
    """Cửa sổ chính của NEU Lunch Chatbot."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🍜 NEU Lunch Chatbot – Trưa Nay Ăn Gì?")
        self.resize(1340, 840)
        self.setMinimumSize(960, 640)

        # Trạng thái bot & luồng nền
        self.bot: Optional[Chatbot] = None
        self.is_trained = False
        self.train_thread: Optional[TrainingThread] = None
        self.eval_thread: Optional[EvaluationThread] = None
        self._last_response: Optional[Dict[str, Any]] = None
        self._eval_loaded_once = False
        self._trained_model_params: Dict[str, Any] = {
            "alpha": 0.5,
            "ngram_range": (1, 2),
            "remove_stopwords": True,
        }

        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._init_bot_async()

    # ── Khởi tạo bot không chặn UI ────────────────────────────────────────────
    def _init_bot_async(self):
        QTimer.singleShot(150, self._try_load_bot)

    def _new_bot(self) -> Chatbot:
        """Tạo Chatbot mới với đường dẫn cấu hình trung tâm (data/dishes_v2.csv)."""
        return Chatbot(
            model_path=DEFAULT_MODEL_PATH,
            dishes_path=DEFAULT_DISHES_PATH,
            profiles_path=DEFAULT_PROFILES_PATH,
            crowd_config_path=DEFAULT_CROWD_CONFIG_PATH,
            confidence_threshold=self.spin_threshold.value(),
            default_top_k=self.spin_topk.value(),
        )

    def _try_load_bot(self):
        try:
            self.bot = self._new_bot()
            if self.bot.classifier.pipeline is None:
                self.bot = None
                raise RuntimeError("chưa có mô hình đã huấn luyện và không tìm thấy data/intents.csv")
            self.is_trained = True
            self._sync_ui_from_classifier(self.bot.classifier)
            self._status("✅ Mô hình đã được tải. Sẵn sàng tư vấn!")
            self._append_bot_message(
                "Xin chào! Mình là <b>NEU Lunch Bot</b> 🍜<br>"
                "Hãy cho mình biết bạn muốn ăn gì trưa nay nhé!<br>"
                "<i>(Ví dụ: \"đi 4 người lúc 12h ăn gì rẻ dưới 35k không cay\" hoặc \"tìm chỗ nào vắng ăn chay\")</i>"
            )
            self._ensure_eval_populated()
        except Exception as e:
            self._status(f"⚠️ Chưa có mô hình: {e}. Nhấn 'Huấn luyện lại' để bắt đầu.")
            self._append_bot_message(
                "⚠️ Chưa có mô hình huấn luyện.<br>"
                "Vui lòng nhấn <b>Huấn luyện lại</b> ở tab <i>Tham số</i> để bắt đầu."
            )

    def _sync_ui_from_classifier(self, clf: NeedClassifier) -> None:
        """Đồng bộ tham số hiển thị ở tab Tham số với mô hình vừa nạp/huấn luyện."""
        self._trained_model_params = {
            "alpha": round(float(getattr(clf, "alpha", 0.5)), 4),
            "ngram_range": tuple(getattr(clf, "ngram_range", (1, 2))),
            "remove_stopwords": bool(getattr(clf, "remove_stopwords", True)),
        }
        self.spin_alpha.blockSignals(True)
        self.spin_alpha.setValue(self._trained_model_params["alpha"])
        self.spin_alpha.blockSignals(False)

        self.chk_remove_sw.blockSignals(True)
        self.chk_remove_sw.setChecked(self._trained_model_params["remove_stopwords"])
        self.chk_remove_sw.blockSignals(False)

        idx = self.cmb_ngram.findData(self._trained_model_params["ngram_range"])
        if idx >= 0:
            self.cmb_ngram.blockSignals(True)
            self.cmb_ngram.setCurrentIndex(idx)
            self.cmb_ngram.blockSignals(False)

        self.lbl_model_dirty.setVisible(False)

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {C_BORDER}; }}")
        root.addWidget(splitter)

        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([475, 845])

        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(f"background:{C_SURFACE}; color:{C_TEXT_DIM};")
        self.status_bar.showMessage("Đang khởi động...")

    # ── PANEL TRÁI: CHAT & BỘ LỌC ────────────────────────────────────────────
    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(400)
        vbox = QVBoxLayout(panel)
        vbox.setSpacing(8)
        vbox.setContentsMargins(0, 0, 0, 0)

        # Title
        title = QLabel("🍜  NEU Lunch Chatbot")
        title.setStyleSheet(f"font-size:16px; font-weight:bold; color:{C_ACCENT}; padding:4px 0;")
        vbox.addWidget(title)

        # Chat area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_scroll.setStyleSheet(f"border:1px solid {C_BORDER}; border-radius:10px; background:{C_SURFACE};")

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setSpacing(4)
        self.chat_layout.setContentsMargins(8, 8, 8, 8)
        self.chat_layout.addStretch()
        self.chat_scroll.setWidget(self.chat_container)
        vbox.addWidget(self.chat_scroll, 1)

        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet(f"background:{C_SURFACE}; border-radius:10px; border:1px solid {C_BORDER};")
        input_vbox = QVBoxLayout(input_frame)
        input_vbox.setContentsMargins(8, 8, 8, 8)
        input_vbox.setSpacing(6)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Nhập câu hỏi... (Enter để gửi)")
        self.input_box.setMinimumHeight(38)
        self.input_box.returnPressed.connect(self._on_send)
        input_vbox.addWidget(self.input_box)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.btn_send = QPushButton("▶  Gửi")
        self.btn_send.setMinimumHeight(36)
        self.btn_send.clicked.connect(self._on_send)
        btn_row.addWidget(self.btn_send, 2)

        self.btn_clear = QPushButton("🗑 Xóa hội thoại")
        self.btn_clear.setToolTip("Xóa toàn bộ lịch sử tin nhắn trong khung hội thoại")
        self.btn_clear.setMinimumHeight(36)
        self.btn_clear.setStyleSheet(
            f"QPushButton {{ background:{C_CARD}; color:{C_TEXT}; border:1px solid {C_BORDER}; "
            f"border-radius:8px; padding:6px 12px; font-size:12px; font-weight:bold; }}"
            f"QPushButton:hover {{ border-color:{C_ACCENT}; color:#fff; }}"
        )
        self.btn_clear.clicked.connect(self._clear_chat)
        btn_row.addWidget(self.btn_clear, 1)

        input_vbox.addLayout(btn_row)
        vbox.addWidget(input_frame)

        # ── Bộ lọc: Ngân sách mỗi người, Số người đi ăn, Khung giờ đi ăn, các ô tích cứng ──
        filter_frame = QFrame()
        filter_frame.setStyleSheet(f"background:{C_SURFACE}; border-radius:10px; border:1px solid {C_BORDER};")
        filter_vbox = QVBoxLayout(filter_frame)
        filter_vbox.setContentsMargins(10, 8, 10, 8)
        filter_vbox.setSpacing(5)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        # (a) Ngân sách mỗi người (nghìn đồng)
        lbl_budget = QLabel("Ngân sách mỗi người (nghìn đồng):")
        lbl_budget.setStyleSheet(f"color:{C_TEXT}; border:none;")
        self.in_budget = QLineEdit()
        self.in_budget.setPlaceholderText("20 - 200")
        self.in_budget.setToolTip("Nhập số từ 20 đến 200 (ví dụ 50 nghĩa là 50.000đ/người). Để trống = không giới hạn.")
        self.in_budget.setValidator(QIntValidator(0, 999999, self))
        self.in_budget.setFixedWidth(82)
        lbl_budget_unit = QLabel("k/người")
        lbl_budget_unit.setStyleSheet(f"color:{C_TEXT_DIM}; border:none;")

        budget_input_row = QHBoxLayout()
        budget_input_row.setSpacing(4)
        budget_input_row.setContentsMargins(0, 0, 0, 0)
        budget_input_row.addWidget(self.in_budget)
        budget_input_row.addWidget(lbl_budget_unit)
        budget_input_row.addStretch()

        grid.addWidget(lbl_budget, 0, 0)
        grid.addLayout(budget_input_row, 0, 1)

        self.lbl_budget_err = QLabel("")
        self.lbl_budget_err.setWordWrap(True)
        self.lbl_budget_err.setVisible(False)
        self.lbl_budget_err.setStyleSheet(f"color:{C_ERROR}; font-size:11px; border:none;")
        grid.addWidget(self.lbl_budget_err, 1, 0, 1, 2)

        # (b) Số người đi ăn + Khung giờ đi ăn (cùng cụm, không cắt chữ)
        lbl_people = QLabel("Số người đi ăn:")
        lbl_people.setStyleSheet(f"color:{C_TEXT}; border:none;")
        self.in_people = QLineEdit("1")
        self.in_people.setPlaceholderText("1 - 10")
        self.in_people.setToolTip("Số nguyên từ 1 đến 10 (mặc định 1 người).")
        self.in_people.setValidator(QIntValidator(0, 999999, self))
        self.in_people.setFixedWidth(64)
        lbl_people_unit = QLabel("người")
        lbl_people_unit.setStyleSheet(f"color:{C_TEXT_DIM}; border:none;")

        people_input_row = QHBoxLayout()
        people_input_row.setSpacing(4)
        people_input_row.setContentsMargins(0, 0, 0, 0)
        people_input_row.addWidget(self.in_people)
        people_input_row.addWidget(lbl_people_unit)
        people_input_row.addStretch()

        grid.addWidget(lbl_people, 2, 0)
        grid.addLayout(people_input_row, 2, 1)

        self.lbl_people_err = QLabel("")
        self.lbl_people_err.setWordWrap(True)
        self.lbl_people_err.setVisible(False)
        self.lbl_people_err.setStyleSheet(f"color:{C_ERROR}; font-size:11px; border:none;")
        grid.addWidget(self.lbl_people_err, 3, 0, 1, 2)

        lbl_slot = QLabel("Khung giờ đi ăn:")
        lbl_slot.setStyleSheet(f"color:{C_TEXT}; border:none;")
        self.cmb_time_slot = QComboBox()
        self.cmb_time_slot.addItem("Tự động theo đồng hồ", "tu_dong")
        self.cmb_time_slot.addItem("Cao điểm (11h30-13h)", "cao_diem")
        self.cmb_time_slot.addItem("Bình thường (11h-11h30, 13h-14h)", "binh_thuong")
        self.cmb_time_slot.addItem("Vắng (khung giờ khác)", "vang")
        self.cmb_time_slot.setToolTip("Chọn khung giờ đi ăn để ước tính độ đông, thời gian chờ và số chỗ trống.")
        grid.addWidget(lbl_slot, 4, 0)
        grid.addWidget(self.cmb_time_slot, 4, 1)

        filter_vbox.addLayout(grid)

        # Hàng bộ lọc cứng: Không cay + Chỉ món chay + Chỉ quán đủ chỗ cho cả nhóm
        chk_row1 = QHBoxLayout()
        chk_row1.setSpacing(14)
        chk_row1.setContentsMargins(0, 2, 0, 0)
        self.chk_no_spicy = QCheckBox("Không cay")
        self.chk_no_spicy.setStyleSheet(f"color:{C_TEXT}; border:none;")
        self.chk_veg = QCheckBox("Chỉ món chay")
        self.chk_veg.setStyleSheet(f"color:{C_TEXT}; border:none;")
        self.chk_enough_seats = QCheckBox("Chỉ quán đủ chỗ cho cả nhóm")
        self.chk_enough_seats.setToolTip("Lọc bỏ các quán có số chỗ trống ước tính nhỏ hơn số người đi ăn")
        self.chk_enough_seats.setStyleSheet(f"color:{C_TEXT}; border:none;")
        chk_row1.addWidget(self.chk_no_spicy)
        chk_row1.addWidget(self.chk_veg)
        chk_row1.addWidget(self.chk_enough_seats)
        chk_row1.addStretch()
        filter_vbox.addLayout(chk_row1)

        self.in_budget.textChanged.connect(self._validate_filter_inputs)
        self.in_people.textChanged.connect(self._validate_filter_inputs)

        vbox.addWidget(filter_frame)
        return panel

    def _validate_filter_inputs(self) -> bool:
        """
        Kiểm tra hợp lệ trực tiếp (inline) cho ô Ngân sách mỗi người và Số người đi ăn.
        Ngoài khoảng hoặc không phải số nguyên -> viền đỏ + dòng lỗi đỏ ngay dưới ô + khóa nút Gửi.
        """
        _, budget_err = parse_budget_k(self.in_budget.text())
        _, people_err = parse_num_people(self.in_people.text())

        normal_style = f"background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; padding:4px 8px; color:{C_TEXT};"
        error_style = f"background:{C_CARD}; border:1.5px solid {C_ERROR}; border-radius:8px; padding:4px 8px; color:{C_TEXT};"

        if budget_err:
            self.in_budget.setStyleSheet(error_style)
            self.lbl_budget_err.setText(f"⚠️ {budget_err}")
            self.lbl_budget_err.setVisible(True)
        else:
            self.in_budget.setStyleSheet(normal_style)
            self.lbl_budget_err.setText("")
            self.lbl_budget_err.setVisible(False)

        if people_err:
            self.in_people.setStyleSheet(error_style)
            self.lbl_people_err.setText(f"⚠️ {people_err}")
            self.lbl_people_err.setVisible(True)
        else:
            self.in_people.setStyleSheet(normal_style)
            self.lbl_people_err.setText("")
            self.lbl_people_err.setVisible(False)

        valid = (budget_err is None) and (people_err is None)
        self.btn_send.setEnabled(valid)
        return valid

    # ── PANEL PHẢI: TABS ──────────────────────────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        self.tabs = QTabWidget()
        vbox.addWidget(self.tabs)

        self.tabs.addTab(self._build_steps_tab(),  "📊 Bước xử lý")
        self.tabs.addTab(self._build_charts_tab(), "📈 Biểu đồ")
        self.tabs.addTab(self._build_params_tab(), "⚙️ Tham số")
        self.tabs.addTab(self._build_eval_tab(),   "🎯 Đánh giá")
        self.tabs.addTab(self._build_data_tab(),   "🗂️ Quản lý dữ liệu")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        return panel

    def _on_tab_changed(self, index: int) -> None:
        """Khi mở tab Đánh giá (index 3), tự động nạp cache hoặc chạy đánh giá nếu chưa có."""
        if index == 3:
            self._refresh_feedback_summary_label()
            self._ensure_eval_populated()

    # ── TAB: QUẢN LÝ DỮ LIỆU ─────────────────────────────────────────────────
    def _build_data_tab(self) -> QWidget:
        from gui.data_tab import DataManagerTab
        self.data_tab = DataManagerTab(
            dishes_path=DEFAULT_DISHES_PATH,
            intents_path=DEFAULT_INTENTS_PATH,
        )
        self.data_tab.dishes_changed.connect(self._on_dishes_changed)
        self.data_tab.intents_changed.connect(
            lambda: self._status("ℹ️ Đã thêm câu mẫu mới – hãy huấn luyện lại để mô hình học dữ liệu mới.")
        )
        self.data_tab.retrain_requested.connect(self._on_train)
        return self.data_tab

    def _on_dishes_changed(self):
        """Nạp lại danh sách món cho bộ gợi ý ngay sau khi sửa dishes_v2.csv (không cần huấn luyện lại)."""
        if self.bot is not None:
            try:
                self.bot.recommender.load_data()
                self._status("✅ Đã cập nhật danh sách món ăn cho bộ gợi ý.")
            except Exception as e:
                QMessageBox.warning(self, "Lỗi nạp dữ liệu món", str(e))

    # ── TAB: BƯỚC XỬ LÝ ──────────────────────────────────────────────────────
    def _build_steps_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{C_SURFACE}; border:none;")

        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setSpacing(10)
        vbox.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(inner)

        def _make_step_group(title: str) -> tuple:
            grp = QGroupBox(title)
            lay = QVBoxLayout(grp)
            lay.setSpacing(4)
            lay.setContentsMargins(8, 12, 8, 8)
            return grp, lay

        # 1. Tiền xử lý
        grp1, lay1 = _make_step_group("1️⃣  Tiền xử lý chuỗi")
        self.lbl_orig   = QLabel("—")
        self.lbl_norm   = QLabel("—")
        self.lbl_tokens = QLabel("—")
        self.lbl_no_sw  = QLabel("—")
        for lbl, text in [
            (self.lbl_orig,   "Câu gốc:"),
            (self.lbl_norm,   "Chuẩn hóa:"),
            (self.lbl_tokens, "Tokens:"),
            (self.lbl_no_sw,  "Tokens (bỏ SW):"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b style='color:{C_ACCENT2}'>{text}</b>"))
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{C_TEXT}; background:{C_CARD}; border-radius:4px; padding:4px 8px;")
            row.addWidget(lbl, 1)
            lay1.addLayout(row)
        vbox.addWidget(grp1)

        # 2. TF-IDF
        grp2, lay2 = _make_step_group("2️⃣  Từ khóa TF-IDF nổi bật")
        self.tbl_tfidf = self._make_table(["Từ / Cụm từ", "TF-IDF"])
        self.tbl_tfidf.setFixedHeight(140)
        lay2.addWidget(self.tbl_tfidf)
        vbox.addWidget(grp2)

        # 3. Xác suất nhãn
        grp3, lay3 = _make_step_group("3️⃣  Xác suất 8 nhãn ý định")
        self.tbl_probs = self._make_table(["Nhãn", "Xác suất"])
        self.tbl_probs.setFixedHeight(200)
        lay3.addWidget(self.tbl_probs)
        vbox.addWidget(grp3)

        # 4. Ràng buộc & lọc
        grp4, lay4 = _make_step_group("4️⃣  Ràng buộc trích xuất & các bước lọc")
        self.lbl_max_price    = QLabel("—")
        self.lbl_group_budget = QLabel("—")
        self.lbl_excl_spicy   = QLabel("—")
        self.lbl_veg_only     = QLabel("—")
        self.lbl_excl_types   = QLabel("—")
        self.lbl_filter_steps = QLabel("—")
        for lbl, txt in [
            (self.lbl_max_price,    "Ngân sách/người:"),
            (self.lbl_group_budget, "Số người & tổng:"),
            (self.lbl_excl_spicy,   "Loại trừ cay:"),
            (self.lbl_veg_only,     "Chỉ món chay:"),
            (self.lbl_excl_types,   "Loại trừ món:"),
            (self.lbl_filter_steps, "Số món qua lọc:"),
        ]:
            row = QHBoxLayout()
            title_lbl = QLabel(f"<b style='color:{C_ACCENT2}'>{txt}</b>")
            title_lbl.setFixedWidth(130)
            row.addWidget(title_lbl)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{C_TEXT}; background:{C_CARD}; border-radius:4px; padding:4px 8px;")
            row.addWidget(lbl, 1)
            lay4.addLayout(row)
        vbox.addWidget(grp4)

        # 5. Độ đông và nhóm (MỚI)
        grp_crowd, lay_crowd = _make_step_group("5️⃣  Độ đông và nhóm")
        self.lbl_crowd_people = QLabel("—")
        self.lbl_crowd_slot   = QLabel("—")
        self.lbl_crowd_a      = QLabel("—")
        for lbl, txt in [
            (self.lbl_crowd_people, "Số người đi ăn:"),
            (self.lbl_crowd_slot,   "Khung giờ đi ăn:"),
            (self.lbl_crowd_a,      "Hệ số phạt độ đông (a):"),
        ]:
            row = QHBoxLayout()
            title_lbl = QLabel(f"<b style='color:{C_ACCENT2}'>{txt}</b>")
            title_lbl.setFixedWidth(155)
            row.addWidget(title_lbl)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{C_TEXT}; background:{C_CARD}; border-radius:4px; padding:4px 8px;")
            row.addWidget(lbl, 1)
            lay_crowd.addLayout(row)

        self.tbl_crowd_dishes = self._make_table(
            ["Món", "Độ đông hiệu dụng", "TG chờ (phút)", "Chỗ trống", "Chỗ ngồi nhóm", "Cảnh báo"]
        )
        self.tbl_crowd_dishes.setFixedHeight(145)
        lay_crowd.addWidget(self.tbl_crowd_dishes)
        vbox.addWidget(grp_crowd)

        # 6. Trọng số 7 tiêu chí
        grp5, lay5 = _make_step_group("6️⃣  Trọng số 7 tiêu chí đang dùng")
        self.tbl_weights = self._make_table(["Tiêu chí", "Trọng số"])
        self.tbl_weights.setFixedHeight(210)
        lay5.addWidget(self.tbl_weights)
        vbox.addWidget(grp5)

        # 7. Điểm đóng góp 7 tiêu chí từng món
        grp6, lay6 = _make_step_group("7️⃣  Điểm đóng góp 7 tiêu chí theo món")
        self.tbl_scores = self._make_table(
            ["Món", "Tổng", "Giá", "Khoảng cách", "Thời gian", "Độ no", "Dinh dưỡng", "Độ ngon", "Chỗ ngồi"]
        )
        self.tbl_scores.setFixedHeight(155)
        lay6.addWidget(self.tbl_scores)
        vbox.addWidget(grp6)

        vbox.addStretch()
        return scroll

    # ── TAB: BIỂU ĐỒ ─────────────────────────────────────────────────────────
    def _build_charts_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{C_SURFACE}; border:none;")

        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.setSpacing(8)
        vbox.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(widget)

        title = QLabel("📈  Trực quan hoá kết quả gợi ý (7 tiêu chí, thời gian chờ & chỗ trống theo nhóm)")
        title.setStyleSheet(f"font-size:14px; font-weight:bold; color:{C_ACCENT};")
        vbox.addWidget(title)

        grp_top = QGroupBox("1. Điểm đóng góp của 7 tiêu chí cho mỗi món (Stacked Bar)")
        top_lay = QVBoxLayout(grp_top)
        self.canvas_stacked = MplCanvas(figsize=(7, 3.0))
        self.canvas_stacked.setMinimumHeight(230)
        top_lay.addWidget(self.canvas_stacked)
        vbox.addWidget(grp_top)

        row_mid = QHBoxLayout()
        row_mid.setSpacing(8)

        grp_wait = QGroupBox("2. Thời gian chờ ước tính theo số người (1-10)")
        wait_lay = QVBoxLayout(grp_wait)
        self.canvas_wait_curve = MplCanvas(figsize=(4.8, 2.8))
        self.canvas_wait_curve.setMinimumHeight(220)
        wait_lay.addWidget(self.canvas_wait_curve)
        row_mid.addWidget(grp_wait, 1)

        grp_seats = QGroupBox("3. Chỗ trống ước tính so với số người đi ăn")
        seats_lay = QVBoxLayout(grp_seats)
        self.canvas_seats = MplCanvas(figsize=(4.8, 2.8))
        self.canvas_seats.setMinimumHeight(220)
        seats_lay.addWidget(self.canvas_seats)
        row_mid.addWidget(grp_seats, 1)

        vbox.addLayout(row_mid)

        grp_bot = QGroupBox("4. Xác suất phân loại ý định (Intent Probability)")
        bot_lay = QVBoxLayout(grp_bot)
        self.canvas_probs = MplCanvas(figsize=(7, 2.6))
        self.canvas_probs.setMinimumHeight(200)
        bot_lay.addWidget(self.canvas_probs)
        vbox.addWidget(grp_bot)

        return scroll

    # ── TAB: THAM SỐ ─────────────────────────────────────────────────────────
    def _build_params_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{C_SURFACE}; border:none;")
        inner = QWidget()
        vbox = QVBoxLayout(inner)
        vbox.setSpacing(12)
        vbox.setContentsMargins(12, 12, 12, 12)
        scroll.setWidget(inner)

        # ─── Tham số mô hình ───────────────────────────────────────────────
        grp_model = QGroupBox("🤖  Tham số mô hình phân loại")
        form = QFormLayout(grp_model)
        form.setSpacing(10)
        form.setContentsMargins(12, 16, 12, 12)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.01, 10.0)
        self.spin_alpha.setSingleStep(0.1)
        self.spin_alpha.setValue(0.5)
        self.spin_alpha.setToolTip("Hệ số làm mịn Laplace α cho Naive Bayes (mặc định 0.5)")
        form.addRow("Alpha (Naive Bayes):", self.spin_alpha)

        self.cmb_ngram = QComboBox()
        self.cmb_ngram.addItem("Unigram + Bigram (1, 2) — Mặc định", (1, 2))
        self.cmb_ngram.addItem("Chỉ Unigram (1, 1)", (1, 1))
        self.cmb_ngram.addItem("Unigram đến Trigram (1, 3)", (1, 3))
        self.cmb_ngram.setToolTip("Phạm vi cụm từ (N-gram) trích xuất bởi TF-IDF")
        form.addRow("Cấu hình N-gram:", self.cmb_ngram)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.05, 1.0)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.30)
        self.spin_threshold.setToolTip("Ngưỡng tin cậy. Dưới ngưỡng → fallback 'cân bằng'")
        form.addRow("Ngưỡng tin cậy:", self.spin_threshold)

        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 10)
        self.spin_topk.setValue(3)
        form.addRow("Top-K gợi ý:", self.spin_topk)

        sw_box = QWidget()
        sw_lay = QVBoxLayout(sw_box)
        sw_lay.setContentsMargins(0, 0, 0, 0)
        sw_lay.setSpacing(4)
        self.chk_remove_sw = QCheckBox("Bật bỏ stopword")
        self.chk_remove_sw.setChecked(True)
        self.lbl_sw_caption = QLabel(
            "Bỏ các từ phổ biến ít mang nghĩa (là, thì, mà...). "
            "Các từ quan trọng như không, rẻ, gần, nhanh luôn được giữ."
        )
        self.lbl_sw_caption.setWordWrap(True)
        self.lbl_sw_caption.setStyleSheet(f"color:{C_TEXT_DIM}; font-size:11px;")
        sw_lay.addWidget(self.chk_remove_sw)
        sw_lay.addWidget(self.lbl_sw_caption)
        form.addRow("Tiền xử lý:", sw_box)

        self.lbl_model_dirty = QLabel("⚠️ Tham số mô hình đã đổi, cần huấn luyện lại")
        self.lbl_model_dirty.setWordWrap(True)
        self.lbl_model_dirty.setVisible(False)
        self.lbl_model_dirty.setStyleSheet(
            f"color:{C_WARNING}; background:{C_CARD}; border:1px solid {C_WARNING}; "
            f"border-radius:6px; padding:6px 10px; font-weight:bold;"
        )
        form.addRow("", self.lbl_model_dirty)

        self.spin_alpha.valueChanged.connect(self._on_model_param_changed)
        self.cmb_ngram.currentIndexChanged.connect(self._on_model_param_changed)
        self.chk_remove_sw.toggled.connect(self._on_model_param_changed)

        vbox.addWidget(grp_model)

        # ─── Hệ số độ đông & chỗ ngồi (crowd_config.json) ──────────────────
        grp_crowd = QGroupBox("🚦  Hệ số mô hình độ đông & chỗ ngồi (config/crowd_config.json)")
        c_lay = QVBoxLayout(grp_crowd)
        c_lay.setSpacing(8)
        c_lay.setContentsMargins(12, 16, 12, 12)

        c_grid = QGridLayout()
        c_grid.setHorizontalSpacing(12)
        c_grid.setVerticalSpacing(6)

        default_crowd = load_crowd_config(DEFAULT_CROWD_CONFIG_PATH)
        slots_default = default_crowd["he_so_khung_gio"]

        self.spin_crowd_a = QDoubleSpinBox()
        self.spin_crowd_a.setRange(0.0, 5.0)
        self.spin_crowd_a.setDecimals(2)
        self.spin_crowd_a.setSingleStep(0.1)
        self.spin_crowd_a.setValue(float(default_crowd["a"]))
        self.spin_crowd_a.setToolTip("Hệ số a: mức tăng thời gian chờ theo độ đông hiệu dụng")

        self.spin_crowd_b = QDoubleSpinBox()
        self.spin_crowd_b.setRange(0.0, 1.0)
        self.spin_crowd_b.setDecimals(2)
        self.spin_crowd_b.setSingleStep(0.02)
        self.spin_crowd_b.setValue(float(default_crowd["b"]))
        self.spin_crowd_b.setToolTip("Hệ số b: mức tăng thời gian chờ khi thêm mỗi người trong nhóm")

        self.spin_crowd_occ = QDoubleSpinBox()
        self.spin_crowd_occ.setRange(0.0, 1.0)
        self.spin_crowd_occ.setDecimals(2)
        self.spin_crowd_occ.setSingleStep(0.05)
        self.spin_crowd_occ.setValue(float(default_crowd["he_so_chiem_cho"]))
        self.spin_crowd_occ.setToolTip("he_so_chiem_cho: tỉ lệ chỗ ngồi bị chiếm tối đa khi độ đông = 5")

        self.spin_slot_peak = QDoubleSpinBox()
        self.spin_slot_peak.setRange(0.0, 2.0)
        self.spin_slot_peak.setDecimals(2)
        self.spin_slot_peak.setSingleStep(0.1)
        self.spin_slot_peak.setValue(float(slots_default["cao_diem"]))

        self.spin_slot_normal = QDoubleSpinBox()
        self.spin_slot_normal.setRange(0.0, 2.0)
        self.spin_slot_normal.setDecimals(2)
        self.spin_slot_normal.setSingleStep(0.1)
        self.spin_slot_normal.setValue(float(slots_default["binh_thuong"]))

        self.spin_slot_quiet = QDoubleSpinBox()
        self.spin_slot_quiet.setRange(0.0, 2.0)
        self.spin_slot_quiet.setDecimals(2)
        self.spin_slot_quiet.setSingleStep(0.1)
        self.spin_slot_quiet.setValue(float(slots_default["vang"]))

        c_grid.addWidget(QLabel("Hệ số độ đông (a):"), 0, 0)
        c_grid.addWidget(self.spin_crowd_a, 0, 1)
        c_grid.addWidget(QLabel("Hệ số quy mô nhóm (b):"), 0, 2)
        c_grid.addWidget(self.spin_crowd_b, 0, 3)
        c_grid.addWidget(QLabel("Hệ số chiếm chỗ:"), 0, 4)
        c_grid.addWidget(self.spin_crowd_occ, 0, 5)

        c_grid.addWidget(QLabel("Khung cao điểm:"), 1, 0)
        c_grid.addWidget(self.spin_slot_peak, 1, 1)
        c_grid.addWidget(QLabel("Khung bình thường:"), 1, 2)
        c_grid.addWidget(self.spin_slot_normal, 1, 3)
        c_grid.addWidget(QLabel("Khung vắng:"), 1, 4)
        c_grid.addWidget(self.spin_slot_quiet, 1, 5)
        c_lay.addLayout(c_grid)

        self.btn_reset_crowd = QPushButton("↺  Đặt lại mặc định từ crowd_config.json")
        self.btn_reset_crowd.setStyleSheet(f"background:{C_CARD}; color:{C_TEXT}; border:1px solid {C_BORDER};")
        self.btn_reset_crowd.clicked.connect(self._reset_crowd_params)
        c_lay.addWidget(self.btn_reset_crowd)
        vbox.addWidget(grp_crowd)

        # ─── Trọng số 7 tiêu chí ───────────────────────────────────────────
        grp_weights = QGroupBox("⚖️  Trọng số 7 tiêu chí (tùy chỉnh)")
        w_vbox = QVBoxLayout(grp_weights)
        w_vbox.setSpacing(8)
        w_vbox.setContentsMargins(12, 16, 12, 12)

        self.weight_sliders: Dict[str, QSlider] = {}
        self.weight_labels:  Dict[str, QLabel]  = {}

        for crit in CRITERIA_ORDER:
            row = QHBoxLayout()
            name_lbl = QLabel(f"{CRITERIA_NAMES[crit]}:")
            name_lbl.setFixedWidth(110)
            name_lbl.setStyleSheet(f"color:{C_TEXT};")
            row.addWidget(name_lbl)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(14)
            sl.setFixedHeight(20)
            sl.valueChanged.connect(self._on_weight_changed)
            self.weight_sliders[crit] = sl
            row.addWidget(sl, 1)

            val_lbl = QLabel("0.14")
            val_lbl.setFixedWidth(42)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet(f"color:{C_ACCENT2}; font-weight:bold;")
            self.weight_labels[crit] = val_lbl
            row.addWidget(val_lbl)

            w_vbox.addLayout(row)

        w_btn_row = QHBoxLayout()
        w_btn_row.setSpacing(6)

        btn_auto_weights = QPushButton("🔄  Dùng trọng số tự động theo nhu cầu")
        btn_auto_weights.setToolTip("Tự động cập nhật trọng số khớp với nhãn vừa gợi ý")
        btn_auto_weights.clicked.connect(self._use_auto_weights)
        w_btn_row.addWidget(btn_auto_weights, 2)

        btn_reset_weights = QPushButton("↺  Đặt lại đều")
        btn_reset_weights.setStyleSheet(f"background:{C_CARD}; color:{C_TEXT}; border:1px solid {C_BORDER};")
        btn_reset_weights.clicked.connect(self._reset_weights)
        w_btn_row.addWidget(btn_reset_weights, 1)

        w_vbox.addLayout(w_btn_row)
        vbox.addWidget(grp_weights)

        # ─── Nút Huấn luyện lại ────────────────────────────────────────────
        grp_train = QGroupBox("🔧  Huấn luyện lại mô hình")
        t_vbox = QVBoxLayout(grp_train)
        t_vbox.setSpacing(8)
        t_vbox.setContentsMargins(12, 16, 12, 12)

        self.lbl_train_status = QLabel("Trạng thái: Chưa huấn luyện hoặc đã có mô hình.")
        self.lbl_train_status.setWordWrap(True)
        self.lbl_train_status.setStyleSheet(f"color:{C_TEXT_DIM};")
        t_vbox.addWidget(self.lbl_train_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        t_vbox.addWidget(self.progress_bar)

        self.btn_train = QPushButton("⚡  Huấn luyện lại")
        self.btn_train.setMinimumHeight(40)
        self.btn_train.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {C_ACCENT}, stop:1 {C_ACCENT2});"
            "border-radius:8px; font-size:14px; font-weight:bold; color:#fff;"
        )
        self.btn_train.clicked.connect(self._on_train)
        t_vbox.addWidget(self.btn_train)

        vbox.addWidget(grp_train)
        vbox.addStretch()
        return scroll

    def _get_crowd_params_from_ui(self) -> Dict[str, Any]:
        """Lấy bộ tham số độ đông từ các spinbox ở tab Tham số."""
        return {
            "a": float(self.spin_crowd_a.value()),
            "b": float(self.spin_crowd_b.value()),
            "he_so_chiem_cho": float(self.spin_crowd_occ.value()),
            "he_so_khung_gio": {
                "cao_diem": float(self.spin_slot_peak.value()),
                "binh_thuong": float(self.spin_slot_normal.value()),
                "vang": float(self.spin_slot_quiet.value()),
            },
        }

    def _reset_crowd_params(self) -> None:
        """Đặt lại các hệ số độ đông về mặc định từ config/crowd_config.json."""
        cfg = load_crowd_config(DEFAULT_CROWD_CONFIG_PATH)
        slots = cfg["he_so_khung_gio"]
        self.spin_crowd_a.setValue(float(cfg["a"]))
        self.spin_crowd_b.setValue(float(cfg["b"]))
        self.spin_crowd_occ.setValue(float(cfg["he_so_chiem_cho"]))
        self.spin_slot_peak.setValue(float(slots["cao_diem"]))
        self.spin_slot_normal.setValue(float(slots["binh_thuong"]))
        self.spin_slot_quiet.setValue(float(slots["vang"]))
        self._status("✅ Đã đặt lại hệ số độ đông mặc định từ config/crowd_config.json.")

    def _current_ui_model_params(self) -> Dict[str, Any]:
        return {
            "alpha": round(float(self.spin_alpha.value()), 4),
            "ngram_range": tuple(self.cmb_ngram.currentData() or (1, 2)),
            "remove_stopwords": bool(self.chk_remove_sw.isChecked()),
        }

    def _is_model_dirty(self) -> bool:
        return self._current_ui_model_params() != self._trained_model_params

    def _on_model_param_changed(self) -> None:
        """Cảnh báo khi người dùng đổi alpha, ngram_range hoặc stopword mà chưa huấn luyện lại."""
        dirty = self._is_model_dirty()
        self.lbl_model_dirty.setVisible(dirty)
        if dirty:
            msg = "⚠️ Tham số mô hình đã đổi, cần huấn luyện lại"
            self.lbl_train_status.setText(msg)
            self.lbl_train_status.setStyleSheet(f"color:{C_WARNING}; font-weight:bold;")
            self._status(msg)
        else:
            self.lbl_train_status.setText("✅ Tham số đang khớp với mô hình hiện tại.")
            self.lbl_train_status.setStyleSheet(f"color:{C_TEXT_DIM};")
            self._status("✅ Tham số đang khớp với mô hình hiện tại.")

    # ── TAB: ĐÁNH GIÁ ────────────────────────────────────────────────────────
    def _build_eval_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{C_SURFACE}; border:none;")

        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(widget)

        guide_frame = QFrame()
        guide_frame.setStyleSheet(
            f"QFrame {{ background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:8px; }}"
        )
        g_lay = QVBoxLayout(guide_frame)
        g_lay.setContentsMargins(10, 8, 10, 8)
        g_lay.setSpacing(4)
        self.lbl_eval_guide = QLabel(
            "<b>ℹ️ Hướng dẫn đọc kết quả đánh giá:</b><br>"
            "• Tab này kiểm tra độ chính xác của mô hình trên <code>data/test_intents.csv</code> "
            "(bộ câu kiểm thử độc lập, không dùng khi huấn luyện).<br>"
            "• Nút <b>\"▶ Chạy đánh giá ngay\"</b> sẽ đánh giá theo cấu hình hiện tại ở tab <i>Tham số</i>.<br>"
            "• <b>Accuracy</b>: tỉ lệ câu đoán đúng trên tổng số câu test · "
            "<b>Precision</b>: độ chuẩn xác khi đoán 1 nhãn · "
            "<b>Recall</b>: độ phủ phát hiện đúng nhãn · "
            "<b>F1-Score</b>: trung bình điều hòa giữa Precision & Recall · "
            "<b>Support</b>: số câu kiểm thử thực tế của nhãn."
        )
        self.lbl_eval_guide.setWordWrap(True)
        self.lbl_eval_guide.setStyleSheet(f"color:{C_TEXT}; font-size:12px; border:none;")
        g_lay.addWidget(self.lbl_eval_guide)
        vbox.addWidget(guide_frame)

        top_metrics_row = QHBoxLayout()
        top_metrics_row.setSpacing(12)

        self.lbl_accuracy = QLabel("Accuracy: Đang nạp kết quả...")
        self.lbl_accuracy.setToolTip(
            "Mức đánh giá Accuracy trên tập kiểm thử độc lập:\n"
            "• Xanh lá (>= 85%): Tốt\n"
            "• Vàng (70% – dưới 85%): Khá, cần cải thiện thêm\n"
            "• Đỏ (< 70%): Thấp, cần bổ sung dữ liệu hoặc chỉnh tham số"
        )
        self.lbl_accuracy.setStyleSheet(f"font-size:16px; font-weight:bold; color:{C_SUCCESS}; padding:4px;")
        top_metrics_row.addWidget(self.lbl_accuracy, 1)

        self.lbl_feedback_summary = QLabel("Phản hồi người dùng: Chưa có phản hồi")
        self.lbl_feedback_summary.setStyleSheet(
            f"color:{C_ACCENT2}; background:{C_CARD}; border:1px solid {C_BORDER}; "
            f"border-radius:6px; padding:6px 10px; font-weight:bold; font-size:12px;"
        )
        top_metrics_row.addWidget(self.lbl_feedback_summary)
        vbox.addLayout(top_metrics_row)

        grp_insights = QGroupBox("💡  Nhận định kết quả (tự động phân tích từ số liệu)")
        ins_lay = QVBoxLayout(grp_insights)
        ins_lay.setContentsMargins(10, 12, 10, 8)
        self.lbl_eval_insights = QLabel("Đang tổng hợp nhận định...")
        self.lbl_eval_insights.setWordWrap(True)
        self.lbl_eval_insights.setStyleSheet(f"color:{C_TEXT}; font-size:12.5px; line-height:1.45;")
        ins_lay.addWidget(self.lbl_eval_insights)
        vbox.addWidget(grp_insights)

        grp_report = QGroupBox("📋  Classification Report (từng nhãn ý định)")
        rep_lay = QVBoxLayout(grp_report)
        rep_lay.setContentsMargins(8, 12, 8, 8)
        self.tbl_report = self._make_table(["Nhãn", "Precision", "Recall", "F1-Score", "Support"])
        self.tbl_report.setAlternatingRowColors(True)
        self.tbl_report.setMinimumHeight(215)
        rep_lay.addWidget(self.tbl_report)
        vbox.addWidget(grp_report)

        grp_cm = QGroupBox("🔢  Ma trận nhầm lẫn (Confusion Matrix)")
        cm_lay = QVBoxLayout(grp_cm)
        cm_lay.setContentsMargins(8, 12, 8, 8)
        self.canvas_cm = MplCanvas(figsize=(7, 4.2))
        self.canvas_cm.setMinimumHeight(280)
        cm_lay.addWidget(self.canvas_cm)
        vbox.addWidget(grp_cm, 1)

        self.eval_progress = QProgressBar()
        self.eval_progress.setRange(0, 0)
        self.eval_progress.setVisible(False)
        vbox.addWidget(self.eval_progress)

        self.btn_run_eval = QPushButton("▶  Chạy đánh giá ngay (theo tham số ở tab Tham số)")
        self.btn_run_eval.setMinimumHeight(38)
        self.btn_run_eval.clicked.connect(self._run_evaluation)
        vbox.addWidget(self.btn_run_eval)

        self._refresh_feedback_summary_label()
        return scroll

    def _feedback_csv_path(self) -> str:
        return DEFAULT_FEEDBACK_PATH

    def _refresh_feedback_summary_label(self) -> None:
        summary = feedback_summary(self._feedback_csv_path())
        self.lbl_feedback_summary.setText(format_feedback_summary(summary))

    def _on_feedback_submitted(self, meta: Dict[str, Any]) -> None:
        """Ghi phản hồi người dùng vào data/feedback.csv (UTF-8 BOM) và cập nhật thống kê."""
        try:
            append_feedback(
                path=self._feedback_csv_path(),
                question=meta.get("question", ""),
                predicted_label=meta.get("predicted_label", ""),
                confidence=float(meta.get("confidence", 0.0)),
                top1_dish=meta.get("top1_dish", ""),
                rating=meta.get("rating", RATING_HELPFUL),
                correct_label=meta.get("correct_label", ""),
                note=meta.get("note", ""),
            )
            self._refresh_feedback_summary_label()
            self._status("✅ Đã lưu phản hồi vào data/feedback.csv.")
        except Exception as e:
            self._status(f"⚠️ Không thể lưu phản hồi: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # HELPER: TẠO BẢNG
    # ══════════════════════════════════════════════════════════════════════════
    def _make_table(self, headers: List[str]) -> QTableWidget:
        tbl = QTableWidget(0, len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet("font-size:12px;")
        return tbl

    def _fill_table(self, tbl: QTableWidget, rows: List[List[str]]):
        tbl.setRowCount(len(rows))
        for r, row_data in enumerate(rows):
            for c, val in enumerate(row_data):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                tbl.setItem(r, c, item)

    # ══════════════════════════════════════════════════════════════════════════
    # CHAT LOGIC
    # ══════════════════════════════════════════════════════════════════════════
    def _on_send(self):
        if not self._validate_filter_inputs():
            self._status("⚠️ Vui lòng sửa ô Ngân sách / Số người hợp lệ trước khi gửi câu hỏi.")
            return

        text = self.input_box.text().strip()
        if not text:
            self._shake_input()
            return
        if not self.is_trained or self.bot is None:
            QMessageBox.warning(
                self, "Chưa có mô hình",
                "Vui lòng nhấn 'Huấn luyện lại' ở tab Tham số trước khi chat.",
            )
            return

        budget_k, _ = parse_budget_k(self.in_budget.text())
        max_price_per_person = budget_k_to_vnd(budget_k)
        num_people, _ = parse_num_people(self.in_people.text())
        no_spicy = self.chk_no_spicy.isChecked()
        veg_only = self.chk_veg.isChecked()
        enough_seats = self.chk_enough_seats.isChecked()
        ui_slot = str(self.cmb_time_slot.currentData() or "tu_dong")

        self.input_box.clear()
        self._append_user_message(text)
        self.btn_send.setEnabled(False)

        self.bot.confidence_threshold = self.spin_threshold.value()
        self.bot.default_top_k = self.spin_topk.value()
        custom_w = self._get_custom_weights() if self._weights_customized() else None
        crowd_params = self._get_crowd_params_from_ui()

        try:
            resp = self.bot.respond(
                text,
                confidence_threshold=self.spin_threshold.value(),
                custom_weights=custom_w,
                top_k=self.spin_topk.value(),
                remove_stopwords=None,
                max_price_per_person=max_price_per_person,
                num_people=num_people,
                vegetarian_only=veg_only,
                exclude_spicy=no_spicy,
                time_slot=ui_slot,
                require_enough_seats=enough_seats,
                crowd_params=crowd_params,
            )

            # Tự động cập nhật ô giao diện nếu câu nói nêu rõ chay, số người hoặc khung giờ
            from_text = resp.get("constraints", {}).get("from_text", {})
            if from_text.get("vegetarian_only") and not self.chk_veg.isChecked():
                self.chk_veg.blockSignals(True)
                self.chk_veg.setChecked(True)
                self.chk_veg.blockSignals(False)

            if from_text.get("num_people") is not None:
                self.in_people.blockSignals(True)
                self.in_people.setText(str(from_text["num_people"]))
                self.in_people.blockSignals(False)

            if from_text.get("time_slot") is not None:
                idx_slot = self.cmb_time_slot.findData(from_text["time_slot"])
                if idx_slot >= 0:
                    self.cmb_time_slot.blockSignals(True)
                    self.cmb_time_slot.setCurrentIndex(idx_slot)
                    self.cmb_time_slot.blockSignals(False)

            self._last_response = resp
            self._update_steps_tab(resp)
            self._update_charts(resp)

            top1_name = ""
            rec = resp.get("recommendation")
            if rec and rec.get("top_k"):
                top1_name = rec["top_k"][0]["dish"]["ten_mon"]
            elif rec and rec.get("near_matches"):
                top1_name = rec["near_matches"][0]["dish"]["ten_mon"]

            fb_meta = {
                "question": text,
                "predicted_label": resp.get("classification", {}).get("label", ""),
                "confidence": resp.get("classification", {}).get("confidence", 0.0),
                "top1_dish": top1_name,
            }
            self._append_bot_message(resp["reply"], feedback_meta=fb_meta)

        except Exception as e:
            self._append_bot_message(f"⚠️ Lỗi xử lý: {str(e)}")
        finally:
            self._validate_filter_inputs()
            self.input_box.setFocus()

    def _append_user_message(self, text: str):
        bubble = ChatBubble(text, is_user=True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _append_bot_message(self, text: str, feedback_meta: Optional[Dict[str, Any]] = None):
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = text.replace("\n", "<br>")
        bubble = ChatBubble(text, is_user=False, feedback_meta=feedback_meta)
        if feedback_meta is not None:
            bubble.feedback_submitted.connect(self._on_feedback_submitted)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        QTimer.singleShot(
            50,
            lambda: self.chat_scroll.verticalScrollBar().setValue(
                self.chat_scroll.verticalScrollBar().maximum()
            ),
        )

    def _clear_chat(self):
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _shake_input(self):
        """Hiệu ứng rung nhẹ khi input rỗng."""
        orig = self.input_box.styleSheet()
        self.input_box.setStyleSheet(orig + f"border-color:{C_ERROR};")
        QTimer.singleShot(400, lambda: self.input_box.setStyleSheet(orig))

    # ══════════════════════════════════════════════════════════════════════════
    # CẬP NHẬT TAB BƯỚC XỬ LÝ
    # ══════════════════════════════════════════════════════════════════════════
    def _update_steps_tab(self, resp: Dict[str, Any]):
        prep = resp.get("preprocessing", {})
        clf  = resp.get("classification", {})
        con  = resp.get("constraints", {})
        rec  = resp.get("recommendation")

        # 1. Tiền xử lý
        self.lbl_orig.setText(prep.get("original", "—"))
        self.lbl_norm.setText(prep.get("normalized", "—"))
        self.lbl_tokens.setText(str(prep.get("tokens", [])))
        self.lbl_no_sw.setText(str(prep.get("tokens_no_stopwords", [])))

        # 2. TF-IDF
        tfidf = clf.get("tfidf_terms", [])[:15]
        self._fill_table(
            self.tbl_tfidf,
            [[t["term"], f"{t['tfidf']:.4f}"] for t in tfidf],
        )

        # 3. Xác suất
        probs = clf.get("probabilities", [])
        rows = []
        for p in probs:
            lbl = LABEL_NAMES.get(p["label"], p["label"])
            rows.append([lbl, f"{p['prob']:.4f}  ({p['prob']*100:.1f}%)"])
        self._fill_table(self.tbl_probs, rows)

        # 4. Ràng buộc & nguồn
        sources = con.get("sources", {}) or {}
        mp = con.get("max_price")
        mp_src = sources.get("budget")
        if mp:
            src_str = f" ({mp_src})" if mp_src else ""
            self.lbl_max_price.setText(f"{mp:,.0f}đ/người{src_str}")
        else:
            self.lbl_max_price.setText("Không giới hạn")

        num_people = int(con.get("num_people") or 1)
        people_src = sources.get("num_people") or "từ ô nhập"
        total_budget = con.get("group_budget")
        people_note = (con.get("intermediate") or {}).get("num_people_note")
        note_suffix = f" | ⚠️ {people_note}" if people_note else ""
        if total_budget:
            self.lbl_group_budget.setText(
                f"{num_people} người ({people_src}) — Tổng ngân sách nhóm: {total_budget:,.0f}đ{note_suffix}"
            )
        else:
            self.lbl_group_budget.setText(
                f"{num_people} người ({people_src}) — Tổng ngân sách nhóm: Không giới hạn{note_suffix}"
            )

        sp_src = sources.get("spicy")
        if con.get("exclude_spicy"):
            self.lbl_excl_spicy.setText(f"Có ({sp_src})" if sp_src else "Có")
        else:
            self.lbl_excl_spicy.setText("Không")

        veg_src = sources.get("vegetarian")
        if con.get("vegetarian_only"):
            self.lbl_veg_only.setText(f"Có ({veg_src})" if veg_src else "Có")
        else:
            self.lbl_veg_only.setText("Không")

        exc = con.get("exclude_types", [])
        self.lbl_excl_types.setText(", ".join(exc) if exc else "Không ràng buộc")

        if rec and rec.get("filter_steps"):
            step_parts = [f"{s['label']}: {s['count']}" for s in rec["filter_steps"]]
            self.lbl_filter_steps.setText(" → ".join(step_parts))
        else:
            self.lbl_filter_steps.setText("—")

        # 5. Độ đông và nhóm
        slot = con.get("time_slot", "cao_diem")
        slot_src = sources.get("time_slot") or "tự động theo đồng hồ"
        slot_label = TIME_SLOT_LABELS_VI.get(slot, slot)
        self.lbl_crowd_people.setText(f"{num_people} người ({people_src}){note_suffix}")

        if rec and rec.get("crowd_info"):
            ci = rec["crowd_info"]
            self.lbl_crowd_slot.setText(
                f"{slot_label} ({slot_src}) — hệ số khung giờ = {ci['he_so_khung_gio']:.2f}"
            )
            if ci.get("uu_tien_vang"):
                self.lbl_crowd_a.setText(
                    f"a = {ci['a_effective']:.2f} (nhân đôi từ {ci['a_base']:.2f} do câu nói ưu tiên quán vắng/yên tĩnh)"
                )
            else:
                self.lbl_crowd_a.setText(f"a = {ci['a_effective']:.2f} (mặc định, b = {ci['b']:.2f})")
        else:
            self.lbl_crowd_slot.setText(f"{slot_label} ({slot_src})")
            self.lbl_crowd_a.setText("—")

        if rec:
            items_for_table = rec.get("top_k") or rec.get("near_matches") or []
            crowd_rows = []
            for item in items_for_table:
                d = item["dish"]
                dd_hd = float(item.get("do_dong_hieu_dung", d.get("do_dong_hieu_dung", 0.0)))
                tg_cho = float(item.get("thoi_gian_cho", d.get("thoi_gian_cho", 0.0)))
                cho_tr = float(item.get("cho_trong", d.get("cho_trong", 0.0)))
                cn_nhom = float(item.get("cho_ngoi_nhom", d.get("cho_ngoi_nhom", 1.0)))
                warn = item.get("seat_warning") or d.get("seat_warning") or "Đủ chỗ"
                crowd_rows.append([
                    d["ten_mon"][:26],
                    f"{dd_hd:.2f} ({item.get('crowd_label', '')})",
                    f"{tg_cho:.1f}p (gốc {d['thoi_gian_phuc_vu_phut']:.0f}p)",
                    f"{cho_tr:.1f}/{d.get('so_cho_ngoi', 30)}",
                    f"{cn_nhom:.4f}",
                    f"⚠️ {warn}" if warn != "Đủ chỗ" else "✅ Đủ chỗ",
                ])
            self._fill_table(self.tbl_crowd_dishes, crowd_rows)
        else:
            self._fill_table(self.tbl_crowd_dishes, [])

        # 6. Trọng số 7 tiêu chí đang dùng
        if rec:
            w = rec.get("weights_used", {})
            w_rows = [[CRITERIA_NAMES.get(k, k), f"{w[k]:.3f}"] for k in CRITERIA_ORDER if k in w]
            self._fill_table(self.tbl_weights, w_rows)

        # 7. Điểm đóng góp 7 tiêu chí từng món
        if rec:
            items_for_table = rec.get("top_k") or rec.get("near_matches") or []
            score_rows = []
            for item in items_for_table:
                d = item["dish"]
                c = item["contributions"]
                score_rows.append([
                    d["ten_mon"][:26],
                    f"{item['score']:.3f}",
                    f"{c.get('gia', 0):.3f}",
                    f"{c.get('khoang_cach', 0):.3f}",
                    f"{c.get('thoi_gian', 0):.3f}",
                    f"{c.get('do_no', 0):.3f}",
                    f"{c.get('dinh_duong', 0):.3f}",
                    f"{c.get('do_ngon', 0):.3f}",
                    f"{c.get('cho_ngoi', 0):.3f}",
                ])
            self._fill_table(self.tbl_scores, score_rows)

    # ══════════════════════════════════════════════════════════════════════════
    # VẼ BIỂU ĐỒ
    # ══════════════════════════════════════════════════════════════════════════
    def _update_charts(self, resp: Dict[str, Any]):
        self._draw_stacked_bar(resp)
        self._draw_wait_time_curve(resp)
        self._draw_available_seats_bar(resp)
        self._draw_prob_bar(resp)

    def _draw_stacked_bar(self, resp: Dict[str, Any]):
        """Vẽ biểu đồ cột xếp chồng đóng góp 7 tiêu chí."""
        self.canvas_stacked.fig.clf()
        rec = resp.get("recommendation")
        items = (rec.get("top_k") or rec.get("near_matches")) if rec else None
        if not items:
            ax = self.canvas_stacked.fig.add_subplot(111)
            ax.set_facecolor(C_CARD)
            ax.text(
                0.5, 0.5, "Không có dữ liệu gợi ý",
                ha="center", va="center", color=C_TEXT_DIM,
                fontsize=12, fontfamily=VIET_FONT,
            )
            self.canvas_stacked.draw()
            return

        top = items
        criteria = CRITERIA_ORDER
        colors = ["#7C6AF7", "#50C5B7", "#F4B942", "#3ECA8D", "#F47F4C", "#E05CB0", "#5BC0EB"]
        labels = [d["dish"]["ten_mon"][:20] for d in top]

        data = np.zeros((len(criteria), len(top)))
        for j, item in enumerate(top):
            c = item["contributions"]
            for i, crit in enumerate(criteria):
                data[i][j] = c.get(crit, 0)

        ax = self.canvas_stacked.fig.add_subplot(111)
        bottom = np.zeros(len(top))
        x = np.arange(len(top))

        for i, (crit, color) in enumerate(zip(criteria, colors)):
            ax.bar(
                x, data[i], bottom=bottom, color=color, width=0.52,
                label=CRITERIA_NAMES[crit], alpha=0.9,
            )
            bottom += data[i]

        for j, item in enumerate(top):
            ax.text(
                j, float(bottom[j]) + 0.005, f"{item['score']:.3f}",
                ha="center", va="bottom", color=C_TEXT, fontsize=9,
                fontweight="bold", fontfamily=VIET_FONT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            labels, fontsize=9, color=C_TEXT,
            rotation=8, ha="right", fontfamily=VIET_FONT,
        )
        ax.set_ylim(0, max(1.0, float(np.max(bottom)) * 1.22))
        ax.legend(
            loc="upper right", ncol=2, fontsize=8, framealpha=0.3,
            labelcolor=C_TEXT, facecolor=C_CARD,
        )
        chart_title = (
            "Điểm đóng góp 7 tiêu chí"
            if (rec and rec.get("top_k"))
            else "Điểm đóng góp 7 tiêu chí (3 món gần điều kiện nhất)"
        )
        _ax_style(ax, title=chart_title, ylabel="Điểm")
        self.canvas_stacked.fig.tight_layout()
        self.canvas_stacked.draw()

    def _draw_wait_time_curve(self, resp: Dict[str, Any]):
        """Vẽ biểu đồ Thời gian chờ ước tính theo số người (1-10) cho Top-3 món ở khung giờ đang chọn."""
        self.canvas_wait_curve.fig.clf()
        rec = resp.get("recommendation")
        items = (rec.get("top_k") or rec.get("near_matches")) if rec else None
        if not items:
            ax = self.canvas_wait_curve.fig.add_subplot(111)
            ax.set_facecolor(C_CARD)
            ax.text(0.5, 0.5, "Chưa có dữ liệu", ha="center", va="center", color=C_TEXT_DIM, fontfamily=VIET_FONT)
            self.canvas_wait_curve.draw()
            return

        top3 = items[:3]
        ci = rec.get("crowd_info", {})
        khung_gio = ci.get("khung_gio", "cao_diem")
        prefer_quiet = bool(ci.get("uu_tien_vang", False))
        cur_people = int(ci.get("so_nguoi", 1))
        crowd_cfg = self._get_crowd_params_from_ui()

        ax = self.canvas_wait_curve.fig.add_subplot(111)
        people_axis = list(range(1, 11))
        line_colors = ["#7C6AF7", "#50C5B7", "#F4B942"]

        for idx, item in enumerate(top3):
            dish = item["dish"]
            waits = [
                compute_crowd_metrics(
                    dish, so_nguoi=p, khung_gio=khung_gio, crowd_cfg=crowd_cfg, prefer_quiet=prefer_quiet
                )["thoi_gian_cho"]
                for p in people_axis
            ]
            ax.plot(
                people_axis, waits, marker="o", markersize=4, linewidth=2,
                color=line_colors[idx % len(line_colors)],
                label=dish["ten_mon"][:18],
            )

        ax.axvline(cur_people, color=C_ERROR, linestyle="--", linewidth=1.2, label=f"Hiện tại ({cur_people} người)")
        ax.set_xticks(people_axis)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.3, labelcolor=C_TEXT, facecolor=C_CARD)
        slot_short = TIME_SLOT_SHORT_VI.get(khung_gio, khung_gio)
        _ax_style(
            ax,
            title=f"TG chờ theo số người ({slot_short})",
            xlabel="Số người đi ăn (1–10)",
            ylabel="TG chờ (phút)",
        )
        self.canvas_wait_curve.fig.tight_layout()
        self.canvas_wait_curve.draw()

    def _draw_available_seats_bar(self, resp: Dict[str, Any]):
        """Vẽ biểu đồ Chỗ trống ước tính so với số người đi ăn cho Top-3 món."""
        self.canvas_seats.fig.clf()
        rec = resp.get("recommendation")
        items = (rec.get("top_k") or rec.get("near_matches")) if rec else None
        if not items:
            ax = self.canvas_seats.fig.add_subplot(111)
            ax.set_facecolor(C_CARD)
            ax.text(0.5, 0.5, "Chưa có dữ liệu", ha="center", va="center", color=C_TEXT_DIM, fontfamily=VIET_FONT)
            self.canvas_seats.draw()
            return

        top3 = items[:3]
        ci = rec.get("crowd_info", {})
        cur_people = int(ci.get("so_nguoi", 1))

        ax = self.canvas_seats.fig.add_subplot(111)
        labels = [it["dish"]["ten_mon"][:16] for it in top3]
        avail = [float(it.get("cho_trong", it["dish"].get("cho_trong", 0.0))) for it in top3]
        bar_colors = [C_SUCCESS if a >= cur_people else C_WARNING for a in avail]

        x = np.arange(len(top3))
        bars = ax.bar(x, avail, color=bar_colors, width=0.5, alpha=0.9, label="Chỗ trống ước tính")
        ax.axhline(cur_people, color=C_ERROR, linestyle="--", linewidth=1.5, label=f"Số người đi ăn ({cur_people})")

        for b_rect, a_val in zip(bars, avail):
            ax.text(
                b_rect.get_x() + b_rect.get_width() / 2,
                a_val + 0.4,
                f"~{a_val:.1f}",
                ha="center", va="bottom", color=C_TEXT, fontsize=8.5, fontweight="bold", fontfamily=VIET_FONT,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, color=C_TEXT, rotation=8, ha="right", fontfamily=VIET_FONT)
        max_y = max(max(avail, default=1.0), float(cur_people)) * 1.25 + 1.0
        ax.set_ylim(0, max_y)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.3, labelcolor=C_TEXT, facecolor=C_CARD)
        _ax_style(ax, title="Chỗ trống ước tính so với số người", ylabel="Số chỗ ngồi")
        self.canvas_seats.fig.tight_layout()
        self.canvas_seats.draw()

    def _draw_prob_bar(self, resp: Dict[str, Any]):
        """Vẽ biểu đồ cột ngang xác suất nhãn."""
        self.canvas_probs.fig.clf()
        probs = resp.get("classification", {}).get("probabilities", [])
        if not probs:
            self.canvas_probs.draw()
            return

        ax = self.canvas_probs.fig.add_subplot(111)
        names = [LABEL_NAMES.get(p["label"], p["label"]) for p in probs]
        vals  = [p["prob"] for p in probs]
        effective = resp.get("effective_intent", "")

        bar_colors = []
        for p in probs:
            if p["label"] == effective:
                bar_colors.append(C_ACCENT)
            elif p["prob"] > 0.15:
                bar_colors.append(C_ACCENT2)
            else:
                bar_colors.append(C_BORDER)

        y = np.arange(len(names))
        bars = ax.barh(y, vals, color=bar_colors, height=0.55, alpha=0.9)

        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v*100:.1f}%", va="center", color=C_TEXT, fontsize=9,
                fontfamily=VIET_FONT,
            )

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9, color=C_TEXT, fontfamily=VIET_FONT)
        ax.set_xlim(0, max(vals) * 1.25 if vals else 1)
        ax.invert_yaxis()
        _ax_style(ax, title="Xác suất phân loại ý định", xlabel="Xác suất")
        self.canvas_probs.fig.tight_layout()
        self.canvas_probs.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # HUẤN LUYỆN
    # ══════════════════════════════════════════════════════════════════════════
    def _on_train(self):
        intents_path = DEFAULT_INTENTS_PATH
        if not os.path.exists(intents_path):
            QMessageBox.critical(self, "Lỗi", f"Không tìm thấy file:\n{intents_path}")
            return

        self.btn_train.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_train_status.setText("⏳ Đang huấn luyện mô hình...")
        self.lbl_train_status.setStyleSheet(f"color:{C_TEXT_DIM};")
        self._status("⏳ Đang huấn luyện...")
        if hasattr(self, "data_tab"):
            self.data_tab.set_training_state(True, "⏳ Đang huấn luyện mô hình...")

        model_save = DEFAULT_MODEL_PATH
        os.makedirs(DEFAULT_OUTPUTS_DIR, exist_ok=True)

        ui_params = self._current_ui_model_params()
        self.train_thread = TrainingThread(
            alpha=ui_params["alpha"],
            model_type="nb",
            intents_path=intents_path,
            model_save_path=model_save,
            ngram_range=ui_params["ngram_range"],
            remove_stopwords=ui_params["remove_stopwords"],
        )
        self.train_thread.progress.connect(self.lbl_train_status.setText)
        self.train_thread.finished.connect(self._on_train_done)
        self.train_thread.start()

    def _on_train_done(self, result: dict):
        self.progress_bar.setVisible(False)
        self.btn_train.setEnabled(True)
        if hasattr(self, "data_tab"):
            icon = "✅" if result["success"] else "❌ Lỗi:"
            self.data_tab.set_training_state(False, f"{icon} {result['message']}")

        if result["success"]:
            self.lbl_train_status.setText(f"✅ {result['message']}")
            self.lbl_train_status.setStyleSheet(f"color:{C_SUCCESS};")
            self._status(f"✅ {result['message']}")
            if result.get("classifier"):
                self.bot = self._new_bot()
                self.bot.classifier = result["classifier"]
                self.is_trained = True
                self._sync_ui_from_classifier(self.bot.classifier)
                if hasattr(self, "data_tab"):
                    self.data_tab.refresh_intent_counts()
            self._append_bot_message(
                "✅ Đã huấn luyện lại mô hình thành công! Bạn có thể bắt đầu chat nhé."
            )
            self._run_evaluation()
        else:
            self.lbl_train_status.setText(f"❌ Lỗi: {result['message']}")
            self.lbl_train_status.setStyleSheet(f"color:{C_ERROR};")
            self._status(f"❌ Lỗi huấn luyện: {result['message']}")
            QMessageBox.critical(self, "Lỗi huấn luyện", result["message"])

    # ══════════════════════════════════════════════════════════════════════════
    # ĐÁNH GIÁ (AUTO-LOAD HOẶC CHẠY TRONG QTHREAD)
    # ══════════════════════════════════════════════════════════════════════════
    def _ensure_eval_populated(self) -> None:
        if self._eval_loaded_once:
            return
        train_path = DEFAULT_INTENTS_PATH
        test_path = DEFAULT_TEST_INTENTS_PATH
        cache_path = os.path.join(DEFAULT_OUTPUTS_DIR, "eval_result.json")

        ui_params = self._current_ui_model_params()
        sig = make_signature(
            alpha=ui_params["alpha"],
            ngram_range=ui_params["ngram_range"],
            remove_stopwords=ui_params["remove_stopwords"],
            train_path=train_path,
            test_path=test_path,
            model_type="nb",
        )
        cached = load_eval_cache(cache_path, signature=sig) or load_eval_cache(cache_path, signature=None)
        if cached is not None:
            self._display_eval_payload(cached)
            self._eval_loaded_once = True
        else:
            self._run_evaluation(use_ui_params=False)

    def _run_evaluation(self, checked: bool = False, use_ui_params: bool = True) -> None:
        train_path = DEFAULT_INTENTS_PATH
        test_path = DEFAULT_TEST_INTENTS_PATH
        outputs_dir = DEFAULT_OUTPUTS_DIR
        if not os.path.exists(test_path):
            QMessageBox.warning(self, "Thiếu dữ liệu", f"Không tìm thấy file kiểm thử:\n{test_path}")
            return

        if self.eval_thread is not None and self.eval_thread.isRunning():
            return

        ui_params = self._current_ui_model_params()
        self.btn_run_eval.setEnabled(False)
        self.eval_progress.setVisible(True)
        self.lbl_accuracy.setText("⏳ Đang chạy đánh giá trên data/test_intents.csv...")
        self._status("⏳ Đang đánh giá mô hình trên tập kiểm thử...")

        self.eval_thread = EvaluationThread(
            train_path=train_path,
            test_path=test_path,
            outputs_dir=outputs_dir,
            alpha=ui_params["alpha"],
            ngram_range=ui_params["ngram_range"],
            remove_stopwords=ui_params["remove_stopwords"],
            classifier=self.bot.classifier if self.bot else None,
            use_custom_params=use_ui_params,
            parent=self,
        )
        self.eval_thread.finished.connect(self._on_eval_done)
        self.eval_thread.start()

    def _on_eval_done(self, res: dict) -> None:
        self.eval_progress.setVisible(False)
        self.btn_run_eval.setEnabled(True)
        if not res.get("success") or not res.get("payload"):
            msg = res.get("message", "Lỗi không xác định")
            self.lbl_accuracy.setText(f"❌ Lỗi đánh giá: {msg}")
            self._status(f"❌ Lỗi đánh giá: {msg}")
            return

        payload = res["payload"]
        self._eval_loaded_once = True
        self._display_eval_payload(payload)
        self._status(
            f"✅ Đánh giá xong: {format_accuracy_text(payload['accuracy'], payload['n_correct'], payload['n_total'])}"
        )

    def _display_eval_payload(self, payload: Dict[str, Any]) -> None:
        acc = float(payload["accuracy"])
        n_correct = int(payload.get("n_correct", 0))
        n_total = int(payload.get("n_total", 0))
        labels = list(payload.get("labels", []))
        report = payload.get("report_dict", {})
        cm = payload.get("confusion_matrix", [])
        sig = payload.get("signature", {})
        alpha_used = sig.get("alpha", self.spin_alpha.value())

        acc_text = format_accuracy_text(acc, n_correct, n_total)
        lvl = accuracy_level(acc)
        color = C_SUCCESS if lvl == LEVEL_GOOD else (C_WARNING if lvl == LEVEL_WARN else C_ERROR)
        self.lbl_accuracy.setText(acc_text)
        self.lbl_accuracy.setStyleSheet(f"font-size:17px; font-weight:bold; color:{color}; padding:4px;")

        insights_text = build_insights(
            accuracy=acc,
            n_correct=n_correct,
            n_total=n_total,
            labels=labels,
            report_dict=report,
            confusion=cm,
            label_names=LABEL_NAMES,
            alpha=alpha_used,
        )
        self.lbl_eval_insights.setText(insights_text)

        rep_rows = []
        for lbl in labels:
            if lbl in report:
                m = report[lbl]
                rep_rows.append([
                    LABEL_NAMES.get(lbl, lbl),
                    f"{m.get('precision', 0):.3f}",
                    f"{m.get('recall', 0):.3f}",
                    f"{m.get('f1-score', 0):.3f}",
                    str(int(m.get("support", 0))),
                ])
        self._fill_table(self.tbl_report, rep_rows)

        self._draw_confusion_matrix({"confusion_matrix": cm, "labels": labels})
        self._refresh_feedback_summary_label()

    def _draw_confusion_matrix(self, result: dict):
        self.canvas_cm.fig.clf()
        cm = np.array(result["confusion_matrix"])
        labels = [LABEL_NAMES.get(l, l) for l in result["labels"]]

        ax = self.canvas_cm.fig.add_subplot(111)
        ax.imshow(cm, cmap="Blues", aspect="auto")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(
            labels, rotation=30, ha="right",
            fontsize=8.5, color=C_TEXT, fontfamily=VIET_FONT,
        )
        ax.set_yticklabels(labels, fontsize=8.5, color=C_TEXT, fontfamily=VIET_FONT)

        cm_max = cm.max() if cm.size and cm.max() > 0 else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = int(cm[i, j])
                txt_color = "#FFFFFF" if val > cm_max * 0.45 else "#1E1E2E"
                ax.text(
                    j, i, str(val), ha="center", va="center",
                    color=txt_color, fontsize=9, fontweight="bold",
                    fontfamily=VIET_FONT,
                )

        _ax_style(ax, title="Ma trận nhầm lẫn trên tập kiểm thử độc lập", xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế")
        self.canvas_cm.fig.tight_layout()
        self.canvas_cm.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # TRỌNG SỐ (7 TIÊU CHÍ)
    # ══════════════════════════════════════════════════════════════════════════
    def _on_weight_changed(self):
        raw = {k: sl.value() for k, sl in self.weight_sliders.items()}
        total = sum(raw.values()) or 1
        for k, sl in self.weight_sliders.items():
            norm = raw[k] / total
            self.weight_labels[k].setText(f"{norm:.2f}")

    def _get_custom_weights(self) -> Dict[str, float]:
        raw = {k: sl.value() for k, sl in self.weight_sliders.items()}
        total = sum(raw.values()) or 1
        return {k: v / total for k, v in raw.items()}

    def _weights_customized(self) -> bool:
        """Kiểm tra xem người dùng có tuỳ chỉnh trọng số không (khác đều 1/7)."""
        vals = [sl.value() for sl in self.weight_sliders.values()]
        mean_v = sum(vals) / len(vals)
        return any(abs(v - mean_v) > 3 for v in vals)

    def _use_auto_weights(self):
        """Đọc profile từ nhãn vừa gợi ý và áp lên các thanh trượt 7 tiêu chí."""
        if not self.bot:
            return
        intent = None
        if self._last_response:
            intent = self._last_response.get("effective_intent")
        profile_weights = self.bot.recommender.profiles.get(intent or "can_bang", {})
        if not profile_weights:
            return
        for crit in CRITERIA_ORDER:
            w = profile_weights.get(crit, 1 / 7)
            sl = self.weight_sliders[crit]
            sl.blockSignals(True)
            sl.setValue(int(round(w * 100)))
            sl.blockSignals(False)
        self._on_weight_changed()
        self._status(f"✅ Đã áp dụng trọng số profile '{intent or 'can_bang'}'")

    def _reset_weights(self):
        for sl in self.weight_sliders.values():
            sl.blockSignals(True)
            sl.setValue(14)
            sl.blockSignals(False)
        self._on_weight_changed()

    # ══════════════════════════════════════════════════════════════════════════
    # TIỆN ÍCH
    # ══════════════════════════════════════════════════════════════════════════
    def _status(self, msg: str):
        self.status_bar.showMessage(msg)

    def closeEvent(self, event):
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.quit()
            self.train_thread.wait(2000)
        if self.eval_thread and self.eval_thread.isRunning():
            self.eval_thread.quit()
            self.eval_thread.wait(2000)
        plt.close("all")
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def launch_gui():
    """Khởi chạy ứng dụng GUI."""
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("NEU Lunch Chatbot")

    window = LunchChatbotGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
