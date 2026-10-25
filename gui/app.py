"""
Giao diện desktop PyQt6 cho NEU Lunch Chatbot.
Bố cục: Trái - khung chat; Phải - QTabWidget (Bước xử lý | Biểu đồ | Tham số | Đánh giá).
Huấn luyện mô hình chạy trong QThread để không đơ giao diện.
Hỗ trợ tiếng Việt có dấu đầy đủ trên biểu đồ matplotlib.
"""

import os
import sys
import re
import json
import time
import math
from typing import Dict, Any, Optional, List

# ── Đường dẫn project ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── PyQt6 ──────────────────────────────────────────────────────────────────────
from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QTimer, QSize)
from PyQt6.QtGui  import (QFont, QColor, QPalette, QIcon, QTextCursor,
                           QPixmap, QPainter, QBrush, QPen)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QLineEdit, QTextEdit, QGroupBox,
    QTabWidget, QScrollArea, QTableWidget, QTableWidgetItem,
    QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QHeaderView,
    QSizePolicy, QFrame, QMessageBox, QProgressBar, QFormLayout,
    QGridLayout, QToolButton, QAbstractItemView
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
def _find_viet_font():
    """Tìm font hỗ trợ tiếng Việt trong hệ thống."""
    candidates = [
        "DejaVu Sans", "Arial", "Tahoma", "Segoe UI",
        "Times New Roman", "Verdana",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "DejaVu Sans"

VIET_FONT = _find_viet_font()
plt.rcParams["font.family"] = VIET_FONT

# ── Src modules ────────────────────────────────────────────────────────────────
try:
    from src.chatbot import Chatbot
    from src.classifier import NeedClassifier
    from src.evaluate import ModelEvaluator
except ImportError:
    sys.path.insert(0, _ROOT)
    from src.chatbot import Chatbot
    from src.classifier import NeedClassifier
    from src.evaluate import ModelEvaluator

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
    "tiet_kiem":      "Tiết kiệm",
    "healthy":        "Healthy",
    "nhanh_gon":      "Nhanh gọn",
    "no_bung":        "No bụng",
    "ngon_doi_gio":   "Ngon đổi gió",
    "can_bang":       "Cân bằng",
    "chao_hoi":       "Chào hỏi",
    "cam_on_tam_biet":"Cảm ơn/Bye",
}

CRITERIA_NAMES = {
    "gia":         "Giá",
    "khoang_cach": "Khoảng cách",
    "thoi_gian":   "Thời gian",
    "do_no":       "Độ no",
    "dinh_duong":  "Dinh dưỡng",
    "do_ngon":     "Độ ngon",
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
    padding: 8px 18px;
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
    padding: 8px 18px;
    font-weight: bold;
}}
QPushButton:hover {{ background: #6A5ADF; }}
QPushButton:pressed {{ background: #5A4ACF; }}
QPushButton:disabled {{ background: {C_BORDER}; color: {C_TEXT_DIM}; }}
QLineEdit, QTextEdit {{
    background: {C_CARD};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 6px 10px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
}}
QLineEdit:focus, QTextEdit:focus {{
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
# QTHREAD HUẤN LUYỆN
# ══════════════════════════════════════════════════════════════════════════════
class TrainingThread(QThread):
    """Huấn luyện mô hình trong luồng nền để GUI không bị đơ."""
    finished   = pyqtSignal(dict)   # {success, message, classifier}
    progress   = pyqtSignal(str)

    def __init__(self, alpha: float, model_type: str,
                 intents_path: str, model_save_path: str, parent=None):
        super().__init__(parent)
        self.alpha = alpha
        self.model_type = model_type
        self.intents_path = intents_path
        self.model_save_path = model_save_path

    def run(self):
        try:
            self.progress.emit("⏳ Đang tiền xử lý dữ liệu...")
            clf = NeedClassifier(model_type=self.model_type, alpha=self.alpha)
            self.progress.emit("⏳ Đang fit TF-IDF và mô hình Naive Bayes...")
            summary = clf.train(self.intents_path)
            self.progress.emit("💾 Đang lưu mô hình vào outputs/...")
            clf.save(self.model_save_path)
            self.progress.emit("✅ Hoàn tất!")
            self.finished.emit({"success": True, "message":
                f"Huấn luyện xong! {summary['num_samples']} mẫu, "
                f"{summary['num_classes']} nhãn, mô hình: {summary['model_type'].upper()}",
                "classifier": clf})
        except Exception as e:
            self.finished.emit({"success": False, "message": str(e), "classifier": None})


# ══════════════════════════════════════════════════════════════════════════════
# WIDGET BONG BÓNG CHAT
# ══════════════════════════════════════════════════════════════════════════════
class ChatBubble(QWidget):
    """Widget hiển thị một tin nhắn trong khung chat dưới dạng bong bóng."""
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.is_user = is_user
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
            # Bot avatar
            av = QLabel("🤖")
            av.setFixedSize(32, 32)
            av.setAlignment(Qt.AlignmentFlag.AlignCenter)
            av.setStyleSheet(f"background:{C_CARD}; border-radius:16px; font-size:18px;")
            outer.addWidget(av)
            outer.addWidget(lbl)
            outer.addStretch()


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
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        # Trạng thái bot
        self.bot: Optional[Chatbot] = None
        self.is_trained = False
        self.train_thread: Optional[TrainingThread] = None
        self._last_response: Optional[Dict[str, Any]] = None

        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._init_bot_async()

    # ── Khởi tạo bot không chặn UI ────────────────────────────────────────────
    def _init_bot_async(self):
        QTimer.singleShot(200, self._try_load_bot)

    def _new_bot(self) -> Chatbot:
        """Tạo Chatbot mới với đường dẫn tuyệt đối (không phụ thuộc thư mục chạy)."""
        return Chatbot(
            model_path=os.path.join(_ROOT, "outputs", "classifier_model.joblib"),
            dishes_path=os.path.join(_ROOT, "data", "dishes.csv"),
            profiles_path=os.path.join(_ROOT, "config", "profiles.json"),
            confidence_threshold=self.spin_threshold.value(),
            default_top_k=self.spin_topk.value()
        )

    def _try_load_bot(self):
        try:
            self.bot = self._new_bot()
            if self.bot.classifier.pipeline is None:
                self.bot = None
                raise RuntimeError("chưa có mô hình đã huấn luyện và không tìm thấy data/intents.csv")
            self.is_trained = True
            self._status("✅ Mô hình đã được tải. Sẵn sàng tư vấn!")
            self._append_bot_message(
                "Xin chào! Mình là <b>NEU Lunch Bot</b> 🍜<br>"
                "Hãy cho mình biết bạn muốn ăn gì trưa nay nhé!<br>"
                "<i>(Ví dụ: \"ăn gì rẻ dưới 35k không cay gần ktx neu\")</i>"
            )
        except Exception as e:
            self._status(f"⚠️ Chưa có mô hình: {e}. Nhấn 'Huấn luyện' để bắt đầu.")
            self._append_bot_message(
                "⚠️ Chưa có mô hình huấn luyện.<br>"
                "Vui lòng nhấn <b>Huấn luyện lại</b> ở tab <i>Tham số</i> để bắt đầu."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Splitter: trái chat | phải tabs ────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {C_BORDER}; }}")
        root.addWidget(splitter)

        splitter.addWidget(self._build_chat_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([440, 820])

        # ── Status bar ─────────────────────────────────────────────────────
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(f"background:{C_SURFACE}; color:{C_TEXT_DIM};")
        self.status_bar.showMessage("Đang khởi động...")

    # ── PANEL TRÁI: CHAT ──────────────────────────────────────────────────────
    def _build_chat_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(340)
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

        btn_clear = QPushButton("🗑")
        btn_clear.setToolTip("Xóa cuộc hội thoại")
        btn_clear.setFixedSize(36, 36)
        btn_clear.setStyleSheet(f"background:{C_CARD}; color:{C_TEXT_DIM}; border-radius:8px; font-size:16px;")
        btn_clear.clicked.connect(self._clear_chat)
        btn_row.addWidget(btn_clear)

        input_vbox.addLayout(btn_row)
        vbox.addWidget(input_frame)

        # Budget quick filter
        budget_row = QHBoxLayout()
        budget_row.setSpacing(6)
        budget_row.addWidget(QLabel("Ngân sách (tuỳ chọn):"))
        self.spin_budget = QSpinBox()
        self.spin_budget.setRange(0, 200)
        self.spin_budget.setSuffix("k")
        self.spin_budget.setSpecialValueText("Không giới hạn")
        self.spin_budget.setValue(0)
        self.spin_budget.setFixedWidth(120)
        budget_row.addWidget(self.spin_budget)
        self.chk_no_spicy = QCheckBox("Không cay")
        self.chk_no_spicy.setStyleSheet(f"color:{C_TEXT};")
        budget_row.addWidget(self.chk_no_spicy)
        budget_row.addStretch()
        vbox.addLayout(budget_row)

        return panel

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
        return panel

    # ── TAB: QUẢN LÝ DỮ LIỆU ─────────────────────────────────────────────────
    def _build_data_tab(self) -> QWidget:
        from gui.data_tab import DataManagerTab
        self.data_tab = DataManagerTab(
            dishes_path=os.path.join(_ROOT, "data", "dishes.csv"),
            intents_path=os.path.join(_ROOT, "data", "intents.csv"),
        )
        self.data_tab.dishes_changed.connect(self._on_dishes_changed)
        self.data_tab.intents_changed.connect(
            lambda: self._status("ℹ️ Đã thêm câu mẫu mới – hãy huấn luyện lại để mô hình học dữ liệu mới."))
        self.data_tab.retrain_requested.connect(self._on_train)
        return self.data_tab

    def _on_dishes_changed(self):
        """Nạp lại danh sách món cho bộ gợi ý ngay sau khi sửa dishes.csv (không cần huấn luyện lại)."""
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
        self.lbl_orig      = QLabel("—")
        self.lbl_norm      = QLabel("—")
        self.lbl_tokens    = QLabel("—")
        self.lbl_no_sw     = QLabel("—")
        for lbl, text in [(self.lbl_orig, "Câu gốc:"),
                          (self.lbl_norm, "Chuẩn hóa:"),
                          (self.lbl_tokens, "Tokens:"),
                          (self.lbl_no_sw, "Tokens (bỏ SW):")]:
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
        self.tbl_tfidf.setFixedHeight(160)
        lay2.addWidget(self.tbl_tfidf)
        vbox.addWidget(grp2)

        # 3. Xác suất nhãn
        grp3, lay3 = _make_step_group("3️⃣  Xác suất 8 nhãn ý định")
        self.tbl_probs = self._make_table(["Nhãn", "Xác suất"])
        self.tbl_probs.setFixedHeight(230)
        lay3.addWidget(self.tbl_probs)
        vbox.addWidget(grp3)

        # 4. Ràng buộc
        grp4, lay4 = _make_step_group("4️⃣  Ràng buộc trích xuất")
        self.lbl_max_price   = QLabel("—")
        self.lbl_excl_spicy  = QLabel("—")
        self.lbl_excl_types  = QLabel("—")
        for lbl, txt in [
            (self.lbl_max_price,  "Ngân sách:"),
            (self.lbl_excl_spicy, "Loại trừ cay:"),
            (self.lbl_excl_types, "Loại trừ món:"),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b style='color:{C_ACCENT2}'>{txt}</b>"))
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{C_TEXT}; background:{C_CARD}; border-radius:4px; padding:4px 8px;")
            row.addWidget(lbl, 1)
            lay4.addLayout(row)
        vbox.addWidget(grp4)

        # 5. Trọng số
        grp5, lay5 = _make_step_group("5️⃣  Trọng số tiêu chí đang dùng")
        self.tbl_weights = self._make_table(["Tiêu chí", "Trọng số"])
        self.tbl_weights.setFixedHeight(195)
        lay5.addWidget(self.tbl_weights)
        vbox.addWidget(grp5)

        # 6. Điểm từng món
        grp6, lay6 = _make_step_group("6️⃣  Điểm đóng góp từng tiêu chí theo món")
        self.tbl_scores = self._make_table(
            ["Món", "Tổng", "Giá", "Khoảng cách", "Thời gian", "Độ no", "Dinh dưỡng", "Độ ngon"]
        )
        self.tbl_scores.setFixedHeight(150)
        lay6.addWidget(self.tbl_scores)
        vbox.addWidget(grp6)

        vbox.addStretch()
        return scroll

    # ── TAB: BIỂU ĐỒ ─────────────────────────────────────────────────────────
    def _build_charts_tab(self) -> QWidget:
        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.setSpacing(6)
        vbox.setContentsMargins(6, 6, 6, 6)

        title = QLabel("📈  Trực quan hoá kết quả gợi ý")
        title.setStyleSheet(f"font-size:14px; font-weight:bold; color:{C_ACCENT};")
        vbox.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{C_BORDER};}}")

        # Biểu đồ 1: Điểm đóng góp tiêu chí (stacked bar)
        grp_top = QGroupBox("Điểm đóng góp của từng tiêu chí cho mỗi món (Stacked Bar)")
        top_lay = QVBoxLayout(grp_top)
        self.canvas_stacked = MplCanvas(figsize=(7, 3.2))
        top_lay.addWidget(self.canvas_stacked)
        splitter.addWidget(grp_top)

        # Biểu đồ 2: Xác suất nhãn (horizontal bar)
        grp_bot = QGroupBox("Xác suất phân loại ý định (Intent Probability)")
        bot_lay = QVBoxLayout(grp_bot)
        self.canvas_probs = MplCanvas(figsize=(7, 2.8))
        bot_lay.addWidget(self.canvas_probs)
        splitter.addWidget(grp_bot)

        splitter.setSizes([380, 280])
        vbox.addWidget(splitter, 1)
        return widget

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
        grp_model = QGroupBox("🤖  Tham số mô hình")
        form = QFormLayout(grp_model)
        form.setSpacing(10)
        form.setContentsMargins(12, 16, 12, 12)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.01, 10.0)
        self.spin_alpha.setSingleStep(0.1)
        self.spin_alpha.setValue(0.5)
        self.spin_alpha.setToolTip("Smoothing α cho Naive Bayes (mặc định 0.5)")
        form.addRow("Alpha (Naive Bayes):", self.spin_alpha)

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

        self.chk_remove_sw = QCheckBox("Bật bỏ stopword")
        self.chk_remove_sw.setChecked(True)
        form.addRow("Tiền xử lý:", self.chk_remove_sw)

        vbox.addWidget(grp_model)

        # ─── Trọng số tiêu chí ─────────────────────────────────────────────
        grp_weights = QGroupBox("⚖️  Trọng số 6 tiêu chí (tùy chỉnh)")
        w_vbox = QVBoxLayout(grp_weights)
        w_vbox.setSpacing(8)
        w_vbox.setContentsMargins(12, 16, 12, 12)

        self.weight_sliders: Dict[str, QSlider] = {}
        self.weight_labels:  Dict[str, QLabel]  = {}

        criteria_order = ["gia", "khoang_cach", "thoi_gian", "do_no", "dinh_duong", "do_ngon"]
        for crit in criteria_order:
            row = QHBoxLayout()
            name_lbl = QLabel(f"{CRITERIA_NAMES[crit]}:")
            name_lbl.setFixedWidth(110)
            name_lbl.setStyleSheet(f"color:{C_TEXT};")
            row.addWidget(name_lbl)

            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 100)
            sl.setValue(17)   # bắt đầu đều
            sl.setFixedHeight(20)
            sl.valueChanged.connect(self._on_weight_changed)
            self.weight_sliders[crit] = sl
            row.addWidget(sl, 1)

            val_lbl = QLabel("0.17")
            val_lbl.setFixedWidth(42)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet(f"color:{C_ACCENT2}; font-weight:bold;")
            self.weight_labels[crit] = val_lbl
            row.addWidget(val_lbl)

            w_vbox.addLayout(row)

        # Nút điều khiển trọng số
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

    # ── TAB: ĐÁNH GIÁ ────────────────────────────────────────────────────────
    def _build_eval_tab(self) -> QWidget:
        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.setSpacing(8)
        vbox.setContentsMargins(10, 10, 10, 10)

        title = QLabel("🎯  Kết quả đánh giá mô hình trên tập kiểm thử")
        title.setStyleSheet(f"font-size:14px; font-weight:bold; color:{C_ACCENT};")
        vbox.addWidget(title)

        # Accuracy summary
        self.lbl_accuracy = QLabel("Accuracy: —")
        self.lbl_accuracy.setStyleSheet(f"font-size:18px; font-weight:bold; color:{C_SUCCESS}; padding:8px;")
        vbox.addWidget(self.lbl_accuracy)

        # Classification report
        grp_report = QGroupBox("📋  Classification Report")
        rep_lay = QVBoxLayout(grp_report)
        rep_lay.setContentsMargins(8, 12, 8, 8)
        self.tbl_report = self._make_table(["Nhãn", "Precision", "Recall", "F1-Score", "Support"])
        self.tbl_report.setAlternatingRowColors(True)
        rep_lay.addWidget(self.tbl_report)
        vbox.addWidget(grp_report)

        # Confusion matrix
        grp_cm = QGroupBox("🔢  Confusion Matrix")
        cm_lay = QVBoxLayout(grp_cm)
        cm_lay.setContentsMargins(8, 12, 8, 8)
        self.canvas_cm = MplCanvas(figsize=(7, 4.5))
        cm_lay.addWidget(self.canvas_cm)
        vbox.addWidget(grp_cm, 1)

        btn_run_eval = QPushButton("▶  Chạy đánh giá ngay")
        btn_run_eval.setMinimumHeight(36)
        btn_run_eval.clicked.connect(self._run_evaluation)
        vbox.addWidget(btn_run_eval)

        return widget

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
        tbl.setStyleSheet(f"font-size:12px;")
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
        text = self.input_box.text().strip()
        if not text:
            self._shake_input()
            return
        if not self.is_trained or self.bot is None:
            QMessageBox.warning(self, "Chưa có mô hình",
                "Vui lòng nhấn 'Huấn luyện lại' ở tab Tham số trước khi chat.")
            return

        self.input_box.clear()
        self._append_user_message(text)
        self.btn_send.setEnabled(False)

        # Áp dụng tham số override
        budget = self.spin_budget.value() * 1000 if self.spin_budget.value() > 0 else None
        no_spicy = self.chk_no_spicy.isChecked()
        self.bot.confidence_threshold = self.spin_threshold.value()
        self.bot.default_top_k = self.spin_topk.value()

        # Custom weights
        custom_w = self._get_custom_weights()

        # Inject override vào extractor
        try:
            resp = self.bot.respond(
                text,
                confidence_threshold=self.spin_threshold.value(),
                custom_weights=custom_w if self._weights_customized() else None,
                top_k=self.spin_topk.value(),
                remove_stopwords=self.chk_remove_sw.isChecked()
            )

            # Nếu budget/no_spicy được set trong UI, override vào constraints
            if budget is not None:
                if resp["constraints"].get("max_price") is None:
                    resp["constraints"]["max_price"] = budget
            if no_spicy:
                resp["constraints"]["exclude_spicy"] = True

            # Nếu constraints bị override, chạy lại recommender
            if (budget is not None or no_spicy) and resp["recommendation"] is not None:
                effective = resp["effective_intent"]
                rec = self.bot.recommender.recommend(
                    intent=effective,
                    constraints=resp["constraints"],
                    custom_weights=custom_w if self._weights_customized() else None,
                    top_k=self.spin_topk.value()
                )
                resp["recommendation"] = rec
                resp["reply"] = self._rebuild_reply(resp)

            self._last_response = resp
            self._update_steps_tab(resp)
            self._update_charts(resp)
            self._append_bot_message(resp["reply"])

        except Exception as e:
            self._append_bot_message(f"⚠️ Lỗi xử lý: {str(e)}")
        finally:
            self.btn_send.setEnabled(True)
            self.input_box.setFocus()

    def _rebuild_reply(self, resp: Dict[str, Any]) -> str:
        """Tạo lại câu trả lời sau khi recommendation bị override."""
        rec = resp["recommendation"]
        top = rec["top_k"]
        explanation = resp["explanation"]
        if not top:
            return "Sau khi lọc theo các ràng buộc bổ sung, không tìm thấy món phù hợp. Bạn thử nới lỏng điều kiện xem!"
        lines = [f"Dạ, {explanation} Dưới đây là {len(top)} gợi ý:\n"]
        for i, item in enumerate(top, 1):
            d = item["dish"]
            cay = "🌶️ Cay" if d["cay"] == 1 else "🟢 Không cay"
            lines.append(
                f"{i}. 🍽️ <b>{d['ten_mon']}</b> ({d['loai'].capitalize()})<br>"
                f"&nbsp;&nbsp;&nbsp;💰 {d['gia_vnd']:,.0f}đ | 📍 {d['khoang_cach_m']:.0f}m | "
                f"⏱️ ~{d['thoi_gian_phuc_vu_phut']:.0f}p | ⭐ {item['score']:.2f} | {cay}<br>"
                f"&nbsp;&nbsp;&nbsp;📝 {d['mo_ta']}"
            )
        lines.append("<br>Chúc bạn ngon miệng! 😊")
        return "<br>".join(lines)

    def _append_user_message(self, text: str):
        bubble = ChatBubble(text, is_user=True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _append_bot_message(self, text: str):
        # Chuyển markdown bold **...** sang <b>...</b>
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        # Chuyển \n sang <br>
        text = text.replace("\n", "<br>")
        bubble = ChatBubble(text, is_user=False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        QTimer.singleShot(50, lambda: self.chat_scroll.verticalScrollBar().setValue(
            self.chat_scroll.verticalScrollBar().maximum()
        ))

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
        self._fill_table(self.tbl_tfidf,
            [[t["term"], f"{t['tfidf']:.4f}"] for t in tfidf])

        # 3. Xác suất
        probs = clf.get("probabilities", [])
        rows = []
        for p in probs:
            lbl = LABEL_NAMES.get(p["label"], p["label"])
            rows.append([lbl, f"{p['prob']:.4f}  ({p['prob']*100:.1f}%)"])
        self._fill_table(self.tbl_probs, rows)

        # 4. Ràng buộc
        mp = con.get("max_price")
        self.lbl_max_price.setText(f"{mp:,.0f}đ" if mp else "Không giới hạn")
        self.lbl_excl_spicy.setText("✅ Không ăn cay" if con.get("exclude_spicy") else "Không ràng buộc")
        exc = con.get("exclude_types", [])
        self.lbl_excl_types.setText(", ".join(exc) if exc else "Không ràng buộc")

        # 5. Trọng số đang dùng
        if rec:
            w = rec.get("weights_used", {})
            w_rows = [[CRITERIA_NAMES.get(k, k), f"{v:.3f}"] for k, v in w.items()]
            self._fill_table(self.tbl_weights, w_rows)

        # 6. Điểm từng tiêu chí
        if rec:
            score_rows = []
            for item in rec.get("top_k", []):
                d = item["dish"]
                c = item["contributions"]
                score_rows.append([
                    d["ten_mon"][:28],
                    f"{item['score']:.3f}",
                    f"{c.get('gia',0):.3f}",
                    f"{c.get('khoang_cach',0):.3f}",
                    f"{c.get('thoi_gian',0):.3f}",
                    f"{c.get('do_no',0):.3f}",
                    f"{c.get('dinh_duong',0):.3f}",
                    f"{c.get('do_ngon',0):.3f}",
                ])
            self._fill_table(self.tbl_scores, score_rows)

    # ══════════════════════════════════════════════════════════════════════════
    # VẼ BIỂU ĐỒ
    # ══════════════════════════════════════════════════════════════════════════
    def _update_charts(self, resp: Dict[str, Any]):
        self._draw_stacked_bar(resp)
        self._draw_prob_bar(resp)

    def _draw_stacked_bar(self, resp: Dict[str, Any]):
        """Vẽ biểu đồ cột xếp chồng đóng góp tiêu chí."""
        self.canvas_stacked.fig.clf()
        rec = resp.get("recommendation")
        if not rec or not rec.get("top_k"):
            ax = self.canvas_stacked.fig.add_subplot(111)
            ax.set_facecolor(C_CARD)
            ax.text(0.5, 0.5, "Không có dữ liệu gợi ý",
                    ha="center", va="center", color=C_TEXT_DIM,
                    fontsize=12, fontfamily=VIET_FONT)
            self.canvas_stacked.draw()
            return

        top = rec["top_k"]
        criteria = ["gia", "khoang_cach", "thoi_gian", "do_no", "dinh_duong", "do_ngon"]
        colors = ["#7C6AF7", "#50C5B7", "#F4B942", "#3ECA8D", "#F47F4C", "#E05CB0"]
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
            bars = ax.bar(x, data[i], bottom=bottom, color=color, width=0.55,
                         label=CRITERIA_NAMES[crit], alpha=0.9)
            bottom += data[i]

        # Tổng điểm trên đỉnh cột
        for j, item in enumerate(top):
            ax.text(j, float(bottom[j]) + 0.005, f"{item['score']:.3f}",
                    ha="center", va="bottom", color=C_TEXT, fontsize=9,
                    fontweight="bold", fontfamily=VIET_FONT)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color=C_TEXT,
                           rotation=10, ha="right", fontfamily=VIET_FONT)
        ax.set_ylim(0, max(1.0, float(np.max(bottom)) * 1.2))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.3,
                  labelcolor=C_TEXT, facecolor=C_CARD)
        _ax_style(ax, title="Điểm đóng góp tiêu chí", ylabel="Điểm")
        self.canvas_stacked.fig.tight_layout()
        self.canvas_stacked.draw()

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

        # Giá trị
        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{v*100:.1f}%", va="center", color=C_TEXT, fontsize=9,
                    fontfamily=VIET_FONT)

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
        intents_path = os.path.join(_ROOT, "data", "intents.csv")
        if not os.path.exists(intents_path):
            QMessageBox.critical(self, "Lỗi", f"Không tìm thấy file:\n{intents_path}")
            return

        self.btn_train.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_train_status.setText("⏳ Đang huấn luyện mô hình...")
        self._status("⏳ Đang huấn luyện...")
        if hasattr(self, "data_tab"):
            self.data_tab.set_training_state(True, "⏳ Đang huấn luyện mô hình...")

        model_save = os.path.join(_ROOT, "outputs", "classifier_model.joblib")
        os.makedirs(os.path.join(_ROOT, "outputs"), exist_ok=True)

        self.train_thread = TrainingThread(
            alpha=self.spin_alpha.value(),
            model_type="nb",
            intents_path=intents_path,
            model_save_path=model_save
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
            self._status(f"✅ {result['message']}")
            # Cập nhật bot với mô hình mới (đồng thời nạp lại dishes.csv mới nhất)
            if result.get("classifier"):
                self.bot = self._new_bot()
                self.bot.classifier = result["classifier"]
                self.is_trained = True
                if hasattr(self, "data_tab"):
                    self.data_tab.refresh_intent_counts()
            self._append_bot_message(
                "✅ Đã huấn luyện lại mô hình thành công! Bạn có thể bắt đầu chat nhé."
            )
        else:
            self.lbl_train_status.setText(f"❌ Lỗi: {result['message']}")
            self._status(f"❌ Lỗi huấn luyện: {result['message']}")
            QMessageBox.critical(self, "Lỗi huấn luyện", result["message"])

    # ══════════════════════════════════════════════════════════════════════════
    # ĐÁNH GIÁ
    # ══════════════════════════════════════════════════════════════════════════
    def _run_evaluation(self):
        if not self.is_trained or self.bot is None:
            QMessageBox.warning(self, "Chưa có mô hình",
                "Vui lòng huấn luyện mô hình trước khi đánh giá.")
            return

        test_path = os.path.join(_ROOT, "data", "test_intents.csv")
        if not os.path.exists(test_path):
            QMessageBox.warning(self, "Thiếu dữ liệu",
                f"Không tìm thấy file test:\n{test_path}")
            return

        try:
            evaluator = ModelEvaluator(test_csv_path=test_path)
            result = evaluator.evaluate(self.bot.classifier)

            # Accuracy
            acc = result["accuracy"]
            self.lbl_accuracy.setText(f"Accuracy: {acc:.2%}")
            color = C_SUCCESS if acc >= 0.8 else C_WARNING if acc >= 0.6 else C_ERROR
            self.lbl_accuracy.setStyleSheet(
                f"font-size:18px; font-weight:bold; color:{color}; padding:8px;"
            )

            # Classification report table
            report = result.get("report_dict", {})
            labels = result.get("labels", [])
            rep_rows = []
            for lbl in labels:
                if lbl in report:
                    m = report[lbl]
                    rep_rows.append([
                        LABEL_NAMES.get(lbl, lbl),
                        f"{m.get('precision', 0):.3f}",
                        f"{m.get('recall', 0):.3f}",
                        f"{m.get('f1-score', 0):.3f}",
                        str(int(m.get('support', 0)))
                    ])
            self._fill_table(self.tbl_report, rep_rows)

            # Confusion matrix plot
            self._draw_confusion_matrix(result)

            self._status(f"✅ Đánh giá xong. Accuracy = {acc:.2%}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi đánh giá", str(e))

    def _draw_confusion_matrix(self, result: dict):
        self.canvas_cm.fig.clf()
        cm     = np.array(result["confusion_matrix"])
        labels = [LABEL_NAMES.get(l, l) for l in result["labels"]]

        ax = self.canvas_cm.fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right",
                           fontsize=8, color=C_TEXT, fontfamily=VIET_FONT)
        ax.set_yticklabels(labels, fontsize=8, color=C_TEXT, fontfamily=VIET_FONT)

        cm_max = cm.max() if cm.max() > 0 else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm[i, j]
                color = "white" if val > cm_max * 0.5 else C_TEXT
                ax.text(j, i, str(val), ha="center", va="center",
                        color=color, fontsize=9, fontweight="bold",
                        fontfamily=VIET_FONT)

        _ax_style(ax, title="Confusion Matrix", xlabel="Dự đoán", ylabel="Thực tế")
        self.canvas_cm.fig.tight_layout()
        self.canvas_cm.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # TRỌNG SỐ
    # ══════════════════════════════════════════════════════════════════════════
    def _on_weight_changed(self):
        """Cập nhật nhãn hiển thị khi thanh trượt thay đổi."""
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
        """Kiểm tra xem người dùng có tuỳ chỉnh trọng số không (khác đều 1/6)."""
        vals = [sl.value() for sl in self.weight_sliders.values()]
        mean_v = sum(vals) / len(vals)
        return any(abs(v - mean_v) > 3 for v in vals)

    def _use_auto_weights(self):
        """Đọc profile từ nhãn vừa gợi ý và áp lên các thanh trượt."""
        if not self.bot:
            return
        intent = None
        if self._last_response:
            intent = self._last_response.get("effective_intent")
        profile_weights = self.bot.recommender.profiles.get(intent or "can_bang", {})
        if not profile_weights:
            return
        crit_order = ["gia", "khoang_cach", "thoi_gian", "do_no", "dinh_duong", "do_ngon"]
        for crit in crit_order:
            w = profile_weights.get(crit, 1/6)
            sl = self.weight_sliders[crit]
            sl.blockSignals(True)
            sl.setValue(int(round(w * 100)))
            sl.blockSignals(False)
        self._on_weight_changed()
        self._status(f"✅ Đã áp dụng trọng số profile '{intent or 'can_bang'}'")

    def _reset_weights(self):
        for sl in self.weight_sliders.values():
            sl.blockSignals(True)
            sl.setValue(17)
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
