"""
Tab "Quản lý dữ liệu" của giao diện PyQt6:
  (1) Bảng xem/thêm/sửa/xóa món ăn -> lưu vào data/dishes_v2.csv (có kiểm tra hợp lệ 15 cột, gồm
      do_dong_cao_diem 1-5, so_cho_ngoi > 0, nguon_du_lieu ∈ {mô phỏng, quan sát, khảo sát}).
  (2) Form thêm câu mẫu mới vào data/intents.csv với nhãn chọn từ danh sách.
  (3) Nút "Huấn luyện lại" để cập nhật mô hình ngay sau khi sửa dữ liệu.
Logic đọc/ghi/kiểm tra nằm ở src/data_manager.py; lớp này chỉ lo giao diện.
"""

import os
from typing import Dict, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton, QLineEdit,
    QComboBox, QSpinBox, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QGridLayout, QProgressBar,
)

from src import data_manager as dm
from src.config import ALLOWED_DATA_SOURCES, DEFAULT_DO_DONG_CAO_DIEM, DEFAULT_SO_CHO_NGOI, DEFAULT_NGUON_DU_LIEU

_TABLE_HEADERS = [
    "ID", "Tên món", "Loại", "Giá (đ)", "Khoảng cách (m)", "Thời gian (p)",
    "Độ no", "Dinh dưỡng", "Độ ngon", "Cay", "Chay",
    "Độ đông (1-5)", "Chỗ ngồi", "Mô tả", "Nguồn",
]


class DataManagerTab(QWidget):
    """Widget quản lý dữ liệu món ăn và câu mẫu ý định."""

    dishes_changed = pyqtSignal()     # Đã lưu thay đổi món ăn -> cửa sổ chính nạp lại recommender
    intents_changed = pyqtSignal()    # Đã thêm câu mẫu -> cần huấn luyện lại để có hiệu lực
    retrain_requested = pyqtSignal()  # Người dùng bấm "Huấn luyện lại"

    def __init__(self, dishes_path: str, intents_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.dishes_path = dishes_path
        self.intents_path = intents_path
        self._selected_id: Optional[int] = None
        self._build_ui()
        self.reload_dishes()
        self.refresh_intent_counts()

    # ── Dựng giao diện ────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # (1) Món ăn
        grp_dish = QGroupBox("🍽️  Danh sách món ăn (data/dishes_v2.csv)")
        dish_lay = QVBoxLayout(grp_dish)
        dish_lay.setContentsMargins(8, 14, 8, 8)

        self.tbl_dishes = QTableWidget(0, len(_TABLE_HEADERS))
        self.tbl_dishes.setHorizontalHeaderLabels(_TABLE_HEADERS)
        self.tbl_dishes.verticalHeader().setVisible(False)
        self.tbl_dishes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl_dishes.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_dishes.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl_dishes.setAlternatingRowColors(True)
        self.tbl_dishes.setMinimumHeight(185)
        header = self.tbl_dishes.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(13, QHeaderView.ResizeMode.Stretch)
        self.tbl_dishes.itemSelectionChanged.connect(self._on_row_selected)
        dish_lay.addWidget(self.tbl_dishes)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self.in_name = QLineEdit()
        self.in_name.setPlaceholderText("Ví dụ: Cơm tấm sườn ngõ Tự Do")
        self.in_type = QComboBox()
        self.in_type.setEditable(True)
        self.in_price = self._spin(1000, 1_000_000, 35000, step=1000, suffix=" đ")
        self.in_dist = self._spin(0, 20_000, 300, step=50, suffix=" m")
        self.in_time = self._spin(0, 180, 8, suffix=" phút")
        self.in_full = self._spin(1, 5, 3)
        self.in_nutri = self._spin(1, 5, 3)
        self.in_taste = self._spin(1, 5, 3)
        self.in_crowd = self._spin(1, 5, DEFAULT_DO_DONG_CAO_DIEM)
        self.in_crowd.setToolTip("Mức độ đông giờ cao điểm trưa (1 = rất vắng, 5 = rất đông)")
        self.in_seats = self._spin(1, 1000, DEFAULT_SO_CHO_NGOI, step=5, suffix=" chỗ")
        self.in_seats.setToolTip("Tổng số chỗ ngồi của quán (> 0)")
        self.in_source = QComboBox()
        for src_opt in ALLOWED_DATA_SOURCES:
            self.in_source.addItem(src_opt, src_opt)
        self.in_spicy = QCheckBox("Món cay")
        self.in_veg = QCheckBox("Món chay")
        self.in_desc = QLineEdit()
        self.in_desc.setPlaceholderText("Mô tả ngắn về món / quán")

        form.addWidget(QLabel("Tên món:"), 0, 0)
        form.addWidget(self.in_name, 0, 1, 1, 3)
        form.addWidget(QLabel("Loại:"), 0, 4)
        form.addWidget(self.in_type, 0, 5)
        form.addWidget(QLabel("Giá:"), 1, 0)
        form.addWidget(self.in_price, 1, 1)
        form.addWidget(QLabel("Khoảng cách:"), 1, 2)
        form.addWidget(self.in_dist, 1, 3)
        form.addWidget(QLabel("Thời gian:"), 1, 4)
        form.addWidget(self.in_time, 1, 5)
        form.addWidget(QLabel("Độ no (1-5):"), 2, 0)
        form.addWidget(self.in_full, 2, 1)
        form.addWidget(QLabel("Dinh dưỡng (1-5):"), 2, 2)
        form.addWidget(self.in_nutri, 2, 3)
        form.addWidget(QLabel("Độ ngon (1-5):"), 2, 4)
        form.addWidget(self.in_taste, 2, 5)
        form.addWidget(QLabel("Độ đông (1-5):"), 3, 0)
        form.addWidget(self.in_crowd, 3, 1)
        form.addWidget(QLabel("Số chỗ ngồi:"), 3, 2)
        form.addWidget(self.in_seats, 3, 3)
        form.addWidget(QLabel("Nguồn dữ liệu:"), 3, 4)
        form.addWidget(self.in_source, 3, 5)
        form.addWidget(self.in_spicy, 4, 0)
        form.addWidget(self.in_veg, 4, 1)
        form.addWidget(QLabel("Mô tả:"), 4, 2)
        form.addWidget(self.in_desc, 4, 3, 1, 3)
        dish_lay.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("➕  Thêm mới")
        self.btn_update = QPushButton("💾  Cập nhật món đang chọn")
        self.btn_delete = QPushButton("🗑  Xóa món đang chọn")
        self.btn_clear = QPushButton("✖  Bỏ chọn / làm trống form")
        for b in (self.btn_add, self.btn_update, self.btn_delete, self.btn_clear):
            btn_row.addWidget(b)
        self.btn_add.clicked.connect(self._on_add_dish)
        self.btn_update.clicked.connect(self._on_update_dish)
        self.btn_delete.clicked.connect(self._on_delete_dish)
        self.btn_clear.clicked.connect(self._clear_form)
        dish_lay.addLayout(btn_row)

        self.lbl_dish_status = QLabel("")
        self.lbl_dish_status.setWordWrap(True)
        dish_lay.addWidget(self.lbl_dish_status)
        root.addWidget(grp_dish, 3)
        self._update_button_state()

        # (2) Câu mẫu
        grp_intent = QGroupBox("💬  Thêm câu mẫu mới (data/intents.csv)")
        int_lay = QVBoxLayout(grp_intent)
        int_lay.setContentsMargins(8, 14, 8, 8)

        row = QHBoxLayout()
        self.in_sentence = QLineEdit()
        self.in_sentence.setPlaceholderText("Nhập câu người dùng có thể nói, ví dụ: trưa nay muốn ăn gì rẻ dưới 30k")
        self.in_sentence.returnPressed.connect(self._on_add_intent)
        self.in_label = QComboBox()
        for key, display in dm.INTENT_LABELS.items():
            self.in_label.addItem(f"{display}  ({key})", key)
        self.btn_add_intent = QPushButton("➕  Thêm câu mẫu")
        self.btn_add_intent.clicked.connect(self._on_add_intent)
        row.addWidget(self.in_sentence, 3)
        row.addWidget(self.in_label, 1)
        row.addWidget(self.btn_add_intent)
        int_lay.addLayout(row)

        self.lbl_intent_counts = QLabel("")
        self.lbl_intent_counts.setWordWrap(True)
        int_lay.addWidget(self.lbl_intent_counts)
        self.lbl_intent_status = QLabel("")
        self.lbl_intent_status.setWordWrap(True)
        int_lay.addWidget(self.lbl_intent_status)
        root.addWidget(grp_intent, 1)

        # (3) Huấn luyện lại
        grp_train = QGroupBox("⚡  Cập nhật mô hình")
        tr_lay = QVBoxLayout(grp_train)
        tr_lay.setContentsMargins(8, 14, 8, 8)
        self.lbl_retrain_hint = QLabel(
            "Thay đổi món ăn có hiệu lực ngay khi lưu. Thêm câu mẫu cần <b>huấn luyện lại</b> "
            "để mô hình học dữ liệu mới (không cần khởi động lại ứng dụng)."
        )
        self.lbl_retrain_hint.setWordWrap(True)
        tr_lay.addWidget(self.lbl_retrain_hint)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        tr_lay.addWidget(self.progress)
        self.btn_retrain = QPushButton("⚡  Huấn luyện lại")
        self.btn_retrain.setMinimumHeight(38)
        self.btn_retrain.clicked.connect(self.retrain_requested.emit)
        tr_lay.addWidget(self.btn_retrain)
        self.lbl_train_status = QLabel("")
        self.lbl_train_status.setWordWrap(True)
        tr_lay.addWidget(self.lbl_train_status)
        root.addWidget(grp_train)

    @staticmethod
    def _spin(low: int, high: int, value: int, step: int = 1, suffix: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    # ── Món ăn: đọc dữ liệu / hiển thị ───────────────────────────────────────
    def reload_dishes(self) -> None:
        """Đọc lại dishes CSV và hiển thị lên bảng; báo lỗi thân thiện nếu thiếu/hỏng file."""
        try:
            dishes = dm.load_dishes(self.dishes_path)
        except (FileNotFoundError, ValueError) as e:
            self.tbl_dishes.setRowCount(0)
            self._set_status(self.lbl_dish_status, f"❌ {e}", error=True)
            self._set_dish_controls_enabled(False)
            return
        self._set_dish_controls_enabled(True)

        self.tbl_dishes.setRowCount(len(dishes))
        types = set()
        for r, d in enumerate(dishes):
            types.add(d["loai"])
            values = [
                d["id"],
                d["ten_mon"],
                d["loai"],
                f"{int(d['gia_vnd']):,}" if d["gia_vnd"].isdigit() else d["gia_vnd"],
                d["khoang_cach_m"],
                d["thoi_gian_phuc_vu_phut"],
                d["do_no"],
                d["dinh_duong"],
                d["do_ngon"],
                "Có" if d["cay"] == "1" else "Không",
                "Có" if d.get("chay", "0") == "1" else "Không",
                d.get("do_dong_cao_diem", str(DEFAULT_DO_DONG_CAO_DIEM)),
                d.get("so_cho_ngoi", str(DEFAULT_SO_CHO_NGOI)),
                d["mo_ta"],
                d.get("nguon_du_lieu", DEFAULT_NGUON_DU_LIEU),
            ]
            for c, v in enumerate(values):
                item = QTableWidgetItem(str(v))
                if c not in (1, 13):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tbl_dishes.setItem(r, c, item)

        current_type = self.in_type.currentText()
        self.in_type.clear()
        self.in_type.addItems(sorted(types))
        self.in_type.setEditText(current_type)
        self._set_status(self.lbl_dish_status, f"Đã tải {len(dishes)} món.")

    def _set_dish_controls_enabled(self, enabled: bool) -> None:
        for w in (self.btn_add, self.btn_update, self.btn_delete, self.btn_clear):
            w.setEnabled(enabled)
        if enabled:
            self._update_button_state()

    def _update_button_state(self) -> None:
        has_sel = self._selected_id is not None
        self.btn_update.setEnabled(has_sel)
        self.btn_delete.setEnabled(has_sel)

    def _on_row_selected(self) -> None:
        rows = self.tbl_dishes.selectionModel().selectedRows()
        if not rows:
            self._selected_id = None
            self._update_button_state()
            return
        r = rows[0].row()
        text = lambda c: self.tbl_dishes.item(r, c).text()
        self._selected_id = int(text(0))
        self.in_name.setText(text(1))
        self.in_type.setEditText(text(2))
        self.in_price.setValue(int(text(3).replace(",", "")))
        self.in_dist.setValue(int(text(4)))
        self.in_time.setValue(int(text(5)))
        self.in_full.setValue(int(text(6)))
        self.in_nutri.setValue(int(text(7)))
        self.in_taste.setValue(int(text(8)))
        self.in_spicy.setChecked(text(9) == "Có")
        self.in_veg.setChecked(text(10) == "Có")
        self.in_crowd.setValue(int(text(11)))
        self.in_seats.setValue(int(text(12)))
        self.in_desc.setText(text(13))
        src_val = text(14)
        idx = self.in_source.findData(src_val)
        if idx >= 0:
            self.in_source.setCurrentIndex(idx)
        self._update_button_state()

    def _clear_form(self) -> None:
        self.tbl_dishes.clearSelection()
        self._selected_id = None
        self.in_name.clear()
        self.in_type.setEditText("")
        self.in_price.setValue(35000)
        self.in_dist.setValue(300)
        self.in_time.setValue(8)
        self.in_full.setValue(3)
        self.in_nutri.setValue(3)
        self.in_taste.setValue(3)
        self.in_crowd.setValue(DEFAULT_DO_DONG_CAO_DIEM)
        self.in_seats.setValue(DEFAULT_SO_CHO_NGOI)
        self.in_source.setCurrentIndex(0)
        self.in_spicy.setChecked(False)
        self.in_veg.setChecked(False)
        self.in_desc.clear()
        self._update_button_state()

    def _read_form(self) -> Dict[str, object]:
        return {
            "ten_mon": self.in_name.text(),
            "loai": self.in_type.currentText(),
            "gia_vnd": self.in_price.value(),
            "khoang_cach_m": self.in_dist.value(),
            "thoi_gian_phuc_vu_phut": self.in_time.value(),
            "do_no": self.in_full.value(),
            "dinh_duong": self.in_nutri.value(),
            "do_ngon": self.in_taste.value(),
            "cay": 1 if self.in_spicy.isChecked() else 0,
            "chay": 1 if self.in_veg.isChecked() else 0,
            "do_dong_cao_diem": self.in_crowd.value(),
            "so_cho_ngoi": self.in_seats.value(),
            "mo_ta": self.in_desc.text(),
            "nguon_du_lieu": self.in_source.currentData() or self.in_source.currentText(),
        }

    # ── Món ăn: thêm / sửa / xóa ─────────────────────────────────────────────
    def _on_add_dish(self) -> None:
        try:
            added = dm.add_dish(self.dishes_path, self._read_form())
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Dữ liệu không hợp lệ", str(e))
            return
        self.reload_dishes()
        self._clear_form()
        self._set_status(self.lbl_dish_status, f"✅ Đã thêm món '{added['ten_mon']}' (id {added['id']}).")
        self.dishes_changed.emit()

    def _on_update_dish(self) -> None:
        if self._selected_id is None:
            return
        try:
            dm.update_dish(self.dishes_path, self._selected_id, self._read_form())
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Dữ liệu không hợp lệ", str(e))
            return
        sel_id = self._selected_id
        self.reload_dishes()
        self._select_by_id(sel_id)
        self._set_status(self.lbl_dish_status, f"✅ Đã cập nhật món id {sel_id}.")
        self.dishes_changed.emit()

    def _on_delete_dish(self) -> None:
        if self._selected_id is None:
            return
        name = self.in_name.text()
        answer = QMessageBox.question(
            self, "Xác nhận xóa", f"Xóa món '{name}' khỏi danh sách món ăn?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            dm.delete_dish(self.dishes_path, self._selected_id)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Không thể xóa", str(e))
            return
        self.reload_dishes()
        self._clear_form()
        self._set_status(self.lbl_dish_status, f"✅ Đã xóa món '{name}'.")
        self.dishes_changed.emit()

    def _select_by_id(self, dish_id: int) -> None:
        for r in range(self.tbl_dishes.rowCount()):
            if self.tbl_dishes.item(r, 0).text() == str(dish_id):
                self.tbl_dishes.selectRow(r)
                return

    # ── Câu mẫu ───────────────────────────────────────────────────────────────
    def _on_add_intent(self) -> None:
        label = self.in_label.currentData()
        try:
            cleaned = dm.add_intent(self.intents_path, self.in_sentence.text(), label)
        except (ValueError, OSError) as e:
            self._set_status(self.lbl_intent_status, f"❌ {e}", error=True)
            return
        self.in_sentence.clear()
        self.refresh_intent_counts()
        self._set_status(self.lbl_intent_status,
                         f"✅ Đã thêm câu mẫu vào nhãn '{label}': “{cleaned}”. Bấm “Huấn luyện lại” để áp dụng.")
        self.intents_changed.emit()

    def refresh_intent_counts(self) -> None:
        """Cập nhật dòng thống kê số câu mẫu theo nhãn."""
        try:
            counts = dm.count_intents_by_label(self.intents_path)
        except (FileNotFoundError, ValueError) as e:
            self.lbl_intent_counts.setText(f"❌ {e}")
            self.btn_add_intent.setEnabled(False)
            return
        self.btn_add_intent.setEnabled(True)
        parts = [f"{dm.INTENT_LABELS[k]}: <b>{v}</b>" for k, v in counts.items()]
        self.lbl_intent_counts.setText("Số câu mẫu hiện có — " + " · ".join(parts))

    # ── Huấn luyện: cửa sổ chính gọi để đồng bộ trạng thái ────────────────────
    def set_training_state(self, busy: bool, message: str = "") -> None:
        """Cập nhật nút/thanh tiến trình/thông báo khi huấn luyện bắt đầu hoặc kết thúc."""
        self.btn_retrain.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.lbl_train_status.setText(message)

    # ── Tiện ích ──────────────────────────────────────────────────────────────
    @staticmethod
    def _set_status(label: QLabel, text: str, error: bool = False) -> None:
        label.setText(text)
        label.setStyleSheet("color:#F47F4C;" if error else "color:#908CAA;")
