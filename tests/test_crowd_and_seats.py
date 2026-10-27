"""
Kiểm thử toàn diện cho tính năng tích hợp Độ đông (do_dong_cao_diem) và Chỗ ngồi (so_cho_ngoi)
gắn với Số người đi ăn (so_nguoi) và Khung giờ (khung_gio) — Mục A, B, C, D, E:
1. Kiểm thử tính tay (hand-calculated) cho thoi_gian_cho, cho_trong, cho_ngoi_nhom với so_nguoi=1 và so_nguoi=6
   ở cả 3 khung giờ (cao_diem, binh_thuong, vang) và khi bật uu_tien_vang (nhân đôi hệ số a).
2. Kiểm thử khi so_nguoi=1 thì cho_ngoi_nhom gần như bằng nhau (= 1.0) giữa các món, đảm bảo xếp hạng tương thích.
3. Kiểm thử trích xuất ngôn ngữ tự nhiên:
   - so_nguoi: chữ số & chữ tiếng Việt 1-10 ('đi 4 người', '4 đứa', 'cả nhóm 6 bạn', 'một mình'=1, 'hai đứa'=2),
     ngoài khoảng ('đi 15 người') thì bỏ qua và ghi chú.
   - khung_gio: '12h', '12h30', '11 giờ rưỡi', '13h15', '10h15'; không nhầm với '40k' hay '4 người'.
   - uu_tien_vang: 'chỗ nào vắng', 'đỡ đông', 'không đông', 'yên tĩnh'; không nhầm 'đông vui';
     'không phải chạy bộ' không nhầm thành chay.
4. Kiểm thử độ ưu tiên câu nói so với ô nhập UI, bộ lọc cứng 'Chỉ quán đủ chỗ cho cả nhóm' và tương thích ngược CSV cũ.
"""

import pytest

from src.config import infer_time_slot, load_crowd_config
from src.extractor import extract_constraints
from src.recommender import DishRecommender, compute_crowd_metrics, merge_constraints
from src.chatbot import Chatbot
from src.io_utils import write_csv_rows


# ══════════════════════════════════════════════════════════════════════════════
# 1. KIỂM THỬ TÍNH TAY CÔNG THỨC ĐỘ ĐÔNG, THỜI GIAN CHỜ & CHỖ NGỒI NHÓM
# ══════════════════════════════════════════════════════════════════════════════
def test_hand_calculated_crowd_formulas_so_nguoi_1_and_6():
    """
    Ví dụ tính tay với hệ số thật trong config/crowd_config.json:
      he_so_khung_gio = {cao_diem: 1.0, binh_thuong: 0.6, vang: 0.3}
      a = 1.0, b = 0.10, he_so_chiem_cho = 0.9

    Xét món mẫu:
      thoi_gian_phuc_vu_phut = 10.0, do_dong_cao_diem = 4, so_cho_ngoi = 20

    * Khung giờ cao_diem (he_so = 1.0):
      - do_dong_hieu_dung = 4 * 1.0 = 4.0
      - cho_trong = 20 * (1 - 0.9 * 4.0 / 5) = 20 * (1 - 0.72) = 20 * 0.28 = 5.6 chỗ
      + Với so_nguoi = 1:
        thoi_gian_cho = 10 * (1 + 1.0 * 4.0 / 5) * (1 + 0.10 * 0) = 10 * 1.8 * 1.0 = 18.0 phút
        cho_ngoi_nhom = min(1, 5.6 / 1) = 1.0 (Đủ chỗ, không cảnh báo)
      + Với so_nguoi = 6:
        thoi_gian_cho = 10 * 1.8 * (1 + 0.10 * 5) = 18.0 * 1.5 = 27.0 phút
        cho_ngoi_nhom = min(1, 5.6 / 6) = 5.6 / 6 = 0.9333
        Vì cho_trong = 5.6 < 6 -> cảnh báo 'Có thể phải chờ chỗ'

    * Khung giờ binh_thuong (he_so = 0.6):
      - do_dong_hieu_dung = 4 * 0.6 = 2.4
      - cho_trong = 20 * (1 - 0.9 * 2.4 / 5) = 20 * (1 - 0.432) = 11.36 chỗ
      + Với so_nguoi = 1:
        thoi_gian_cho = 10 * (1 + 1.0 * 2.4 / 5) * 1.0 = 10 * 1.48 = 14.8 phút
        cho_ngoi_nhom = min(1, 11.36 / 1) = 1.0
      + Với so_nguoi = 6:
        thoi_gian_cho = 14.8 * 1.5 = 22.2 phút
        cho_ngoi_nhom = min(1, 11.36 / 6) = 1.0 (Đủ chỗ)

    * Khung giờ vang (he_so = 0.3):
      - do_dong_hieu_dung = 4 * 0.3 = 1.2
      - cho_trong = 20 * (1 - 0.9 * 1.2 / 5) = 20 * (1 - 0.216) = 15.68 chỗ
      + Với so_nguoi = 1:
        thoi_gian_cho = 10 * (1 + 1.0 * 1.2 / 5) * 1.0 = 10 * 1.24 = 12.4 phút
        cho_ngoi_nhom = 1.0
      + Với so_nguoi = 6:
        thoi_gian_cho = 12.4 * 1.5 = 18.6 phút
        cho_ngoi_nhom = 1.0
    """
    cfg = load_crowd_config()
    dish = {
        "id": 1,
        "ten_mon": "Bún chả mẫu",
        "thoi_gian_phuc_vu_phut": 10.0,
        "do_dong_cao_diem": 4,
        "so_cho_ngoi": 20,
    }

    # 1. Cao điểm: so_nguoi = 1 và so_nguoi = 6
    m_peak_1 = compute_crowd_metrics(dish, so_nguoi=1, khung_gio="cao_diem", crowd_cfg=cfg)
    assert m_peak_1["do_dong_hieu_dung"] == pytest.approx(4.0)
    assert m_peak_1["thoi_gian_cho"] == pytest.approx(18.0)
    assert m_peak_1["cho_trong"] == pytest.approx(5.6)
    assert m_peak_1["cho_ngoi_nhom"] == pytest.approx(1.0)
    assert m_peak_1["seat_warning"] is None

    m_peak_6 = compute_crowd_metrics(dish, so_nguoi=6, khung_gio="cao_diem", crowd_cfg=cfg)
    assert m_peak_6["do_dong_hieu_dung"] == pytest.approx(4.0)
    assert m_peak_6["thoi_gian_cho"] == pytest.approx(27.0)
    assert m_peak_6["cho_trong"] == pytest.approx(5.6)
    assert m_peak_6["cho_ngoi_nhom"] == pytest.approx(5.6 / 6.0, abs=1e-4)
    assert m_peak_6["seat_warning"] == "Có thể phải chờ chỗ"

    # 2. Bình thường: so_nguoi = 1 và so_nguoi = 6
    m_norm_1 = compute_crowd_metrics(dish, so_nguoi=1, khung_gio="binh_thuong", crowd_cfg=cfg)
    assert m_norm_1["do_dong_hieu_dung"] == pytest.approx(2.4)
    assert m_norm_1["thoi_gian_cho"] == pytest.approx(14.8)
    assert m_norm_1["cho_trong"] == pytest.approx(11.36)
    assert m_norm_1["cho_ngoi_nhom"] == pytest.approx(1.0)
    assert m_norm_1["seat_warning"] is None

    m_norm_6 = compute_crowd_metrics(dish, so_nguoi=6, khung_gio="binh_thuong", crowd_cfg=cfg)
    assert m_norm_6["thoi_gian_cho"] == pytest.approx(22.2)
    assert m_norm_6["cho_trong"] == pytest.approx(11.36)
    assert m_norm_6["cho_ngoi_nhom"] == pytest.approx(1.0)
    assert m_norm_6["seat_warning"] is None

    # 3. Vắng: so_nguoi = 1 và so_nguoi = 6
    m_quiet_1 = compute_crowd_metrics(dish, so_nguoi=1, khung_gio="vang", crowd_cfg=cfg)
    assert m_quiet_1["do_dong_hieu_dung"] == pytest.approx(1.2)
    assert m_quiet_1["thoi_gian_cho"] == pytest.approx(12.4)
    assert m_quiet_1["cho_trong"] == pytest.approx(15.68)
    assert m_quiet_1["cho_ngoi_nhom"] == pytest.approx(1.0)

    m_quiet_6 = compute_crowd_metrics(dish, so_nguoi=6, khung_gio="vang", crowd_cfg=cfg)
    assert m_quiet_6["thoi_gian_cho"] == pytest.approx(18.6)
    assert m_quiet_6["cho_trong"] == pytest.approx(15.68)
    assert m_quiet_6["cho_ngoi_nhom"] == pytest.approx(1.0)

    # 4. Khi bật prefer_quiet (uu_tien_vang): hệ số a nhân đôi (a_eff = 2.0)
    # thoi_gian_cho ở cao_diem, 1 người = 10 * (1 + 2.0 * 4.0 / 5) * 1.0 = 26.0 phút
    m_prefer_quiet = compute_crowd_metrics(dish, so_nguoi=1, khung_gio="cao_diem", crowd_cfg=cfg, prefer_quiet=True)
    assert m_prefer_quiet["a_effective"] == pytest.approx(2.0)
    assert m_prefer_quiet["thoi_gian_cho"] == pytest.approx(26.0)


def test_infer_time_slot_rules():
    """Kiểm tra suy luận khung_gio từ đồng hồ theo đúng quy tắc mục B."""
    assert infer_time_slot("11:30") == "cao_diem"
    assert infer_time_slot("12:15") == "cao_diem"
    assert infer_time_slot("13:00") == "cao_diem"

    assert infer_time_slot("11:00") == "binh_thuong"
    assert infer_time_slot("11:29") == "binh_thuong"
    assert infer_time_slot("13:01") == "binh_thuong"
    assert infer_time_slot("14:00") == "binh_thuong"

    assert infer_time_slot("10:59") == "vang"
    assert infer_time_slot("14:01") == "vang"
    assert infer_time_slot("09:00") == "vang"


# ══════════════════════════════════════════════════════════════════════════════
# 2. KIỂM THỬ TƯƠNG THÍCH XẾP HẠNG KHI SO_NGUOI = 1 vs SO_NGUOI = 6
# ══════════════════════════════════════════════════════════════════════════════
def test_so_nguoi_1_ranking_compatibility_and_group_effect():
    """
    Khi so_nguoi = 1:
    - Ở khung giờ bình thường và vắng, 100% (50/50) món ăn trong dishes.csv có cho_trong >= 1
      nên cho_ngoi_nhom = 1.0 bằng nhau tuyệt đối cho mọi món (chuẩn hóa cho_ngoi = 1.0 đồng nhất).
    - Ở khung giờ cao điểm, >= 96% (48/50) món có cho_ngoi_nhom = 1.0.
    Khi tăng lên so_nguoi = 6 ở giờ cao điểm:
    - Các quán nhỏ/đông bị giảm cho_ngoi_nhom (< 1.0) và các quán rộng chỗ được ưu tiên tăng hạng.
    """
    rec = DishRecommender()

    res_norm_1 = rec.recommend("can_bang", num_people=1, time_slot="binh_thuong", top_k=50)
    cho_ngoi_vals_norm = [it["cho_ngoi_nhom"] for it in res_norm_1["all_ranked"]]
    assert all(v == pytest.approx(1.0) for v in cho_ngoi_vals_norm)

    res_peak_1 = rec.recommend("can_bang", num_people=1, time_slot="cao_diem", top_k=50)
    full_seat_score_count = sum(1 for it in res_peak_1["all_ranked"] if it["cho_ngoi_nhom"] == pytest.approx(1.0))
    assert full_seat_score_count >= 48  # 48/50 món đạt đúng 1.0 khi đi 1 người

    # Khi đi nhóm 8 người vào giờ cao điểm, top 3 gợi ý có số chỗ trống trung bình cao hơn hẳn khi đi 1 người
    res_peak_8 = rec.recommend("can_bang", num_people=8, time_slot="cao_diem", top_k=3)
    avg_seats_1 = sum(it["cho_trong"] for it in res_peak_1["top_k"][:3]) / 3.0
    avg_seats_8 = sum(it["cho_trong"] for it in res_peak_8["top_k"][:3]) / 3.0
    assert avg_seats_8 >= avg_seats_1


# ══════════════════════════════════════════════════════════════════════════════
# 3. KIỂM THỬ TRÍCH XUẤT SO_NGUOI, KHUNG_GIO, UU_TIEN_VANG & PHÂN BIỆT TỪ ĐỒNG ÂM
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("sentence,expected_people", [
    ("trưa nay đi 4 người ăn gì ngon", 4),
    ("bọn mình 4 đứa tìm quán bún", 4),
    ("cả nhóm 6 bạn muốn ăn cơm", 6),
    ("hôm nay ăn một mình thôi", 1),
    ("di an mot minh", 1),
    ("hai đứa đi ăn trưa", 2),
    ("nhóm năm người cần quán rộng", 5),
    ("đi mười người ăn lẩu", 10),
])
def test_extract_num_people_valid(sentence, expected_people):
    res = extract_constraints(sentence)
    assert res["num_people"] == expected_people
    assert res["intermediate"]["num_people_note"] is None


@pytest.mark.parametrize("sentence", [
    "đi 15 người ăn liên hoan",
    "nhóm 25 bạn tìm quán",
    "0 người",
])
def test_extract_num_people_out_of_range_ignored_with_note(sentence):
    res = extract_constraints(sentence)
    assert res["num_people"] is None
    assert res["intermediate"]["num_people_note"] is not None
    assert "ngoài khoảng" in res["intermediate"]["num_people_note"]


@pytest.mark.parametrize("sentence,expected_slot,expected_hhmm", [
    ("trưa nay 12h đi ăn bún chả", "cao_diem", "12:00"),
    ("hẹn nhau lúc 12h30 ở cổng trường", "cao_diem", "12:30"),
    ("tầm 11 giờ rưỡi mình ra ăn", "cao_diem", "11:30"),
    ("13h15 mới tan ca học", "binh_thuong", "13:15"),
    ("ăn sớm lúc 11h15 nhé", "binh_thuong", "11:15"),
    ("10h30 đói bụng quá ăn luôn", "vang", "10:30"),
    ("14h30 chiều mới đi ăn", "vang", "14:30"),
])
def test_extract_time_slot_and_no_confusion_with_budget_or_people(sentence, expected_slot, expected_hhmm):
    res = extract_constraints(sentence)
    assert res["time_slot"] == expected_slot
    assert res["parsed_time"] == expected_hhmm


def test_budget_and_people_do_not_trigger_time_slot():
    """Câu có '40k' và '4 người' nhưng không nói giờ thì time_slot phải là None."""
    res = extract_constraints("đi 4 người ăn gì dưới 40k không cay")
    assert res["max_price"] == 40000
    assert res["num_people"] == 4
    assert res["time_slot"] is None


@pytest.mark.parametrize("sentence", [
    "tìm chỗ nào vắng ngồi ăn trưa",
    "quán nào đỡ đông một chút",
    "tìm quán không đông nhé",
    "muốn tìm chỗ yên tĩnh ăn trưa",
    "cho nao vang de ngoi",
    "tim quan do dong",
])
def test_extract_prefer_quiet_positive(sentence):
    res = extract_constraints(sentence)
    assert res["prefer_quiet"] is True
    assert res["uu_tien_vang"] is True


@pytest.mark.parametrize("sentence", [
    "tìm quán nào đông vui nhộn nhịp",
    "đi ăn càng đông càng vui",
    "mình thích đông vui",
    "ăn bún đậu mắm tôm ngõ tự do",
])
def test_extract_prefer_quiet_negative_dong_vui(sentence):
    """'đông vui' tuyệt đối không bị nhận nhầm thành ưu tiên vắng."""
    res = extract_constraints(sentence)
    assert res["prefer_quiet"] is False


def test_khong_phai_chay_bo_not_confused_with_vegetarian():
    """'không phải chạy bộ' hoặc 'chạy bộ' không bao giờ nhận nhầm thành món chay."""
    for s in [
        "tìm quán gần trường để không phải chạy bộ xa",
        "khong phai chay bo giua trua nang",
        "vừa chạy bộ xong muốn ăn phở",
    ]:
        res = extract_constraints(s)
        assert res["vegetarian_only"] is False


# ══════════════════════════════════════════════════════════════════════════════
# 4. KIỂM THỬ CHATBOT, ĐỘ ƯU TIÊN CÂU NÓI, LỌC ĐỦ CHỖ & TƯƠNG THÍCH NGƯỢC CSV
# ══════════════════════════════════════════════════════════════════════════════
def test_text_overrides_ui_for_people_and_time_slot():
    bot = Chatbot()
    resp = bot.respond(
        "nhóm mình đi 6 người lúc 12h15 tìm quán ăn no bụng, chỗ nào vắng nhé",
        num_people=1,          # UI đang để 1 người
        time_slot="vang",      # UI đang để Vắng
    )
    con = resp["constraints"]
    assert con["num_people"] == 6
    assert con["sources"]["num_people"] == "từ câu nói"
    assert con["time_slot"] == "cao_diem"
    assert "từ câu nói" in con["sources"]["time_slot"]
    assert con["prefer_quiet"] is True

    ci = resp["recommendation"]["crowd_info"]
    assert ci["a_effective"] == pytest.approx(2.0 * ci["a_base"])
    assert "Chờ ước tính" in resp["reply"]
    assert "Độ đông:" in resp["reply"]
    assert "nhóm 6 người" in resp["explanation"].lower()


def test_require_enough_seats_filter_and_no_match_fallback():
    rec = DishRecommender()
    # Với 10 người ở giờ cao điểm, bật require_enough_seats=True sẽ lọc bỏ các quán có cho_trong < 10
    res = rec.recommend(
        "tiet_kiem",
        num_people=10,
        time_slot="cao_diem",
        require_enough_seats=True,
    )
    assert res["filtered_dishes_count"] < res["total_dishes"]
    assert all(it["cho_trong"] >= 10.0 for it in res["all_ranked"])

    # Kết hợp điều kiện thật chặt (chỉ món chay + dưới 25k + đủ 10 chỗ giờ cao điểm) để kiểm tra fallback
    res_no = rec.recommend(
        "healthy",
        max_price_per_person=25000,
        num_people=10,
        vegetarian_only=True,
        time_slot="cao_diem",
        require_enough_seats=True,
    )
    if res_no["filtered_dishes_count"] == 0:
        assert res_no["no_match"] is not None
        assert len(res_no["near_matches"]) == 3


def test_backward_compatibility_missing_crowd_and_seats_columns(tmp_path):
    """Đọc file CSV cũ thiếu cả 2 cột mới (do_dong_cao_diem, so_cho_ngoi) -> mặc định 3 và 30."""
    legacy_csv = tmp_path / "dishes_old.csv"
    header = [
        "id", "ten_mon", "loai", "gia_vnd", "khoang_cach_m",
        "thoi_gian_phuc_vu_phut", "do_no", "dinh_duong", "do_ngon", "cay", "chay", "mo_ta",
    ]
    rows = [
        [1, "Phở bò cũ", "phở", 40000, 200, 8, 4, 4, 4, 0, 0, "file cũ không có cột độ đông"],
    ]
    write_csv_rows(str(legacy_csv), header, rows)

    rec = DishRecommender(dishes_path=str(legacy_csv))
    assert len(rec.dishes) == 1
    assert rec.dishes[0]["do_dong_cao_diem"] == 3
    assert rec.dishes[0]["so_cho_ngoi"] == 30
    assert rec.dishes[0]["nguon_du_lieu"] == "mô phỏng"


def test_gui_crowd_and_seats_integration(monkeypatch):
    """Kiểm thử giao diện PyQt6 (offscreen): khung giờ, đủ chỗ, 7 thanh trượt, bảng độ đông, 4 biểu đồ và tab dữ liệu 15 cột."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from gui.app import LunchChatbotGUI

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(LunchChatbotGUI, "_init_bot_async", lambda self: None)
    win = LunchChatbotGUI()
    win._try_load_bot()

    # 1. Kiểm tra các điều khiển mới ở khung lọc và tab Tham số
    assert win.cmb_time_slot.count() == 4
    assert hasattr(win, "chk_enough_seats")
    assert len(win.weight_sliders) == 7
    assert "cho_ngoi" in win.weight_sliders
    assert win.tbl_scores.columnCount() == 9  # Món + Tổng + 7 tiêu chí
    assert win.tbl_crowd_dishes.columnCount() == 6
    assert win.data_tab.tbl_dishes.columnCount() == 15

    # 2. Gửi câu hỏi có số người, giờ đi ăn và ưu tiên vắng -> kiểm tra tự động cập nhật UI, bước xử lý và biểu đồ
    win.input_box.setText("đi 6 người lúc 12h15 tìm chỗ nào vắng ăn cơm")
    win._on_send()
    assert win.in_people.text() == "6"
    assert win.cmb_time_slot.currentData() == "cao_diem"
    assert win.tbl_crowd_dishes.rowCount() >= 1
    assert win.tbl_scores.rowCount() >= 1
    assert "nhân đôi" in win.lbl_crowd_a.text()

    # 3. Kiểm tra nút đặt lại hệ số độ đông mặc định
    win.spin_crowd_a.setValue(2.5)
    win._reset_crowd_params()
    assert win.spin_crowd_a.value() == pytest.approx(1.0)
    win.close()

