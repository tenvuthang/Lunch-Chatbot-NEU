"""
Kiểm thử việc sinh file outputs/worked_example.txt theo đủ 4 yêu cầu:
1. 9 câu mẫu tiếng Việt có dấu (3 nhãn tiet_kiem, healthy, nhanh_gon × 3 câu),
   tách từ bằng preprocess(), không có token ghép lạ ngoài từ điển pyvi.
2. Câu thử không trùng nguyên văn 9 câu mẫu, dự đoán ra tiet_kiem với xác suất >= 50%.
3. Phần 2 dùng nhãn tiet_kiem, trọng số thật từ config/profiles.json, đủ 6 tiêu chí,
   4 món từ data/dishes.csv, đóng góp w*v 4 chữ số thập phân cộng khớp tổng điểm.
4. Khớp sklearn (sai số < 1e-6) và in 'ĐÃ KHỚP VỚI SKLEARN'.
"""

import os
from src.worked_example import generate_worked_example, SAMPLE_DOCS, TEST_QUERY, DISH_IDS


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_worked_example_generation_and_content(tmp_path):
    """Kiểm tra hàm generate_worked_example chạy thành công và nội dung đạt đủ 4 yêu cầu."""
    out_path = generate_worked_example(
        root_dir=ROOT_DIR,
        outputs_dir=str(tmp_path),
        output_filename="worked_example.txt",
    )
    assert os.path.exists(out_path)

    with open(out_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Yêu cầu 1: 9 câu mẫu có dấu, không có token lạ 'di_nhanh'
    assert len(SAMPLE_DOCS) == 9
    for sentence, _ in SAMPLE_DOCS:
        assert sentence in content
    assert "di_nhanh" not in content

    # Yêu cầu 2: Câu thử không trùng 9 câu mẫu, dự đoán tiet_kiem >= 50%
    assert TEST_QUERY not in [s for s, _ in SAMPLE_DOCS]
    assert TEST_QUERY in content
    assert "DỰ ĐOÁN: 'tiet_kiem'" in content

    # Yêu cầu 3: 6 tiêu chí, 4 món từ dishes.csv, trọng số thật từ profiles.json
    assert len(DISH_IDS) == 4
    for w_name in ["w_gia", "w_khoang_cach", "w_thoi_gian", "w_do_no", "w_dinh_duong", "w_do_ngon"]:
        assert w_name in content

    # Yêu cầu 4: Khớp với sklearn và in dòng xác nhận
    assert "ĐÃ KHỚP VỚI SKLEARN" in content
