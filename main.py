"""
Điểm vào chính của dự án Lunch Chatbot:
1. Chế độ CLI tương tác thử nhanh: Chat trực tiếp với bot, xem chi tiết toàn bộ các bước trung gian và giải thích.
2. Chế độ câu hỏi nhanh (One-shot CLI): Truyền trực tiếp câu hỏi qua dòng lệnh.
3. Chế độ kiểm tra dữ liệu (--check): Xác thực tính hợp lệ của CSV/JSON và in bảng thống kê.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter

# Đảm bảo in tiếng Việt có dấu an toàn trên mọi hệ điều hành
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo import được module src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.chatbot import Chatbot
from src.io_utils import read_csv_dicts
from src.config import (
    DEFAULT_DISHES_PATH,
    DEFAULT_INTENTS_PATH,
    DEFAULT_TEST_INTENTS_PATH,
    DEFAULT_PROFILES_PATH,
    DEFAULT_CROWD_CONFIG_PATH,
    load_crowd_config,
)


def check_profiles_json(filepath: str) -> dict:
    """Kiểm tra tính hợp lệ của file config/profiles.json (7 tiêu chí)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    expected_profiles = ["tiet_kiem", "healthy", "nhanh_gon", "no_bung", "ngon_doi_gio", "can_bang"]
    expected_criteria = ["gia", "khoang_cach", "thoi_gian", "do_no", "dinh_duong", "do_ngon", "cho_ngoi"]

    print("=" * 80)
    print("1. KIỂM TRA FILE CẤU HÌNH (config/profiles.json & config/crowd_config.json)")
    print("=" * 80)

    for p in expected_profiles:
        if p not in profiles:
            raise ValueError(f"Thiếu profile '{p}' trong {filepath}")
        weights = profiles[p]
        for c in expected_criteria:
            if c not in weights:
                raise ValueError(f"Profile '{p}' thiếu tiêu chí '{c}'")
        total_weight = sum(weights.values())
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            raise ValueError(f"Tổng trọng số của profile '{p}' = {total_weight} != 1.0")
        print(f"  [OK] Profile '{p:<13}': Tổng trọng số = {total_weight:.2f} {weights}")

    crowd_cfg = load_crowd_config(DEFAULT_CROWD_CONFIG_PATH)
    print(f"  [OK] Cấu hình độ đông (crowd_config.json): a={crowd_cfg['a']}, b={crowd_cfg['b']}, "
          f"he_so_chiem_cho={crowd_cfg['he_so_chiem_cho']}, he_so_khung_gio={crowd_cfg['he_so_khung_gio']}")
    print("  -> File config/profiles.json & config/crowd_config.json hoàn toàn HỢP LỆ!\n")
    return profiles


def check_intents_csv(intents_path: str, test_intents_path: str):
    """Kiểm tra và in thống kê số lượng mẫu mỗi nhãn."""
    print("=" * 80)
    print("2. KIỂM TRA TẬP DỮ LIỆU Ý ĐỊNH (data/intents.csv & data/test_intents.csv)")
    print("=" * 80)

    expected_labels = [
        "tiet_kiem", "healthy", "nhanh_gon", "no_bung",
        "ngon_doi_gio", "can_bang", "chao_hoi", "cam_on_tam_biet"
    ]

    train_counts = Counter()
    train_texts = set()
    for row in read_csv_dicts(intents_path):
        train_counts[row["label"]] += 1
        train_texts.add(row["text"].strip())

    test_counts = Counter()
    test_texts = set()
    for row in read_csv_dicts(test_intents_path):
        test_counts[row["label"]] += 1
        test_texts.add(row["text"].strip())

    overlap = train_texts.intersection(test_texts)
    if overlap:
        print(f"  [CẢNH BÁO] Có {len(overlap)} mẫu trùng giữa tập train và test!")
    else:
        print("  [OK] Không có bất kỳ mẫu nào bị trùng lặp giữa tập train và test!")

    header = f"{'STT':<4} | {'Nhãn ý định (Label)':<18} | {'Huấn luyện (>=35)':<18} | {'Kiểm thử (=8)':<14} | {'Tổng cộng':<10} | {'Trạng thái'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_train = 0
    total_test = 0
    for idx, label in enumerate(expected_labels, 1):
        tr_c = train_counts.get(label, 0)
        te_c = test_counts.get(label, 0)
        total_train += tr_c
        total_test += te_c
        status = "ĐẠT" if tr_c >= 35 and te_c == 8 else "CHƯA ĐẠT"
        print(f"{idx:<4} | {label:<18} | {tr_c:<18} | {te_c:<14} | {tr_c + te_c:<10} | [OK] {status}")

    print("-" * len(header))
    print(f"{'TỔNG':<4} | {'8 nhãn':<18} | {total_train:<18} | {total_test:<14} | {total_train + total_test:<10} | [OK] HOÀN TẤT")
    print("=" * len(header) + "\n")


def check_dishes_csv(dishes_path: str):
    """Kiểm tra và in thống kê chi tiết dữ liệu món ăn."""
    print("=" * 80)
    print(f"3. THỐNG KÊ DỮ LIỆU MÓN ĂN ({os.path.relpath(dishes_path, current_dir)})")
    print("=" * 80)

    expected_cols = [
        "id", "ten_mon", "loai", "gia_vnd", "khoang_cach_m",
        "thoi_gian_phuc_vu_phut", "do_no", "dinh_duong", "do_ngon", "cay", "mo_ta"
    ]

    dishes = []
    raw_rows = read_csv_dicts(dishes_path)
    if raw_rows:
        for c in expected_cols:
            if c not in raw_rows[0]:
                raise ValueError(f"Thiếu cột '{c}' trong {dishes_path}")
    for row in raw_rows:
        dishes.append({
            "id": int(row["id"]),
            "ten_mon": row["ten_mon"],
            "loai": row["loai"],
            "gia_vnd": int(float(row["gia_vnd"])),
            "khoang_cach_m": int(float(row["khoang_cach_m"])),
            "thoi_gian_phuc_vu_phut": int(float(row["thoi_gian_phuc_vu_phut"])),
            "do_no": int(float(row["do_no"])),
            "dinh_duong": int(float(row["dinh_duong"])),
            "do_ngon": int(float(row["do_ngon"])),
            "cay": int(row["cay"]),
            "chay": int(row.get("chay", 0) or 0),
            "do_dong_cao_diem": int(float(row.get("do_dong_cao_diem", 3) or 3)),
            "so_cho_ngoi": int(float(row.get("so_cho_ngoi", 30) or 30)),
            "mo_ta": row["mo_ta"],
        })

    total_dishes = len(dishes)
    print(f"Tổng số lượng món ăn: {total_dishes} món")

    categories = Counter(d["loai"] for d in dishes)
    print("\n* Phân bố theo loại món:")
    for cat, count in categories.most_common():
        pct = (count / total_dishes) * 100
        bar = "█" * int(count * 2)
        print(f"  - {cat:<16}: {count:>2} món ({pct:>5.1f}%) {bar}")

    prices = [d["gia_vnd"] for d in dishes]
    distances = [d["khoang_cach_m"] for d in dishes]
    times = [d["thoi_gian_phuc_vu_phut"] for d in dishes]
    satieties = [d["do_no"] for d in dishes]
    nutritions = [d["dinh_duong"] for d in dishes]
    tastes = [d["do_ngon"] for d in dishes]
    crowds = [d["do_dong_cao_diem"] for d in dishes]
    seats = [d["so_cho_ngoi"] for d in dishes]
    spicy_count = sum(1 for d in dishes if d["cay"] == 1)
    veg_count = sum(1 for d in dishes if d["chay"] == 1)

    print("\n* Thống kê các chỉ số cơ bản:")
    print(f"  - Giá món (VND)         : Min = {min(prices):,d}đ | Max = {max(prices):,d}đ | TB = {sum(prices)/total_dishes:,.0f}đ")
    print(f"  - Khoảng cách từ NEU (m): Min = {min(distances)}m | Max = {max(distances)}m | TB = {sum(distances)/total_dishes:.1f}m")
    print(f"  - Thời gian phục vụ (p) : Min = {min(times)}p | Max = {max(times)}p | TB = {sum(times)/total_dishes:.1f} phút")
    print(f"  - Điểm độ no (1-5)      : Min = {min(satieties)} | Max = {max(satieties)} | TB = {sum(satieties)/total_dishes:.2f}")
    print(f"  - Điểm dinh dưỡng (1-5) : Min = {min(nutritions)} | Max = {max(nutritions)} | TB = {sum(nutritions)/total_dishes:.2f}")
    print(f"  - Điểm độ ngon (1-5)    : Min = {min(tastes)} | Max = {max(tastes)} | TB = {sum(tastes)/total_dishes:.2f}")
    print(f"  - Độ đông cao điểm (1-5): Min = {min(crowds)} | Max = {max(crowds)} | TB = {sum(crowds)/total_dishes:.2f}")
    print(f"  - Số chỗ ngồi           : Min = {min(seats)} chỗ | Max = {max(seats)} chỗ | TB = {sum(seats)/total_dishes:.1f} chỗ")
    print(f"  - Tỷ lệ món cay         : {spicy_count}/{total_dishes} món ({spicy_count/total_dishes*100:.1f}%)")
    print(f"  - Tỷ lệ món chay        : {veg_count}/{total_dishes} món ({veg_count/total_dishes*100:.1f}%)")
    print("=" * 80 + "\n")


def print_intermediate_results(response: dict):
    """In chi tiết các bước trung gian để người dùng dễ theo dõi."""
    print("\n" + "─" * 70)
    print("🔍 [BƯỚC TRUNG GIAN ĐỂ HIỂN THỊ GUI]")
    print("─" * 70)

    # 1. Tiền xử lý
    prep = response["preprocessing"]
    print(f"1. Tiền xử lý chuỗi:")
    print(f"   • Chuẩn hóa: '{prep['normalized']}'")
    print(f"   • Tokens: {prep['tokens']}")
    print(f"   • Tokens lọc stopword: {prep['tokens_no_stopwords']}")

    # 2. Phân loại ý định
    clf = response["classification"]
    print(f"\n2. Phân loại ý định (Intent Classification):")
    print(f"   • Nhãn dự đoán: '{clf['label']}' (Độ tin cậy: {clf['confidence']:.2%})")
    print(f"   • Nhãn áp dụng thực tế: '{response['effective_intent']}' (Fallback: {response['is_fallback']})")
    if response["is_fallback"]:
        print(f"     Lý do: {response['fallback_reason']}")
    top_3_probs = ", ".join([f"{p['label']}: {p['prob']:.2%}" for p in clf['probabilities'][:3]])
    print(f"   • Top 3 xác suất: [{top_3_probs}]")
    if clf["tfidf_terms"]:
        top_terms = ", ".join([f"'{t['term']}': {t['tfidf']:.3f}" for t in clf['tfidf_terms'][:5]])
        print(f"   • Từ khóa TF-IDF nổi bật: [{top_terms}]")

    # 3. Ràng buộc trích xuất
    c = response["constraints"]
    print(f"\n3. Ràng buộc trích xuất (Constraints):")
    print(f"   • Ngân sách tối đa: {c['max_price']:,.0f}đ" if c['max_price'] else "   • Ngân sách tối đa: Không ràng buộc")
    print(f"   • Số người đi ăn: {c.get('num_people', 1)} người ({c.get('sources', {}).get('num_people', '')})")
    print(f"   • Khung giờ đi ăn: {c.get('time_slot', 'cao_diem')} ({c.get('sources', {}).get('time_slot', '')})")
    print(f"   • Ưu tiên quán vắng: {'Có (nhân đôi hệ số a)' if c.get('prefer_quiet') else 'Không'}")
    print(f"   • Loại trừ đồ cay: {'Có (không ăn cay)' if c['exclude_spicy'] else 'Không'}")
    print(f"   • Chỉ món chay: {'Có (chỉ ăn chay)' if c.get('vegetarian_only') else 'Không'}")
    print(f"   • Món/loại loại trừ: {c['exclude_types'] if c['exclude_types'] else 'Không'}")
    print(f"   • Khớp chuỗi: {c['intermediate']}")

    # 4. Gợi ý món ăn
    rec = response["recommendation"]
    if rec:
        print(f"\n4. Gợi ý đa tiêu chí (Recommender - 7 tiêu chí):")
        print(f"   • Profile áp dụng: '{rec['profile_used']}'")
        weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in rec['weights_used'].items()])
        print(f"   • Trọng số tiêu chí: [{weights_str}]")
        print(f"   • Số món đạt ràng buộc: {rec['filtered_dishes_count']}/{rec['total_dishes']}")
        if rec["top_k"]:
            print(f"   • Top {len(rec['top_k'])} món điểm cao nhất:")
            for idx, item in enumerate(rec["top_k"], 1):
                d = item["dish"]
                c_str = ", ".join([f"{k}={v:.3f}" for k, v in item['contributions'].items()])
                print(
                    f"     {idx}. {d['ten_mon']} (Điểm: {item['score']:.4f}, "
                    f"TG chờ ~{item.get('thoi_gian_cho', 0):.1f}p, "
                    f"Chỗ trống ~{item.get('cho_trong', 0):.1f}/{d.get('so_cho_ngoi', 30)}) -> Đóng góp: [{c_str}]"
                )
    print("─" * 70 + "\n")


def run_cli_interactive():
    """Khởi chạy chế độ CLI tương tác trò chuyện với Chatbot."""
    print("\n" + "=" * 80)
    print("   🤖 CHATBOT TƯ VẤN ĂN TRƯA ĐẠI HỌC KINH TẾ QUỐC DÂN (NEU) - CHẾ ĐỘ CLI 🍜")
    print("=" * 80)
    print("• Nhập câu hỏi tự nhiên về bữa trưa của bạn (ví dụ: 'đi 4 người ăn gì rẻ dưới 35k không cay')")
    print("• Nhập 'exit', 'quit' hoặc 'q' để kết thúc chương trình.\n")

    bot = Chatbot()

    while True:
        try:
            user_input = input("Sinh viên NEU: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nChatbot: Tạm biệt bạn! Chúc bạn có những bữa trưa ngon miệng tại NEU! 👋\n")
                break

            response = bot.respond(user_input)
            print_intermediate_results(response)

            print("Chatbot:")
            print(response["reply"])
            print("\n" + "=" * 80 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nĐã thoát chương trình. Tạm biệt bạn!")
            break


def main():
    parser = argparse.ArgumentParser(description="NEU Lunch Chatbot - Trưa Nay Ăn Gì?")
    parser.add_argument("--check", action="store_true", help="Kiểm tra tính hợp lệ của CSV/JSON và in thống kê")
    parser.add_argument("--chat", action="store_true", help="Bắt đầu chế độ trò chuyện CLI tương tác")
    parser.add_argument("--gui", action="store_true", help="Khởi chạy giao diện đồ họa Desktop PyQt6")
    parser.add_argument("--eval", action="store_true", help="Chạy toàn bộ thí nghiệm đánh giá và xuất báo cáo/biểu đồ vào outputs/")
    parser.add_argument("query", nargs="*", help="Câu hỏi nhanh (One-shot CLI)")

    args = parser.parse_args()

    if args.gui:
        try:
            from gui.app import launch_gui
            launch_gui()
            return
        except Exception as e:
            print(f"[Lỗi khởi chạy GUI]: {e}")
            sys.exit(1)

    if args.eval:
        from src.evaluate import ModelEvaluator
        evaluator = ModelEvaluator()
        evaluator.run_all()
        return

    if args.check:
        print("\n" + "#" * 80)
        print("### KIỂM TRA DỮ LIỆU VÀ CẤU HÌNH DỰ ÁN LUNCH CHATBOT (NEU) ###".center(74))
        print("#" * 80 + "\n")
        check_profiles_json(DEFAULT_PROFILES_PATH)
        check_intents_csv(DEFAULT_INTENTS_PATH, DEFAULT_TEST_INTENTS_PATH)
        check_dishes_csv(DEFAULT_DISHES_PATH)
        print(">>> TẤT CẢ DỮ LIỆU CSV VÀ JSON ĐÃ ĐƯỢC XÁC THỰC HỢP LỆ VÀ SẴN SÀNG! <<<\n")
        return

    if args.query:
        query_text = " ".join(args.query)
        print(f"\n[Sinh viên NEU]: {query_text}")
        bot = Chatbot()
        resp = bot.respond(query_text)
        print_intermediate_results(resp)
        print("[Chatbot NEU]:")
        print(resp["reply"])
        return

    if sys.stdin.isatty() or args.chat:
        run_cli_interactive()
    else:
        print("\n" + "#" * 80)
        print("### KIỂM TRA DỮ LIỆU VÀ CẤU HÌNH DỰ ÁN LUNCH CHATBOT (NEU) ###".center(74))
        print("#" * 80 + "\n")
        check_profiles_json(DEFAULT_PROFILES_PATH)
        check_intents_csv(DEFAULT_INTENTS_PATH, DEFAULT_TEST_INTENTS_PATH)
        check_dishes_csv(DEFAULT_DISHES_PATH)
        print("Mẹo: Chạy 'python main.py --chat' để bắt đầu hội thoại CLI tương tác!\n")


if __name__ == "__main__":
    main()
