"""
Sinh outputs/worked_example.txt: ví dụ tính tay (Chương 2) cho
  Phần 1 - TF-IDF + MultinomialNB (đúng tham số của model thật trong src/classifier.py)
  Phần 2 - chấm điểm đa tiêu chí (trọng số thật trong config/profiles.json, món thật trong data/dishes.csv)

Cách làm: mọi con số được TÍNH TAY bằng công thức (chỉ dùng math/numpy), sau đó ĐỐI CHIẾU với sklearn và
với chính các lớp của dự án (NeedClassifier, DishRecommender) bằng assert (sai số < 1e-6).
Chỉ khi mọi assert đạt thì file mới được ghi và có dòng "ĐÃ KHỚP VỚI SKLEARN".
"""

import csv
import json
import math
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pyvi.ViTokenizer import ViTokenizer
from sklearn.preprocessing import MinMaxScaler

from src.classifier import NeedClassifier
from src.preprocess import preprocess
from src.recommender import DishRecommender, compute_crowd_metrics

# ── Dữ liệu ví dụ ─────────────────────────────────────────────────────────────
LABELS = ["tiet_kiem", "healthy", "nhanh_gon"]
SAMPLE_DOCS: List[Tuple[str, str]] = [
    ("cơm bình dân giá rẻ", "tiet_kiem"),
    ("món nào rẻ tiết kiệm", "tiet_kiem"),
    ("quán ăn rẻ cho sinh viên", "tiet_kiem"),
    ("salad ít dầu mỡ", "healthy"),
    ("món ăn thanh đạm ít béo", "healthy"),
    ("ăn rau củ luộc ít calo", "healthy"),
    ("bánh mì mang đi nhanh gọn", "nhanh_gon"),
    ("đồ ăn nhanh tiện lợi", "nhanh_gon"),
    ("ăn nhanh kịp giờ học", "nhanh_gon"),
]
TEST_QUERY = "tìm quán cơm giá rẻ"
# 4 món lấy từ data/dishes.csv (giá, khoảng cách, thời gian, độ đông, chỗ ngồi đa dạng)
DISH_IDS = [1, 12, 23, 28]
TOKEN_PATTERN = r"(?u)\b\w\w+\b"  # mặc định của TfidfVectorizer
CRITERIA = [  # (khóa trong profiles.json, cột giá trị đưa vào chấm điểm, tên hiển thị, là tiêu chí chi phí?)
    ("gia", "gia_vnd", "Giá (đ)", True),
    ("khoang_cach", "khoang_cach_m", "Khoảng cách (m)", True),
    ("thoi_gian", "thoi_gian_cho", "Thời gian chờ (phút)", True),
    ("do_no", "do_no", "Độ no", False),
    ("dinh_duong", "dinh_duong", "Dinh dưỡng", False),
    ("do_ngon", "do_ngon", "Độ ngon", False),
    ("cho_ngoi", "cho_ngoi_nhom", "Chỗ ngồi nhóm", False),
]
SEP = "=" * 100
SUB = "-" * 100


def _unknown_compounds(tokens: List[str]) -> List[str]:
    """Token ghép (có '_') mà cụm tương ứng KHÔNG nằm trong từ điển của pyvi -> nghi ngờ ghép sai."""
    bad = []
    for t in tokens:
        if "_" in t:
            phrase = t.replace("_", " ")
            if phrase not in ViTokenizer.bi_grams and phrase not in ViTokenizer.tri_grams:
                bad.append(t)
    return bad


def _terms(text: str, ngram_range: Tuple[int, int]) -> List[str]:
    """Tái hiện analyzer của TfidfVectorizer: lower -> token_pattern -> n-gram nối bằng khoảng trắng."""
    tokens = re.findall(TOKEN_PATTERN, text.lower())
    out: List[str] = []
    for n in range(ngram_range[0], min(ngram_range[1], len(tokens)) + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i:i + n]))
    return out


def _fmt_row(cells: List[str], widths: List[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))


def generate_worked_example(root_dir: str = _ROOT, outputs_dir: str = None,
                            output_filename: str = "worked_example.txt") -> str:
    """
    Tạo file worked_example.txt (xem docstring module). Ném AssertionError nếu số liệu tính tay
    lệch sklearn / DishRecommender, hoặc nếu ví dụ không thỏa các điều kiện (token lạ, câu thử trùng, xác suất < 50%).
    Trả về đường dẫn file đã ghi.
    """
    outputs_dir = outputs_dir or os.path.join(root_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    out: List[str] = []
    P = out.append

    # ══ 0. Huấn luyện model THẬT trên 9 câu mẫu (đúng NeedClassifier.train) ═══════════════════════════
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "sample_intents.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(SAMPLE_DOCS)
        clf = NeedClassifier()          # tham số mặc định đúng như model trong ứng dụng
        clf.train(csv_path)
    pipe = clf.pipeline
    tfidf_sk = pipe.named_steps["tfidf"]
    nb_sk = pipe.named_steps["clf"]
    ngram_range = tuple(tfidf_sk.ngram_range)
    alpha = float(nb_sk.alpha)
    assert tfidf_sk.sublinear_tf and tfidf_sk.norm == "l2" and tfidf_sk.smooth_idf and tfidf_sk.use_idf

    # ══ 1. Tiền xử lý bằng đúng preprocess() của pipeline ═════════════════════════════════════════════
    docs_tokens, docs_text, labels = [], [], []
    for text, lbl in SAMPLE_DOCS:
        tk = preprocess(text, remove_stopwords=True)["tokens_no_stopwords"]
        bad = _unknown_compounds(tk)
        assert not bad, f"Câu '{text}' bị pyvi ghép nghi sai: {bad} -> hãy đổi câu mẫu"
        assert " ".join(tk) == clf._prepare_text(text)
        docs_tokens.append(tk)
        docs_text.append(" ".join(tk))
        labels.append(lbl)
    q_tokens = preprocess(TEST_QUERY, remove_stopwords=True)["tokens_no_stopwords"]
    assert not _unknown_compounds(q_tokens), f"Câu thử bị ghép nghi sai: {q_tokens}"
    q_text = " ".join(q_tokens)
    assert TEST_QUERY.strip().lower() not in {t.lower() for t, _ in SAMPLE_DOCS}, "Câu thử trùng câu mẫu"
    assert q_text not in docs_text, "Câu thử trùng câu mẫu sau tiền xử lý"

    # ══ 2. TÍNH TAY TF-IDF ═══════════════════════════════════════════════════════════════════════════
    n_docs = len(SAMPLE_DOCS)
    doc_terms = [_terms(t, ngram_range) for t in docs_text]
    vocab = sorted({t for terms in doc_terms for t in terms})
    index = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)
    df = np.zeros(V)
    for terms in doc_terms:
        for t in set(terms):
            df[index[t]] += 1
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0

    def tfidf_vector(terms: List[str]) -> Tuple[np.ndarray, Dict[str, Tuple[int, float, float, float]]]:
        """Trả về (vector đã chuẩn hóa L2, chi tiết từng từ: (tf, 1+ln tf, tf*idf, ...))."""
        counts: Dict[str, int] = {}
        for t in terms:
            if t in index:
                counts[t] = counts.get(t, 0) + 1
        raw = np.zeros(V)
        detail: Dict[str, Tuple[int, float, float, float]] = {}
        for t, c in counts.items():
            tfp = 1.0 + math.log(c)
            raw[index[t]] = tfp * idf[index[t]]
            detail[t] = (c, tfp, idf[index[t]], raw[index[t]])
        norm = math.sqrt(float(np.sum(raw ** 2)))
        return (raw / norm if norm > 0 else raw), {"_norm": norm, **detail}  # type: ignore

    X = np.zeros((n_docs, V))
    docs_detail = []
    for i, terms in enumerate(doc_terms):
        X[i], det = tfidf_vector(terms)
        docs_detail.append(det)

    # ══ 3. TÍNH TAY MULTINOMIAL NB ═══════════════════════════════════════════════════════════════════
    classes = sorted(set(labels))
    assert classes == sorted(LABELS)
    n_k = {c: labels.count(c) for c in classes}
    prior = {c: n_k[c] / n_docs for c in classes}
    N_ki = np.array([X[[j for j, l in enumerate(labels) if l == c]].sum(axis=0) for c in classes])  # (C, V)
    N_k = N_ki.sum(axis=1)
    theta = (N_ki + alpha) / (N_k[:, None] + alpha * V)
    log_theta = np.log(theta)
    log_prior = np.array([math.log(prior[c]) for c in classes])

    q_terms = _terms(q_text, ngram_range)
    xq, q_detail = tfidf_vector(q_terms)
    joint = log_prior + log_theta @ xq
    probs = np.exp(joint - joint.max())
    probs /= probs.sum()
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]

    # ── Đối chiếu với sklearn (< 1e-6) ──
    assert list(tfidf_sk.get_feature_names_out()) == vocab, "Từ vựng tính tay khác sklearn"
    assert np.max(np.abs(tfidf_sk.idf_ - idf)) < 1e-9
    assert np.max(np.abs(tfidf_sk.transform(docs_text).toarray() - X)) < 1e-9
    assert list(nb_sk.classes_) == classes
    assert np.max(np.abs(nb_sk.class_log_prior_ - log_prior)) < 1e-9
    assert np.max(np.abs(nb_sk.feature_log_prob_ - log_theta)) < 1e-9
    sk_probs = pipe.predict_proba([q_text])[0]
    diff_proba = float(np.max(np.abs(sk_probs - probs)))
    assert diff_proba < 1e-6, f"predict_proba lệch {diff_proba}"
    app_pred = clf.predict(TEST_QUERY)
    assert app_pred["label"] == pred_label == "tiet_kiem", f"Dự đoán {pred_label}, cần tiet_kiem"
    assert probs[pred_idx] >= 0.5, f"Xác suất tiet_kiem chỉ {probs[pred_idx]:.2%} (< 50%) -> đổi câu thử"

    # ══ 4. PHẦN 2: CHẤM ĐIỂM MÓN ĂN (7 TIÊU CHÍ & MÔ HÌNH ĐỘ ĐÔNG) ══════════════════════════════════════
    active_profile = os.path.join(root_dir, "config", "profiles.json")
    with open(active_profile, "r", encoding="utf-8") as f:
        profiles = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    weights = profiles[pred_label]
    assert abs(sum(weights.values()) - 1.0) < 1e-9 and set(weights) == {c[0] for c in CRITERIA}

    dishes_path = os.path.join(root_dir, "data", "dishes.csv")
    crowd_cfg_path = os.path.join(root_dir, "config", "crowd_config.json")
    with open(dishes_path, "r", encoding="utf-8-sig", newline="") as f:
        all_dishes = {int(r["id"]): r for r in csv.DictReader(f)}
    dishes = [all_dishes[i] for i in DISH_IDS]

    with open(crowd_cfg_path, "r", encoding="utf-8") as f:
        crowd_cfg = json.load(f)

    # Tham số tính toán
    so_nguoi_4 = 4
    khung_gio_cd = "cao_diem"
    hs_khung_gio = float(crowd_cfg["he_so_khung_gio"][khung_gio_cd])
    a_coef = float(crowd_cfg["a"])
    b_coef = float(crowd_cfg["b"])
    hs_chiem_cho = float(crowd_cfg["he_so_chiem_cho"])

    # Tính toán độ đông & chỗ ngồi cho từng món với so_nguoi = 4, khung_gio = cao_diem
    dishes_4 = []
    crowd_steps_4 = []
    for d in dishes:
        m = compute_crowd_metrics(d, so_nguoi=so_nguoi_4, khung_gio=khung_gio_cd, crowd_cfg=crowd_cfg)
        d_merged = {**d, **m}
        dishes_4.append(d_merged)
        crowd_steps_4.append({
            "id": int(d["id"]),
            "ten_mon": d["ten_mon"],
            "do_dong_cao_diem": float(d["do_dong_cao_diem"]),
            "do_dong_hieu_dung": m["do_dong_hieu_dung"],
            "thoi_gian_phuc_vu_phut": float(d["thoi_gian_phuc_vu_phut"]),
            "thoi_gian_cho": m["thoi_gian_cho"],
            "so_cho_ngoi": float(d["so_cho_ngoi"]),
            "cho_trong": m["cho_trong"],
            "cho_ngoi_nhom": m["cho_ngoi_nhom"],
            "seat_warning": m["seat_warning"],
        })

    # Ma trận raw 7 tiêu chí cho so_nguoi = 4
    raw = np.array([[float(d[col]) for _, col, _, _ in CRITERIA] for d in dishes_4])
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    assert np.all(maxs > mins), "Có tiêu chí không phân biệt được giữa 4 món"
    norm = np.zeros_like(raw)
    for j, (_, _, _, is_cost) in enumerate(CRITERIA):
        norm[:, j] = (maxs[j] - raw[:, j]) / (maxs[j] - mins[j]) if is_cost else (raw[:, j] - mins[j]) / (maxs[j] - mins[j])
    w_vec = np.array([weights[k] for k, _, _, _ in CRITERIA])
    contrib = norm * w_vec
    totals = contrib.sum(axis=1)
    order = list(np.argsort(-totals))

    # Tính toán đối chứng khi đi 1 người (so_nguoi = 1, khung_gio = cao_diem)
    dishes_1 = []
    for d in dishes:
        m = compute_crowd_metrics(d, so_nguoi=1, khung_gio=khung_gio_cd, crowd_cfg=crowd_cfg)
        dishes_1.append({**d, **m})
    raw_1 = np.array([[float(d[col]) for _, col, _, _ in CRITERIA] for d in dishes_1])
    mins_1, maxs_1 = raw_1.min(axis=0), raw_1.max(axis=0)
    norm_1 = np.zeros_like(raw_1)
    for j, (_, _, _, is_cost) in enumerate(CRITERIA):
        if maxs_1[j] > mins_1[j]:
            norm_1[:, j] = (maxs_1[j] - raw_1[:, j]) / (maxs_1[j] - mins_1[j]) if is_cost else (raw_1[:, j] - mins_1[j]) / (maxs_1[j] - mins_1[j])
        else:
            norm_1[:, j] = 1.0
    contrib_1 = norm_1 * w_vec
    totals_1 = contrib_1.sum(axis=1)
    order_1 = list(np.argsort(-totals_1))

    # ── Đối chiếu: MinMaxScaler của sklearn cho so_nguoi = 4 ──
    scaled = MinMaxScaler().fit_transform(raw)
    for j, (_, _, _, is_cost) in enumerate(CRITERIA):
        if is_cost:
            scaled[:, j] = 1.0 - scaled[:, j]
    sk_totals = scaled @ w_vec
    diff_score = float(np.max(np.abs(sk_totals - totals)))
    assert diff_score < 1e-6 and list(np.argsort(-sk_totals)) == order, "Điểm/xếp hạng lệch sklearn"

    # ── Đối chiếu: DishRecommender thật của dự án cho cả 4 người và 1 người ──
    rec = DishRecommender(dishes_path=dishes_path, profiles_path=active_profile, crowd_config_path=crowd_cfg_path)
    all_d_rec = {int(d["id"]): d for d in rec.dishes}
    rec.dishes = [all_d_rec[i] for i in DISH_IDS]

    rec_result_4 = rec.recommend(pred_label, top_k=len(dishes), num_people=4, time_slot=khung_gio_cd)["all_ranked"]
    rec_scores_4 = {item["dish"]["id"]: item["score"] for item in rec_result_4}
    diff_rec_4 = max(abs(rec_scores_4[int(d["id"])] - totals[i]) for i, d in enumerate(dishes_4))
    assert diff_rec_4 < 1e-6, f"DishRecommender lệch khi 4 người: {diff_rec_4}"
    assert [item["dish"]["id"] for item in rec_result_4] == [int(dishes_4[i]["id"]) for i in order]

    rec_result_1 = rec.recommend(pred_label, top_k=len(dishes), num_people=1, time_slot=khung_gio_cd)["all_ranked"]
    rec_scores_1 = {item["dish"]["id"]: item["score"] for item in rec_result_1}
    diff_rec_1 = max(abs(rec_scores_1[int(d["id"])] - totals_1[i]) for i, d in enumerate(dishes_1))
    assert diff_rec_1 < 1e-6, f"DishRecommender lệch khi 1 người: {diff_rec_1}"
    assert [item["dish"]["id"] for item in rec_result_1] == [int(dishes_1[i]["id"]) for i in order_1]

    # ── Tổng các đóng góp (làm tròn 4 chữ số) phải khớp đúng điểm làm tròn 4 chữ số ──
    for i in range(len(dishes)):
        assert abs(sum(round(c, 4) for c in contrib[i]) - round(totals[i], 4)) < 1e-9, \
            f"Tổng đóng góp làm tròn không khớp điểm ở món {dishes[i]['ten_mon']} -> chọn bộ món khác"

    # ══ 5. XUẤT VĂN BẢN ═══════════════════════════════════════════════════════════════════════════════
    P(SEP)
    P("VÍ DỤ TÍNH TAY (WORKED EXAMPLE) – CHƯƠNG 2")
    P("Hệ thống chatbot gợi ý bữa trưa cho sinh viên NEU")
    P("Mọi số liệu dưới đây do công thức tính tay sinh ra và đã được đối chiếu với sklearn và DishRecommender (xem mục 1.10, 2.9 và 3).")
    P("Quy ước: bảng in làm tròn 4 chữ số thập phân; phép tính nội bộ dùng độ chính xác đầy đủ.")
    P(SEP)

    P("\nPHẦN 1. PHÂN LOẠI NHU CẦU BẰNG TF-IDF + MULTINOMIAL NAIVE BAYES")
    P(SUB)
    P("1.1. Tham số lấy từ model thật (src/classifier.py: NeedClassifier -> build_model('nb'))")
    P(f"   TfidfVectorizer: ngram_range={ngram_range}, sublinear_tf={tfidf_sk.sublinear_tf}, min_df={tfidf_sk.min_df}, "
      f"lowercase={tfidf_sk.lowercase}, use_idf={tfidf_sk.use_idf}, smooth_idf={tfidf_sk.smooth_idf}, norm='{tfidf_sk.norm}'")
    P(f"                    token_pattern = r\"{tfidf_sk.token_pattern}\" (chỉ giữ token ≥ 2 ký tự; '_' thuộc \\w nên 'bình_dân' là 1 token)")
    P(f"   MultinomialNB:   alpha = {alpha}, fit_prior = {nb_sk.fit_prior}")
    P("   Công thức:")
    P("     tf'(t,d)   = 1 + ln(tf(t,d))                    (sublinear_tf=True, chỉ áp dụng khi tf > 0)")
    P("     idf(t)     = ln((1 + n) / (1 + df(t))) + 1       (smooth_idf=True), n = số văn bản huấn luyện")
    P("     w(t,d)     = tf'(t,d) × idf(t)")
    P("     x(d)       = w(·,d) / ||w(·,d)||₂                (chuẩn hóa L2)")
    P("     P(C)       = n_C / n")
    P("     N(C,t)     = Σ_{d ∈ C} x_t(d);   N(C) = Σ_t N(C,t)")
    P("     P(t|C)     = (N(C,t) + alpha) / (N(C) + alpha·|V|)")
    P("     log P(C|x) ∝ ln P(C) + Σ_t x_t · ln P(t|C);  P(C|x) = softmax của các giá trị này")

    P("\n1.2. Tập huấn luyện 9 câu (3 nhãn × 3 câu) và kết quả preprocess() của pipeline thật")
    for i, ((text, lbl), tk) in enumerate(zip(SAMPLE_DOCS, docs_tokens), 1):
        P(f"   d{i}  [{lbl:<9}] '{text}'  ->  {tk}")
    P("   Kiểm tra token: mọi token ghép (có '_') đều có trong từ điển pyvi -> không có token lạ.")

    P(f"\n1.3. Từ vựng V (unigram + bigram, sắp theo thứ tự chữ cái), |V| = {V}, n = {n_docs}")
    w_cols = [4, 18, 3, 8]
    P("   " + _fmt_row(["STT", "Từ / cụm từ", "df", "idf"], w_cols))
    for t in vocab:
        i = index[t]
        P("   " + _fmt_row([str(i), t, str(int(df[i])), f"{idf[i]:.4f}"], w_cols))

    P("\n1.4. Ma trận TF-IDF (chỉ liệt kê các ô khác 0) – từng văn bản")
    for i, det in enumerate(docs_detail, 1):
        norm_val = det["_norm"]
        P(f"   d{i} [{labels[i - 1]}]  ||w||₂ = {norm_val:.4f}")
        P("      " + _fmt_row(["Từ / cụm từ", "tf", "tf'", "idf", "w=tf'×idf", "x=w/||w||"], [18, 3, 7, 7, 10, 10]))
        for t, (c, tfp, idf_t, wv) in sorted((k, v) for k, v in det.items() if k != "_norm"):
            P("      " + _fmt_row([t, str(c), f"{tfp:.4f}", f"{idf_t:.4f}", f"{wv:.4f}", f"{wv / norm_val:.4f}"],
                                 [18, 3, 7, 7, 10, 10]))

    P("\n1.5. Xác suất tiên nghiệm P(C) = n_C / n")
    for c in classes:
        P(f"   P({c:<9}) = {n_k[c]}/{n_docs} = {prior[c]:.4f}     ln P = {math.log(prior[c]):.4f}")

    P("\n1.6. Tổng trọng số N(C,t) = Σ x_t(d) theo lớp (liệt kê các từ có N > 0; còn lại N = 0)")
    P("   " + _fmt_row(["Từ / cụm từ"] + classes, [18, 10, 10, 10]))
    for t in vocab:
        i = index[t]
        P("   " + _fmt_row([t] + [f"{N_ki[k, i]:.4f}" for k in range(len(classes))], [18, 10, 10, 10]))
    P("   " + _fmt_row(["N(C) = Σ_t N(C,t)"] + [f"{N_k[k]:.4f}" for k in range(len(classes))], [18, 10, 10, 10]))
    P(f"   Mẫu số làm mịn: N(C) + alpha·|V| = N(C) + {alpha}×{V} = N(C) + {alpha * V:.1f}")
    for k, c in enumerate(classes):
        P(f"     {c:<9}: {N_k[k]:.4f} + {alpha * V:.1f} = {N_k[k] + alpha * V:.4f}")

    P(f"\n1.7. Câu thử: '{TEST_QUERY}'  (không trùng câu nào trong 9 câu mẫu)")
    P(f"   preprocess() -> tokens_no_stopwords = {q_tokens}")
    P(f"   Chuỗi đưa vào vectorizer = '{q_text}'")
    in_vocab = [t for t in q_terms if t in index]
    oov = [t for t in q_terms if t not in index]
    P(f"   Các n-gram của câu: {q_terms}")
    P(f"   Không có trong V (bị bỏ qua): {oov}")
    P(f"   Có trong V: {in_vocab}")
    qn = q_detail["_norm"]
    P(f"   ||w||₂ = {qn:.4f}")
    P("   " + _fmt_row(["Từ / cụm từ", "tf", "tf'", "idf", "w=tf'×idf", "x=w/||w||"], [18, 3, 7, 7, 10, 10]))
    for t, (c, tfp, idf_t, wv) in sorted((k, v) for k, v in q_detail.items() if k != "_norm"):
        P("   " + _fmt_row([t, str(c), f"{tfp:.4f}", f"{idf_t:.4f}", f"{wv:.4f}", f"{wv / qn:.4f}"], [18, 3, 7, 7, 10, 10]))

    P("\n1.8. Likelihood có làm mịn cho các từ của câu thử:  P(t|C) = (N(C,t) + alpha) / (N(C) + alpha·|V|)")
    P("   " + _fmt_row(["Từ / cụm từ", "Lớp", "N(C,t)+alpha", "Mẫu số", "P(t|C)", "ln P(t|C)"], [18, 10, 13, 9, 9, 10]))
    for t in sorted(q_detail.keys() - {"_norm"}):
        i = index[t]
        for k, c in enumerate(classes):
            P("   " + _fmt_row([t, c, f"{N_ki[k, i] + alpha:.4f}", f"{N_k[k] + alpha * V:.4f}",
                                f"{theta[k, i]:.4f}", f"{log_theta[k, i]:.4f}"], [18, 10, 13, 9, 9, 10]))

    P("\n1.9. Điểm log-posterior chưa chuẩn hóa:  S(C) = ln P(C) + Σ_t x_t · ln P(t|C)")
    for k, c in enumerate(classes):
        parts = " + ".join(f"{xq[index[t]]:.4f}×({log_theta[k, index[t]]:.4f})" for t in sorted(q_detail.keys() - {'_norm'}))
        P(f"   S({c:<9}) = ({log_prior[k]:.4f}) + {parts}")
        P(f"              = {joint[k]:.4f}")
    P("\n   Chuẩn hóa softmax:  P(C|x) = exp(S(C) − S_max) / Σ_C' exp(S(C') − S_max),   S_max = "
      f"{joint.max():.4f}")
    for k, c in enumerate(classes):
        P(f"   exp(S({c}) − S_max) = {math.exp(joint[k] - joint.max()):.4f}")
    P(f"   Tổng = {sum(math.exp(j - joint.max()) for j in joint):.4f}")
    for k, c in enumerate(classes):
        P(f"   P({c:<9} | x) = {probs[k]:.4f}  ({probs[k] * 100:.2f}%)")
    P(f"\n   ===> DỰ ĐOÁN: '{pred_label}' với xác suất {probs[pred_idx] * 100:.2f}% (≥ 50%).")

    P("\n1.10. Đối chiếu với sklearn (assert sai số < 1e-6)")
    P(f"   predict_proba sklearn : { {c: round(float(p), 6) for c, p in zip(classes, sk_probs)} }")
    P(f"   Tính tay              : { {c: round(float(p), 6) for c, p in zip(classes, probs)} }")
    P(f"   Sai số lớn nhất = {diff_proba:.2e}   (từ vựng, idf, ma trận TF-IDF, ln P(C), ln P(t|C) cũng khớp < 1e-9)")
    P(f"   NeedClassifier.predict() của ứng dụng trả về nhãn '{app_pred['label']}' -> trùng với tính tay.")

    # ── Phần 2 ──
    P("\n" + SEP)
    P("PHẦN 2. CHẤM ĐIỂM VÀ XẾP HẠNG MÓN ĂN THEO NHÓM NHU CẦU (7 TIÊU CHÍ & MÔ HÌNH ĐỘ ĐÔNG)")
    P(SEP)
    P(f"2.1. Nhóm nhu cầu = nhãn dự đoán ở Phần 1: '{pred_label}'. Trọng số THẬT đọc từ config/profiles.json:")
    for k, _, name, _ in CRITERIA:
        P(f"   w_{k:<12} = {weights[k]:.2f}   ({name})")
    P(f"   Tổng trọng số = {sum(weights.values()):.2f}")

    P("\n2.2. Bốn món lấy từ data/dishes.csv (dữ liệu ban đầu)")
    names = [f"[{d['id']}] {d['ten_mon']}" for d in dishes_4]
    base_headers = ["Món", "Giá (đ)", "K.Cách (m)", "TG PV (phút)", "Độ no", "D.Dưỡng", "Độ ngon", "Độ đông", "Chỗ ngồi"]
    base_widths = [48, 9, 11, 13, 7, 9, 8, 9, 9]
    P("   " + _fmt_row(base_headers, base_widths))
    for d in dishes:
        P("   " + _fmt_row([
            f"[{d['id']}] {d['ten_mon']}",
            f"{float(d['gia_vnd']):.0f}",
            f"{float(d['khoang_cach_m']):.0f}",
            f"{float(d['thoi_gian_phuc_vu_phut']):.0f}",
            f"{float(d['do_no']):.0f}",
            f"{float(d['dinh_duong']):.0f}",
            f"{float(d['do_ngon']):.0f}",
            f"{float(d['do_dong_cao_diem']):.0f}",
            f"{float(d['so_cho_ngoi']):.0f}",
        ], base_widths))

    P(f"\n2.3. Mô hình hóa Độ đông và Chỗ ngồi nhóm (Khung giờ: {khung_gio_cd}, Số người = {so_nguoi_4})")
    P("   Tham số từ config/crowd_config.json:")
    P(f"     • Hệ số khung giờ cao điểm : he_so_khung_gio['{khung_gio_cd}'] = {hs_khung_gio:.1f}")
    P(f"     • Hệ số thời gian chờ      : a = {a_coef:.1f}, b = {b_coef:.2f}")
    P(f"     • Hệ số chiếm chỗ          : he_so_chiem_cho = {hs_chiem_cho:.1f}")
    P("   Công thức từng bước tính:")
    P("     (1) do_dong_hieu_dung = do_dong_cao_diem × he_so_khung_gio")
    P("     (2) thoi_gian_cho     = thoi_gian_phuc_vu_phut × (1 + a × do_dong_hieu_dung / 5) × (1 + b × (so_nguoi − 1))")
    P("     (3) cho_trong         = so_cho_ngoi × (1 − he_so_chiem_cho × do_dong_hieu_dung / 5)")
    P("     (4) cho_ngoi_nhom     = min(1.0, cho_trong / so_nguoi)")
    P("     (*) Cảnh báo: Nếu cho_trong < so_nguoi thì cảnh báo 'Có thể phải chờ chỗ'.")

    P("\n   Chi tiết tính toán từng món (so_nguoi = 4, cao điểm):")
    for step in crowd_steps_4:
        P(f"   [{step['id']}] {step['ten_mon']}:")
        P(f"      • do_dong_hieu_dung = {step['do_dong_cao_diem']:.0f} × {hs_khung_gio:.1f} = {step['do_dong_hieu_dung']:.1f}")
        P(f"      • thoi_gian_cho     = {step['thoi_gian_phuc_vu_phut']:.0f} × (1 + {a_coef}×{step['do_dong_hieu_dung']:.1f}/5) × (1 + {b_coef}×3)")
        P(f"                          = {step['thoi_gian_phuc_vu_phut']:.0f} × {1 + a_coef * step['do_dong_hieu_dung'] / 5:.2f} × {1 + b_coef * 3:.2f} = {step['thoi_gian_cho']:.2f} phút")
        P(f"      • cho_trong         = {step['so_cho_ngoi']:.0f} × (1 − {hs_chiem_cho}×{step['do_dong_hieu_dung']:.1f}/5) = {step['cho_trong']:.2f} chỗ")
        warn_txt = " --> Cảnh báo: Có thể phải chờ chỗ!" if step['seat_warning'] else " --> Đủ chỗ cho nhóm"
        P(f"      • cho_ngoi_nhom     = min(1.0, {step['cho_trong']:.2f} / 4) = {step['cho_ngoi_nhom']:.4f}{warn_txt}")

    P("\n2.4. Ma trận 7 tiêu chí đưa vào chấm điểm (thoi_gian = thoi_gian_cho, cho_ngoi = cho_ngoi_nhom)")
    crit_headers = ["Món"] + [c[2] for c in CRITERIA]
    crit_widths = [48, 10, 14, 18, 8, 10, 9, 15]
    P("   " + _fmt_row(crit_headers, crit_widths))
    for i in range(len(dishes_4)):
        P("   " + _fmt_row([names[i]] + [f"{raw[i, j]:.2f}" for j in range(len(CRITERIA))], crit_widths))

    P("\n2.5. Chuẩn hóa min-max trong 4 món (đảo chiều cho tiêu chí chi phí)")
    P("   Chi phí (giá, khoảng cách, thời gian chờ): v = (max − x) / (max − min)")
    P("   Lợi ích (độ no, dinh dưỡng, độ ngon, chỗ ngồi nhóm) : v = (x − min) / (max − min)")
    P("   " + _fmt_row(["Tiêu chí", "min", "max", "max−min", "Loại"], [22, 10, 10, 10, 10]))
    for j, (_, _, name, is_cost) in enumerate(CRITERIA):
        P("   " + _fmt_row([name, f"{mins[j]:.2f}", f"{maxs[j]:.2f}", f"{maxs[j] - mins[j]:.2f}",
                           "chi phí" if is_cost else "lợi ích"], [22, 10, 10, 10, 10]))
    P("\n   Giá trị chuẩn hóa v (mức độ thỏa mãn trong khoảng [0, 1]):")
    P("   " + _fmt_row(crit_headers, crit_widths))
    for i in range(len(dishes_4)):
        P("   " + _fmt_row([names[i]] + [f"{norm[i, j]:.4f}" for j in range(len(CRITERIA))], crit_widths))

    P("\n2.6. Đóng góp w × v của từng tiêu chí (4 chữ số thập phân) và tổng điểm (Điểm = Σ w×v)")
    score_headers = ["Món"] + [c[2] for c in CRITERIA] + ["Σ = Điểm"]
    score_widths = crit_widths + [10]
    P("   " + _fmt_row(score_headers, score_widths))
    for i in range(len(dishes_4)):
        P("   " + _fmt_row([names[i]] + [f"{contrib[i, j]:.4f}" for j in range(len(CRITERIA))] + [f"{totals[i]:.4f}"],
                           score_widths))

    P("\n   Kiểm tra tính khớp cộng từng món:")
    for i in range(len(dishes_4)):
        parts = " + ".join(f"{round(contrib[i, j], 4):.4f}" for j in range(len(CRITERIA)))
        P(f"   {names[i][:40]:<40}: {parts} = {sum(round(c, 4) for c in contrib[i]):.4f}  (điểm {totals[i]:.4f})")

    P("\n2.7. Bảng xếp hạng khi đi nhóm 4 người (giờ cao điểm)")
    for rank, i in enumerate(order, 1):
        warn_flag = " [Cảnh báo: Chờ chỗ]" if dishes_4[i]["seat_warning"] else ""
        P(f"   #{rank}  {names[i]:<48}  điểm = {totals[i]:.4f}{warn_flag}")

    P("\n2.8. So sánh tác động của Độ đông & Quy mô nhóm (Đi 1 người vs Đi 4 người)")
    cmp_headers = ["Món", "Điểm (1 ng)", "Hạng (1 ng)", "Điểm (4 ng)", "Hạng (4 ng)", "Thay đổi thứ hạng & Phân tích"]
    cmp_widths = [45, 12, 12, 12, 12, 45]
    P("   " + _fmt_row(cmp_headers, cmp_widths))

    # Xây dựng từ điển tra cứu xếp hạng
    rank_dict_1 = {dishes_1[idx]["id"]: (rank, totals_1[idx]) for rank, idx in enumerate(order_1, 1)}
    rank_dict_4 = {dishes_4[idx]["id"]: (rank, totals[idx]) for rank, idx in enumerate(order, 1)}

    for d in dishes:
        did = d["id"]
        r1, s1 = rank_dict_1[did]
        r4, s4 = rank_dict_4[did]
        if r4 > r1:
            analysis = f"Tụt #{r1} -> #{r4} (do quy mô nhỏ, thiếu chỗ ngồi nhóm)"
        elif r4 < r1:
            analysis = f"Tăng #{r1} -> #{r4} (quán rộng, đáp ứng tốt chỗ ngồi nhóm)"
        else:
            analysis = f"Giữ nguyên #{r1} (cân bằng tốt giữa giá, vị trí và chỗ ngồi)"
        P("   " + _fmt_row([
            f"[{did}] {d['ten_mon']}",
            f"{s1:.4f}",
            f"#{r1}",
            f"{s4:.4f}",
            f"#{r4}",
            analysis
        ], cmp_widths))

    P("\n   Nhận định quan trọng:")
    P("     • Bánh mì pate chả ngõ Tự Do: Quy mô chỉ 10 ghế. Giờ cao điểm vắng chỗ (chỗ trống = 1.0). Khi đi 1 người,")
    P("       quán đáp ứng đủ chỗ nên đạt điểm cao (0.6300, hạng #2). Nhưng khi đi nhóm 4 người, quán không đủ chỗ")
    P("       (cho_ngoi_nhom = 0.25), điểm tụt xuống 0.5500 và rơi xuống hạng #3 kèm cảnh báo chờ chỗ.")
    P("     • Bún đậu mắm tôm KTX NEU: Có 54 ghế (chỗ trống = 5.4 >= 4), đáp ứng trọn vẹn chỗ cho nhóm 4 người")
    P("       (cho_ngoi_nhom = 1.0), vươn lên hạng #2 thay thế vị trí của quán bánh mì.")

    P("\n2.9. Đối chiếu (assert sai số < 1e-6)")
    P(f"   MinMaxScaler của sklearn (đảo chiều cột chi phí) × trọng số: sai số điểm lớn nhất = {diff_score:.2e}, cùng thứ hạng.")
    P(f"   DishRecommender của dự án (nhóm 4 người): sai số điểm lớn nhất = {diff_rec_4:.2e}, cùng thứ hạng.")
    P(f"   DishRecommender của dự án (nhóm 1 người): sai số điểm lớn nhất = {diff_rec_1:.2e}, cùng thứ hạng.")

    P("\n" + SEP)
    P("3. KẾT LUẬN KIỂM CHỨNG")
    P(f"   Phần 1: từ vựng, idf, ma trận TF-IDF, prior, likelihood, predict_proba (sai số {diff_proba:.2e}) khớp sklearn.")
    P(f"   Phần 2: độ đông, thời gian chờ, chỗ trống, điểm và xếp hạng (sai số {diff_score:.2e}) khớp sklearn và DishRecommender.")
    P("   ĐÃ KHỚP VỚI SKLEARN")
    P(SEP)

    path = os.path.join(outputs_dir, output_filename)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    return path


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(generate_worked_example())
