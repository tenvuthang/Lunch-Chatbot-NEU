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
from src.recommender import DishRecommender

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
# 4 món lấy từ data/dishes.csv (giá, khoảng cách, thời gian... đều có phân bố khác nhau, mỗi món một loại)
DISH_IDS = [1, 12, 28, 34]
TOKEN_PATTERN = r"(?u)\b\w\w+\b"  # mặc định của TfidfVectorizer
CRITERIA = [  # (khóa trong profiles.json, cột trong dishes.csv, tên hiển thị, là tiêu chí chi phí?)
    ("gia", "gia_vnd", "Giá (đ)", True),
    ("khoang_cach", "khoang_cach_m", "Khoảng cách (m)", True),
    ("thoi_gian", "thoi_gian_phuc_vu_phut", "Thời gian (phút)", True),
    ("do_no", "do_no", "Độ no", False),
    ("dinh_duong", "dinh_duong", "Dinh dưỡng", False),
    ("do_ngon", "do_ngon", "Độ ngon", False),
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
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
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

    # ══ 4. PHẦN 2: CHẤM ĐIỂM MÓN ĂN ══════════════════════════════════════════════════════════════════
    with open(os.path.join(root_dir, "config", "profiles.json"), "r", encoding="utf-8") as f:
        profiles = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    weights = profiles[pred_label]
    assert abs(sum(weights.values()) - 1.0) < 1e-9 and set(weights) == {c[0] for c in CRITERIA}
    with open(os.path.join(root_dir, "data", "dishes.csv"), "r", encoding="utf-8", newline="") as f:
        all_dishes = {int(r["id"]): r for r in csv.DictReader(f)}
    dishes = [all_dishes[i] for i in DISH_IDS]
    raw = np.array([[float(d[col]) for _, col, _, _ in CRITERIA] for d in dishes])
    mins, maxs = raw.min(axis=0), raw.max(axis=0)
    assert np.all(maxs > mins), "Có tiêu chí không phân biệt được giữa 4 món"
    norm = np.zeros_like(raw)
    for j, (_, _, _, is_cost) in enumerate(CRITERIA):
        norm[:, j] = (maxs[j] - raw[:, j]) / (maxs[j] - mins[j]) if is_cost else (raw[:, j] - mins[j]) / (maxs[j] - mins[j])
    w_vec = np.array([weights[k] for k, _, _, _ in CRITERIA])
    contrib = norm * w_vec
    totals = contrib.sum(axis=1)
    order = list(np.argsort(-totals))

    # ── Đối chiếu: MinMaxScaler của sklearn ──
    scaled = MinMaxScaler().fit_transform(raw)
    for j, (_, _, _, is_cost) in enumerate(CRITERIA):
        if is_cost:
            scaled[:, j] = 1.0 - scaled[:, j]
    sk_totals = scaled @ w_vec
    diff_score = float(np.max(np.abs(sk_totals - totals)))
    assert diff_score < 1e-6 and list(np.argsort(-sk_totals)) == order, "Điểm/xếp hạng lệch sklearn"
    # ── Đối chiếu: DishRecommender thật của dự án ──
    rec = DishRecommender(dishes_path=os.path.join(root_dir, "data", "dishes.csv"),
                          profiles_path=os.path.join(root_dir, "config", "profiles.json"))
    rec.dishes = [{**{k: v for k, v in d.items()}, "id": int(d["id"]),
                   **{c: float(d[c]) for _, c, _, _ in CRITERIA}, "cay": int(d["cay"])} for d in dishes]
    rec._calculate_stats()
    rec_result = rec.recommend(pred_label, top_k=len(dishes))["all_ranked"]
    rec_scores = {item["dish"]["id"]: item["score"] for item in rec_result}
    diff_rec = max(abs(rec_scores[int(d["id"])] - totals[i]) for i, d in enumerate(dishes))
    assert diff_rec < 1e-6, f"DishRecommender lệch {diff_rec}"
    assert [item["dish"]["id"] for item in rec_result] == [int(dishes[i]["id"]) for i in order]
    # ── Tổng các đóng góp (làm tròn 4 chữ số) phải khớp đúng điểm làm tròn 4 chữ số ──
    for i in range(len(dishes)):
        assert abs(sum(round(c, 4) for c in contrib[i]) - round(totals[i], 4)) < 1e-9, \
            f"Tổng đóng góp làm tròn không khớp điểm ở món {dishes[i]['ten_mon']} -> chọn bộ món khác"

    # ══ 5. XUẤT VĂN BẢN ═══════════════════════════════════════════════════════════════════════════════
    P(SEP)
    P("VÍ DỤ TÍNH TAY (WORKED EXAMPLE) – CHƯƠNG 2")
    P("Hệ thống chatbot gợi ý bữa trưa cho sinh viên NEU")
    P("Mọi số liệu dưới đây do công thức tính tay sinh ra và đã được đối chiếu với sklearn (xem mục 1.10, 2.6 và 3).")
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
    P("PHẦN 2. CHẤM ĐIỂM VÀ XẾP HẠNG MÓN ĂN THEO NHÓM NHU CẦU")
    P(SEP)
    P(f"2.1. Nhóm nhu cầu = nhãn dự đoán ở Phần 1: '{pred_label}'. Trọng số THẬT đọc từ config/profiles.json:")
    for k, _, name, _ in CRITERIA:
        P(f"   w_{k:<12} = {weights[k]:.2f}   ({name})")
    P(f"   Tổng trọng số = {sum(weights.values()):.2f}")

    P("\n2.2. Bốn món lấy từ data/dishes.csv (dùng đủ 6 tiêu chí)")
    names = [f"[{d['id']}] {d['ten_mon']}" for d in dishes]
    P("   " + _fmt_row(["Món"] + [c[2] for c in CRITERIA], [56] + [17, 17, 17, 8, 11, 8]))
    for i, d in enumerate(dishes):
        P("   " + _fmt_row([names[i]] + [f"{raw[i, j]:.0f}" for j in range(len(CRITERIA))], [56] + [17, 17, 17, 8, 11, 8]))

    P("\n2.3. Chuẩn hóa min-max trong 4 món (đảo chiều cho tiêu chí chi phí)")
    P("   Chi phí (giá, khoảng cách, thời gian): v = (max − x) / (max − min)")
    P("   Lợi ích (độ no, dinh dưỡng, độ ngon) : v = (x − min) / (max − min)")
    P("   " + _fmt_row(["Tiêu chí", "min", "max", "max−min", "Loại"], [18, 8, 8, 8, 10]))
    for j, (_, _, name, is_cost) in enumerate(CRITERIA):
        P("   " + _fmt_row([name, f"{mins[j]:.0f}", f"{maxs[j]:.0f}", f"{maxs[j] - mins[j]:.0f}",
                           "chi phí" if is_cost else "lợi ích"], [18, 8, 8, 8, 10]))
    P("\n   Giá trị chuẩn hóa v:")
    P("   " + _fmt_row(["Món"] + [c[2] for c in CRITERIA], [56] + [17, 17, 17, 8, 11, 8]))
    for i in range(len(dishes)):
        P("   " + _fmt_row([names[i]] + [f"{norm[i, j]:.4f}" for j in range(len(CRITERIA))], [56] + [17, 17, 17, 8, 11, 8]))

    P("\n2.4. Đóng góp w × v của từng tiêu chí (4 chữ số thập phân) và điểm tổng  Điểm = Σ w×v")
    P("   " + _fmt_row(["Món"] + [c[2] for c in CRITERIA] + ["Σ = Điểm"], [56] + [17, 17, 17, 8, 11, 8] + [9]))
    for i in range(len(dishes)):
        P("   " + _fmt_row([names[i]] + [f"{contrib[i, j]:.4f}" for j in range(len(CRITERIA))] + [f"{totals[i]:.4f}"],
                           [56] + [17, 17, 17, 8, 11, 8] + [9]))
    P("\n   Kiểm tra cộng từng món:")
    for i in range(len(dishes)):
        parts = " + ".join(f"{round(contrib[i, j], 4):.4f}" for j in range(len(CRITERIA)))
        P(f"   {names[i][:45]:<45}: {parts} = {sum(round(c, 4) for c in contrib[i]):.4f}  (điểm {totals[i]:.4f})")

    P("\n2.5. Bảng xếp hạng")
    for rank, i in enumerate(order, 1):
        P(f"   #{rank}  {names[i]:<56}  điểm = {totals[i]:.4f}")

    P("\n2.6. Đối chiếu (assert sai số < 1e-6)")
    P(f"   MinMaxScaler của sklearn (đảo chiều cột chi phí) × trọng số: sai số điểm lớn nhất = {diff_score:.2e}, cùng thứ hạng.")
    P(f"   DishRecommender của dự án (4 món trên): sai số điểm lớn nhất = {diff_rec:.2e}, cùng thứ hạng.")

    P("\n" + SEP)
    P("3. KẾT LUẬN KIỂM CHỨNG")
    P(f"   Phần 1: từ vựng, idf, ma trận TF-IDF, prior, likelihood, predict_proba (sai số {diff_proba:.2e}) khớp sklearn.")
    P(f"   Phần 2: điểm và xếp hạng (sai số {diff_score:.2e}) khớp sklearn và DishRecommender.")
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
