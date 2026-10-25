# 🍜 NEU Lunch Chatbot - Trưa Nay Ăn Gì?

> **Hệ thống Chatbot tiếng Việt tư vấn bữa trưa thông minh dành riêng cho sinh viên Trường Đại học Kinh tế Quốc dân (NEU - Hà Nội).**  
> Dự án ứng dụng kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) với TF-IDF kết hợp mô hình học máy Multinomial Naive Bayes (scikit-learn) để phân loại nhu cầu người dùng, tích hợp thuật toán chấm điểm đa tiêu chí (MCDM) và giao diện trực quan PyQt6.  
> Đây là sản phẩm phục vụ bài tập lớn môn **Trí tuệ nhân tạo (Artificial Intelligence)**.

---

## 🌟 Tính năng chính

1. **Nhận biết nhu cầu ăn uống (Intent Classification)**:
   - Tự động phân loại câu hỏi tiếng Việt vào 8 nhóm ý định: `tiet_kiem` (Tiết kiệm), `healthy` (Ăn lành mạnh/healthy), `nhanh_gon` (Nhanh gọn vội học), `no_bung` (Ăn no chắc bụng), `ngon_doi_gio` (Ngon đổi gió/tụ tập), `can_bang` (Cân bằng dinh dưỡng), cùng 2 nhãn giao tiếp xã giao `chao_hoi` (Chào hỏi) và `cam_on_tam_biet` (Cảm ơn / Tạm biệt).
   - Cơ chế Fallback an toàn: Tự động chuyển về nhóm `can_bang` khi độ tin cậy thấp hơn ngưỡng cấu hình (Confidence Threshold).

2. **Trích xuất ràng buộc thực thể thông minh (Entity & Constraint Extraction)**:
   - Trích xuất ngân sách tối đa bằng biểu thức chính quy (hỗ trợ đa dạng cách diễn đạt: *"dưới 40k"*, *"tầm 35 nghìn"*, *"30.000đ"*, *"tối đa 50k"*).
   - Nhận diện yêu cầu loại trừ đồ cay (*"không ăn cay"*, *"bớt cay"*).
   - Nhận diện yêu cầu loại trừ loại món ăn (*"không ăn bún"*, *"chán cơm rồi"*).

3. **Gợi ý và Chấm điểm đa tiêu chí (Multi-Criteria Recommendation)**:
   - Đánh giá trên 6 tiêu chí thực tế: **Giá tiền**, **Khoảng cách từ NEU**, **Thời gian phục vụ**, **Độ no**, **Dinh dưỡng**, **Độ ngon**.
   - Áp dụng kỹ thuật chuẩn hóa Min-Max: Đảo chiều thang đo đối với các tiêu chí chi phí (giá, khoảng cách, thời gian càng thấp thì điểm càng cao).
   - Cho phép tùy biến trọng số linh hoạt: Chọn hồ sơ trọng số tự động theo nhu cầu (Profile-based) hoặc tự chỉnh 6 thanh trượt trên giao diện.

4. **Giao diện Desktop hiện đại (PyQt6 GUI)**:
   - Khung chat hai chiều dạng bong bóng hội thoại trực quan kèm icon và giải thích lý do gợi ý rõ ràng.
   - Minh bạch hóa toàn bộ các bước tính toán trung gian: Câu gốc $\to$ Chuẩn hóa $\to$ Token $\to$ Trọng số TF-IDF $\to$ Xác suất 8 nhãn $\to$ Ràng buộc trích xuất $\to$ Bảng đóng góp từng tiêu chí của từng món.
   - Tích hợp biểu đồ cột xếp chồng Matplotlib trực tiếp vào giao diện, hiển thị trực quan đóng góp của 6 tiêu chí vào tổng điểm từng món ăn.

5. **Quản lý dữ liệu tương tác (Data Management)**:
   - Tab **🗂️ Quản lý dữ liệu**: Hỗ trợ xem, tìm kiếm, thêm, sửa, xóa món ăn trong `data/dishes.csv` kèm kiểm tra tính hợp lệ nghiêm ngặt.
   - Thêm câu mẫu mới vào `data/intents.csv` theo từng nhãn ý định.
   - Nút **Huấn luyện lại mô hình** ngay trên giao diện chạy qua luồng ngầm (`QThread`), cập nhật mô hình tức thì mà không gây đơ ứng dụng.

---

## 📁 Cấu trúc thư mục

Cấu trúc dự án thực tế được tổ chức khoa học và phân tách rõ ràng theo từng module:

```text
lunch_chatbot/
├── config/
│   └── profiles.json               # Cấu hình trọng số 6 tiêu chí cho các nhóm nhu cầu
├── data/
│   ├── dishes.csv                  # Dữ liệu 40 món/quán ăn trưa mẫu quanh NEU
│   ├── intents.csv                 # Dữ liệu huấn luyện ý định (8 nhãn, 304 mẫu)
│   └── test_intents.csv            # Dữ liệu kiểm thử độc lập (8 nhãn, 64 mẫu)
├── gui/
│   ├── __init__.py
│   ├── app.py                      # Giao diện chính PyQt6 (Chat, Tham số, Các bước xử lý, Biểu đồ, Đánh giá)
│   └── data_tab.py                 # Tab giao diện Quản lý dữ liệu món ăn và câu mẫu
├── outputs/
│   ├── .gitkeep
│   ├── classifier_model.joblib     # Mô hình Naive Bayes đã huấn luyện và lưu trữ
│   ├── confusion_matrix.png        # Biểu đồ ma trận nhầm lẫn trên tập test
│   ├── exp_alpha.png               # Biểu đồ ảnh hưởng của siêu tham số alpha
│   ├── exp_ngram.png               # Biểu đồ ảnh hưởng của n-gram range
│   ├── exp_stopwords.png           # Biểu đồ so sánh có/không loại bỏ stopword
│   ├── misclassified_samples.csv   # Danh sách câu dự đoán sai trên tập test
│   ├── model_comparison.csv        # Bảng so sánh chỉ số giữa 3 thuật toán ML
│   ├── model_comparison.png        # Biểu đồ so sánh MultinomialNB, LogReg, LinearSVC
│   └── worked_example.txt          # Ví dụ tính toán từng bước chi tiết (Chương 2)
├── src/
│   ├── __init__.py
│   ├── chatbot.py                  # Bộ điều phối trung tâm kết nối toàn bộ pipeline
│   ├── classifier.py               # Huấn luyện và phân loại nhu cầu (TF-IDF + MultinomialNB/LogReg/SVM)
│   ├── data_manager.py             # Xử lý CRUD món ăn và thêm mẫu ý định có kiểm tra hợp lệ
│   ├── evaluate.py                 # Module đánh giá mô hình, kiểm thử chéo và xuất báo cáo/biểu đồ
│   ├── extractor.py                # Trích xuất ràng buộc thực thể (ngân sách, loại trừ, cay)
│   ├── preprocess.py               # Chuẩn hóa văn bản tiếng Việt, tách từ (PyVi) và lọc stopword
│   ├── recommender.py              # Thuật toán chuẩn hóa min-max và chấm điểm đa tiêu chí
│   └── worked_example.py           # Sinh ví dụ tính tay khớp tuyệt đối với scikit-learn
├── tests/
│   ├── __init__.py
│   ├── test_data_manager.py        # Kiểm thử CRUD dữ liệu món ăn và ý định
│   ├── test_extractor.py           # Kiểm thử trích xuất ràng buộc thực thể
│   ├── test_preprocess.py          # Kiểm thử tiền xử lý và tách từ tiếng Việt
│   ├── test_recommender.py         # Kiểm thử thuật toán chuẩn hóa và xếp hạng gợi ý
│   └── test_worked_example.py      # Kiểm thử sinh số liệu tính tay đối chiếu sklearn
├── .gitignore                      # Cấu hình bỏ qua tệp tạm, môi trường ảo và tệp nén
├── main.py                         # Điểm vào chính hỗ trợ CLI, GUI, kiểm tra dữ liệu, đánh giá
├── package.py                      # Script tự động đóng gói dự án thành tệp ZIP
├── README.md                       # Tài liệu hướng dẫn dự án
└── requirements.txt                # Danh sách thư viện phụ thuộc
```

---

## ⚙️ Yêu cầu môi trường & Cài đặt

- **Phiên bản Python**: Khuyến nghị sử dụng **Python 3.10 – 3.14** (Hệ thống đã kiểm thử và chạy ổn định trên **Python 3.14.7**).
- **Hệ điều hành**: Windows, macOS, hoặc Linux.

### Các bước cài đặt:

1. **Clone repository về máy**:
   ```bash
   git clone https://github.com/tenvuthang/Lunch-Chatbot-NEU.git
   cd Lunch-Chatbot-NEU
   ```

2. **Khởi tạo và kích hoạt môi trường ảo (khuyến nghị)**:
   ```bash
   # Tạo môi trường ảo
   python -m venv venv

   # Kích hoạt môi trường ảo:
   # Trên Windows (PowerShell / CMD):
   venv\Scripts\activate
   # Trên Linux / macOS:
   source venv/bin/activate
   ```

3. **Cài đặt các gói thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Hướng dẫn chạy chương trình

Dự án hỗ trợ đầy đủ các phương thức khởi chạy từ giao diện đồ họa đến dòng lệnh:

### 1. Khởi chạy Giao diện Desktop GUI (PyQt6)
Chạy ứng dụng với đầy đủ giao diện tương tác, trực quan hóa biểu đồ và bảng quản lý dữ liệu:
```bash
python main.py --gui
# hoặc chạy trực tiếp:
python gui/app.py
```

### 2. Chế độ Chat dòng lệnh tương tác (CLI Interactive)
Trò chuyện trực tiếp với chatbot trong terminal, bot sẽ phản hồi kèm phân tích chi tiết các bước xử lý trung gian:
```bash
python main.py --chat
```

### 3. Chế độ hỏi nhanh một câu (One-shot CLI)
Truyền trực tiếp câu hỏi qua tham số dòng lệnh để nhận ngay kết quả:
```bash
python main.py "tìm quán cơm bình dân ngõ Tự Do giá dưới 35k không ăn cay"
```

### 4. Kiểm tra tính hợp lệ của dữ liệu (Data Integrity Check)
Xác thực cấu hình `profiles.json`, tập mẫu `intents.csv`, `test_intents.csv` và in thống kê chi tiết `dishes.csv`:
```bash
python main.py --check
```

### 5. Chạy đánh giá mô hình và xuất báo cáo/biểu đồ
Tự động huấn luyện, đánh giá độc lập, chạy 5-Fold Cross Validation, so sánh 3 thuật toán và xuất hình ảnh vào thư mục `outputs/`:
```bash
python main.py --eval
# hoặc chạy trực tiếp:
python src/evaluate.py
```

### 6. Chạy bộ kiểm thử tự động (Unit Tests)
Dự án bao gồm 31 bài test tự động bao phủ toàn bộ các module (tiền xử lý, trích xuất thực thể, gợi ý đa tiêu chí, CRUD dữ liệu, tính tay đối chiếu sklearn):
```bash
pytest -v
```

### 7. Đóng gói mã nguồn dự án thành file ZIP
Tự động quét và đóng gói các thư mục mã nguồn, tài liệu, kết quả đánh giá (bỏ qua cache rác):
```bash
python main.py --package
# hoặc chạy trực tiếp:
python package.py
```

---

## 📊 Mô tả dữ liệu & Hướng dẫn thay thế dữ liệu thật

### 1. Dữ liệu huấn luyện và kiểm thử ý định
- **`data/intents.csv`**: Gồm **304 mẫu câu** phân bổ đều cho 8 nhãn ý định (38 mẫu/nhãn). Dữ liệu được biên soạn tự nhiên với cách diễn đạt của sinh viên, bao gồm câu tiếng Việt có dấu, không dấu, câu viết tắt (*"ko cay"*, *"sv"*, *"k"*), câu chứa ràng buộc ngân sách và yêu cầu loại trừ.
- **`data/test_intents.csv`**: Gồm **64 mẫu câu độc lập** (8 mẫu/nhãn), hoàn toàn không trùng lặp câu nào với tập huấn luyện, dùng để kiểm thử khách quan độ chuẩn xác của mô hình.

### 2. Dữ liệu danh sách món ăn (`data/dishes.csv`)
Tập tin chứa 40 món ăn tiêu biểu thuộc nhiều nhóm (cơm, bún, phở, bánh mì, mì, đồ ăn nhanh, salad, cơm văn phòng, khác) với 11 cột thuộc tính:
- `id`: Mã định danh món ăn (1 đến 40).
- `ten_mon`: Tên món ăn hoặc quán ăn.
- `loai`: Nhóm món ăn.
- `gia_vnd`: Giá tiền một suất ăn (dao động từ 20.000đ đến 80.000đ).
- `khoang_cach_m`: Khoảng cách đi bộ từ cổng KTX/Trường NEU (80m đến 900m).
- `thoi_gian_phuc_vu_phut`: Thời gian chế biến và chờ phục vụ (5 đến 15 phút).
- `do_no`: Mức độ no sau khi ăn (thang đo 1 - 5).
- `dinh_duong`: Giá trị dinh dưỡng, độ cân bằng rau/thịt (thang đo 1 - 5).
- `do_ngon`: Đánh giá độ ngon miệng, hương vị (thang đo 1 - 5).
- `cay`: Món có vị cay hay không (1: Cay, 0: Không cay).
- `mo_ta`: Mô tả ngắn gọn về đặc điểm của món/quán.

> [!NOTE]
> **LƯU Ý VỀ DỮ LIỆU MÓN ĂN**:
> Tập dữ liệu `data/dishes.csv` là **DỮ LIỆU MẪU TỰ BIÊN SOẠN / GIẢ LẬP** phản ánh chân thực môi trường ẩm thực xung quanh Đại học Kinh tế Quốc dân (khu vực ngõ Tự Do, đường Trần Đại Nghĩa, KTX NEU, phố Lê Thanh Nghị, Tạ Quang Bửu, Đại La).  
> **Cách thay bằng dữ liệu thực tế:**
> 1. **Cách 1 (Trực quan)**: Khởi chạy giao diện (`python main.py --gui`), chuyển sang tab **🗂️ Quản lý dữ liệu** để thêm, sửa, xóa món ăn trực tiếp. Dữ liệu sẽ được lưu tự động vào `data/dishes.csv` và hệ thống gợi ý cập nhật ngay lập tức.
> 2. **Cách 2 (Thủ công)**: Mở file `data/dishes.csv` bằng Excel hoặc trình soạn thảo văn bản, chỉnh sửa hoặc bổ sung thêm các dòng dữ liệu thật theo đúng định dạng 11 cột nêu trên.

---

## 📈 Kết quả đánh giá mô hình

Toàn bộ số liệu dưới đây được trích xuất trực tiếp từ các file kết quả thực nghiệm trong thư mục `outputs/`:

### 1. So sánh hiệu năng 3 mô hình học máy (`outputs/model_comparison.csv`)

| Mô hình Học máy | Test Accuracy | Test Macro F1 | Test Macro Precision | Test Macro Recall | 5-Fold CV Accuracy (Mean ± Std) | 5-Fold CV F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (chính)** | **85.94%** (0.8594) | **0.8606** | **0.8705** | **0.8594** | **76.94% ± 5.39%** | **0.7700** |
| **Logistic Regression** | 89.06% (0.8906) | 0.8916 | 0.9111 | 0.8906 | 80.56% ± 5.95% | 0.8019 |
| **Linear Support Vector (LinearSVC)** | 93.75% (0.9375) | 0.9384 | 0.9520 | 0.9375 | 86.49% ± 4.99% | 0.8621 |

> **Nhận xét**:  
> - **Multinomial Naive Bayes** (mô hình cốt lõi của hệ thống) đạt **85.94% độ chính xác** trên tập kiểm thử độc lập (chỉ phân loại sai 9/64 mẫu). Mô hình có ưu điểm vượt trội về tốc độ huấn luyện/suy luận tính bằng mili-giây, cung cấp phân phối xác suất dự đoán rõ ràng (`predict_proba`) rất thuận lợi cho việc giải thích lý do gợi ý trên giao diện và tính toán minh bạch từng bước.
> - **LinearSVC** và **Logistic Regression** đạt độ chính xác cao hơn (89.06% – 93.75%), cho thấy không gian đặc trưng TF-IDF phân tách tuyến tính rất tốt.

### 2. Ma trận nhầm lẫn của Multinomial Naive Bayes trên tập kiểm thử

![Ma trận nhầm lẫn Naive Bayes](outputs/confusion_matrix.png)

### 3. Khớp kiểm chứng tính tay với lý thuyết (Chương 2)
Trong tệp `outputs/worked_example.txt`, toàn bộ các bước tính toán thủ công:
- Bước 1: Tính toán từ vựng, Smooth IDF, ma trận TF-IDF (L2-norm), xác suất tiên nghiệm $P(C)$, likelihood có làm mịn Laplace $\alpha = 0.5$, và suy diễn xác suất hậu nghiệm $P(C|x)$ cho câu thử `"tìm quán cơm giá rẻ"`.
- Bước 2: Chuẩn hóa Min-Max đa tiêu chí (đảo chiều chi phí), tính toán lượng đóng góp $w \times v$ và xếp hạng 4 món ăn theo nhóm nhu cầu `tiet_kiem`.
- Kết quả tính toán bằng công thức toán học hoàn toàn trùng khớp với đầu ra của thư viện `scikit-learn` với sai số tuyệt đối $< 10^{-15}$ (`ĐÃ KHỚP VỚI SKLEARN`).

---

## 🔍 Hạn chế và Hướng phát triển

1. **Mở rộng dữ liệu hội thoại**: Quy mô tập dữ liệu ý định hiện tại ở mức thực nghiệm học tập (368 câu mẫu). Hướng phát triển tiếp theo là triển khai khảo sát diện rộng tới sinh viên NEU để thu thập ngữ liệu thực tế phong phú hơn.
2. **Nâng cấp mô hình hiểu ngôn ngữ (NLU)**: Hiện tại phần trích xuất thực thể (ngân sách, loại trừ) dựa trên Regex và từ khóa luật. Có thể tích hợp mô hình nhận dạng thực thể tiếng Việt sâu hơn như PhoBERT-NER để hiểu các cấu trúc ngữ nghĩa phức tạp và phủ định lồng nhau.
3. **Cá nhân hóa theo hành vi người dùng**: Hiện tại trọng số tiêu chí được áp dụng cố định theo từng profile. Có thể mở rộng cơ chế phản hồi tương tác (Like/Dislike, đánh giá sao món ăn) kết hợp thuật toán Contextual Bandits để tự động tối ưu trọng số riêng cho từng người dùng theo thời gian.
4. **Tích hợp dữ liệu thời gian thực**: Kết nối API bản đồ (Google Maps / OpenStreetMap) để tính khoảng cách và thời gian đi bộ chính xác, tích hợp giờ mở cửa và trạng thái đông bàn của quán ăn theo thời gian thực.

---

## 📜 Giấy phép (License)

Dự án được phân phối dưới giấy phép mã nguồn mở **[MIT License](https://opensource.org/licenses/MIT)**. Bạn hoàn toàn có thể tự do sử dụng, chỉnh sửa và phát triển tiếp cho mục đích học tập và nghiên cứu.
