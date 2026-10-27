# 📘 HƯỚNG DẪN DÀNH CHO NGƯỜI MỚI BẮT ĐẦU (NON-TECH GUIDE)
## Hệ thống Chatbot Tư Vấn Bữa Trưa Cho Sinh Viên NEU

Chào bạn! Nếu bạn **không chuyên về công nghệ (non-tech)**, chưa từng học sâu về lập trình hay Trí tuệ nhân tạo (AI), tài liệu này được viết riêng để giúp bạn làm quen, sử dụng và làm chủ công cụ này một cách dễ dàng nhất — từ việc trò chuyện tìm món ăn, tinh chỉnh theo ý thích, cho đến việc "dạy" (huấn luyện) thêm cho bot thông minh hơn.

---

## 🧭 MỤC LỤC
1. [Chatbot này là gì và giúp gì cho bạn?](#1-chatbot-này-là-gì-và-giúp-gì-cho-bạn)
2. [Cài đặt siêu nhanh trong 3 bước](#2-cài-đặt-siêu-nhanh-trong-3-bước)
3. [Làm quen với giao diện ứng dụng (Cửa sổ trực quan)](#3-làm-quen-với-giao-diện-ứng-dụng)
4. [Hướng dẫn sử dụng hàng ngày: Hỏi và tìm món ăn](#4-hướng-dẫn-sử-dụng-hàng-ngày-hỏi-và-tìm-món-ăn)
5. [Tùy chỉnh "tính cách" và tham số của bot (Không cần biết code)](#5-tùy-chỉnh-tính-cách-và-tham-số-của-bot)
6. [Quản lý danh sách món ăn: Thêm, sửa, xóa quán ăn](#6-quản-lý-danh-sách-món-ăn-thêm-sửa-xóa-quán-ăn)
7. ["Dạy" thêm câu hỏi mới và Huấn luyện lại bot (Training)](#7-dạy-thêm-câu-hỏi-mới-và-huấn-luyện-lại-bot-training)
8. [Kiểm tra xem bot học giỏi đến đâu (Đánh giá mô hình)](#8-kiểm-tra-xem-bot-học-giỏi-đến-đâu-đánh-giá-mô-hình)
9. [Các lỗi thường gặp và cách tự xử lý đơn giản](#9-các-lỗi-thường-gặp-và-cách-tự-xử-lý-đơn-giản)

---

## 1. Chatbot này là gì và giúp gì cho bạn?

Hãy tưởng tượng bạn có một **"người bạn sành ăn"** cắm chốt ngay tại cổng trường Đại học Kinh tế Quốc dân (NEU). Mỗi khi đến trưa và bạn chưa biết ăn gì, bạn chỉ cần nhắn cho bạn ấy:

* *"Trưa nay cháy túi rồi, có gì ăn dưới 25k gần trường không?"*
* *"Muốn ăn đồ healthy ít dầu mỡ, không ăn cay nhé!"*
* *"Sắp vào ca học chiều, có món gì ăn nhanh mang đi được không?"*

Bot sẽ:
1. **Hiểu được tâm lý & nhu cầu**: Bạn đang muốn ăn **tiết kiệm**, ăn **healthy**, ăn **nhanh gọn**, hay muốn **ăn no**, **đổi gió ăn ngon**...
2. **Lọc chuẩn theo yêu cầu**: Tự bóc tách ngân sách tối đa (ví dụ dưới 30k) và tránh các món bạn kiêng (ví dụ không ăn cay, không ăn bún).
3. **Chấm điểm thông minh trên 6 tiêu chí**: Cân đối giữa Giá tiền, Khoảng cách đi bộ, Thời gian đợi, Độ no bụng, Dinh dưỡng và Độ ngon miệng để chọn ra những quán ăn phù hợp nhất quanh các khu vực quen thuộc: ngõ Tự Do, Trần Đại Nghĩa, KTX NEU, Lê Thanh Nghị...

---

## 2. Cài đặt siêu nhanh trong 3 bước

Nếu bạn vừa tải thư mục dự án về máy tính Windows, chỉ cần làm lần lượt 3 bước sau:

### Bước 1: Cài đặt Python (nếu máy chưa có)
1. Tải Python phiên bản 3.10 đến 3.14 từ trang chủ: [python.org/downloads](https://www.python.org/downloads/)
2. Khi mở file cài đặt, **BẮT BUỘC TÍCH VÀO Ô**: ☑️ **"Add python.exe to PATH"** (hoặc "Add Python to environment variables") ở màn hình đầu tiên trước khi bấm **Install Now**.

### Bước 2: Mở cửa sổ dòng lệnh tại thư mục dự án
1. Mở thư mục `Lunch_chatbot` trên máy tính của bạn.
2. Nhấp chuột vào thanh địa chỉ đường dẫn ở phía trên cùng của cửa sổ thư mục, gõ chữ `cmd` (hoặc `powershell`) rồi ấn phím **Enter**. Cửa sổ màu đen hiện ra.

### Bước 3: Cài đặt thư viện cần thiết
Dán dòng lệnh sau vào cửa sổ đó rồi ấn **Enter**:
```bash
pip install -r requirements.txt
```
> ⏱️ *Quá trình này chỉ mất 1-2 phút trong lần đầu tiên để máy tính tải các công cụ hỗ trợ giao diện và học máy.*

---

## 3. Làm quen với giao diện ứng dụng

Để mở giao diện trực quan với đầy đủ nút bấm và biểu đồ đẹp mắt, bạn chỉ cần gõ lệnh sau và ấn **Enter**:

```bash
python main.py --gui
```
*(Hoặc gõ: `python gui/app.py`)*

Một cửa sổ hiện đại sẽ mở lên, được chia làm 2 nửa rõ ràng:

```text
┌────────────────────────────────────────┬────────────────────────────────────────┐
│             NỬA BÊN TRÁI               │              NỬA BÊN PHẢI              │
│       (KHUNG TRÒ CHUYỆN & NHẬP)        │      (CÁC TAB ĐIỀU KHIỂN & PHÂN TÍCH)  │
├────────────────────────────────────────┼────────────────────────────────────────┤
│                                        │ [Tab 1] 🔍 Các bước xử lý              │
│  [Bot]: Chào bạn! Trưa nay bạn muốn... │ [Tab 2] 📊 Biểu đồ trực quan           │
│                                        │ [Tab 3] ⚙️ Tham số thuật toán          │
│  [Bạn]: Tìm quán cơm bình dân dưới 30k │ [Tab 4] 📈 Đánh giá mô hình            │
│                                        │ [Tab 5] 🗂️ Quản lý dữ liệu             │
│  [Bot]: Mình ưu tiên giá rẻ...         │                                        │
│         1. Cơm bình dân ngõ Tự Do      │ (Bấm qua lại giữa các tab để xem       │
│         2. Cơm tấm sườn nướng...       │  chi tiết bên trong)                   │
├────────────────────────────────────────┴────────────────────────────────────────┤
│ [Ô nhập câu hỏi của bạn...]                                  [ Nút GỬI ]        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Hướng dẫn sử dụng hàng ngày: Hỏi và tìm món ăn

### Cách đặt câu hỏi để bot hiểu tốt nhất
Bot hiểu được tiếng Việt tự nhiên có dấu hoặc không dấu, thậm chí cả từ viết tắt thông dụng:

| Bạn muốn gì? | Ví dụ câu bạn có thể gõ vào khung chat |
| :--- | :--- |
| **Tiết kiệm tiền** | *"Có món nào rẻ hạt dẻ dưới 25k gần trường không?"* |
| **Ăn uống healthy** | *"Trưa nay muốn ăn thanh đạm ít dầu mỡ, nhiều rau"* |
| **Vội giờ học ca chiều** | *"Sắp muộn học rồi, gợi ý món gì lấy ngay mang đi được"* |
| **Muốn ăn thật no** | *"Đói cồn cào rồi, quán nào cơm nhiều thịt nhiều ăn no căng"* |
| **Rủ bạn đổi gió** | *"Trưa nay có tiền muốn ăn ngon đổi gió quanh KTX"* |
| **Có yêu cầu kiêng cữ** | *"Tìm quán bún dưới 40k nhưng không ăn cay nhé"* |

### Đọc câu trả lời của bot
Sau khi bạn ấn nút **Gửi** (hoặc ấn phím **Enter** trên bàn phím):
- Bot sẽ giải thích lý do ngắn gọn: *"Mình ưu tiên tối đa tiêu chí giá hạt dẻ, tiết kiệm chi phí cho sinh viên..."*
- Danh sách món ăn được gợi ý kèm đầy đủ: **Tên quán/món**, **Giá tiền**, **Khoảng cách đi bộ từ cổng NEU**, **Thời gian chờ**, **Mức độ cay (🌶️ hay 🟢)** và **Mô tả ngắn**.

### Xem "Não bộ" của bot hoạt động (Tại Tab "🔍 Các bước xử lý")
Nếu bạn tò mò tại sao bot lại chọn quán đó, hãy bấm sang tab **"Các bước xử lý"** ở bên phải:
- Bạn sẽ thấy bot tách câu bạn nói thành từng từ thế nào.
- Bot tính xác suất câu đó thuộc nhóm nào (ví dụ: `tiet_kiem` 68%, `healthy` 15%).
- Bot tự nhận diện ngân sách (ví dụ: tối đa 35.000đ) và lọc bỏ các món không hợp.

### Xem biểu đồ trực quan (Tại Tab "📊 Biểu đồ")
- Bấm vào tab **Biểu đồ**, bạn sẽ thấy đồ thị hình cột phân tích chi tiết: vì sao món Cơm bình dân đạt điểm cao (phần màu xanh dương của "Giá rẻ" đóng góp rất lớn).

---

## 5. Tùy chỉnh "tính cách" và tham số của bot

Nếu bạn muốn can thiệp để bot tìm món theo sở thích riêng của bạn thay vì theo mặc định, hãy mở tab **"⚙️ Tham số thuật toán"**:

```text
┌────────────────────────────────────────────────────────┐
│  ⚙️ BẢNG ĐIỀU CHỈNH THAM SỐ                           │
│                                                        │
│  Số món muốn gợi ý (Top-K): [ 3 món ▼ ]                │
│  Độ mở từ mới (Alpha):      [====●=============] 0.5   │
│  Ngưỡng tự tin:             [========●=========] 0.25  │
│                                                        │
│  ⚖️ CHỈNH 6 THANH TRƯỢT ƯU TIÊN:                       │
│  - Giá rẻ:         [====================●] 0.35        │
│  - Gần trường:     [========●============] 0.15        │
│  - Phục vụ nhanh:  [=====●===============] 0.10        │
│  - Ăn no bụng:     [===========●=========] 0.20        │
│  - Dinh dưỡng:     [=====●===============] 0.10        │
│  - Vị ngon miệng:  [=====●===============] 0.10        │
│                                                        │
│  [ Dùng tự động theo nhóm ]      [ Đặt lại mặc định ]  │
└────────────────────────────────────────────────────────┘
```

### Ý nghĩa của các nút điều chỉnh (Giải thích dễ hiểu):
1. **Số lượng gợi ý (Top-K)**: Bạn muốn bot đưa ra 3 quán, 4 quán hay 5 quán.
2. **Ngưỡng tự tin (Confidence Threshold)**:
   - Nếu bạn nói một câu quá mơ hồ (ví dụ: *"Trưa nay đói bụng"*), độ tự tin phân loại có thể rất thấp.
   - Nếu dưới ngưỡng này, bot sẽ không cố đoán bừa mà thông minh chuyển về chế độ **"Cân bằng"** (hài hòa cả giá, ngon, no và khoảng cách).
3. **6 Thanh trượt trọng số (Bạn coi trọng điều gì nhất?)**:
   - Bạn là người đặt tiêu chí **Giá rẻ** lên hàng đầu? Hãy kéo thanh trượt "Giá" lên cao.
   - Bạn đang vội lên lớp học ca chiều? Hãy kéo thanh trượt "Thời gian phục vụ" và "Khoảng cách" lên cao.
   - Bạn có thể ấn nút **"Dùng trọng số tự động theo nhóm nhu cầu"** bất cứ lúc nào để máy tự phân bổ tối ưu theo ý định câu nói của bạn.

---

## 6. Quản lý danh sách món ăn: Thêm, sửa, xóa quán ăn

Xung quanh cổng NEU có quán bún mới mở hoặc quán cơm quen thuộc vừa tăng giá thêm 5.000đ? Bạn có thể tự cập nhật dữ liệu ngay trên ứng dụng mà **không cần mở file Excel**:

1. Bấm vào tab **"🗂️ Quản lý dữ liệu"** ở góc phải màn hình.
2. Nửa trên là bảng danh sách toàn bộ các quán ăn hiện có.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 📋 DANH SÁCH MÓN ĂN QUANH NEU                                           │
│ [🔍 Tìm kiếm món/quán...]                                               │
├─────┬──────────────────────────┬──────────┬──────────┬──────────┬───────┤
│ ID  │ Tên món / Quán           │ Loại     │ Giá (đ)  │ Cách NEU │ Cay?  │
├─────┼──────────────────────────┼──────────┼──────────┼──────────┼───────┤
│ 1   │ Cơm bình dân tự chọn     │ cơm      │ 25,000   │ 150m     │ Không │
│ 2   │ Bún chả nướng than hoa   │ bún      │ 35,000   │ 200m     │ Không │
└─────┴──────────────────────────┴──────────┴──────────┴──────────┴───────┘
```

### Cách thực hiện:
- **Để sửa thông tin một quán**: Bấm chuột vào dòng của quán đó trong bảng. Các thông tin sẽ tự điền vào khung nhập phía dưới. Bạn sửa lại (ví dụ đổi giá từ `25,000` thành `30,000`), sau đó bấm nút **"💾 Cập nhật món ăn"**.
- **Để thêm một quán mới toanh**: Bấm nút **"Làm mới form"**, điền tên quán, địa chỉ, chọn loại món (cơm, bún, phở, bánh mì...), điền giá tiền và chấm điểm ước lượng (1-5 sao) rồi bấm nút **"➕ Thêm món mới"**.
- **Để xóa một quán đã đóng cửa**: Bấm chọn quán đó trên bảng và bấm nút **"🗑️ Xóa món ăn"** (chọn Yes khi hộp thoại hỏi lại).

> 💡 *Mọi thay đổi sẽ được lưu ngay vào hệ thống. Ở lượt chat tiếp theo, bot sẽ sử dụng ngay danh sách quán mới này để tư vấn cho bạn!*

---

## 7. "Dạy" thêm câu hỏi mới và Huấn luyện lại bot (Training)

Nếu bạn thử gõ một câu mà thấy bot đoán chưa đúng ý (ví dụ: bạn gõ *"tìm quán trà sữa bánh ngọt"* nhưng bot lại nghĩ là bạn muốn ăn no), bạn có thể **dạy cho bot học thêm câu đó**:

### Bước 1: Thêm câu hỏi mẫu
1. Tại tab **"🗂️ Quản lý dữ liệu"**, cuộn xuống phần **"Thêm câu mẫu huấn luyện ý định"** ở phía dưới bên phải.
2. Nhập câu của bạn vào ô: `tìm quán trà sữa bánh ngọt đổi gió`.
3. Chọn nhóm nhu cầu đúng từ menu thả xuống: ví dụ chọn **`ngon_doi_gio` (Ngon đổi gió / Tụ tập)**.
4. Bấm nút **"➕ Thêm câu mẫu"**. Máy sẽ báo đã thêm câu thành công vào kho tri thức của bot!

### Bước 2: Bấm nút "Huấn luyện lại" (Retrain)
Sau khi thêm một hoặc nhiều câu mẫu, bạn chỉ cần bấm nút:
> 🚀 **"Huấn luyện lại mô hình (Retrain)"** *(Màu xanh lá cây nổi bật)*

- **Điều gì sẽ xảy ra?** Mô hình học máy sẽ tự động đọc lại dữ liệu, cập nhật thuật toán thông minh hơn chỉ trong khoảng **2 đến 3 giây**.
- Giao diện vẫn hoạt động mượt mà không bị treo đơ.
- Ngay sau đó, bạn quay lại khung chat gõ lại câu tương tự, bot đã "thông minh" hơn và hiểu chính xác ý của bạn!

---

## 8. Kiểm tra xem bot học giỏi đến đâu (Đánh giá mô hình)

Làm sao để biết bot tư vấn chuẩn xác được bao nhiêu phần trăm?

### Cách 1: Xem trực tiếp trên giao diện
1. Mở tab **"📈 Đánh giá"** trên giao diện.
2. Bạn sẽ thấy ngay:
   - **Độ chính xác (Accuracy)**: Bot của chúng ta đạt độ chính xác lên tới **85.94%** (nghĩa là trong 100 câu hỏi độc lập của sinh viên, bot trả lời chuẩn xác khoảng 86 câu).
   - **Ảnh Ma trận nhầm lẫn (Confusion Matrix)**: Biểu đồ màu sắc thể hiện các nhãn. Những ô nằm trên đường chéo có màu càng đậm chứng tỏ bot nhận diện nhãn đó càng chuẩn.

### Cách 2: Chạy đánh giá tự động bằng lệnh (Tùy chọn)
Nếu muốn xuất toàn bộ báo cáo chi tiết và biểu đồ mới nhất ra thư mục `outputs/`, bạn mở cửa sổ dòng lệnh và gõ:
```bash
python main.py --eval
```
Chương trình sẽ tự động kiểm tra so sánh 3 mô hình học máy khác nhau và in ra bảng tổng kết rõ ràng.

---

## 9. Các lỗi thường gặp và cách tự xử lý đơn giản

| Hiện tượng | Nguyên nhân | Cách khắc phục siêu nhanh |
| :--- | :--- | :--- |
| Gõ `python` máy báo lỗi *'python' is not recognized...* | Chưa tích chọn "Add Python to PATH" khi cài Python. | Mở lại file cài Python đã tải về, chọn **Modify** hoặc cài lại, nhớ tích dấu ☑️ vào ô **"Add python.exe to PATH"**. |
| Báo lỗi *ModuleNotFoundError: No module named 'PyQt6'...* | Chưa cài đặt các thư viện cần thiết. | Chạy lại lệnh: `pip install -r requirements.txt`. |
| Cửa sổ giao diện không hiện lên khi gõ lệnh | Đang chạy trong môi trường không có màn hình đồ họa. | Đảm bảo bạn đang mở terminal trực tiếp trên máy tính Windows cá nhân của mình. |
| Bot gợi ý quán ăn giá chưa chính xác | Quán vừa tăng/giảm giá trên thực tế. | Mở tab **"🗂️ Quản lý dữ liệu"**, tìm quán đó và sửa lại giá đúng rồi bấm "Cập nhật". |
| Bấm nút "Gửi" nhưng không thấy phản hồi | Bạn vô tình để trống ô nhập câu hỏi. | Hãy gõ vài từ khóa (ví dụ: *"cơm tấm"*, *"bún bò dưới 35k"*) rồi bấm Gửi lại nhé. |

---

> 🎓 **Lời kết**:  
> Dự án Chatbot tư vấn bữa trưa NEU được thiết kế với tiêu chí: **khoa học trong thuật toán, nhưng thân thiện và dễ dùng với mọi người**. Bạn không cần phải là một lập trình viên để sử dụng hay bổ sung dữ liệu cho bot. Chúc bạn có những bữa trưa ngon miệng và tìm được những quán ăn ưng ý nhất quanh NEU! 🍲✨
