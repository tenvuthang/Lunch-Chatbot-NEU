"""
Script đóng gói mã nguồn, dữ liệu, cấu hình, kết quả đánh giá và tài liệu dự án
thành file nén zip 'lunch_chatbot.zip' (người dùng có thể giải nén hoặc chuyển đổi sang .rar).
Tự động loại bỏ các thư mục rác như __pycache__, .pytest_cache, file tạm .pyc, v.v.
"""

import os
import sys
import zipfile
from datetime import datetime

# Đảm bảo in tiếng Việt chuẩn
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def should_exclude(rel_path: str, filename: str) -> bool:
    """Kiểm tra xem file hoặc thư mục có nằm trong danh sách loại trừ hay không."""
    normalized_path = rel_path.replace("\\", "/").lower()
    
    # Các thư mục loại trừ
    excluded_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".venv",
        "venv",
        ".idea",
        ".vscode",
        ".system_generated",
        ".gemini",
        "scratch"
    ]
    for d in excluded_dirs:
        if f"/{d}/" in f"/{normalized_path}/" or normalized_path.startswith(f"{d}/") or normalized_path == d:
            return True

    # Các phần mở rộng loại trừ
    excluded_extensions = [
        ".pyc",
        ".pyo",
        ".pyd",
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz"
    ]
    for ext in excluded_extensions:
        if filename.lower().endswith(ext):
            return True

    return False


def package_project(root_dir: str = ".", output_zip: str = "lunch_chatbot.zip") -> str:
    """Đóng gói toàn bộ dự án vào file ZIP chuẩn."""
    abs_root = os.path.abspath(root_dir)
    zip_path = os.path.join(abs_root, output_zip)

    # Nếu file zip cũ đã tồn tại, xóa để tạo mới
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("=" * 80)
    print(f"BẮT ĐẦU ĐÓNG GÓI DỰ ÁN LUNCH CHATBOT: {output_zip}")
    print("=" * 80)
    print(f"Thư mục gốc: {abs_root}\n")

    files_added = []
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for foldername, subfolders, filenames in os.walk(abs_root):
            # Lọc bỏ subfolders tại chỗ để không duyệt sâu vào
            subfolders[:] = [
                d for d in subfolders
                if not should_exclude(os.path.relpath(os.path.join(foldername, d), abs_root), d)
            ]

            for filename in filenames:
                file_full_path = os.path.join(foldername, filename)
                rel_path = os.path.relpath(file_full_path, abs_root)

                # Bỏ qua chính file zip đầu ra và các file loại trừ
                if should_exclude(rel_path, filename) or file_full_path == zip_path:
                    continue

                # Đặt prefix folder lunch_chatbot/ trong zip để khi giải nén ra thư mục gọn gàng
                arcname = os.path.join("lunch_chatbot", rel_path).replace("\\", "/")
                zipf.write(file_full_path, arcname)

                f_size = os.path.getsize(file_full_path)
                total_uncompressed_bytes += f_size
                files_added.append((arcname, f_size))

    # In danh sách các file được đóng gói
    print(f"{'STT':<4} | {'Đường dẫn file trong ZIP':<55} | {'Dung lượng'}")
    print("-" * 75)
    for idx, (arcname, f_size) in enumerate(files_added, 1):
        if f_size >= 1024 * 1024:
            size_str = f"{f_size / (1024 * 1024):.2f} MB"
        elif f_size >= 1024:
            size_str = f"{f_size / 1024:.1f} KB"
        else:
            size_str = f"{f_size} B"
        print(f"{idx:<4} | {arcname:<55} | {size_str:>10}")

    zip_size = os.path.getsize(zip_path)
    zip_size_mb = zip_size / (1024 * 1024)
    ratio = (1 - zip_size / total_uncompressed_bytes) * 100 if total_uncompressed_bytes > 0 else 0

    print("-" * 75)
    print(f"Tổng số files đã đóng gói : {len(files_added)} files")
    print(f"Dung lượng chưa nén        : {total_uncompressed_bytes / (1024 * 1024):.2f} MB")
    print(f"Dung lượng file ZIP        : {zip_size_mb:.2f} MB ({zip_size:,d} bytes)")
    print(f"Tỷ lệ nén giảm được        : {ratio:.1f}%")
    print("=" * 80)
    print(f"✅ ĐÃ TẠO FILE NÉN THÀNH CÔNG TẠI: {zip_path}")
    print("👉 Bạn có thể đổi tên, gửi đi hoặc dùng WinRAR nén lại thành file .rar bất cứ lúc nào.")
    print("=" * 80 + "\n")

    return zip_path


if __name__ == "__main__":
    current_proj_dir = os.path.dirname(os.path.abspath(__file__))
    package_project(current_proj_dir, "lunch_chatbot.zip")
