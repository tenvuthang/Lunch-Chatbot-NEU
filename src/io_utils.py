"""
Module tiện ích tập trung xử lý đọc/ghi các file CSV và dữ liệu dạng bảng.
Đảm bảo thống nhất:
1. Mọi thao tác đọc CSV đều dùng encoding='utf-8-sig' (tự động nhận diện và bóc tách
   BOM nếu có, đồng thời đọc mượt mà các file UTF-8 chuẩn không có BOM).
2. Mọi thao tác ghi CSV đều dùng encoding='utf-8-sig', lineterminator='\\n' (để Microsoft
   Excel trên Windows hiển thị đúng 100% tiếng Việt có dấu khi mở trực tiếp).
3. Hỗ trợ cơ chế ghi nguyên tử (atomic write) qua file tạm để tránh hỏng dữ liệu.
"""

import csv
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

CSV_ENCODING = "utf-8-sig"
CSV_LINETERMINATOR = "\n"


# ══════════════════════════════════════════════════════════════════════════════
# ĐỌC CSV
# ══════════════════════════════════════════════════════════════════════════════
def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    """
    Đọc file CSV thành danh sách các dictionary (mỗi dòng một dict).
    Tự động chuẩn hóa tên cột (strip khoảng trắng) và giá trị.
    Dùng encoding='utf-8-sig' để an toàn với cả file có BOM lẫn không BOM.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {path}")

    with open(path, "r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        # Bóc tách tên trường đã strip
        raw_fieldnames = reader.fieldnames or []
        clean_fieldnames = [col.strip() if col else "" for col in raw_fieldnames]

        results: List[Dict[str, str]] = []
        for row in reader:
            clean_row: Dict[str, str] = {}
            for orig_col, clean_col in zip(raw_fieldnames, clean_fieldnames):
                val = row.get(orig_col, "")
                clean_row[clean_col] = val.strip() if val is not None else ""
            results.append(clean_row)
        return results


def read_csv_rows(path: str) -> List[List[str]]:
    """
    Đọc file CSV thành danh sách các dòng (mỗi dòng là danh sách chuỗi).
    Dùng encoding='utf-8-sig'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {path}")

    with open(path, "r", encoding=CSV_ENCODING, newline="") as f:
        reader = csv.reader(f)
        return [[cell.strip() for cell in row] for row in reader]


def read_csv_dataframe(path: str, **kwargs: Any) -> pd.DataFrame:
    """
    Đọc file CSV bằng pandas với encoding='utf-8-sig' mặc định.
    Cho phép truyền thêm các tham số tùy chọn của pd.read_csv.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file CSV: {path}")

    kwargs.setdefault("encoding", CSV_ENCODING)
    return pd.read_csv(path, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# GHI CSV
# ══════════════════════════════════════════════════════════════════════════════
def write_csv_rows(
    path: str,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
    atomic: bool = True,
    lineterminator: str = CSV_LINETERMINATOR,
) -> None:
    """
    Ghi dữ liệu dạng bảng (header + danh sách hàng) vào file CSV với chuẩn
    UTF-8 có BOM (utf-8-sig) và ký tự xuống dòng '\\n'.
    Nếu atomic=True: ghi ra file tạm cùng thư mục rồi đổi tên để đảm bảo an toàn.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    def _do_write(file_obj: Any) -> None:
        writer = csv.writer(file_obj, lineterminator=lineterminator)
        writer.writerow(header)
        writer.writerows(rows)

    if atomic:
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding=CSV_ENCODING, newline="") as f:
                _do_write(f)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
    else:
        with open(path, "w", encoding=CSV_ENCODING, newline="") as f:
            _do_write(f)


def append_csv_row(
    path: str,
    header: Sequence[str],
    row: Sequence[Any],
    lineterminator: str = CSV_LINETERMINATOR,
) -> None:
    """
    Nối thêm một dòng vào cuối file CSV (tạo file kèm header nếu chưa có).
    File mới được tạo với UTF-8 BOM (utf-8-sig); khi nối vào file đã có nội dung thì KHÔNG ghi
    thêm BOM (tránh BOM lạc giữa file làm hỏng dữ liệu), vẫn dùng '\\n' xuống dòng.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    is_new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    encoding = CSV_ENCODING if is_new else "utf-8"
    with open(path, "a", encoding=encoding, newline="") as f:
        writer = csv.writer(f, lineterminator=lineterminator)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def write_csv_dicts(
    path: str,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, Any]],
    atomic: bool = True,
    lineterminator: str = CSV_LINETERMINATOR,
) -> None:
    """
    Ghi danh sách dictionary vào CSV với UTF-8 BOM (utf-8-sig) và xuống dòng '\\n'.
    """
    extracted_rows = [[row.get(col, "") for col in fieldnames] for row in rows]
    write_csv_rows(path, header=fieldnames, rows=extracted_rows, atomic=atomic, lineterminator=lineterminator)


def write_csv_dataframe(
    df: pd.DataFrame,
    path: str,
    index: bool = False,
    lineterminator: str = CSV_LINETERMINATOR,
    **kwargs: Any,
) -> None:
    """
    Ghi DataFrame pandas ra file CSV với UTF-8 BOM (utf-8-sig) và xuống dòng '\\n'.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    kwargs.setdefault("encoding", CSV_ENCODING)
    kwargs.setdefault("lineterminator", lineterminator)
    kwargs.setdefault("index", index)
    df.to_csv(path, **kwargs)
