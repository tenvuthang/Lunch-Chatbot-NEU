"""
Kiểm thử module src/io_utils.py:
1. Đọc file CSV có BOM lẫn không có BOM đều trả về đúng chuỗi tiếng Việt có dấu.
2. Ghi file CSV luôn sinh mã hóa UTF-8 BOM (utf-8-sig) và ký tự xuống dòng '\\n'.
3. Round-trip với dictionary và pandas DataFrame.
"""

import os
import pandas as pd
from src.io_utils import (
    read_csv_dicts,
    read_csv_rows,
    read_csv_dataframe,
    write_csv_rows,
    write_csv_dicts,
    write_csv_dataframe,
    CSV_ENCODING,
)


def test_write_and_read_utf8_sig_rows(tmp_path):
    p = str(tmp_path / "test_rows.csv")
    header = ["id", "tên_món", "giá"]
    rows = [["1", "Cơm bình dân", "25000"], ["2", "Bún chả", "35000"]]

    write_csv_rows(p, header, rows)

    # Kiểm tra byte đầu là BOM
    with open(p, "rb") as f:
        raw = f.read()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 3

    # Đọc lại bằng read_csv_rows và read_csv_dicts
    loaded_rows = read_csv_rows(p)
    assert loaded_rows[0] == ["id", "tên_món", "giá"]
    assert loaded_rows[1] == ["1", "Cơm bình dân", "25000"]

    loaded_dicts = read_csv_dicts(p)
    assert len(loaded_dicts) == 2
    assert loaded_dicts[0]["tên_món"] == "Cơm bình dân"
    assert loaded_dicts[1]["giá"] == "35000"


def test_read_csv_without_bom(tmp_path):
    """File UTF-8 thuần không có BOM vẫn được đọc chính xác."""
    p = str(tmp_path / "no_bom.csv")
    content = "mã,món\n1,Phở bò tái lăn\n2,Bánh mì xíu mại\n"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    loaded = read_csv_dicts(p)
    assert len(loaded) == 2
    assert loaded[0]["món"] == "Phở bò tái lăn"
    assert loaded[1]["món"] == "Bánh mì xíu mại"


def test_dataframe_io(tmp_path):
    p = str(tmp_path / "df.csv")
    df = pd.DataFrame([
        {"nhãn": "tiet_kiem", "độ_chính_xác": 0.8594},
        {"nhãn": "healthy", "độ_chính_xác": 0.8906},
    ])
    write_csv_dataframe(df, p, index=False)

    with open(p, "rb") as f:
        raw = f.read()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw

    df_loaded = read_csv_dataframe(p)
    assert list(df_loaded.columns) == ["nhãn", "độ_chính_xác"]
    assert df_loaded.iloc[0]["nhãn"] == "tiet_kiem"
    assert df_loaded.iloc[0]["độ_chính_xác"] == 0.8594
