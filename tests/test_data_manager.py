"""
Kiểm thử src/data_manager.py: kiểm tra hợp lệ và CRUD món ăn, thêm câu mẫu ý định.
Dùng bản sao dữ liệu trong thư mục tạm, không đụng tới data/ thật.
"""

import shutil
import pytest

from src import data_manager as dm


@pytest.fixture
def paths(tmp_path):
    d = tmp_path / "dishes.csv"
    i = tmp_path / "intents.csv"
    shutil.copy("data/dishes.csv", d)
    shutil.copy("data/intents.csv", i)
    return str(d), str(i)


def _dish(**over):
    base = {"ten_mon": "Món thử", "loai": "cơm", "gia_vnd": 30000, "khoang_cach_m": 200,
            "thoi_gian_phuc_vu_phut": 7, "do_no": 4, "dinh_duong": 3, "do_ngon": 4, "cay": 0, "mo_ta": "thử"}
    base.update(over)
    return base


def test_validate_dish_ok():
    assert dm.validate_dish(_dish()) == []


@pytest.mark.parametrize("field,value", [
    ("ten_mon", "  "), ("loai", ""), ("gia_vnd", 50), ("gia_vnd", "abc"), ("gia_vnd", "1500.5"),
    ("do_no", 0), ("do_ngon", 6), ("cay", 2), ("khoang_cach_m", -1), ("thoi_gian_phuc_vu_phut", 999),
])
def test_validate_dish_rejects(field, value):
    assert dm.validate_dish(_dish(**{field: value})) != []


def test_add_update_delete_dish(paths):
    dishes_path, _ = paths
    before = dm.load_dishes(dishes_path)
    added = dm.add_dish(dishes_path, _dish())
    assert added["id"] == str(len(before) + 1)
    assert len(dm.load_dishes(dishes_path)) == len(before) + 1

    dm.update_dish(dishes_path, int(added["id"]), _dish(gia_vnd=45000, cay=1))
    updated = [d for d in dm.load_dishes(dishes_path) if d["id"] == added["id"]][0]
    assert updated["gia_vnd"] == "45000" and updated["cay"] == "1"

    dm.delete_dish(dishes_path, int(added["id"]))
    assert len(dm.load_dishes(dishes_path)) == len(before)


def test_dish_duplicate_and_missing(paths):
    dishes_path, _ = paths
    existing = dm.load_dishes(dishes_path)[0]["ten_mon"]
    with pytest.raises(ValueError):
        dm.add_dish(dishes_path, _dish(ten_mon=existing.upper()))
    with pytest.raises(ValueError):
        dm.update_dish(dishes_path, 9999, _dish())
    with pytest.raises(ValueError):
        dm.delete_dish(dishes_path, 9999)
    with pytest.raises(ValueError):
        dm.add_dish(dishes_path, _dish(gia_vnd=1))


def test_recommender_reads_saved_file(paths):
    """File sau khi lưu vẫn đọc được bằng DishRecommender (đúng định dạng/cột)."""
    from src.recommender import DishRecommender
    dishes_path, _ = paths
    dm.add_dish(dishes_path, _dish(ten_mon="Món kiểm thử đặc biệt"))
    rec = DishRecommender(dishes_path=dishes_path, profiles_path="config/profiles.json")
    assert any(d["ten_mon"] == "Món kiểm thử đặc biệt" for d in rec.dishes)


def test_add_intent(paths):
    _, intents_path = paths
    counts = dm.count_intents_by_label(intents_path)
    cleaned = dm.add_intent(intents_path, "  trưa  nay ăn gì   rẻ   ", "tiet_kiem")
    assert cleaned == "trưa nay ăn gì rẻ"
    assert dm.count_intents_by_label(intents_path)["tiet_kiem"] == counts["tiet_kiem"] + 1
    with pytest.raises(ValueError):
        dm.add_intent(intents_path, "TRƯA NAY ĂN GÌ RẺ", "healthy")  # trùng
    with pytest.raises(ValueError):
        dm.add_intent(intents_path, "", "healthy")
    with pytest.raises(ValueError):
        dm.add_intent(intents_path, "câu hợp lệ nhưng nhãn sai", "khong_ton_tai")


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        dm.load_dishes(str(tmp_path / "none.csv"))
    with pytest.raises(FileNotFoundError):
        dm.load_intents(str(tmp_path / "none.csv"))
