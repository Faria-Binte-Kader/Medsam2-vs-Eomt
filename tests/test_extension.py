import pandas as pd
import pytest

import process_data as P


def _base():
    sp = P.make_splits(range(1, 21))  # 14 train / 3 val / 3 test (frozen, as in manifests/splits.json)
    return sp


def _inv(patients):
    return pd.DataFrame({"patient_id": list(patients), "eval_set": "primary"})


def test_extension_keeps_frozen_split_and_nested_subsets():
    base = _base()
    ext = P.extend_splits(base, _inv(range(1, 41)), new_sizes=[20, "all"])
    assert ext["val"] == base["val"] and ext["test"] == base["test"]
    assert ext["train"][:len(base["train"])] == base["train"]
    new = ext["extension"]["new_train_patients"]
    assert sorted(new) == list(range(21, 41)) and set(new).isdisjoint(base["val"] + base["test"])
    for k, v in base["train_subsets"].items():
        if k != "all":
            assert ext["train_subsets"][k] == v  # existing nested subsets are untouched
    old_all = str(len(base["train"]))
    assert ext["train_subsets"][old_all] == base["train_subsets"]["all"]
    assert ext["train_subsets"]["all"] == ext["train"] and len(ext["train"]) == len(base["train"]) + 20
    assert ext["counts"]["val"] == len(base["val"]) and ext["counts"]["test"] == len(base["test"])


def test_extension_is_seeded_and_deterministic_and_skips_oversize():
    base = _base()
    a = P.extend_splits(base, _inv(range(1, 31)), new_sizes=[500, "all"])
    b = P.extend_splits(base, _inv(range(1, 31)), new_sizes=[500, "all"])
    assert a == b and "500" not in a["train_subsets"] and any("skipped" in n for n in a["notes"])


def test_extension_refuses_if_a_frozen_patient_is_no_longer_primary():
    base = _base()
    with pytest.raises(ValueError):
        P.extend_splits(base, _inv(p for p in range(1, 41) if p != base["test"][0]))
