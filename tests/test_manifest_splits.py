import json
import zipfile

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
import process_data as P


def write(path, arr, spacing=(0.8, 0.8, 2.0), origin=(-100.0, 90.0, 10.0), direction=None):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if direction:
        img.SetDirection(direction)
    sitk.WriteImage(img, str(path))
    return path


SHAPE = (6, 16, 20)
FLIPY = (1, 0, 0, 0, -1, 0, 0, 0, 1)  # same orientation as the real scans


def box(z0, z1, y0, y1, x0, x1):
    m = np.zeros(SHAPE, np.int16)
    m[z0:z1, y0:y1, x0:x1] = 1
    return m


@pytest.fixture
def archives(tmp_path):
    src, raw = tmp_path / "src", tmp_path / "raw"
    src.mkdir(); raw.mkdir()
    ct = np.random.default_rng(0).integers(-500, 500, SHAPE).astype(np.int16)
    with zipfile.ZipFile(raw / "ct_scans_1_4_wawtace_09_05_24.zip", "w") as zc:
        for p, ph, org in [(1, 1, (-100.0, 90.0, 10.0)), (1, 2, (-101.0, 90.0, 10.0)), (2, 2, (-100.0, 90.0, 10.0)),
                           (3, 1, (-100.0, 90.0, 10.0))]:
            f = write(src / f"{p}_{ph}.nii.gz", ct, origin=org, direction=FLIPY)
            zc.write(f, f"{p}/{p}_{ph}_scan.nii.gz")
    root = "tumor_masks_wawtace_v1_08_05_2024"
    with zipfile.ZipFile(raw / "tumor_masks_wawtace_v1_08_05_2024.zip", "w") as zm:
        def add(name, arr, **kw):
            f = write(src / name.replace("/", "_"), arr, direction=FLIPY, **kw)
            zm.write(f, f"{root}/{name}")
        add("1/1_1_0_tumor_seg.nrrd", box(1, 3, 2, 5, 2, 6))
        add("1/1_1_1_tumor_seg.nrrd", box(2, 5, 4, 8, 5, 9))             # overlaps instance 0
        add("1/1_1_2_tumor_seg.nrrd", box(1, 3, 2, 5, 2, 6))             # byte-identical duplicate of 0
        add("1/1_1_3_tumor_seg.nrrd", np.zeros(SHAPE, np.int16))         # empty
        add("2/2_2_0_tumor_seg.nrrd", box(1, 3, 2, 5, 2, 6), origin=(-90.0, 90.0, 10.0))  # grid mismatch
        add("3/3_1_0_tumor_seg.nrrd", box(0, 2, 1, 3, 1, 3))
        add("3/3_1_0.nrrd", box(0, 2, 1, 3, 1, 3))                       # stray alias
        add("9/9_2_0_tumor_seg.nrrd", box(0, 2, 1, 3, 1, 3))             # CT not in archive
    pd.DataFrame({"patient_id": [1, 1, 2, 3], "ct_phase": [1, 2, 2, 1], "ct_file_name": ["1_1", "1_2", "2_2", "3_1"],
                  "tumor_count": [4, 0, 1, 1], "slice_thickness": [2.0, 2.0, 2.0, 2.0]}).to_csv(
        raw / "ct_hcc_metadata_v2.csv", index=False)
    return raw, tmp_path / "ext"


def test_manifest_end_to_end(archives):
    raw, ext = archives
    inv, exc = P.build_manifest(raw, ext)
    r = {(x.patient_id, x.phase): x for x in inv.itertuples()}
    assert len(inv) == 4
    a = r[(1, 1)]
    assert a.usable and a.n_instances_used == 2 and a.n_mask_files == 4
    expected_union = int((box(1, 3, 2, 5, 2, 6) | box(2, 5, 4, 8, 5, 9)).sum())
    assert a.union_voxels == expected_union and a.instance_overlap_voxels > 0
    assert not r[(1, 2)].usable and "NOT assumed tumor-negative" in r[(1, 2)].reason  # unannotated phase
    assert not r[(2, 2)].usable and r[(2, 2)].reason.startswith("mask_grid_mismatch")
    assert r[(3, 1)].usable
    reasons = set(exc.reason)
    assert {"duplicate_instance", "empty_instance", "mask_grid_mismatch", "nonconforming_mask_name",
            "ct_not_in_downloaded_archive"} <= reasons
    alias = exc[exc.reason == "nonconforming_mask_name"].iloc[0]
    assert "identical bytes" in alias.detail
    assert abs(a.union_volume_ml - expected_union * 0.8 * 0.8 * 2.0 / 1000) < 1e-6


def test_zip_header_reader_agrees_with_simpleitk(archives):
    raw, ext = archives
    assert P.cross_check_ct_header(raw / "ct_scans_1_4_wawtace_09_05_24.zip", "1/1_1_scan.nii.gz", ext)


def test_union_mask_and_overlap():
    u, ov = P.union_mask([box(0, 2, 0, 2, 0, 2), box(1, 3, 0, 2, 0, 2)])
    assert u.sum() == 12 and ov == 4 and u.dtype == np.uint8


def test_splits_disjoint_deterministic_nested():
    pats = list(range(100, 156))  # 56 patients, like archive 1/4
    s = P.make_splits(pats, seed=7)
    assert s["counts"] == {"train": 40, "val": 8, "test": 8}
    assert set(s["train"]).isdisjoint(s["val"]) and set(s["train"]).isdisjoint(s["test"]) and set(s["val"]).isdisjoint(s["test"])
    assert sorted(s["train"] + s["val"] + s["test"]) == pats
    assert P.make_splits(pats, seed=7) == s and P.make_splits(pats, seed=8)["test"] != s["test"]
    sub = s["train_subsets"]
    assert list(sub) == ["1", "2", "5", "10", "20", "all"] and sub["all"] == s["train"]
    for a, b in zip(["1", "2", "5", "10", "20"], ["2", "5", "10", "20", "all"]):
        assert sub[a] == sub[b][:len(sub[a])]  # nested
    assert set(sub["all"]) <= set(s["train"])


def test_splits_small_cohort_reduces_subsets_and_documents():
    s = P.make_splits(range(10), seed=1)
    assert "20" not in s["train_subsets"] and any("skipped" in n for n in s["notes"])
