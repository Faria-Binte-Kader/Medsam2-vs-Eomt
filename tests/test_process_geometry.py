import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
import process_data as P


def make(path, arr, spacing=(0.7, 0.7, 2.0), origin=(-100.0, -90.0, 30.0)):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(img, str(path))


def test_header_axis_order_and_same_grid(tmp_path):
    ct = np.zeros((5, 8, 10), np.int16)  # numpy (z, y, x)
    make(tmp_path / "ct.nii.gz", ct)
    m = np.zeros((5, 8, 10), np.uint8)
    m[2, 3:5, 4:7] = 1
    make(tmp_path / "m.nrrd", m)
    hc, hm = P.header(tmp_path / "ct.nii.gz"), P.header(tmp_path / "m.nrrd")
    assert hc["size_xyz"] == [10, 8, 5]  # header is (x, y, z)
    ok, why = P.same_grid(hc, hm)
    assert ok and not why


def test_different_grid_is_reported(tmp_path):
    make(tmp_path / "a.nii.gz", np.zeros((5, 8, 10), np.int16))
    make(tmp_path / "b.nrrd", np.zeros((5, 8, 10), np.uint8), origin=(-100.0, -90.0, 40.0))
    make(tmp_path / "c.nrrd", np.zeros((4, 8, 10), np.uint8))
    a = P.header(tmp_path / "a.nii.gz")
    ok, why = P.same_grid(a, P.header(tmp_path / "b.nrrd"))
    assert not ok and any("origin" in w for w in why)
    ok, why = P.same_grid(a, P.header(tmp_path / "c.nrrd"))
    assert not ok and any("size" in w for w in why)


def test_mask_stats_values_and_bbox(tmp_path):
    m = np.zeros((5, 8, 10), np.uint8)
    m[2, 3:5, 4:7] = 1
    make(tmp_path / "m.nrrd", m)
    s = P.mask_stats(tmp_path / "m.nrrd")
    assert s["values"] == {0: 5 * 8 * 10 - 6, 1: 6}
    assert s["bbox_zyx_min_max"] == [[2, 3, 4], [2, 4, 6]]
