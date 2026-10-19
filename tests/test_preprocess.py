import json
import zipfile

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("nibabel")
pd = pytest.importorskip("pandas")
from PIL import Image

import config as C
import process_data as P

FLIPY = (1, 0, 0, 0, -1, 0, 0, 0, 1)


def test_window_known_values_and_upstream_equivalence():
    lo, hi = -140.0, 260.0
    x = np.array([-3024, -140, -141, 60, 259, 260, 3000])
    assert P.window_to_uint8(x, lo, hi).tolist() == [0, 0, 0, 127, 254, 255, 255]  # 127.5 truncates like np.uint8
    vol = np.random.default_rng(0).integers(-1000, 600, (4, 8, 8))  # spans the window, like real CT
    pre = np.clip(vol, lo, hi)
    upstream = np.uint8((pre - np.min(pre)) / (np.max(pre) - np.min(pre)) * 255.0)
    assert (P.window_to_uint8(vol, lo, hi) == upstream).all()


def upstream_input(array, size):  # copy of the upstream resize + normalisation logic
    d = array.shape[0]
    res = np.zeros((d, 3, size, size))
    for i in range(d):
        res[i] = np.array(Image.fromarray(array[i].astype(np.uint8)).convert("RGB").resize((size, size))).transpose(2, 0, 1)
    res = res / 255.0
    mean = np.array((0.485, 0.456, 0.406))[None, :, None, None]
    std = np.array((0.229, 0.224, 0.225))[None, :, None, None]
    return (res - mean) / std


@pytest.mark.parametrize("hw", [(32, 48), (512, 512)])
def test_model_input_matches_upstream_logic(hw):
    a = np.random.default_rng(1).integers(0, 256, (3, *hw)).astype(np.uint8)
    got = P.to_medsam2_input(a, 512)
    assert got.shape == (3, 3, 512, 512) and got.dtype == np.float32
    assert np.allclose(got, upstream_input(a, 512), atol=1e-5)


GEOM = {"size_xyz": [64, 64, 5], "spacing_xyz": [0.7, 0.7, 2.5], "origin_xyz": [-50.0, 40.0, 100.0],
        "direction": [1, 0, 0, 0, -1, 0, 0, 0, 1]}


def test_restore_identity_at_native_size_and_nearest_roundtrip():
    m = np.zeros((5, 64, 64), np.uint8); m[1:4, 10:30, 20:40] = 1
    assert (P.restore_to_native(m, GEOM) == m).all()  # 512-style case: no resize happens
    up = np.stack([np.array(Image.fromarray(s).resize((512, 512), Image.NEAREST)) for s in m])  # model resolution
    back = P.restore_to_native(up, GEOM)
    assert back.shape == m.shape and (back == m).all() and set(np.unique(back)) <= {0, 1}
    with pytest.raises(ValueError):
        P.restore_to_native(m[:3], GEOM)


def test_write_native_mask_keeps_ct_geometry(tmp_path):
    ct = sitk.GetImageFromArray(np.zeros((5, 64, 64), np.int16))
    ct.SetSpacing(GEOM["spacing_xyz"]); ct.SetOrigin(GEOM["origin_xyz"]); ct.SetDirection(GEOM["direction"])
    sitk.WriteImage(ct, str(tmp_path / "ct.nii.gz"))
    m = np.zeros((5, 64, 64), np.uint8); m[2, 5:9, 7:12] = 1
    for ext in (".nii.gz", ".nrrd"):
        P.write_native_mask(m, GEOM, tmp_path / f"pred{ext}")
        ok, why = P.same_grid(P.header(tmp_path / f"pred{ext}"), P.header(tmp_path / "ct.nii.gz"))
        assert ok, why
        assert (sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / f"pred{ext}"))) == m).all()


def test_audit_flags_wrong_offset_and_passes_plausible_hu():
    rng = np.random.default_rng(0)
    hu = np.full((6, 32, 32), -1000.0, np.float32)
    hu[:, 4:28, 4:28] = rng.normal(60, 15, (6, 24, 24))
    tumor = np.zeros((6, 32, 32), np.uint8); tumor[2:4, 10:18, 10:18] = 1
    hu[tumor > 0] = 90
    ok = P.audit_hu(np.rint(hu), tumor)
    assert ok["hu_plausible"] and 85 < ok["tumor_median"] < 95 and ok["tumor_frac_outside_window"] == 0
    shifted = P.audit_hu(np.rint(hu) + 1024, tumor)  # a missing -1024 offset must be caught
    assert not shifted["hu_plausible"] and not shifted["air_ok"] and not shifted["tumor_ok"]
    assert not P.audit_hu(hu + 0.5, tumor)["integer_valued"]


def test_hist_percentiles_match_numpy():
    a = np.random.default_rng(0).integers(-3024, 1500, (4, 20, 20)).astype(np.int32)
    got = P.hist_percentiles(a, [0.5, 50, 99.5])
    ref = np.percentile(a, [0.5, 50, 99.5], method="inverted_cdf")
    assert np.allclose(got, ref)


@pytest.fixture
def realistic(tmp_path, monkeypatch):
    src, raw = tmp_path / "src", tmp_path / "raw"
    src.mkdir(); raw.mkdir()
    rng = np.random.default_rng(0)
    shape = (6, 32, 32)

    def ct(offset=0):
        hu = np.full(shape, -1000.0); hu[:, 4:28, 4:28] = rng.normal(60, 15, (6, 24, 24)); hu[1:4, 10:18, 10:18] = 90
        return (np.rint(hu) + offset).astype(np.int16)

    def write(path, arr):
        img = sitk.GetImageFromArray(arr); img.SetSpacing((0.8, 0.8, 2.0)); img.SetOrigin((-10.0, 20.0, 5.0))
        img.SetDirection(FLIPY); sitk.WriteImage(img, str(path)); return path

    m = np.zeros(shape, np.int16); m[1:4, 10:18, 10:18] = 1
    with zipfile.ZipFile(raw / "ct_scans_1_4_wawtace_09_05_24.zip", "w") as z:
        z.write(write(src / "a.nii.gz", ct()), "1/1_1_scan.nii.gz")
        z.write(write(src / "b.nii.gz", ct(1024)), "2/2_1_scan.nii.gz")  # wrong offset -> implausible
    with zipfile.ZipFile(raw / "tumor_masks_wawtace_v1_08_05_2024.zip", "w") as z:
        z.write(write(src / "m1.nrrd", m), "r/1/1_1_0_tumor_seg.nrrd")
        z.write(write(src / "m2.nrrd", m), "r/2/2_1_0_tumor_seg.nrrd")
    pd.DataFrame({"patient_id": [1, 2], "ct_phase": [1, 1], "ct_file_name": ["1_1", "2_1"], "tumor_count": [1, 1],
                  "slice_thickness": [2.0, 2.0]}).to_csv(raw / "ct_hcc_metadata_v2.csv", index=False)
    monkeypatch.setattr(C, "RAW_DIR", raw)
    monkeypatch.setattr(C, "CT_ARCHIVES", ["ct_scans_1_4_wawtace_09_05_24.zip"])
    inv, _ = P.build_manifest(raw, tmp_path / "ext", secondary={})
    return inv, tmp_path / "ext", tmp_path / "cache", m


def test_preprocess_case_end_to_end(realistic):
    inv, ext, cache, m = realistic
    row = inv[inv.patient_id == 1].iloc[0]
    arow, status = P.preprocess_case(row, ext, cache)
    assert status == "cached_new" and arow["hu_plausible"]
    case = P.load_case(P.case_dir(row, cache))
    assert case["ct_u8"].dtype == np.uint8 and case["ct_u8"].shape == (6, 32, 32)
    assert (case["mask_union"] == m).all() and case["instances"].shape == (1, 6, 32, 32)
    meta = case["meta"]
    assert meta["geometry"]["spacing_xyz"] == pytest.approx([0.8, 0.8, 2.0]) and meta["window_hu"] == list(C.HU_WINDOW)
    assert meta["source"]["ct_member"] == "1/1_1_scan.nii.gz" and meta["preprocess_version"] == C.PREPROCESS_VERSION
    assert case["ct_u8"][2, 12, 12] == P.window_to_uint8(np.array([90]))[0]
    assert P.preprocess_case(row, ext, cache)[1] == "cached"  # second call hits the cache
    # native-space restoration of a "prediction" equal to the mask reproduces the CT grid exactly
    P.write_native_mask(case["mask_union"], meta["geometry"], cache / "pred.nii.gz")
    assert P.same_grid(P.header(cache / "pred.nii.gz"), meta["geometry"])[0]


def test_implausible_hu_is_flagged_and_not_cached(realistic):
    inv, ext, cache, _ = realistic
    row = inv[inv.patient_id == 2].iloc[0]
    arow, status = P.preprocess_case(row, ext, cache)
    assert status == "implausible_hu_not_cached" and not arow["hu_plausible"]
    assert not (P.case_dir(row, cache) / "meta.json").exists()


def test_stale_cache_is_refused(realistic, monkeypatch):
    inv, ext, cache, _ = realistic
    row = inv[inv.patient_id == 1].iloc[0]
    P.preprocess_case(row, ext, cache)
    monkeypatch.setattr(C, "HU_WINDOW", (-100.0, 200.0))
    with pytest.raises(RuntimeError, match="stale cache"):
        P.load_case(P.case_dir(row, cache))
    assert P.preprocess_case(row, ext, cache)[1] == "cached_new"  # rebuilt for the new window


def test_preprocess_cli_writes_audit_and_reports_flagged(realistic, monkeypatch, capsys, tmp_path):
    import types
    inv, ext, cache, _ = realistic
    out = tmp_path / "man"; out.mkdir()
    inv.to_csv(out / "inventory.csv", index=False)
    monkeypatch.setattr(C, "EXTRACT_DIR", ext)
    monkeypatch.setattr(C, "CACHE_DIR", cache)
    P.cmd_preprocess(types.SimpleNamespace(out_dir=str(out), eval_set="all", patients=None, limit=None,
                                           force=False, keep_implausible=False))
    audit = pd.read_csv(out / "preprocess_audit.csv")
    assert dict(audit.set_index("patient_id").status) == {1: "cached_new", 2: "implausible_hu_not_cached"}
    assert "FLAGGED" in capsys.readouterr().out
    assert (cache / "primary" / "p1_ph1" / "case.npz").exists() and not (cache / "primary" / "p2_ph1").exists()
