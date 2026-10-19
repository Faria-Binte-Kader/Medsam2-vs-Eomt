import zipfile

import process_data as P


def test_parse_real_ct_and_mask_names():
    assert P.parse_ct_name("10/10_2_scan.nii.gz") == {"patient": 10, "phase": 2, "member": "10/10_2_scan.nii.gz"}
    m = P.parse_mask_name("tumor_masks_wawtace_v1_08_05_2024/102/102_1_5_tumor_seg.nrrd")
    assert (m["patient"], m["phase"], m["instance"]) == (102, 1, 5)


def test_nonconforming_and_mismatched_names_rejected():
    assert P.parse_mask_name("tumor_masks_wawtace_v1_08_05_2024/63/63_1_0.nrrd") is None  # real stray alias
    assert P.parse_mask_name("x/10/11_1_0_tumor_seg.nrrd") is None  # folder/patient mismatch
    assert P.parse_ct_name("10/10_4_scan.nii.gz") is None  # phase outside 0-3


def test_parse_archive_names_flags_odd_members(tmp_path):
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("r/63/63_1_0_tumor_seg.nrrd", "a")
        zf.writestr("r/63/63_1_0.nrrd", "a")
    ok, odd = P.parse_archive_names(z, P.parse_mask_name)
    assert len(ok) == 1 and odd == ["r/63/63_1_0.nrrd"]
