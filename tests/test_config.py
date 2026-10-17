import config as C


def test_split_ratios_sum_to_one():
    assert abs(sum(C.SPLIT_RATIOS.values()) - 1.0) < 1e-9


def test_medsam2_pins_well_formed():
    assert len(C.MEDSAM2["commit"]) == 40
    assert len(C.MEDSAM2["sha256"]) == 64
    assert C.MEDSAM2["checkpoint"] == "MedSAM2_latest.pt"


def test_hu_window_increasing():
    assert C.HU_WINDOW[0] < C.HU_WINDOW[1]
