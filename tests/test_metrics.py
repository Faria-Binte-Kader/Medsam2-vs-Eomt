import numpy as np
import pytest

import metrics as M


def cube(shape, z, y, x, n):
    a = np.zeros(shape, bool)
    a[z:z + n, y:y + n, x:x + n] = True
    return a


def test_dice_iou_known_answers():
    a, b = cube((10, 10, 10), 0, 0, 0, 4), cube((10, 10, 10), 2, 0, 0, 4)  # 64 voxels each, 32 shared
    assert M.dice(a, a) == 1.0 and M.iou(a, a) == 1.0
    assert M.dice(a, b) == pytest.approx(0.5) and M.iou(a, b) == pytest.approx(32 / 96)
    assert M.dice(a, ~a & np.ones_like(a)) == 0.0


def test_empty_mask_policy():
    e, a = np.zeros((4, 4, 4), bool), cube((4, 4, 4), 0, 0, 0, 2)
    assert np.isnan(M.dice(e, e)) and np.isnan(M.iou(e, e)) and np.isnan(M.hd95(e, e, (1, 1, 1)))
    assert M.dice(e, a) == 0.0 and M.iou(a, e) == 0.0 and np.isnan(M.hd95(e, a, (1, 1, 1)))


def test_volume_ml_uses_spacing():
    a = cube((10, 10, 10), 0, 0, 0, 5)  # 125 voxels
    assert M.volume_ml(a, (0.8, 0.8, 2.0)) == pytest.approx(125 * 0.8 * 0.8 * 2.0 / 1000)


def test_hd95_known_shift_with_anisotropic_spacing():
    shape = (20, 30, 30)
    a = np.zeros(shape, bool); a[5:10, 5:15, 5:15] = True
    for axis, sp, expect in [(2, (0.5, 1.0, 3.0), 3 * 0.5), (1, (0.5, 1.0, 3.0), 3 * 1.0), (0, (0.5, 1.0, 3.0), 3 * 3.0)]:
        b = np.roll(a, 3, axis=axis)
        # identical-size boxes shifted by 3 voxels: the surface-to-surface distance profile is bounded by 3 voxels
        d = M.hd95(a, b, sp)
        assert 0 < d <= expect + 1e-9
    assert M.hd95(a, a, (1, 1, 1)) == 0.0
    # exact check on 1-voxel-thick slabs: every surface voxel is 3 voxels from the other slab along the shift axis
    s1 = np.zeros(shape, bool); s1[5, 5:15, 5:15] = True
    s2 = np.zeros(shape, bool); s2[8, 5:15, 5:15] = True
    assert M.hd95(s1, s2, (0.5, 1.0, 3.0)) == pytest.approx(9.0)  # 3 slices * 3.0 mm in z


def test_hd95_crop_does_not_change_result():
    rng = np.random.default_rng(0)
    big = np.zeros((40, 60, 60), bool)
    a = big.copy(); a[10:20, 20:35, 20:35] = True
    b = big.copy(); b[12:22, 22:40, 18:33] = True
    small = (slice(8, 24), slice(18, 42), slice(16, 38))
    assert M.hd95(a, b, (0.7, 0.7, 2.5)) == pytest.approx(M.hd95(a[small], b[small], (0.7, 0.7, 2.5)))


def test_size_bins_and_bootstrap_determinism():
    assert [M.size_bin(x) for x in (1, 9.99, 10, 49.9, 50, 900)] == ["<10", "<10", "10-50", "10-50", ">=50", ">=50"]
    v = np.random.default_rng(1).uniform(0, 1, 14)
    assert M.bootstrap_ci(v) == M.bootstrap_ci(v) and M.bootstrap_ci(v)[0] < v.mean() < M.bootstrap_ci(v)[1]
    assert np.isnan(M.bootstrap_ci([np.nan])[0])


def test_summarize_is_per_patient_then_across_patients():
    import pandas as pd
    df = pd.DataFrame({"condition": ["exact"] * 4, "patient_id": [1, 1, 2, 3], "dice": [0.2, 0.4, 0.9, 0.6], "iou": [.1, .2, .8, .4]})
    per, s = M.summarize(df)
    assert per.loc[per.patient_id == 1, "dice"].iloc[0] == pytest.approx(0.3)  # volumes averaged within patient first
    assert s.n_patients.iloc[0] == 3 and s.dice_mean.iloc[0] == pytest.approx((0.3 + 0.9 + 0.6) / 3)
