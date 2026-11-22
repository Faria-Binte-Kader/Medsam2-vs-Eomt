import json
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("SimpleITK")
pytest.importorskip("torch")
pytest.importorskip("matplotlib")

import config as C
import make_figures as F
from test_run_medsam2 import project, run_cli  # noqa: F401  (fixtures/helpers reused)


def test_choose_cases_rule_best_median_worst():
    d = pd.DataFrame({"patient_id": range(14), "condition": "exact", "dice": np.linspace(0.5, 0.96, 14)})
    d = pd.concat([d, d.assign(condition="mild", dice=0.1)])
    got = F.choose_cases(d)
    labels = [(l, int(r.patient_id)) for l, r in got]
    assert labels[0] == ("best", 13) and labels[2] == ("worst", 0)
    assert labels[1] == ("median", 6)  # descending order, index 14 // 2 = 7 -> patient 6 (the lower middle value)
    one = F.choose_cases(d[d.patient_id == 3])
    assert [l for l, _ in one] == ["best"]  # duplicates collapse when there are few cases
    assert F.choose_cases(d.iloc[0:0]) == []


def test_crop_window_is_inside_image_and_contains_masks():
    m = np.zeros((100, 120), bool); m[40:50, 60:75] = True
    y0, y1, x0, x1 = F.crop_window([m], m.shape)
    assert 0 <= y0 < 40 and 50 < y1 <= 100 and 0 <= x0 < 60 and 75 < x1 <= 120
    assert F.crop_window([np.zeros_like(m)], m.shape) == (0, 100, 0, 120)


def test_figures_are_written_from_a_finished_run(project, monkeypatch, tmp_path):  # noqa: F811
    man, out = project
    assert run_cli(monkeypatch, man, "--split", "val", "--no-overlays") == 0
    run = out / "medsam2" / "val_unbounded"
    figs = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--run-dir", str(run), "--out-dir", str(figs),
                                      "--manifest-dir", str(man), "--no-preprocessing"])
    F.main()
    for f in ("fig_examples.png", "fig_sensitivity.png"):
        assert (figs / f).stat().st_size > 5000, f
