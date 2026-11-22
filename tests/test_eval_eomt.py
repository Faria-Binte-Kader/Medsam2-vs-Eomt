import json
import sys

import numpy as np
import pytest

pytest.importorskip("lightning")
pd = pytest.importorskip("pandas")
sitk = pytest.importorskip("SimpleITK")
import torch

import config as C
import eval_eomt as V
import metrics as M
import process_data as P

H = W = 64
Z = 8
GEOM = {"size_xyz": [W, H, Z], "spacing_xyz": [0.8, 0.8, 2.0], "origin_xyz": [-30.0, 20.0, 10.0],
        "direction": [1, 0, 0, 0, -1, 0, 0, 0, 1]}


def test_select_best_takes_highest_and_earlier_step_on_ties():
    assert V.select_best({500: 0.61, 1000: 0.72, 1500: 0.70}) == 1000
    assert V.select_best({500: 0.7, 1000: 0.7}) == 500


@pytest.fixture
def run(tmp_path, monkeypatch):
    man, cache, out = tmp_path / "man", tmp_path / "cache", tmp_path / "out"
    man.mkdir()
    rows = [{"patient_id": i, "phase": 1, "phase_name": "arterial", "eval_set": "primary"} for i in (5, 6, 7)]
    pd.DataFrame(rows).to_csv(man / "inventory.csv", index=False)
    (man / "splits.json").write_text(json.dumps({"train": [], "val": [5, 6], "test": [7], "train_subsets": {}}))
    yy, xx = np.mgrid[:H, :W]
    for r in rows:
        rng = np.random.default_rng(r["patient_id"])
        ct = rng.integers(0, 255, (Z, H, W)).astype(np.uint8)
        m = np.zeros((Z, H, W), np.uint8)
        m[2:6][:, (yy - 30) ** 2 + (xx - 28) ** 2 < 80] = 1
        meta = {"preprocess_version": C.PREPROCESS_VERSION, "window_hu": list(C.HU_WINDOW), "geometry": GEOM,
                "patient_id": r["patient_id"], "phase": 1, "phase_name": "arterial", "eval_set": "primary"}
        P.save_case(P.case_dir(r, cache), ct, m, m[None], [0], meta)
    monkeypatch.setattr(C, "CACHE_DIR", cache)
    run = out / "run"
    (run / "weights").mkdir(parents=True)
    info = {"args": {"backbone": "vit_small_patch14_reg4_dinov2", "img_size": H, "n_patients": "2", "init": "dinov2"}}
    (run / "run_info.json").write_text(json.dumps(info))
    net, _ = V.build_network(info, "cpu")
    for step, seed in ((3, 0), (6, 1)):  # two differently-initialised "checkpoints" in the training callback's format
        torch.manual_seed(seed)
        fresh, _ = V.build_network(info, "cpu")
        sd = {f"network.{k}": v for k, v in fresh.state_dict().items()}
        torch.save({"state_dict": sd, "global_step": step}, run / "weights" / f"step{step}.pt")
    return man, run


def cli(monkeypatch, man, *argv):
    monkeypatch.setattr(sys, "argv", ["eval_eomt.py", "--manifest-dir", str(man), "--device", "cpu", *argv])
    try:
        V.main()
        return 0
    except SystemExit as e:
        return e.code


def test_select_then_val_eval_native_grid_and_metrics_consistent(run, monkeypatch):
    man, run = run
    assert cli(monkeypatch, man, "--run-dir", str(run), "--select") == 0
    best = json.loads((run / "best_step.json").read_text())
    assert best["best_step"] in (3, 6) and set(map(int, best["val_dice_by_step"])) == {3, 6}
    assert "validation patients only" in best["selected_on"]
    assert cli(monkeypatch, man, "--run-dir", str(run), "--split", "val", "--hd95") == 0
    out = run / "eval_val"
    vol = pd.read_csv(out / "volume_metrics.csv")
    assert list(vol.patient_id) == [5, 6] and set(vol.checkpoint) == {f"step{best['best_step']}"}
    assert {"dice", "iou", "hd95_mm", "n_train_patients", "init", "diag_false_positive_slices"} <= set(vol.columns)
    for r in vol.itertuples():  # the reported Dice equals the Dice of the saved native-grid mask against the cached GT
        saved = out / "masks" / f"p{r.patient_id}_ph1.nii.gz"
        ok, why = P.same_grid(P.header(saved), GEOM)
        assert ok, why
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(saved))) > 0
        case = P.load_case(P.case_dir({"eval_set": "primary", "patient_id": r.patient_id, "phase": 1}), keys=("mask_union",))
        assert pred.shape == (Z, H, W) and r.dice == pytest.approx(M.dice(pred, case["mask_union"] > 0))
    assert (out / "summary.csv").exists() and (out / "patient_summary.csv").exists()


def test_test_split_requires_validation_choice_and_rejects_arbitrary_weights(run, monkeypatch):
    man, run = run
    code = cli(monkeypatch, man, "--run-dir", str(run), "--split", "test")
    assert isinstance(code, str) and "--select" in code  # no validation-based choice recorded yet
    code = cli(monkeypatch, man, "--run-dir", str(run), "--split", "test", "--weights", str(run / "weights" / "step3.pt"))
    assert isinstance(code, str) and "not allowed" in code
    assert cli(monkeypatch, man, "--run-dir", str(run), "--select") == 0
    assert cli(monkeypatch, man, "--run-dir", str(run), "--split", "test") == 0
    vol = pd.read_csv(run / "eval_test" / "volume_metrics.csv")
    assert list(vol.patient_id) == [7] and set(vol.split) == {"test"}


def test_wrong_slice_size_is_refused(run):
    man, run = run
    info = json.loads((run / "run_info.json").read_text())
    net, torch_ = V.build_network(info, "cpu")
    with pytest.raises(ValueError, match="resize policy"):
        V.predict_volume(net, torch_, np.zeros((2, 48, 64), np.uint8), "cpu", size=64)
