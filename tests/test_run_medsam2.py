"""Prompt logic and runner plumbing. The StubPredictor below exists ONLY to test orchestration; its outputs are never results."""
import json
import sys

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
sitk = pytest.importorskip("SimpleITK")
torch = pytest.importorskip("torch")

import config as C
import process_data as P
import run_medsam2 as R

GEOM = {"size_xyz": [64, 48, 30], "spacing_xyz": [0.8, 0.8, 2.0], "origin_xyz": [-30.0, 20.0, 10.0],
        "direction": [1, 0, 0, 0, -1, 0, 0, 0, 1]}  # H=48, W=64, Z=30


def blob(z0, z1, y0, y1, x0, x1, shape=(30, 48, 64)):
    m = np.zeros(shape, bool)
    m[z0:z1, y0:y1, x0:x1] = True
    return m


# ---------------------------------------------------------------- prompts
def test_key_slice_and_box_exact_and_border_touching():
    m = blob(5, 9, 10, 20, 30, 40)
    m[7, 10:25, 30:45] = True  # slice 7 has the largest area
    z, box, area = R.key_slice_and_box(m)
    assert (z, box, area) == (7, [30, 10, 44, 24], 15 * 15)
    edge = blob(0, 2, 0, 5, 59, 64)  # touches the top/left/right image borders
    z, box, area = R.key_slice_and_box(edge)
    assert (z, box) == (0, [59, 0, 63, 4])  # first slice on ties
    with pytest.raises(ValueError):
        R.key_slice_and_box(np.zeros((3, 4, 4), bool))


def test_lesions_from_instances_split_components_and_cover_union():
    inst = np.zeros((2, 30, 48, 64), np.uint8)
    inst[0, 2:6, 5:10, 5:10] = 1
    inst[0, 20:24, 30:40, 40:50] = 1  # second blob inside the same instance
    inst[1, 10:14, 20:25, 20:25] = 1
    les = R.lesions_from_instances(inst, [0, 3])
    assert [l["lesion_id"] for l in les] == ["i0c0", "i0c1", "i3c0"]
    assert les[0]["mask"].sum() > les[1]["mask"].sum()  # largest component first
    assert (np.logical_or.reduce([l["mask"] for l in les]) == (inst.max(0) > 0)).all()


def test_perturb_reproducible_clipped_and_logged():
    box, H, W = [2, 3, 20, 25], 48, 64
    prm = C.PROMPT_CONDITIONS["strong"]
    a = R.perturb_box(box, prm, R.make_rng(42, 1, 1, 0, 2), H, W)
    b = R.perturb_box(box, prm, R.make_rng(42, 1, 1, 0, 2), H, W)
    c = R.perturb_box(box, prm, R.make_rng(42, 1, 1, 0, 3), H, W)
    assert a == b and a != c
    assert a["box"][0] >= 0 and a["box"][1] >= 0 and a["box"][2] <= W - 1 and a["box"][3] <= H - 1
    assert a["clipped"] and a["box_before_clip"] != a["box"]  # near the border, so clipping must have happened
    ex = R.perturb_box(box, C.PROMPT_CONDITIONS["exact"], R.make_rng(1), H, W)
    assert ex["box"] == box and not ex["clipped"] and ex["valid"]


def test_perturbation_magnitudes_within_declared_bounds():
    box = [20, 20, 40, 40]
    prm = {"pad": 5, "translate": 3, "jitter": 2}
    for s in range(200):
        r = R.perturb_box(box, prm, R.make_rng(7, s), 512, 512)
        for k, (v, e) in enumerate(zip(r["box_before_clip"], box)):
            lo = (-5 if k < 2 else 5) - 3 - 2
            hi = (-5 if k < 2 else 5) + 3 + 2
            assert lo <= v - e <= hi
        assert abs(r["translate_xy"][0]) <= 3 and all(abs(j) <= 2 for j in r["jitter"])


def test_invalid_prompt_is_recorded_not_repaired():
    r = R.perturb_box([60, 10, 63, 12], {"pad": 0, "translate": 30, "jitter": 0}, R.make_rng(0), 48, 64)
    tiny = R.perturb_box([5, 5, 6, 6], {"pad": -3, "translate": 0, "jitter": 0}, R.make_rng(0), 48, 64)
    assert tiny["valid"] is False and tiny["reason"]
    assert r["valid"] in (True, False)  # may or may not be valid, but a log entry always exists


def test_build_prompts_independent_of_condition_selection():
    inst = np.zeros((1, 30, 48, 64), np.uint8); inst[0, 5:9, 10:20, 30:40] = 1
    les = R.lesions_from_instances(inst, [0])
    allc = R.build_prompts(les, 5, 1, C.PROMPT_CONDITIONS, 48, 64)
    only = R.build_prompts(les, 5, 1, {"strong": C.PROMPT_CONDITIONS["strong"]}, 48, 64)
    assert [p for p in allc if p["condition"] == "strong"][0]["final_box_xyxy"] == only[0]["final_box_xyxy"]
    assert allc[0]["units"] == C.PROMPT["box_units"] and allc[0]["seed_entropy"][0] == C.SEED


def test_box_iou_known():
    assert R.box_iou([0, 0, 9, 9], [0, 0, 9, 9]) == 1.0
    assert R.box_iou([0, 0, 9, 9], [5, 0, 14, 9]) == pytest.approx(50 / 150)
    assert R.box_iou([0, 0, 4, 4], [10, 10, 14, 14]) == 0.0


def test_largest_cc_keeps_only_biggest_blob():
    s = blob(0, 3, 0, 3, 0, 3) | blob(10, 12, 10, 12, 10, 12)
    out = R.largest_cc(s)
    assert out.sum() == 27 and out[0, 0, 0] and not out[10, 10, 10]
    assert not R.largest_cc(np.zeros((3, 3, 3), bool)).any()


# ---------------------------------------------------------------- runner plumbing with a STUB predictor
class StubPredictor:
    """Pretend model: segments the prompted box on frames within +-2 slices of the key frame, in the asked direction."""
    def init_state(self, images, H, W):
        return {"n": images.shape[0], "H": H, "W": W, "box": None, "key": None, "calls": []}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        inference_state["box"], inference_state["key"] = [int(v) for v in box], frame_idx

    def reset_state(self, st):
        st["box"] = None

    def propagate_in_video(self, st, reverse=False, max_frame_num_to_track=None):
        st["calls"].append(reverse)
        k = st["key"]
        frames = range(k, max(k - 3, -1), -1) if reverse else range(k, min(k + 3, st["n"]))
        for f in frames:
            m = torch.full((1, 1, st["H"], st["W"]), -5.0)
            x0, y0, x1, y1 = st["box"]
            m[0, 0, y0:y1 + 1, x0:x1 + 1] = 5.0
            yield f, [1], m


def make_case(tmp, patient=5, phase=1, lesions=((10, 15), (20, 24))):
    shape = (30, 48, 64)
    inst = np.zeros((len(lesions),) + shape, np.uint8)
    for i, (a, b) in enumerate(lesions):
        inst[i, a:b, 8 + 12 * i:18 + 12 * i, 10 + 20 * i:25 + 20 * i] = 1
    union = (inst.max(0) > 0).astype(np.uint8)
    meta = {"preprocess_version": C.PREPROCESS_VERSION, "window_hu": list(C.HU_WINDOW), "patient_id": patient, "phase": phase,
            "phase_name": "arterial", "eval_set": "primary", "geometry": GEOM}
    ct = np.random.default_rng(0).integers(0, 255, shape).astype(np.uint8)
    return {"ct_u8": ct, "mask_union": union, "instances": inst, "instance_ids": list(range(len(lesions))), "meta": meta}


def test_evaluate_case_with_stub(tmp_path):
    case = make_case(tmp_path)
    row = {"eval_set": "primary"}
    conds = {"exact": C.PROMPT_CONDITIONS["exact"]}
    vrows, lrows, prompts = R.evaluate_case(case, row, "val", conds, StubPredictor(), "cpu", tmp_path, hd95=True)
    v = vrows[0]
    # each lesion cross-section is 10x15 = 150 px. Lesion 0: z 10..14, key = 10 (first max); the stub covers key-2..key+2
    # = 8..12, so 3 slices overlap. Lesion 1: z 20..23, key 20, covered 18..22, 3 slices overlap.
    inter, pred_vox, gt_vox = 6 * 150, (5 + 5) * 150, (5 + 4) * 150
    assert v["dice"] == pytest.approx(2 * inter / (pred_vox + gt_vox))
    assert v["n_lesions"] == 2 and v["n_invalid_prompts"] == 0 and v["size_bin"] == "<10"
    assert len(lrows) == 2 and all(r["dice"] > 0 for r in lrows) and np.isfinite(v["hd95_mm"])
    # saved mask is on the CT grid and equals the stub prediction
    ok, why = P.same_grid(P.header(v["mask_path"]), GEOM)
    assert ok, why
    saved = sitk.GetArrayFromImage(sitk.ReadImage(v["mask_path"]))
    assert saved.shape == (30, 48, 64) and saved[8:13, 8:18, 10:25].all() and not saved[13:15].any() and not saved[:8].any()
    assert (tmp_path / "overlays" / "p5_ph1_exact.png").exists()
    assert prompts[0]["key_slice"] == 10 and prompts[0]["exact_box_xyxy"] == [10, 8, 24, 17]


def test_invalid_prompt_counts_as_missed_not_dropped(tmp_path, monkeypatch):
    case = make_case(tmp_path)
    monkeypatch.setitem(C.PROMPT_CONDITIONS, "bad", {"pad": -50, "translate": 0, "jitter": 0})
    vrows, lrows, _ = R.evaluate_case(case, {"eval_set": "primary"}, "val", {"bad": C.PROMPT_CONDITIONS["bad"]},
                                      StubPredictor(), "cpu", tmp_path, overlays=False)
    assert vrows[0]["n_invalid_prompts"] == 2 and vrows[0]["dice"] == 0.0 and vrows[0]["pred_volume_ml"] == 0.0
    assert [r["dice"] for r in lrows] == [0.0, 0.0]


# ---------------------------------------------------------------- CLI: dry run, full stub run, resume, test guard
@pytest.fixture
def project(tmp_path, monkeypatch):
    man, cache, out = tmp_path / "man", tmp_path / "cache", tmp_path / "out"
    man.mkdir()
    rows = []
    for pid in (11, 12, 13):
        rows.append({"patient_id": pid, "phase": 1, "phase_name": "arterial", "eval_set": "primary"})
    rows.append({"patient_id": 90, "phase": 1, "phase_name": "arterial", "eval_set": "secondary"})
    inv = pd.DataFrame(rows)
    inv.to_csv(man / "inventory.csv", index=False)
    (man / "splits.json").write_text(json.dumps({"train": [11], "val": [12], "test": [13]}))
    for r in rows:
        case = make_case(tmp_path, r["patient_id"], 1)
        case["meta"]["eval_set"] = r["eval_set"]
        P.save_case(P.case_dir(r, cache), case["ct_u8"], case["mask_union"], case["instances"], case["instance_ids"], case["meta"])
    monkeypatch.setattr(C, "CACHE_DIR", cache)
    monkeypatch.setattr(C, "OUTPUT_DIR", out)
    monkeypatch.setattr(R, "load_predictor", lambda device: StubPredictor())
    monkeypatch.setattr(R, "resolve_device", lambda d: "cpu")
    return man, out


def run_cli(monkeypatch, man, *argv):
    monkeypatch.setattr(sys, "argv", ["run_medsam2.py", "--manifest-dir", str(man), *argv])
    with pytest.raises(SystemExit) as e:
        R.main()
    return e.value.code


def test_select_cases_respects_splits(project):
    man, _ = project
    inv = pd.read_csv(man / "inventory.csv"); sp = json.loads((man / "splits.json").read_text())
    assert list(R.select_cases("val", inv, sp).patient_id) == [12]
    assert list(R.select_cases("test", inv, sp).patient_id) == [13]
    assert list(R.select_cases("secondary", inv, sp).patient_id) == [90]  # never mixed into val/test


def test_dry_run_writes_reproducible_prompts_without_a_model(project, monkeypatch):
    man, out = project
    monkeypatch.setattr(R, "load_predictor", lambda d: pytest.fail("model must not load in --dry-run"))
    assert run_cli(monkeypatch, man, "--split", "val", "--dry-run") == 0
    p1 = (out / "medsam2" / "val_unbounded_dryrun" / "prompts.jsonl").read_text()
    assert len(p1.splitlines()) == 2 * 3  # 2 lesions x 3 conditions
    import shutil; shutil.rmtree(out / "medsam2" / "val_unbounded_dryrun")
    run_cli(monkeypatch, man, "--split", "val", "--dry-run")
    assert (out / "medsam2" / "val_unbounded_dryrun" / "prompts.jsonl").read_text() == p1
    snap = json.loads((out / "medsam2" / "val_unbounded_dryrun" / "config_snapshot.json").read_text())
    assert snap["oracle_localization"] is True and snap["medsam2"]["checkpoint"] == "MedSAM2_latest.pt"


def test_full_stub_run_resume_and_outputs(project, monkeypatch):
    man, out = project
    assert run_cli(monkeypatch, man, "--split", "val", "--no-overlays") == 0
    d = out / "medsam2" / "val_unbounded"
    vol = pd.read_csv(d / "volume_metrics.csv")
    assert len(vol) == 3 and set(vol.condition) == {"exact", "mild", "strong"} and (vol.split == "val").all()
    for f in ("summary.csv", "summary_by_size.csv", "patient_summary.csv", "lesion_metrics.jsonl", "prompts.jsonl", "config_snapshot.json"):
        assert (d / f).exists(), f
    before = (d / "volume_metrics.csv").read_text()
    run_cli(monkeypatch, man, "--split", "val", "--no-overlays")  # resume: everything already done
    assert (d / "volume_metrics.csv").read_text() == before


def test_test_split_is_guarded_against_config_changes(project, monkeypatch):
    man, out = project
    reg = out / "medsam2" / "test_registry.json"
    # debug runs on the test split do not register a configuration
    assert run_cli(monkeypatch, man, "--split", "test", "--limit", "1", "--no-overlays", "--run-name", "dbg") == 0
    assert not reg.exists()
    assert run_cli(monkeypatch, man, "--split", "test", "--no-overlays") == 0
    assert len(json.loads(reg.read_text())) == 1
    monkeypatch.setattr(C, "HU_WINDOW", (-100.0, 200.0))  # a change made after seeing test results
    code = run_cli(monkeypatch, man, "--split", "test", "--no-overlays", "--run-name", "test_again")
    assert isinstance(code, str) and "already evaluated" in code  # refused with an explanation
    assert not (out / "medsam2" / "test_again" / "volume_metrics.csv").exists()


def test_checkpoint_missing_or_corrupt_is_refused_with_no_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MODELS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="not found"):
        R.verify_checkpoint()
    (tmp_path / C.MEDSAM2["checkpoint"]).write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit, match="pinned size/SHA-256"):
        R.verify_checkpoint()
    good = tmp_path / "good.pt"
    good.write_bytes(b"abc")
    import hashlib
    monkeypatch.setitem(C.MEDSAM2, "checkpoint", "good.pt")
    monkeypatch.setitem(C.MEDSAM2, "size_bytes", 3)
    monkeypatch.setitem(C.MEDSAM2, "sha256", hashlib.sha256(b"abc").hexdigest())
    assert R.verify_checkpoint() == good


class LeakyStub(StubPredictor):
    """Stub that, like an over-propagating model, segments the box on EVERY frame it is asked to track."""
    def propagate_in_video(self, st, reverse=False, max_frame_num_to_track=None):
        st["calls"].append(reverse)
        k, n = st["key"], st["n"]
        mx = n if max_frame_num_to_track is None else max_frame_num_to_track
        frames = range(k, max(k - mx, 0) - 1, -1) if reverse else range(k, min(k + mx, n - 1) + 1)
        for f in frames:
            m = torch.full((1, 1, st["H"], st["W"]), -5.0)
            x0, y0, x1, y1 = st["box"]
            m[0, 0, y0:y1 + 1, x0:x1 + 1] = 5.0
            yield f, [1], m


def test_z_modes_unbounded_leaks_gt_extent_bounds_the_propagation(tmp_path):
    case = make_case(tmp_path, lesions=((10, 15),))  # one cuboid lesion, z 10..14, same cross-section on every slice
    conds = {"exact": C.PROMPT_CONDITIONS["exact"]}
    out = {}
    for mode in ("unbounded", "gt_extent"):
        v, _, pr = R.evaluate_case(case, {"eval_set": "primary"}, "val", conds, LeakyStub(), "cpu", tmp_path / mode,
                                   overlays=False, z_mode=mode)
        out[mode] = (v[0], pr[0])
    u, b = out["unbounded"][0], out["gt_extent"][0]
    assert u["diag_pred_ml_outside_gt_z"] > 0 and u["dice"] < 0.5          # leaks over all 30 slices
    assert b["diag_pred_ml_outside_gt_z"] == 0 and b["dice"] == pytest.approx(1.0)
    assert u["diag_dice_within_gt_z"] == pytest.approx(1.0) and u["diag_gt_z_range"] == "10-14"
    pu, pb = out["unbounded"][1], out["gt_extent"][1]
    assert pu["propagate_forward_max"] is None and pu["gt_z_range"] == [10, 14] and pu["z_mode"] == "unbounded"
    assert (pb["propagate_forward_max"], pb["propagate_reverse_max"]) == (4, 0)  # key slice is 10 (first max)
    # prompts (box, key slice) are identical across z modes: only the propagation range differs
    assert pu["final_box_xyxy"] == pb["final_box_xyxy"] and pu["key_slice"] == pb["key_slice"]


def test_fingerprint_is_independent_of_chosen_z_mode(monkeypatch):
    a = R.config_fingerprint()
    monkeypatch.setitem(C.PROMPT, "z_extent", "gt_extent")
    assert R.config_fingerprint() == a
    monkeypatch.setattr(C, "Z_MODES", ("unbounded",))
    assert R.config_fingerprint() != a


def test_cli_z_mode_separates_run_dirs_and_is_recorded(project, monkeypatch):
    man, out = project
    assert run_cli(monkeypatch, man, "--split", "val", "--z-mode", "gt_extent", "--no-overlays") == 0
    d = out / "medsam2" / "val_gt_extent"
    snap = json.loads((d / "config_snapshot.json").read_text())
    assert snap["z_mode"] == "gt_extent" and "sam2_cuda_extension" in snap["versions"]
    vol = pd.read_csv(d / "volume_metrics.csv")
    assert set(vol.z_mode) == {"gt_extent"} and "diag_pred_ml_outside_gt_z" in vol.columns
    assert not (out / "medsam2" / "val_unbounded").exists()
