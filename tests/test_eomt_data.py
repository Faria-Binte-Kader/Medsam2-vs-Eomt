import json

import numpy as np
import pytest

pytest.importorskip("lightning")
pytest.importorskip("torchvision")
pd = pytest.importorskip("pandas")
import torch

import config as C
import eomt_data as E
import process_data as P

H = W = 64


def make_patient(pid, n_slices=12, tumor_slices=(4, 5, 6), seed=0):
    rng = np.random.default_rng(seed + pid)
    ct = rng.integers(0, 255, (n_slices, H, W)).astype(np.uint8)
    m = np.zeros((n_slices, H, W), np.uint8)
    for z in tumor_slices:
        m[z, 20:36, 24:44] = 1
    return ct, m


@pytest.fixture
def project(tmp_path, monkeypatch):
    man, cache = tmp_path / "man", tmp_path / "cache"
    man.mkdir()
    ids = [1, 2, 3, 4, 5, 6]
    rows = [{"patient_id": i, "phase": 1, "phase_name": "arterial", "eval_set": "primary"} for i in ids]
    pd.DataFrame(rows).to_csv(man / "inventory.csv", index=False)
    (man / "splits.json").write_text(json.dumps({
        "train": [1, 2, 3, 4], "val": [5, 6], "test": [],
        "train_subsets": {"1": [1], "2": [1, 2], "all": [1, 2, 3, 4]}}))
    for r in rows:
        ct, m = make_patient(r["patient_id"], tumor_slices=(4, 5, 6) if r["patient_id"] % 2 else (3, 4))
        meta = {"preprocess_version": C.PREPROCESS_VERSION, "window_hu": list(C.HU_WINDOW)}
        P.save_case(P.case_dir(r, cache), ct, m, m[None], [0], meta)
    return man, cache


def test_slice_index_counts():
    _, m = make_patient(1)
    pos, neg = E.slice_index([m])
    assert len(pos) == 3 and len(neg) == 9 and (0, 4) in pos and (0, 0) in neg


def test_slice_targets_two_classes_background_always_present():
    ct, m = make_patient(1)
    ds = E.SliceDataset([ct], [m > 0], [(0, 4), (0, 0)])
    img, t = ds[0]  # tumor slice
    assert img.shape == (3, H, W) and img.dtype == torch.uint8 and (img[0] == img[1]).all() and (img[1] == img[2]).all()
    assert t["labels"].tolist() == [0, 1] and t["masks"].shape == (2, H, W) and t["is_crowd"].tolist() == [False, False]
    bg, tumor = t["masks"][0], t["masks"][1]
    assert (bg ^ tumor).all() and not (bg & tumor).any() and int(tumor.sum()) == 16 * 20  # exact partition of the slice
    img0, t0 = ds[1]  # tumor-free slice: background only, never empty
    assert t0["labels"].tolist() == [0] and t0["masks"].shape == (1, H, W) and t0["masks"].all()


def test_transforms_do_not_hang_on_tumor_free_slices_and_keep_masks_consistent():
    ct, m = make_patient(1)
    ds = E.SliceDataset([ct], [m > 0], [(0, 4), (0, 0)], E.CTTransforms((H, W), (0.75, 1.33)))
    for _ in range(20):
        for i in (0, 1):
            img, t = ds[i]
            assert img.shape == (3, H, W)
            stacked = t["masks"].sum(0)
            assert stacked.max() <= 1  # classes never overlap (padding after down-scaling is unlabeled)
            assert t["masks"].shape[0] == len(t["labels"]) and t["masks"].flatten(1).any(1).all()


def test_sampler_all_tumor_slices_every_epoch_and_seeded_negatives():
    pos, neg = list(range(10)), list(range(10, 100))
    s = E.EpochSampler(pos, neg, 1.0, seed=7)
    e1, e2 = list(s), list(s)
    assert len(s) == 20 and len(e1) == 20 and set(pos) <= set(e1) and set(pos) <= set(e2)
    assert set(e1) != set(e2) and sorted(e1) != e1  # negatives redrawn, order shuffled
    again = E.EpochSampler(pos, neg, 1.0, seed=7)
    assert list(again) == e1 and list(again) == e2  # same seed -> same stream of epochs
    assert len(E.EpochSampler(pos, neg[:3], 5.0, seed=0)) == 13  # capped by the available negatives


def test_datamodule_nested_subsets_and_validation_isolated(project):
    man, cache = project
    got = {}
    for n in ("1", "2", "all"):
        dm = E.WawTaceSlices(n_patients=n, manifest_dir=str(man), cache_dir=str(cache), num_workers=0, batch_size=4,
                             img_size=(H, W)).setup()
        got[n] = (dm.train_ids, dm.train_counts, dm.val_ids, dm.val_counts)
    assert got["1"][0] == [1] and got["2"][0] == [1, 2] and got["all"][0] == [1, 2, 3, 4]
    assert set(got["all"][0]).isdisjoint(got["all"][2]) and got["all"][2] == [5, 6]  # val patients never in train
    c = got["all"][1]
    assert c["patients"] == 4 and c["slices"] == 48 and c["tumor_slices"] == 3 + 2 + 3 + 2 and c["slices_per_epoch"] == 2 * 10
    assert got["all"][3]["patients"] == 2 and got["all"][3]["tumor_slices"] == 5
    with pytest.raises(ValueError):
        E.subset_patients(man, 7)  # only the frozen nested sizes are allowed


def test_extra_orderings_are_seeded_train_only_and_leave_frozen_order_alone(project):
    man, _ = project
    sp = json.loads((man / "splits.json").read_text())
    frozen, val = E.subset_patients(man, 2, 0)
    assert frozen == sp["train_subsets"]["2"]
    a, b = E.subset_patients(man, 2, 5), E.subset_patients(man, 2, 5)
    assert a == b and set(a[0]) <= set(sp["train"]) and len(set(a[0])) == 2 and a[1] == val  # deterministic, TRAIN only
    assert set(E.subset_patients(man, 1, 5)[0]) <= set(a[0])  # nested within an ordering
    assert any(E.subset_patients(man, 2, k)[0] != frozen for k in range(1, 12))  # orderings really differ from the frozen one
    assert E.subset_patients(man, "all", 5)[0] == sp["train_subsets"]["all"]


def test_dataloaders_yield_eomt_batches(project):
    man, cache = project
    dm = E.WawTaceSlices(n_patients=2, manifest_dir=str(man), cache_dir=str(cache), num_workers=0, batch_size=4,
                         img_size=(H, W)).setup()
    batches = list(dm.train_dataloader())
    assert len(batches) == len(dm.train_sampler) // 4
    imgs, targets = batches[0]
    assert imgs.shape == (4, 3, H, W) and imgs.dtype == torch.uint8 and len(targets) == 4
    assert all({"masks", "labels", "is_crowd"} <= set(t) for t in targets)
    vimgs, vtargets = next(iter(dm.val_dataloader()))  # evaluation collate: tuples of variable-size items
    # a tuple locally, a list when CUDA is present (DataLoader pin_memory converts tuples); EoMT only indexes into it
    assert isinstance(vimgs, (tuple, list)) and vimgs[0].shape == (3, H, W) and len(vtargets) == len(vimgs)
    one = E.WawTaceSlices(n_patients=1, manifest_dir=str(man), cache_dir=str(cache), num_workers=0, batch_size=64,
                          img_size=(H, W)).setup()
    assert len(list(one.train_dataloader())) == 1  # tiny subsets still yield a batch (drop_last off)
