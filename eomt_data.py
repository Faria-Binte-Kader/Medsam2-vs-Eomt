"""EoMT data module for the cached WAW-TACE slices (used by train_eomt.py through EoMT's LightningCLI).

One training sample is one axial slice: windowed uint8 CT (the same slice MedSAM2 sees), gray copied to RGB. EoMT's semantic
interface wants one binary mask per class present, so every slice gets a BACKGROUND mask (class 0) and, if it contains tumor,
a TUMOR mask (class 1). Two reasons for modelling background explicitly (found by reading the pinned EoMT code):
  * its training transforms retry cropping until some mask is non-empty, which never ends for a tumor-free slice;
  * its semantic prediction is an argmax over classes, which with a single tumor class would label every pixel tumor.

Patients come from the frozen split (manifests/splits.json): `n_patients` selects a nested TRAIN subset (1, 2, 5, 10, 20, all);
validation uses the validation patients only. Each epoch uses every tumor slice plus a seeded random sample of tumor-free
slices (`neg_per_pos` per tumor slice), redrawn every epoch. No flip and no colour jitter (gray CT, left-right anatomy).

  python eomt_data.py --n-patients 5          # print slice counts for a subset (no training)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import lightning
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data import Dataset as TorchDataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

import config as C
import process_data as P

sys.path.insert(0, str(C.EOMT_DIR))  # EoMT's own packages (datasets, training, models)
from datasets.lightning_data_module import LightningDataModule  # noqa: E402
from datasets.transforms import Transforms  # noqa: E402


def load_patients(patient_ids, inventory, cache_dir=None):
    """Windowed CT and union mask per patient from the preprocessing cache. Returns (cts, masks, ids)."""
    cts, masks, ids = [], [], []
    for pid in patient_ids:
        row = inventory[(inventory.patient_id == pid) & (inventory.eval_set == "primary")]
        if len(row) != 1:
            raise ValueError(f"patient {pid}: expected exactly one primary inventory row, found {len(row)}")
        case = P.load_case(P.case_dir(row.iloc[0], cache_dir), keys=("ct_u8", "mask_union"))
        cts.append(case["ct_u8"])
        masks.append(case["mask_union"] > 0)
        ids.append(int(pid))
    return cts, masks, ids


def slice_index(masks):
    """(patient index, slice) pairs split into tumor slices and tumor-free slices."""
    pos, neg = [], []
    for p, m in enumerate(masks):
        has = m.any((1, 2))
        for z, h in enumerate(has):
            (pos if h else neg).append((p, z))
    return pos, neg


class SliceDataset(TorchDataset):
    def __init__(self, cts, masks, index, transforms=None):
        self.cts, self.masks, self.index, self.transforms = cts, masks, index, transforms

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        p, z = self.index[i]
        gray = torch.from_numpy(np.ascontiguousarray(self.cts[p][z]))            # (H, W) uint8
        img = tv_tensors.Image(gray[None].expand(3, -1, -1).clone())             # (3, H, W) uint8
        tumor = torch.from_numpy(np.ascontiguousarray(self.masks[p][z]))         # (H, W) bool
        masks, labels = [~tumor], [0]                                            # background is always present
        if tumor.any():
            masks.append(tumor)
            labels.append(1)
        target = {"masks": tv_tensors.Mask(torch.stack(masks)), "labels": torch.tensor(labels),
                  "is_crowd": torch.zeros(len(labels), dtype=torch.bool)}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


class EpochSampler(Sampler):
    """Every tumor slice + a seeded sample of tumor-free slices, reshuffled each epoch (epoch counter advances per iteration)."""

    def __init__(self, pos, neg, neg_per_pos, seed):
        self.pos, self.neg, self.seed, self.epoch = list(pos), list(neg), seed, 0
        self.n_neg = min(len(self.neg), int(round(neg_per_pos * len(self.pos))))

    def __len__(self):
        return len(self.pos) + self.n_neg

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        self.epoch += 1
        neg = torch.randperm(len(self.neg), generator=g)[: self.n_neg].tolist()
        idx = self.pos + [self.neg[i] for i in neg]
        return iter([idx[j] for j in torch.randperm(len(idx), generator=g).tolist()])


class CTTransforms(Transforms):
    """EoMT's scale jitter + random crop, with colour jitter off and no horizontal flip (anatomy is not left-right symmetric)."""

    def __init__(self, img_size, scale_range):
        super().__init__(img_size=img_size, color_jitter_enabled=False, scale_range=scale_range)
        self.random_horizontal_flip = T.RandomHorizontalFlip(p=0.0)


def subset_patients(manifest_dir, n_patients):
    sp = json.loads((Path(manifest_dir) / "splits.json").read_text())
    key = str(n_patients)
    if key not in sp["train_subsets"]:
        raise ValueError(f"n_patients={n_patients} not in the frozen subsets {list(sp['train_subsets'])}")
    return sp["train_subsets"][key], sp["val"]


class WeightsCheckpoint(lightning.pytorch.Callback):
    """Save plain network weights (no optimizer, no pickled modules) every `every_n_steps` and at the end.

    EoMT's module stores the network object in its hyper-parameters, which makes ordinary checkpoints twice as large and
    unloadable with weights_only=True; these files are plain state dicts keyed "network.*".
    """

    def __init__(self, dirpath: str, every_n_steps: int):
        self.dirpath, self.every = Path(dirpath), int(every_n_steps)
        self.saved = set()

    def _save(self, trainer, module):
        step = int(trainer.global_step)
        if step in self.saved or step == 0:
            return
        self.saved.add(step)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        sd = {k.replace("._orig_mod", ""): v.detach().cpu() for k, v in module.state_dict().items() if k.startswith("network.")}
        torch.save({"state_dict": sd, "global_step": step}, self.dirpath / f"step{step}.pt")

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.every > 0 and trainer.global_step % self.every == 0:
            self._save(trainer, module)

    def on_train_end(self, trainer, module):
        self._save(trainer, module)


class WawTaceSlices(LightningDataModule):
    def __init__(
        self,
        n_patients: Union[int, str] = "all",
        manifest_dir: str = str(C.MANIFEST_DIR),
        cache_dir: str = str(C.CACHE_DIR),
        num_workers: int = 4,
        batch_size: int = C.EOMT_TRAIN["batch_size"],
        img_size: tuple[int, int] = C.EOMT_TRAIN["img_size"],
        num_classes: int = C.EOMT_TRAIN["num_classes"],
        neg_per_pos: float = C.EOMT_TRAIN["neg_per_pos"],
        scale_range: tuple[float, float] = C.EOMT_TRAIN["scale_range"],
        seed: int = C.SEED,
        info_path: Optional[str] = None,
    ) -> None:
        super().__init__(path=manifest_dir, batch_size=batch_size, num_workers=num_workers, num_classes=num_classes,
                         img_size=img_size, check_empty_targets=False)
        self.save_hyperparameters(ignore=["_class_path"])
        self.n_patients, self.manifest_dir, self.cache_dir = str(n_patients), manifest_dir, cache_dir
        self.neg_per_pos, self.seed, self.info_path = neg_per_pos, seed, info_path
        self.transforms = CTTransforms(img_size, scale_range)

    def setup(self, stage: Optional[str] = None):
        import pandas as pd
        inv = pd.read_csv(Path(self.manifest_dir) / "inventory.csv")
        train_ids, val_ids = subset_patients(self.manifest_dir, self.n_patients)
        cts, masks, self.train_ids = load_patients(train_ids, inv, self.cache_dir)
        pos, neg = slice_index(masks)
        self.train_dataset = SliceDataset(cts, masks, pos + neg, self.transforms)
        # the sampler indexes the concatenated list: first len(pos) entries are tumor slices
        self.train_sampler = EpochSampler(range(len(pos)), range(len(pos), len(pos) + len(neg)), self.neg_per_pos, self.seed)
        self.train_counts = {"patients": len(train_ids), "slices": len(pos) + len(neg), "tumor_slices": len(pos),
                             "tumor_free_slices": len(neg), "slices_per_epoch": len(self.train_sampler)}
        vcts, vmasks, self.val_ids = load_patients(val_ids, inv, self.cache_dir)
        vpos, vneg = slice_index(vmasks)
        g = np.random.default_rng(self.seed)  # fixed validation subset: all tumor slices + sampled tumor-free slices
        keep = g.permutation(len(vneg))[: min(len(vneg), int(round(self.neg_per_pos * len(vpos))))]
        self.val_dataset = SliceDataset(vcts, vmasks, vpos + [vneg[i] for i in sorted(keep)], None)
        self.val_counts = {"patients": len(val_ids), "tumor_slices": len(vpos), "slices": len(self.val_dataset)}
        if self.info_path:  # record exactly which patients and how many slices this run trained on
            Path(self.info_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.info_path).write_text(json.dumps({"n_patients": self.n_patients, "train_patients": self.train_ids,
                                                        "val_patients": self.val_ids, "train": self.train_counts,
                                                        "val": self.val_counts}, indent=1))
        return self

    def train_dataloader(self):
        n = len(self.train_sampler)
        return DataLoader(self.train_dataset, sampler=self.train_sampler, drop_last=n >= self.dataloader_kwargs["batch_size"],
                          collate_fn=self.train_collate, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=self.eval_collate, **self.dataloader_kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-patients", default="all")
    ap.add_argument("--manifest-dir", default=str(C.MANIFEST_DIR))
    ap.add_argument("--cache-dir", default=str(C.CACHE_DIR))
    a = ap.parse_args()
    dm = WawTaceSlices(n_patients=a.n_patients, manifest_dir=a.manifest_dir, cache_dir=a.cache_dir, num_workers=0)
    dm.setup()
    print("train", dm.train_counts)
    print("val  ", dm.val_counts)


if __name__ == "__main__":
    main()
