"""Fine-tune EoMT for binary liver-tumor slice segmentation on a nested subset of the frozen TRAIN patients.

Builds a run directory with an exact record of the run (run_config.yaml for EoMT's LightningCLI, run_info.json) and launches
EoMT's own `main.py fit` unchanged (pinned clone in EoMT/). Run it inside the `eomt` conda env.

  python train_eomt.py --n-patients 5 --steps 3000 --dry-run     # show what would run, nothing is trained
  python train_eomt.py --n-patients 5 --steps 3000               # ADE20K-pretrained EoMT-L (default init)
  python train_eomt.py --n-patients 5 --steps 3000 --init dinov2 # DINOv2 backbone only (ablation)

Initialisation `ade20k`: the verified EoMT-L ADE20K semantic checkpoint (python fetch_eomt.py), classification head dropped and
re-created for 2 classes (background, tumor). Validation patients are used for monitoring only; the final checkpoint is chosen
afterwards by eval_eomt.py with 3D Dice on the validation patients. The test patients are never read here.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

import config as C



def annealing_schedule(total_steps, num_blocks=4):
    """EoMT's attention-mask annealing as read from its shipped configs: stage i runs from (i+1)T/6 to (i+2)T/6."""
    unit = total_steps / 6
    return ([round((i + 1) * unit) for i in range(num_blocks)], [round((i + 2) * unit) for i in range(num_blocks)])


def warmup_schedule(total_steps):
    """EoMT's defaults (500 head steps, then 1000 backbone steps) were set for ~39k steps; keep those proportions."""
    return [max(20, round(total_steps * 500 / 39120)), max(40, round(total_steps * 1000 / 39120))]


def build_config(a, out_dir):
    out = Path(out_dir)
    starts, ends = annealing_schedule(a.steps)
    val_every = a.val_every or max(1, a.steps // 8)
    ckpt_every = a.ckpt_every or max(1, a.steps // 6)
    gpu = a.device == "cuda"
    ade = a.init == "ade20k"
    cfg = {
        "seed_everything": a.seed,
        "compile_disabled": not a.compile,
        "trainer": {
            "max_steps": a.steps, "max_epochs": -1, "check_val_every_n_epoch": None, "val_check_interval": val_every,
            "default_root_dir": str(out), "accelerator": "gpu" if gpu else "cpu", "devices": 1,
            "precision": "16-mixed" if gpu else "32-true",
            "logger": {"class_path": "lightning.pytorch.loggers.wandb.WandbLogger",
                       "init_args": {"project": "eomt", "name": out.name, "save_dir": str(out), "offline": True}},
            "callbacks": [{"class_path": "eomt_data.WeightsCheckpoint",
                           "init_args": {"dirpath": str(out / "weights"), "every_n_steps": ckpt_every}}],
        },
        "model": {
            "class_path": "training.mask_classification_semantic.MaskClassificationSemantic",
            "init_args": {
                "attn_mask_annealing_enabled": True,
                "attn_mask_annealing_start_steps": starts, "attn_mask_annealing_end_steps": ends,
                "lr": a.lr, "warmup_steps": warmup_schedule(a.steps),
                "ckpt_path": str(C.MODELS_DIR / "eomt_ade20k_semantic_L512.bin") if ade else None,
                "load_ckpt_class_head": False,
                "network": {"class_path": "models.eomt.EoMT",
                            "init_args": {"num_q": 100,
                                          "encoder": {"class_path": "models.vit.ViT", "init_args": {"backbone_name": a.backbone}}}},
            },
        },
        "data": {
            "class_path": "eomt_data.WawTaceSlices",
            "init_args": {"n_patients": a.n_patients, "manifest_dir": str(a.manifest_dir), "cache_dir": str(C.CACHE_DIR),
                          "num_workers": a.num_workers, "batch_size": a.batch_size, "img_size": [a.img_size, a.img_size],
                          "neg_per_pos": a.neg_per_pos, "seed": a.seed, "info_path": str(out / "data_info.json"),
                          "order_seed": a.order_seed},
        },
    }
    return cfg


def eomt_commit():
    try:
        return subprocess.run(["git", "-C", str(C.EOMT_DIR), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def check_setup(a):
    if not (C.EOMT_DIR / "main.py").exists():
        sys.exit(f"EoMT not found at {C.EOMT_DIR}: run `python fetch_eomt.py`")
    head = eomt_commit()
    if head != C.EOMT["commit"]:
        sys.exit(f"EoMT is at {head}, expected the pinned {C.EOMT['commit']}")
    if a.init == "ade20k":
        from fetch_medsam2 import sha256
        m, path = C.EOMT["ade20k_semantic_L512"], C.MODELS_DIR / "eomt_ade20k_semantic_L512.bin"
        if not path.is_file():
            sys.exit(f"{path} missing: run `python fetch_eomt.py` (no fallback weights)")
        if path.stat().st_size != m["size_bytes"] or sha256(path) != m["sha256"]:
            sys.exit(f"{path} does not match the pinned size/SHA-256; refusing to train from unverified weights")
        if a.backbone != C.EOMT["dinov2_backbone"]["timm_name"] or a.img_size != 512:
            sys.exit("the ADE20K checkpoint is EoMT-L at 512 px: use the default --backbone and --img-size 512")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-patients", required=True, help="nested TRAIN subset key in splits.json (1, 2, 5, 10, 20, all; extension adds 68, 100, ...)")
    ap.add_argument("--manifest-dir", default=str(C.MANIFEST_DIR), help="manifests/ (frozen, default) or manifests_ext/ (extension)")
    ap.add_argument("--steps", type=int, required=True, help="total optimizer steps (same for all subset sizes in a comparison)")
    ap.add_argument("--init", choices=["ade20k", "dinov2"], default="ade20k")
    ap.add_argument("--backbone", default=C.EOMT["dinov2_backbone"]["timm_name"])
    ap.add_argument("--img-size", type=int, default=C.EOMT_TRAIN["img_size"][0])
    ap.add_argument("--batch-size", type=int, default=C.EOMT_TRAIN["batch_size"])
    ap.add_argument("--lr", type=float, default=1e-4, help="EoMT's default; decide any change on validation patients only")
    ap.add_argument("--neg-per-pos", type=float, default=C.EOMT_TRAIN["neg_per_pos"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--order-seed", type=int, default=0,
                    help="0 = the frozen nested subsets; k>0 = an extra random ordering of the train patients (variability of small subsets)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, help="steps between validation passes (default steps // 8)")
    ap.add_argument("--ckpt-every", type=int, help="steps between saved weights (default steps // 6)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=C.DEVICE)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (EoMT's default; slower start, faster steps)")
    ap.add_argument("--run-name")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    a.manifest_dir = str(Path(a.manifest_dir).resolve())  # EoMT runs with its own folder as cwd, so a relative path would break
    sp = json.loads((Path(a.manifest_dir) / "splits.json").read_text())
    if a.n_patients not in sp["train_subsets"]:
        sys.exit(f"--n-patients {a.n_patients} not in {Path(a.manifest_dir) / 'splits.json'}: {list(sp['train_subsets'])}")
    name = a.run_name or f"n{a.n_patients}_{a.init}_s{a.steps}_seed{a.seed}" + (f"_order{a.order_seed}" if a.order_seed else "")
    out = C.OUTPUT_DIR / "eomt" / name
    check_setup(a)
    cfg = build_config(a, out)
    cmd = [sys.executable, "main.py", "fit", "-c", str(out / "run_config.yaml")]
    if a.dry_run:
        print(f"run directory: {out}\ncommand (cwd {C.EOMT_DIR}): {' '.join(cmd)}\n")
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "run_info.json").write_text(json.dumps({
        "command": sys.argv, "args": vars(a), "eomt": {"commit": eomt_commit(), "pinned": C.EOMT["commit"]},
        "init": a.init, "init_checkpoint_sha256": C.EOMT["ade20k_semantic_L512"]["sha256"] if a.init == "ade20k" else None,
        "classes": {"0": "background", "1": "tumor"}, "window_hu": C.HU_WINDOW, "frozen_split": str(Path(a.manifest_dir) / "splits.json"),
        "note": "Automatic model: sees only the CT slice, no prompts, no ground-truth localization."}, indent=1, default=str))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(C.ROOT), str(C.EOMT_DIR), os.environ.get("PYTHONPATH", "")]),
               WANDB_MODE="offline", WANDB_DIR=str(out))
    sys.exit(subprocess.run(cmd, cwd=C.EOMT_DIR, env=env).returncode)


if __name__ == "__main__":
    main()
