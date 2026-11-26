"""Volume-level evaluation of a fine-tuned EoMT: run on every slice (no prompts), stack, restore native grid, score with the
shared metrics (same Dice / IoU / HD95 code and the same CSV columns as the MedSAM2 baseline, so the curves overlay).

  python eval_eomt.py --run-dir results/eomt/<run> --select              # val: score every saved checkpoint, record the best
  python eval_eomt.py --run-dir results/eomt/<run> --split test          # test: uses ONLY the checkpoint chosen on validation
  python eval_eomt.py --run-dir results/eomt/<run> --split val --weights results/eomt/<run>/weights/step500.pt --limit 1

The prediction of a slice is the argmax over (background, tumor) of EoMT's per-pixel class scores at the last block, exactly its
own semantic-segmentation read-out; no post-processing is applied (the MedSAM2 baseline's largest-component step exists because
that model is prompted per lesion). Checkpoints are chosen by mean per-patient 3D Dice on the validation patients only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import config as C
import metrics as M
import process_data as P

sys.path.insert(0, str(C.EOMT_DIR))


def build_network(run_info, device):
    """Same architecture as training; weights are loaded afterwards, so no pretrained download is triggered."""
    import torch
    from models.eomt import EoMT
    from models.vit import ViT
    a = run_info["args"]
    enc = ViT(img_size=(a["img_size"], a["img_size"]), backbone_name=a["backbone"], ckpt_path="weights-loaded-separately")
    net = EoMT(encoder=enc, num_classes=C.EOMT_TRAIN["num_classes"], num_q=100)
    return net.to(device).eval(), torch


def load_weights(net, path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)  # plain state dict written by eomt_data.WeightsCheckpoint
    sd = {k[len("network."):]: v for k, v in ck["state_dict"].items()}
    missing = net.load_state_dict(sd, strict=True)
    return int(ck["global_step"]), missing


def predict_volume(net, torch, ct_u8, device, batch_slices=16, size=512):
    """Tumor mask (z, H, W) bool from slice-by-slice inference. Native slices must already be size x size."""
    import torch.nn.functional as F
    from training.mask_classification_semantic import MaskClassificationSemantic
    Z, H, W = ct_u8.shape
    if (H, W) != (size, size):
        raise ValueError(f"slice size {(H, W)} != model input {(size, size)}; resize policy not defined for this case")
    out = np.zeros((Z, H, W), dtype=bool)
    amp = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), amp:
        for s in range(0, Z, batch_slices):
            x = torch.from_numpy(ct_u8[s:s + batch_slices]).to(device)[:, None].expand(-1, 3, -1, -1).float() / 255.0
            mask_logits, class_logits = net(x)
            ml = F.interpolate(mask_logits[-1], (H, W), mode="bilinear")
            scores = MaskClassificationSemantic.to_per_pixel_logits_semantic(ml, class_logits[-1])  # (B, 2, H, W)
            out[s:s + batch_slices] = (scores.argmax(1) == 1).cpu().numpy()
    return out


def evaluate_patients(net, torch, rows, split, device, out_dir, size, tag, hd95=False, batch_slices=16, save_masks=True):
    recs = []
    for _, row in rows.iterrows():
        case = P.load_case(P.case_dir(row), keys=("ct_u8", "mask_union"))
        geom = case["meta"]["geometry"]
        gt = case["mask_union"] > 0
        t0 = time.perf_counter()
        pred = predict_volume(net, torch, case["ct_u8"], device, batch_slices, size)
        if device == "cuda":
            torch.cuda.synchronize()
        runtime = time.perf_counter() - t0
        native = P.restore_to_native(pred, geom).astype(bool)
        if save_masks:
            P.write_native_mask(native, geom, Path(out_dir) / "masks" / f"p{int(row.patient_id)}_ph{int(row.phase)}.nii.gz")
        zs = np.nonzero(gt.any((1, 2)))[0]
        outside = native.copy()
        outside[zs.min():zs.max() + 1] = False
        gt_ml = M.volume_ml(gt, geom["spacing_xyz"])
        recs.append({
            "patient_id": int(row.patient_id), "phase": int(row.phase), "phase_name": row.phase_name, "eval_set": row.eval_set,
            "split": split, "model": "eomt", "checkpoint": tag, "n_lesions": np.nan, "gt_volume_ml": gt_ml,
            "pred_volume_ml": M.volume_ml(native, geom["spacing_xyz"]), "size_bin": M.size_bin(gt_ml, C.SIZE_BINS_ML),
            "dice": M.dice(native, gt), "iou": M.iou(native, gt),
            "hd95_mm": M.hd95(native, gt, geom["spacing_xyz"]) if hd95 else float("nan"),
            "diag_pred_ml_outside_gt_z": M.volume_ml(outside, geom["spacing_xyz"]),
            "diag_dice_within_gt_z": M.dice(native[zs.min():zs.max() + 1], gt[zs.min():zs.max() + 1]),
            "diag_false_positive_slices": int((native.any((1, 2)) & ~gt.any((1, 2))).sum()),
            "runtime_s": runtime, "n_slices": gt.shape[0]})
    return recs


def select_best(step_scores):
    """Checkpoint with the highest mean per-patient validation Dice; ties go to the earlier step."""
    return max(sorted(step_scores), key=lambda s: step_scores[s])


def cases_for(split, inv, splits, patients=None, limit=None):
    if split == "secondary":
        rows = inv[inv.eval_set == "secondary"]
    else:
        rows = inv[(inv.eval_set == "primary") & inv.patient_id.isin(splits[split])]
    if patients:
        rows = rows[rows.patient_id.isin(patients)]
    rows = rows.sort_values(["patient_id", "phase"])
    return rows.head(limit) if limit else rows


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", choices=["train", "val", "test", "secondary"], default="val",
                    help="train is for sanity checks on patients the model was trained on (always written to a *_debug folder)")
    ap.add_argument("--select", action="store_true", help="score every saved checkpoint on the validation patients and record the best")
    ap.add_argument("--weights", help="a specific weights file (not allowed on test/secondary)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=C.DEVICE)
    ap.add_argument("--hd95", action="store_true")
    ap.add_argument("--overwrite-selection", action="store_true", help="with --select: replace an existing best_step.json even if fewer checkpoints remain")
    ap.add_argument("--prune", action="store_true", help="with --select: delete every weights file except the selected one")
    ap.add_argument("--patients", type=int, nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-slices", type=int, default=16)
    ap.add_argument("--manifest-dir", default=str(C.MANIFEST_DIR))
    a = ap.parse_args()

    run = Path(a.run_dir)
    info = json.loads((run / "run_info.json").read_text())
    man = Path(a.manifest_dir)
    inv, splits = pd.read_csv(man / "inventory.csv"), json.loads((man / "splits.json").read_text())
    if a.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            sys.exit("--device cuda requested but CUDA is not available")
    net, torch = build_network(info, a.device)
    size = info["args"]["img_size"]
    best_file = run / "best_step.json"

    if a.select:
        rows = cases_for("val", inv, splits, a.patients, a.limit)
        scores, allrecs = {}, []
        for w in sorted((run / "weights").glob("step*.pt"), key=lambda p: int(p.stem[4:])):
            step, _ = load_weights(net, w)
            recs = evaluate_patients(net, torch, rows, "val", a.device, run / "eval_val" / f"step{step}", size, f"step{step}",
                                     a.hd95, a.batch_slices, save_masks=False)
            allrecs += recs
            scores[step] = float(np.mean([r["dice"] for r in recs]))
            print(f"step {step}: mean validation 3D Dice {scores[step]:.4f} ({len(recs)} patients)")
        if not scores:
            sys.exit(f"no weights found in {run / 'weights'}")
        if best_file.exists() and not a.overwrite_selection:
            old = json.loads(best_file.read_text()).get("val_dice_by_step", {})
            if len(old) > len(scores):  # e.g. after --prune: re-selecting from the one kept checkpoint would erase the full curve
                sys.exit(f"{best_file} already records {len(old)} checkpoints and only {len(scores)} weights remain; "
                         "not overwriting the selection (pass --overwrite-selection to force)")
        best = select_best(scores)
        (run / "eval_val").mkdir(exist_ok=True)
        pd.DataFrame(allrecs).to_csv(run / "eval_val" / "all_checkpoints_volume_metrics.csv", index=False)
        best_file.write_text(json.dumps({"best_step": best, "selected_on": "validation patients only (mean per-patient 3D Dice)",
                                         "val_dice_by_step": scores, "weights": str(run / "weights" / f"step{best}.pt")}, indent=1))
        print(f"best checkpoint on validation: step {best}; recorded in {best_file}")
        if a.prune:
            gone = [w for w in (run / "weights").glob("step*.pt") if int(w.stem[4:]) != best]
            for w in gone:
                w.unlink()
            print(f"pruned {len(gone)} non-selected weights files (the scores for all of them are kept in best_step.json)")
        return

    if a.split in ("test", "secondary"):
        if a.weights:
            sys.exit(f"--weights is not allowed on the {a.split} split: use the checkpoint chosen on validation (--select)")
        if not best_file.exists():
            sys.exit(f"{best_file} missing: run `--select` on the validation patients first")
    weights = Path(a.weights) if a.weights else Path(json.loads(best_file.read_text())["weights"]) if best_file.exists() else None
    if weights is None:
        sys.exit("give --weights, or run --select first")
    step, _ = load_weights(net, weights)
    rows = cases_for(a.split, inv, splits, a.patients, a.limit)
    debug = bool(a.limit or a.patients or a.weights or a.split == "train")
    out = run / f"eval_{a.split}{'_debug' if debug else ''}"
    recs = evaluate_patients(net, torch, rows, a.split, a.device, out, size, f"step{step}", a.hd95, a.batch_slices)
    vol = pd.DataFrame(recs)
    vol.insert(vol.columns.get_loc("model") + 1, "n_train_patients", info["args"]["n_patients"])
    vol.insert(vol.columns.get_loc("model") + 2, "init", info["args"]["init"])
    vol.insert(vol.columns.get_loc("model") + 3, "order_seed", info["args"].get("order_seed", 0))
    vol.to_csv(out / "volume_metrics.csv", index=False)
    vol["condition"] = "automatic"  # no prompt conditions for EoMT; keeps the shared summary code usable
    per_patient, summary = M.summarize(vol, ("dice", "iou"), ("condition",), C.BOOTSTRAP["seed"], C.BOOTSTRAP["n"])
    per_patient.to_csv(out / "patient_summary.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(f"{a.split}: checkpoint step {step}, {len(vol)} patients")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
