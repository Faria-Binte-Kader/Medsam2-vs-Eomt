"""Zero-shot MedSAM2 evaluation: ORACLE-LOCALIZATION PROMPTED baseline (not fully automatic).

The only ground-truth information given to the model is the prompt: for each lesion, a 2D box on its key slice
(see config.PROMPT). Predictions are never corrected using ground truth. Perturbed prompts (mild/strong) are
seeded and logged. Weights are the pinned genuine MedSAM2 checkpoint; there is no fallback model.

  python run_medsam2.py --split val --limit 1 --device cuda            # smoke test on one validation volume
  python run_medsam2.py --split val --dry-run                          # prompts only, no model
  python run_medsam2.py --split val --z-mode gt_extent --device cuda   # bound propagation by the GT z range
  python run_medsam2.py --split test --device cuda                     # final frozen test run (guarded)
  python run_medsam2.py --split secondary --device cuda                # separately reported secondary set
"""
import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

import config as C
import metrics as M
import process_data as P

CONN26 = np.ones((3, 3, 3), dtype=bool)


# ----------------------------------------------------------------------------- prompts (ground-truth derived)
def lesions_from_instances(instances, instance_ids):
    """Split each instance mask into 26-connected components; each component is one lesion (one prompt).

    Components are ordered by size (largest first) within an instance so lesion ids are deterministic.
    """
    out = []
    for inst, iid in zip(instances, instance_ids):
        lab, n = ndimage.label(inst > 0, structure=CONN26)
        comps = sorted(range(1, n + 1), key=lambda k: (-int((lab == k).sum()), k))
        for c, k in enumerate(comps):
            out.append({"lesion_id": f"i{iid}c{c}", "instance_id": int(iid), "component": c, "mask": lab == k})
    return out


def key_slice_and_box(mask):
    """Key slice = axial slice with the largest lesion area (first on ties); box = tight in-plane extent there.

    Box is [x_min, y_min, x_max, y_max] with inclusive pixel indices, the same convention as the upstream
    mask2D_to_bbox. Raises on an empty mask.
    """
    area = mask.sum((1, 2))
    if area.max() == 0:
        raise ValueError("empty lesion mask: cannot derive a prompt")
    z = int(np.argmax(area))
    ys, xs = np.nonzero(mask[z])
    return z, [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], int(area[z])


def make_rng(*parts):
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def perturb_box(box, params, rng, H, W, min_side=2):
    """Pad -> translate -> per-coordinate jitter -> clip to the image. All in pixels, integer valued.

    pad: every side moves out by `pad`. translate: whole box shifts by integers in [-t, t] per axis.
    jitter: each of the 4 coordinates moves by an integer in [-j, j]. Returns a log dict; `valid` is False when
    the clipped box has a side shorter than `min_side` pixels or is inverted (recorded, never silently repaired).
    """
    x0, y0, x1, y1 = box
    p, t, j = params["pad"], params["translate"], params["jitter"]
    x0, y0, x1, y1 = x0 - p, y0 - p, x1 + p, y1 + p
    dx, dy = (int(rng.integers(-t, t + 1)), int(rng.integers(-t, t + 1))) if t else (0, 0)
    jit = [int(v) for v in rng.integers(-j, j + 1, 4)] if j else [0, 0, 0, 0]
    x0, y0, x1, y1 = x0 + dx + jit[0], y0 + dy + jit[1], x1 + dx + jit[2], y1 + dy + jit[3]
    before = [x0, y0, x1, y1]
    after = [min(max(x0, 0), W - 1), min(max(y0, 0), H - 1), min(max(x1, 0), W - 1), min(max(y1, 0), H - 1)]
    valid = (after[2] - after[0] + 1) >= min_side and (after[3] - after[1] + 1) >= min_side
    return {"box": after, "box_before_clip": before, "clipped": after != before, "valid": bool(valid),
            "reason": None if valid else "box smaller than min_side or inverted after clipping",
            "translate_xy": [dx, dy], "jitter": jit}


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
    inter = ix * iy
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / (area(a) + area(b) - inter)


def build_prompts(lesions, patient, phase, conditions, H, W, z_mode="unbounded"):
    """One record per (lesion, condition). GT enters only through each lesion's key slice and exact box."""
    recs = []
    for li, les in enumerate(lesions):
        z, exact, area = key_slice_and_box(les["mask"])
        zs = np.nonzero(les["mask"].any((1, 2)))[0]
        zr = [int(zs.min()), int(zs.max())]
        bounded = z_mode == "gt_extent"
        for cname, params in conditions.items():
            ci = list(C.PROMPT_CONDITIONS).index(cname)  # global index, so resumed/partial runs draw identical boxes
            entropy = [C.SEED, patient, phase, li, ci]
            pr = perturb_box(exact, params, make_rng(*entropy), H, W)
            recs.append({"patient_id": patient, "phase": phase, "lesion_id": les["lesion_id"], "lesion_index": li,
                         "condition": cname, "condition_params": params, "key_slice": z, "gt_area_px_on_key_slice": area,
                         "gt_z_range": zr, "z_mode": z_mode,
                         "propagate_forward_max": zr[1] - z if bounded else None,
                         "propagate_reverse_max": z - zr[0] if bounded else None,
                         "exact_box_xyxy": exact, "final_box_xyxy": pr["box"], "box_before_clip": pr["box_before_clip"],
                         "clipped": pr["clipped"], "valid": pr["valid"], "invalid_reason": pr["reason"],
                         "translate_xy": pr["translate_xy"], "jitter": pr["jitter"],
                         "box_iou_with_exact": round(box_iou(pr["box"], exact), 4) if pr["valid"] else None,
                         "units": C.PROMPT["box_units"], "seed_entropy": entropy})
    return recs


# ----------------------------------------------------------------------------- model
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint():
    ck = C.MEDSAM2
    path = C.MODELS_DIR / ck["checkpoint"]
    if not path.is_file():
        sys.exit(f"MedSAM2 checkpoint not found at {path}. Run `python fetch_medsam2.py`; there is no fallback model.")
    if path.stat().st_size != ck["size_bytes"] or sha256_file(path) != ck["sha256"]:
        sys.exit(f"{path} does not match the pinned size/SHA-256; refusing to run with unverified weights.")
    return path


def resolve_device(requested):
    import torch
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        sys.exit("--device cuda requested but CUDA is not available (use --device cpu for a slow smoke test)")
    return "cpu"


def load_predictor(device):
    import torch
    from sam2.build_sam import build_sam2_video_predictor_npz
    ckpt = verify_checkpoint()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(C.SEED)
    return build_sam2_video_predictor_npz(C.MEDSAM2["model_cfg"], str(ckpt), device=device)


def propagate_lesion(predictor, images, H, W, key, box, device, z_range=None):
    """Box on `key`, propagate forward, reset, same box again, propagate in reverse: upstream semantics.

    With z_range=(z_min, z_max) the propagation itself is bounded (max_frame_num_to_track forward = z_max - key,
    reverse = key - z_min). Returns a bool array (z, H, W) at the native slice size (the predictor resizes to H, W).
    """
    import torch
    seg = np.zeros((images.shape[0], H, W), dtype=bool)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    with torch.inference_mode(), amp:
        state = predictor.init_state(images, H, W)
        for reverse in (False, True):
            predictor.add_new_points_or_box(inference_state=state, frame_idx=key, obj_id=1,
                                            box=np.array(box, dtype=np.float32))
            kw = {} if z_range is None else {"max_frame_num_to_track": (key - z_range[0]) if reverse else (z_range[1] - key)}
            for idx, _ids, logits in predictor.propagate_in_video(state, reverse=reverse, **kw):
                seg[idx] |= (logits[0] > 0.0).cpu().numpy()[0]
            predictor.reset_state(state)
    return seg


def largest_cc(seg):
    lab, n = ndimage.label(seg, structure=CONN26)
    if n == 0:
        return seg
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


# ----------------------------------------------------------------------------- evaluation of one case
def evaluate_case(case, row, split, conditions, predictor, device, out_dir, hd95=False, overlays=True,
                  propagate=propagate_lesion, z_mode="unbounded"):
    """Run every condition on one cached case. Returns (volume rows, lesion rows, prompt records)."""
    meta, gt = case["meta"], case["mask_union"] > 0
    geom = meta["geometry"]
    Z, H, W = gt.shape
    patient, phase = meta["patient_id"], meta["phase"]
    lesions = lesions_from_instances(case["instances"], case["instance_ids"])
    covered = np.zeros_like(gt)
    for les in lesions:
        covered |= les["mask"]
    assert (covered == gt).all(), "lesion components do not reproduce the union ground truth"
    prompts = build_prompts(lesions, patient, phase, conditions, H, W, z_mode)
    images = None
    if predictor is not None:
        import torch
        images = torch.from_numpy(P.to_medsam2_input(case["ct_u8"])).to(device)
    vol_rows, les_rows = [], []
    for cname in conditions:
        pred = np.zeros_like(gt)
        t0 = time.perf_counter()
        n_inv = 0
        for les, pr in zip(lesions, [p for p in prompts if p["condition"] == cname]):
            rec = {"patient_id": patient, "phase": phase, "condition": cname, "lesion_id": les["lesion_id"],
                   "gt_volume_ml": M.volume_ml(les["mask"], geom["spacing_xyz"]), "valid_prompt": pr["valid"]}
            if not pr["valid"]:
                n_inv += 1
                rec.update(dice=0.0, pred_volume_ml=0.0)  # a skipped lesion counts as missed, not dropped
                les_rows.append(rec)
                continue
            if predictor is None:
                continue
            zr = tuple(pr["gt_z_range"]) if z_mode == "gt_extent" else None
            seg = largest_cc(propagate(predictor, images, H, W, pr["key_slice"], pr["final_box_xyxy"], device, z_range=zr))
            pred |= seg
            rec.update(dice=M.dice(seg, les["mask"]), pred_volume_ml=M.volume_ml(seg, geom["spacing_xyz"]))
            les_rows.append(rec)
        if predictor is None:
            continue
        if device == "cuda":
            import torch
            torch.cuda.synchronize()
        runtime = time.perf_counter() - t0
        pred_native = P.restore_to_native(pred, geom).astype(bool)
        mpath = Path(out_dir) / "masks" / row["eval_set"] / f"p{patient}_ph{phase}" / f"{cname}.nii.gz"
        P.write_native_mask(pred_native, geom, mpath)
        gt_ml = M.volume_ml(gt, geom["spacing_xyz"])
        zs = np.nonzero(gt.any((1, 2)))[0]  # DIAGNOSTICS ONLY (use GT, never fed back into the prediction)
        zlo, zhi = int(zs.min()), int(zs.max())
        outside = pred_native.copy()
        outside[zlo:zhi + 1] = False
        vol_rows.append({
            "patient_id": patient, "phase": phase, "phase_name": meta["phase_name"], "eval_set": row["eval_set"],
            "split": split, "z_mode": z_mode, "condition": cname, "n_lesions": len(lesions), "n_invalid_prompts": n_inv,
            "gt_volume_ml": gt_ml, "pred_volume_ml": M.volume_ml(pred_native, geom["spacing_xyz"]),
            "size_bin": M.size_bin(gt_ml, C.SIZE_BINS_ML), "dice": M.dice(pred_native, gt), "iou": M.iou(pred_native, gt),
            "hd95_mm": M.hd95(pred_native, gt, geom["spacing_xyz"]) if hd95 else float("nan"),
            "diag_pred_ml_outside_gt_z": M.volume_ml(outside, geom["spacing_xyz"]),
            "diag_dice_within_gt_z": M.dice(pred_native[zlo:zhi + 1], gt[zlo:zhi + 1]),
            "diag_gt_z_range": f"{zlo}-{zhi}", "runtime_s": runtime, "n_slices": Z, "mask_path": str(mpath)})
        if overlays:
            save_overlay(case, pred_native, [p for p in prompts if p["condition"] == cname],
                         Path(out_dir) / "overlays" / f"p{patient}_ph{phase}_{cname}.png", vol_rows[-1])
    return vol_rows, les_rows, prompts


def save_overlay(case, pred, prompts, path, vrow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = case["mask_union"] > 0
    big = max(prompts, key=lambda p: p["gt_area_px_on_key_slice"])
    z = big["key_slice"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(case["ct_u8"][z], cmap="gray", vmin=0, vmax=255)
    ax.contour(gt[z], levels=[0.5], colors="lime", linewidths=1)
    if pred[z].any():
        ax.contour(pred[z], levels=[0.5], colors="red", linewidths=1)
    b = big["final_box_xyxy"]
    ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor="yellow", lw=1.2))
    ys, xs = np.nonzero(gt[z])
    ax.set_xlim(max(xs.min() - 60, 0), min(xs.max() + 60, gt.shape[2]))
    ax.set_ylim(min(ys.max() + 60, gt.shape[1]), max(ys.min() - 60, 0))
    ax.set_title(f"p{vrow['patient_id']} {vrow['condition']} z={z} Dice={vrow['dice']:.3f}\n"
                 "green=GT red=pred yellow=prompt box", fontsize=8)
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- run management
def config_fingerprint():
    # z_mode is chosen per run and recorded in the snapshot; the set of allowed modes is part of the frozen design
    blob = {"window": C.HU_WINDOW, "conditions": C.PROMPT_CONDITIONS,
            "prompt": {k: v for k, v in C.PROMPT.items() if k != "z_extent"}, "z_modes": C.Z_MODES,
            "medsam2": C.MEDSAM2, "seed": C.SEED, "preprocess": C.PREPROCESS_VERSION, "size_bins": C.SIZE_BINS_ML}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()[:16]


def select_cases(split, inv, splits, patients=None, limit=None):
    if split == "secondary":
        rows = inv[inv.eval_set == "secondary"]
    else:
        rows = inv[(inv.eval_set == "primary") & inv.patient_id.isin(splits[split])]
    if patients:
        rows = rows[rows.patient_id.isin(patients)]
    rows = rows.sort_values(["patient_id", "phase"])
    return rows.head(limit) if limit else rows


def _append_jsonl(path, recs):
    with open(path, "a") as f:
        for r in recs:
            f.write(json.dumps(r, default=str) + "\n")


def _append_csv(path, df):
    import pandas as pd
    df.to_csv(path, mode="a", header=not Path(path).exists(), index=False)


def guard_test_split(args, fp, run_dir):
    """The test set is touched by full runs only under one frozen configuration."""
    reg = C.OUTPUT_DIR / "medsam2" / "test_registry.json"
    runs = json.loads(reg.read_text()) if reg.exists() else {}
    if runs and fp not in runs and not args.allow_test_config_change:
        sys.exit(f"Test split was already evaluated with configuration(s) {sorted(runs)}; current is {fp}. "
                 "Changing prompt policy, window or model after seeing test results is not allowed. Use "
                 "--allow-test-config-change only for a documented, intentional rerun.")
    runs.setdefault(fp, []).append({"run_dir": str(run_dir), "z_mode": args.z_mode, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(runs, indent=1))


def versions():
    import importlib.metadata as md
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    for pkg in ("torch", "SimpleITK", "MedSAM2", "SAM-2"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    try:
        from sam2 import _C  # noqa: F401  (optional compiled extension used for hole filling)
        out["sam2_cuda_extension"] = "loaded"
    except Exception as e:
        out["sam2_cuda_extension"] = f"NOT loaded ({type(e).__name__}): small-hole filling post-processing is skipped"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "val", "test", "secondary"], required=True)
    ap.add_argument("--conditions", nargs="+", default=list(C.PROMPT_CONDITIONS), choices=list(C.PROMPT_CONDITIONS))
    ap.add_argument("--device", default=C.DEVICE, choices=["cuda", "cpu", "auto"])
    ap.add_argument("--patients", type=int, nargs="+")
    ap.add_argument("--limit", type=int, help="debug: first N cases of the split")
    ap.add_argument("--z-mode", choices=list(C.Z_MODES), default=C.PROMPT["z_extent"],
                    help="unbounded: propagate through the whole volume; gt_extent: bound propagation by the lesion's GT z range")
    ap.add_argument("--hd95", action="store_true", help="also compute HD95 (mm); slower")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="derive and write prompts only; no model, no predictions")
    ap.add_argument("--run-name")
    ap.add_argument("--manifest-dir", default=str(C.MANIFEST_DIR))
    ap.add_argument("--allow-test-config-change", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    man = Path(args.manifest_dir)
    inv = pd.read_csv(man / "inventory.csv")
    splits = json.loads((man / "splits.json").read_text())
    todo = select_cases(args.split, inv, splits, args.patients, args.limit)
    if todo.empty:
        sys.exit("no cases selected")
    conditions = {k: C.PROMPT_CONDITIONS[k] for k in args.conditions}
    debug = bool(args.limit or args.patients)
    name = args.run_name or (f"{args.split}_{args.z_mode}" + ("_debug" if debug else "") + ("_dryrun" if args.dry_run else ""))
    run_dir = C.OUTPUT_DIR / "medsam2" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = config_fingerprint()

    snap = run_dir / "config_snapshot.json"
    if snap.exists() and json.loads(snap.read_text())["fingerprint"] != fp:
        sys.exit(f"{run_dir} was created with a different configuration; use --run-name or delete the directory")
    if args.split == "test" and not debug and not args.dry_run:
        guard_test_split(args, fp, run_dir)
    if args.split == "test":
        print("NOTE: test split. Do not tune prompts, window or post-processing on these results.")

    device = None if args.dry_run else resolve_device(args.device)
    predictor = None if args.dry_run else load_predictor(device)
    ver = versions()
    if "NOT loaded" in str(ver.get("sam2_cuda_extension")):
        print(f"WARNING: sam2 CUDA extension {ver['sam2_cuda_extension']}. Rebuild it (README) so all runs use the same post-processing.")
    snap.write_text(json.dumps({
        "fingerprint": fp, "z_mode": args.z_mode, "command": sys.argv, "split": args.split, "n_cases": len(todo), "conditions": conditions,
        "window_hu": C.HU_WINDOW, "prompt_policy": C.PROMPT, "medsam2": C.MEDSAM2, "seed": C.SEED, "device": device,
        "versions": ver, "git_commit": P._git_head(), "oracle_localization": True,
        "note": "Prompted baseline: boxes derive from ground-truth masks; not a fully automatic result."}, indent=1, default=str))

    vpath, lpath, ppath = run_dir / "volume_metrics.csv", run_dir / "lesion_metrics.jsonl", run_dir / "prompts.jsonl"
    done = set()
    if vpath.exists() and not args.dry_run:
        d = pd.read_csv(vpath)
        done = set(zip(d.patient_id, d.phase, d.condition))
    failures = []
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        p, ph = int(row.patient_id), int(row.phase)
        conds = {c: v for c, v in conditions.items() if args.dry_run or (p, ph, c) not in done}
        if not conds:
            print(f"[{i}/{len(todo)}] p{p} phase {ph}: already done, skipping")
            continue
        try:
            case = P.load_case(P.case_dir(row))
            vrows, lrows, prompts = evaluate_case(case, row, args.split, conds, predictor, device, run_dir,
                                                  hd95=args.hd95, overlays=not args.no_overlays, z_mode=args.z_mode)
        except Exception as e:  # keep a long job alive; failures are reported and counted
            failures.append({"patient_id": p, "phase": ph, "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(todo)}] p{p} phase {ph}: FAILED {type(e).__name__}: {e}")
            continue
        _append_jsonl(ppath, prompts)
        if args.dry_run:
            print(f"[{i}/{len(todo)}] p{p} phase {ph}: {len(prompts) // len(conds)} lesion(s), "
                  f"invalid prompts: {sum(not x['valid'] for x in prompts)}")
            continue
        _append_jsonl(lpath, lrows)
        _append_csv(vpath, pd.DataFrame(vrows))
        print(f"[{i}/{len(todo)}] p{p} {row.phase_name}: " + ", ".join(
            f"{r['condition']} dice={r['dice']:.3f} ({r['runtime_s']:.0f}s)" for r in vrows))
    if failures:
        (run_dir / "failures.json").write_text(json.dumps(failures, indent=1))
        print(f"{len(failures)} case(s) failed; see {run_dir / 'failures.json'}")
    if args.dry_run or not vpath.exists():
        sys.exit(1 if failures else 0)

    vol = pd.read_csv(vpath)
    per_patient, summary = M.summarize(vol, ("dice", "iou"), ("condition",), C.BOOTSTRAP["seed"], C.BOOTSTRAP["n"])
    _, by_size = M.summarize(vol, ("dice", "iou"), ("condition", "size_bin"), C.BOOTSTRAP["seed"], C.BOOTSTRAP["n"])
    per_patient.to_csv(run_dir / "patient_summary.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    by_size.to_csv(run_dir / "summary_by_size.csv", index=False)
    print(summary.round(3).to_string(index=False))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
