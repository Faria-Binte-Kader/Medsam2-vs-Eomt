"""Slide-ready figures from the preprocessing cache and a finished MedSAM2 run (no inference is run here).

  python make_figures.py --run-dir results/medsam2/test_unbounded --out-dir figures

Cases are chosen by a fixed rule, not by eye: best, median (lower of the two middle values) and worst patient by
exact-box 3D Dice. Colours: ground truth cyan, MedSAM2 prediction magenta, prompt box yellow. Images are shown in array
order (rows = y, columns = x), not in radiological orientation.
  fig_examples.png     one row per chosen case: CT | + ground truth | + prompt box | + MedSAM2 prediction
  fig_sensitivity.png  the median case with exact / mild / strong boxes and the resulting predictions
  fig_preprocessing.png raw HU | liver window (model input) | + ground truth and prompt box (needs the CT archive)
"""
import argparse
import json
from pathlib import Path

import numpy as np

import config as C
import metrics as M
import process_data as P

GT, PRED, BOX = "#00e5ff", "#ff2bd6", "#ffd600"


def choose_cases(vol, condition="exact"):
    """Best / median / worst patient by 3D Dice for one condition. Returns a list of (label, row), fewer if few cases."""
    d = vol[vol.condition == condition].sort_values("dice", ascending=False).reset_index(drop=True)
    if d.empty:
        return []
    picks = [("best", d.iloc[0]), ("median", d.iloc[len(d) // 2]), ("worst", d.iloc[-1])]  # descending: n//2 is the lower middle
    seen, out = set(), []
    for label, r in picks:
        if int(r.patient_id) not in seen:
            seen.add(int(r.patient_id))
            out.append((label, r))
    return out


def load_pred(run_dir, row, condition):
    import SimpleITK as sitk
    p = Path(run_dir) / "masks" / row["eval_set"] / f"p{int(row['patient_id'])}_ph{int(row['phase'])}" / f"{condition}.nii.gz"
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0


def key_prompt(prompts, patient, condition):
    """Prompt of the lesion with the largest key-slice area (the one shown in the overlays)."""
    cand = [p for p in prompts if p["patient_id"] == patient and p["condition"] == condition]
    return max(cand, key=lambda p: p["gt_area_px_on_key_slice"])


def crop_window(masks2d, shape, margin=40, min_side=120):
    union = np.logical_or.reduce(masks2d)
    if not union.any():
        return 0, shape[0], 0, shape[1]
    ys, xs = np.nonzero(union)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min(), min_side) / 2 + margin
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, shape[0]))
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, shape[1]))
    return y0, y1, x0, x1


def draw(ax, ct2d, win, gt=None, pred=None, box=None, gt_fill=False, title=None):
    y0, y1, x0, x1 = win
    ax.imshow(ct2d[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255, extent=(x0, x1, y1, y0), interpolation="nearest")
    xs, ys = np.arange(x0, x1), np.arange(y0, y1)
    if gt is not None and gt[y0:y1, x0:x1].any():
        if gt_fill:
            ax.contourf(xs, ys, gt[y0:y1, x0:x1], levels=[0.5, 1.5], colors=[GT], alpha=0.35)
        ax.contour(xs, ys, gt[y0:y1, x0:x1], levels=[0.5], colors=[GT], linewidths=1.4)
    if pred is not None and pred[y0:y1, x0:x1].any():
        ax.contour(xs, ys, pred[y0:y1, x0:x1], levels=[0.5], colors=[PRED], linewidths=1.4)
    if box is not None:
        ax.add_patch(__import__("matplotlib").patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                                                fill=False, edgecolor=BOX, lw=1.6))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)


def _case(run_dir, row, prompts):
    case = P.load_case(P.case_dir(row))
    return case, key_prompt(prompts, int(row["patient_id"]), "exact")


def fig_examples(run_dir, chosen, prompts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(chosen), 4, figsize=(11, 2.9 * len(chosen)), squeeze=False)
    for i, (label, r) in enumerate(chosen):
        case, kp = _case(run_dir, r, prompts)
        z = kp["key_slice"]
        ct, gt = case["ct_u8"][z], case["mask_union"][z] > 0
        pred = load_pred(run_dir, r, "exact")[z]
        win = crop_window([gt, pred], gt.shape)
        sd = M.dice(pred, gt)
        draw(axes[i, 0], ct, win, title="Input CT (liver window)" if i == 0 else None)
        draw(axes[i, 1], ct, win, gt=gt, gt_fill=True, title="Ground-truth mask" if i == 0 else None)
        draw(axes[i, 2], ct, win, gt=gt, box=kp["final_box_xyxy"], title="Prompt box from GT (exact)" if i == 0 else None)
        draw(axes[i, 3], ct, win, gt=gt, pred=pred, title="MedSAM2 prediction vs GT" if i == 0 else None)
        axes[i, 0].text(0.02, 0.96, f"{label}: patient {int(r.patient_id)}, {r.phase_name}\n3D Dice {r.dice:.2f} | slice Dice {sd:.2f}",
                        transform=axes[i, 0].transAxes, color="white", fontsize=8, va="top",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2, lw=0))
    fig.suptitle("Key slice of the largest lesion. Cyan = ground truth, magenta = MedSAM2, yellow = prompt box. "
                 "Cases chosen by rule: best / median / worst exact-box Dice.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_sensitivity(run_dir, label_row, prompts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    label, r = label_row
    case = P.load_case(P.case_dir(r))
    vol = __import__("pandas").read_csv(Path(run_dir) / "volume_metrics.csv")
    conds = [c for c in C.PROMPT_CONDITIONS if ((vol.patient_id == r.patient_id) & (vol.condition == c)).any()]
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.9), squeeze=False)
    for ax, c in zip(axes[0], conds):
        kp = key_prompt(prompts, int(r.patient_id), c)
        z = kp["key_slice"]
        gt = case["mask_union"][z] > 0
        pred = load_pred(run_dir, r, c)[z]
        win = crop_window([gt, pred], gt.shape, margin=70)
        d3 = float(vol[(vol.patient_id == r.patient_id) & (vol.condition == c)].dice.iloc[0])
        draw(ax, case["ct_u8"][z], win, gt=gt, pred=pred, box=kp["final_box_xyxy"],
             title=f"{c} box (IoU with exact {kp['box_iou_with_exact']:.2f})\n3D Dice {d3:.2f}")
    fig.suptitle(f"Prompt sensitivity, {label} case (patient {int(r.patient_id)}). Cyan = GT, magenta = MedSAM2, yellow = prompt box",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_preprocessing(run_dir, row, prompts, inv, out):
    """Raw HU vs the liver-windowed model input vs overlay. Reads the CT from the archive (extracted, then removed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    import download_data as D
    case = P.load_case(P.case_dir(row))
    kp = key_prompt(prompts, int(row["patient_id"]), "exact")
    z = kp["key_slice"]
    meta = inv[(inv.patient_id == row["patient_id"]) & (inv.phase == row["phase"])].iloc[0]
    D.safe_extract(C.RAW_DIR / meta["ct_archive"], C.EXTRACT_DIR, meta["ct_member"])
    path = C.EXTRACT_DIR / meta["ct_member"]
    hu = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))[z]
    path.unlink()
    lo, hi = C.HU_WINDOW
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    axes[0].imshow(hu, cmap="gray", vmin=-1024, vmax=1500)
    axes[0].set_title(f"Raw CT in HU\n(display -1024 to 1500, range {int(hu.min())} to {int(hu.max())})", fontsize=9)
    axes[1].imshow(case["ct_u8"][z], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Liver window {lo:.0f} to {hi:.0f} HU\n-> uint8 model input", fontsize=9)
    axes[2].imshow(case["ct_u8"][z], cmap="gray", vmin=0, vmax=255)
    gt = case["mask_union"][z] > 0
    axes[2].contour(gt, levels=[0.5], colors=[GT], linewidths=1.4)
    b = kp["final_box_xyxy"]
    axes[2].add_patch(matplotlib.patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=BOX, lw=1.6))
    axes[2].set_title("+ ground truth (cyan) and\nprompt box (yellow)", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Preprocessing, patient {int(row['patient_id'])} {row['phase_name']}, key slice {z}; native geometry is kept", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--manifest-dir", default=str(C.MANIFEST_DIR))
    ap.add_argument("--no-preprocessing", action="store_true", help="skip the figure that reads the CT archive")
    a = ap.parse_args()
    run = Path(a.run_dir)
    vol = pd.read_csv(run / "volume_metrics.csv")
    prompts = [json.loads(l) for l in (run / "prompts.jsonl").read_text().splitlines()]
    chosen = choose_cases(vol)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    out = Path(a.out_dir)
    print("cases (rule: best / median / worst exact-box Dice):", [(l, int(r.patient_id), round(float(r.dice), 3)) for l, r in chosen])
    fig_examples(run, chosen, prompts, out / "fig_examples.png")
    med = next((c for c in chosen if c[0] == "median"), chosen[0])
    fig_sensitivity(run, med, prompts, out / "fig_sensitivity.png")
    if not a.no_preprocessing:
        inv = pd.read_csv(Path(a.manifest_dir) / "inventory.csv")
        fig_preprocessing(run, med[1], prompts, inv, out / "fig_preprocessing.png")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
