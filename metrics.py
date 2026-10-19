"""Shared 3D segmentation metrics (MedSAM2 now, EoMT later). Inputs are binary arrays in (z, y, x) order.

Empty-mask policy: if prediction and reference are both empty, Dice/IoU/HD95 are NaN (undefined, never 1.0 by default);
if exactly one is empty, Dice and IoU are 0 and HD95 is NaN. Evaluation cases always have a non-empty reference.
"""
import numpy as np
from scipy import ndimage


def dice(pred, gt):
    pred, gt = np.asarray(pred) > 0, np.asarray(gt) > 0
    total = pred.sum() + gt.sum()
    return float("nan") if total == 0 else float(2.0 * (pred & gt).sum() / total)


def iou(pred, gt):
    pred, gt = np.asarray(pred) > 0, np.asarray(gt) > 0
    union = (pred | gt).sum()
    return float("nan") if union == 0 else float((pred & gt).sum() / union)


def volume_ml(mask, spacing_xyz):
    return float((np.asarray(mask) > 0).sum()) * float(np.prod(spacing_xyz)) / 1000.0


def _surface(mask):
    """Boundary voxels: mask voxels with a background 6-neighbour (the array border counts as background)."""
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)


def hd95(pred, gt, spacing_xyz):
    """95th percentile of the pooled directed surface distances (mm), spacing given as (x, y, z).

    Distances are exact Euclidean distances in physical units between surface voxels. The computation is cropped to
    the joint bounding box of both masks, which loses nothing because all surface voxels lie inside it.
    """
    pred, gt = np.asarray(pred) > 0, np.asarray(gt) > 0
    if not pred.any() or not gt.any():
        return float("nan")
    idx = np.nonzero(pred | gt)
    sl = tuple(slice(max(int(i.min()) - 1, 0), int(i.max()) + 2) for i in idx)
    p, g = pred[sl], gt[sl]
    sp = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])  # array axes are z, y, x
    sp_p, sp_g = _surface(p), _surface(g)
    d_to_g = ndimage.distance_transform_edt(~sp_g, sampling=sp)
    d_to_p = ndimage.distance_transform_edt(~sp_p, sampling=sp)
    return float(np.percentile(np.concatenate([d_to_g[sp_p], d_to_p[sp_g]]), 95))


def bootstrap_ci(values, n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI of the mean over patients (resamples patients, not slices or lesions)."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def size_bin(volume_ml_value, edges=(10.0, 50.0)):
    """Predeclared tumor-size bins (mL): '<10', '10-50', '>=50' for the default edges."""
    lo, hi = edges
    if volume_ml_value < lo:
        return f"<{lo:g}"
    return f"{lo:g}-{hi:g}" if volume_ml_value < hi else f">={hi:g}"


def summarize(df, metrics=("dice", "iou"), by=("condition",), seed=42, n_boot=2000):
    """Per-volume rows -> per-patient means -> summary across patients (mean, median, std, bootstrap CI of the mean)."""
    import pandas as pd
    by = list(by)
    per_patient = df.groupby(by + ["patient_id"])[list(metrics)].mean().reset_index()
    rows = []
    for key, g in per_patient.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(by, key), n_patients=len(g))
        for m in metrics:
            lo, hi = bootstrap_ci(g[m].values, n_boot, seed)
            rec.update({f"{m}_mean": g[m].mean(), f"{m}_median": g[m].median(), f"{m}_std": g[m].std(ddof=1),
                        f"{m}_ci95_lo": lo, f"{m}_ci95_hi": hi})
        rows.append(rec)
    return per_patient, pd.DataFrame(rows)
