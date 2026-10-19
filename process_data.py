"""WAW-TACE inspection, inventory, splits and preprocessing.

  python process_data.py names                      # parse archive member names (no pixel data read)
  python process_data.py inspect --patients 10 13   # read real headers/intensities, compare geometry
  python process_data.py manifest                   # inventory.csv + excluded.csv (matching, geometry checks)
  python process_data.py splits                     # splits.json/csv + nested train subsets
  python process_data.py qc --n 12                  # overlay + summary figures
  python process_data.py diagnose                   # read-only: why annotated cases were excluded; lesion components

File-name conventions below were read from the real archive listings:
  CT   <p>/<p>_<phase>_scan.nii.gz
  mask <p>/<p>_<phase>_<instance>_tumor_seg.nrrd
Whether the phase digit in a mask name equals the phase digit in a scan name is a
hypothesis until `inspect` confirms it by geometry; names alone are never trusted.

Axis note: SimpleITK sizes/spacings/origins are (x, y, z); numpy arrays from
GetArrayFromImage are (z, y, x).
"""
import argparse
import gzip
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config as C

CT_RE = re.compile(r"^(?:.*/)?(\d+)/(\d+)_([0-3])_scan\.nii\.gz$")
MASK_RE = re.compile(r"^(?:.*/)?(\d+)/(\d+)_([0-3])_(\d+)_tumor_seg\.nrrd$")
GEOM_TOL = 1e-3


def parse_ct_name(member):
    m = CT_RE.match(member)
    if not m or m.group(1) != m.group(2):
        return None
    return {"patient": int(m.group(1)), "phase": int(m.group(3)), "member": member}


def parse_mask_name(member):
    m = MASK_RE.match(member)
    if not m or m.group(1) != m.group(2):
        return None
    return {"patient": int(m.group(1)), "phase": int(m.group(3)),
            "instance": int(m.group(4)), "member": member}


def parse_archive_names(archive, parser):
    """Return (parsed entries, non-conforming member names) for a ZIP."""
    ok, odd = [], []
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            if i.is_dir():
                continue
            e = parser(i.filename)
            (ok if e else odd).append(e or i.filename)
    return ok, odd


def header(path):
    """Geometry of an image file without loading pixels."""
    import SimpleITK as sitk
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return {
        "file": str(path), "size_xyz": list(r.GetSize()), "spacing_xyz": list(r.GetSpacing()),
        "origin_xyz": list(r.GetOrigin()), "direction": list(r.GetDirection()),
        "pixel_type": sitk.GetPixelIDValueAsString(r.GetPixelID()), "dims": r.GetDimension(),
        "meta": {k: r.GetMetaData(k) for k in r.GetMetaDataKeys()},
    }


def same_grid(a, b, tol=GEOM_TOL):
    """True when two headers describe the same voxel grid; also returns the reasons it differs."""
    why = []
    if a["size_xyz"] != b["size_xyz"]:
        why.append(f"size {a['size_xyz']} vs {b['size_xyz']}")
    for k in ("spacing_xyz", "origin_xyz", "direction"):
        if not np.allclose(a[k], b[k], atol=tol):
            why.append(f"{k} differs")
    return not why, why


def ct_stats(path):
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    p = np.percentile(a, [0.5, 50, 99.5])
    return {"dtype": str(a.dtype), "shape_zyx": list(a.shape), "min": float(a.min()), "max": float(a.max()),
            "p0.5_p50_p99.5": [float(x) for x in p]}


def mask_stats(path):
    import SimpleITK as sitk
    a = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    vals, counts = np.unique(a, return_counts=True)
    out = {"dtype": str(a.dtype), "shape_zyx": list(a.shape),
           "values": {int(v): int(c) for v, c in zip(vals, counts)}}
    nz = np.argwhere(a > 0)
    out["bbox_zyx_min_max"] = [nz.min(0).tolist(), nz.max(0).tolist()] if len(nz) else None
    return out


def cmd_names(_):
    masks, odd_m = parse_archive_names(C.RAW_DIR / "tumor_masks_wawtace_v1_08_05_2024.zip", parse_mask_name)
    cts, odd_c = [], []
    for arc in C.CT_ARCHIVES:
        c, o = parse_archive_names(C.RAW_DIR / arc, parse_ct_name)
        cts, odd_c = cts + c, odd_c + o
    print(f"masks: {len(masks)} files, {len({e['patient'] for e in masks})} patients; non-conforming: {odd_m}")
    print(f"CT archives {C.CT_ARCHIVES}: {len(cts)} scans, {len({e['patient'] for e in cts})} patients; non-conforming: {odd_c[:5]}")
    print("mask phase counts:", dict(sorted(Counter(e["phase"] for e in masks).items())))
    print("CT phase counts:", dict(sorted(Counter(e["phase"] for e in cts).items())))
    ct_keys = {(e["patient"], e["phase"]) for e in cts}
    mask_keys = {(e["patient"], e["phase"]) for e in masks}
    print(f"(patient,phase) with BOTH a scan in the configured archives and a mask (by name only): {len(ct_keys & mask_keys)}")
    print(f"mask (patient,phase) keys whose CT is not in the configured archives: {len(mask_keys - ct_keys)}")
    per = defaultdict(set)
    for e in masks:
        per[e["patient"]].add(e["phase"])
    print("patients with masks in more than one phase:", sum(len(v) > 1 for v in per.values()))


def cmd_inspect(a):
    out_dir = C.EXTRACT_DIR
    import download_data as D
    mask_zip = C.RAW_DIR / "tumor_masks_wawtace_v1_08_05_2024.zip"
    csv = C.RAW_DIR / "ct_hcc_metadata_v2.csv"
    if csv.exists():
        import pandas as pd
        df = pd.read_csv(csv)
        print(f"== metadata CSV: {df.shape[0]} rows, columns: {list(df.columns)}")
        print(df.head(3).to_string())
    report = []
    for p in a.patients:
        D.safe_extract(mask_zip, out_dir, f"*/{p}/{p}_*")
        for arc in C.CT_ARCHIVES:
            D.safe_extract(C.RAW_DIR / arc, out_dir, f"{p}/{p}_*")
    for p in a.patients:
        ct_files = sorted(out_dir.rglob(f"{p}/{p}_*_scan.nii.gz"))
        mask_files = sorted(out_dir.rglob(f"{p}/{p}_*_tumor_seg.nrrd"))
        cts = {parse_ct_name(f"{f.parent.name}/{f.name}")["phase"]: f for f in ct_files}
        for ph, f in sorted(cts.items()):
            h = header(f)
            print(f"\n== patient {p} CT phase {ph}: {f.name}")
            print(json.dumps({k: h[k] for k in ("size_xyz", "spacing_xyz", "origin_xyz", "direction", "pixel_type")}))
            s = ct_stats(f)
            print("intensity:", json.dumps(s))
            report.append({"kind": "ct", "patient": p, "phase": ph, "header": h, "stats": s})
        for f in mask_files:
            e = parse_mask_name(f"{f.parent.name}/{f.name}")
            h = header(f)
            s = mask_stats(f)
            print(f"\n== patient {p} MASK phase {e['phase']} instance {e['instance']}: {f.name}")
            print(json.dumps({k: h[k] for k in ("size_xyz", "spacing_xyz", "origin_xyz", "direction", "pixel_type")}))
            print("labels:", json.dumps(s))
            if e["phase"] in cts:
                ok, why = same_grid(h, header(cts[e["phase"]]))
                print(f"geometry vs CT phase {e['phase']}: {'SAME GRID' if ok else 'DIFFERENT: ' + '; '.join(why)}")
            else:
                ok, why = None, ["no CT of that phase in extracted set"]
                print("geometry vs CT:", why[0])
            report.append({"kind": "mask", "patient": p, **e, "header": h, "stats": s,
                           "same_grid_as_ct": ok, "grid_diff": why})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {a.out}")


# ----------------------------------------------------------------------------- manifest
def ct_header_from_zip(zf, member):
    """CT geometry from the first 348 bytes of the gzip'd NIfTI inside the ZIP (no extraction).

    Returned in SimpleITK's convention (LPS, x/y/z) so it is directly comparable with NRRD
    masks read by SimpleITK. Also reports scl_slope/scl_inter: if they are not 1/0 the stored
    values would not be HU.
    """
    import nibabel as nib
    with zf.open(member) as f, gzip.GzipFile(fileobj=f) as g:
        h = nib.Nifti1Header.from_fileobj(g)
    shape = h.get_data_shape()
    if len(shape) != 3:
        raise ValueError(f"{member}: expected 3D, got shape {shape}")
    A = h.get_best_affine()
    zooms = np.array(h.get_zooms()[:3], dtype=float)
    flip = np.diag([-1.0, -1.0, 1.0])  # RAS -> LPS
    slope, inter = h.get_slope_inter()
    return {
        "size_xyz": [int(x) for x in shape], "spacing_xyz": zooms.tolist(),
        "origin_xyz": (flip @ A[:3, 3]).tolist(), "direction": (flip @ (A[:3, :3] / zooms)).flatten().tolist(),
        "pixel_type": str(h.get_data_dtype()),
        "scl_slope": None if slope is None or np.isnan(slope) else float(slope),
        "scl_inter": None if inter is None or np.isnan(inter) else float(inter),
    }


def union_mask(arrays):
    """Binary union (uint8) of instance masks; also returns the number of voxels covered by >1 instance."""
    stack = np.stack([np.asarray(a) > 0 for a in arrays])
    return stack.any(0).astype(np.uint8), int((stack.sum(0) > 1).sum())


def mask_array(path):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _crc_key(info):
    return (info.CRC, info.file_size)


def build_manifest(raw_dir=None, extract_dir=None, patients=None, ct_archives=None):
    """Match masks to CTs by exact (patient, phase), verify grids, and return (inventory, excluded) DataFrames."""
    import pandas as pd
    import download_data as D
    raw, ext = Path(raw_dir or C.RAW_DIR), Path(extract_dir or C.EXTRACT_DIR)
    ct_zips = [raw / a for a in (ct_archives or C.CT_ARCHIVES)]
    missing = [z.name for z in ct_zips if not z.exists()]
    if missing:
        raise FileNotFoundError(f"configured CT archive(s) not found in {raw}: {missing}")
    mask_zip = raw / "tumor_masks_wawtace_v1_08_05_2024.zip"
    meta = pd.read_csv(raw / "ct_hcc_metadata_v2.csv")
    csv_idx = {(int(r.patient_id), int(r.ct_phase)): r for r in meta.itertuples()}
    excluded = []

    def exc(patient, phase, item, reason, detail=""):
        excluded.append({"patient_id": patient, "phase": phase, "item": item, "reason": reason, "detail": detail})

    ct_hdr = {}
    for ct_zip in ct_zips:
        cts, odd_ct = parse_archive_names(ct_zip, parse_ct_name)
        for name in odd_ct:
            exc(None, None, f"{ct_zip.name}:{name}", "nonconforming_ct_name")
        with zipfile.ZipFile(ct_zip) as zc:
            for e in cts:
                if patients is not None and e["patient"] not in patients:
                    continue
                key = (e["patient"], e["phase"])
                if key in ct_hdr:
                    exc(*key, f"{ct_zip.name}:{e['member']}", "duplicate_ct_for_patient_phase",
                        f"already seen in {ct_hdr[key][0]}; later copy ignored")
                    continue
                ct_hdr[key] = (ct_zip.name, e["member"], ct_header_from_zip(zc, e["member"]))

    masks, odd_m = parse_archive_names(mask_zip, parse_mask_name)
    with zipfile.ZipFile(mask_zip) as zm:
        infos = {i.filename: i for i in zm.infolist()}
    for name in odd_m:
        sib = re.sub(r"\.nrrd$", "_tumor_seg.nrrd", name)
        same = sib in infos and _crc_key(infos[sib]) == _crc_key(infos[name])
        exc(None, None, name, "nonconforming_mask_name",
            f"identical bytes (CRC32+size) to {sib}" if same else f"sibling {sib} present={sib in infos}, not identical")
    groups = defaultdict(list)
    for e in sorted(masks, key=lambda e: (e["patient"], e["phase"], e["instance"])):
        groups[(e["patient"], e["phase"])].append(e)

    for key, es in groups.items():
        if key not in ct_hdr and (patients is None or key[0] in patients):
            exc(*key, f"{len(es)} mask file(s)", "ct_not_in_downloaded_archive",
                "mask has no CT of the same patient+phase in the configured CT archives (not an error)")

    rows = []
    for key in sorted(ct_hdr):
        patient, phase = key
        archive, member, h = ct_hdr[key]
        r = csv_idx.get(key)
        row = {
            "patient_id": patient, "phase": phase, "phase_name": C.PHASES[phase], "ct_archive": archive, "ct_member": member,
            **{f"size_{a}": v for a, v in zip("xyz", h["size_xyz"])},
            **{f"spacing_{a}": v for a, v in zip("xyz", h["spacing_xyz"])},
            **{f"origin_{a}": v for a, v in zip("xyz", h["origin_xyz"])},
            "direction": json.dumps([round(x, 6) + 0.0 for x in h["direction"]]),
            "ct_dtype": h["pixel_type"], "scl_slope": h["scl_slope"], "scl_inter": h["scl_inter"],
            "csv_ct_file_name": getattr(r, "ct_file_name", None), "csv_tumor_count": getattr(r, "tumor_count", None),
            "csv_slice_thickness": getattr(r, "slice_thickness", None),
        }
        row["csv_thickness_matches_z"] = (None if r is None else bool(abs(r.slice_thickness - h["spacing_xyz"][2]) < 0.01))
        if r is None:
            exc(patient, phase, member, "ct_missing_from_metadata_csv")
        es, seen, used, arrays = groups.get(key, []), {}, [], []
        row["n_mask_files"] = len(es)
        reason = None
        for e in es:
            info = infos[e["member"]]
            if _crc_key(info) in seen:
                exc(patient, phase, e["member"], "duplicate_instance",
                    f"identical bytes (CRC32+size) to instance {seen[_crc_key(info)]}; dropped")
                continue
            seen[_crc_key(info)] = e["instance"]
            D.safe_extract(mask_zip, ext, e["member"])
            path = ext / e["member"]
            ok, why = same_grid(header(path), h)
            if not ok:
                reason = "mask_grid_mismatch: " + "; ".join(why)
                exc(patient, phase, e["member"], "mask_grid_mismatch", "; ".join(why))
                continue
            a = mask_array(path)
            labels = set(np.unique(a).tolist())
            if not labels <= {0, 1}:
                reason = f"unexpected_mask_labels {sorted(labels)}"
                exc(patient, phase, e["member"], "unexpected_mask_labels", str(sorted(labels)))
                continue
            if not a.any():
                exc(patient, phase, e["member"], "empty_instance", "dropped")
                continue
            used.append(e)
            arrays.append(a)
        row["instance_ids"] = json.dumps([e["instance"] for e in used])
        row["mask_members"] = json.dumps([e["member"] for e in used])
        row["n_instances_used"] = len(used)
        row["csv_count_matches_files"] = (None if r is None or not es else int(r.tumor_count) == len(es))
        if used:
            u, overlap = union_mask(arrays)
            vox = int(u.sum())
            row.update(instance_voxels=json.dumps([int(a.sum()) for a in arrays]), union_voxels=vox,
                       instance_overlap_voxels=overlap,
                       union_volume_ml=vox * float(np.prod(h["spacing_xyz"])) / 1000.0)
        if reason is None and not es:
            reason = "no_mask_for_phase (unannotated; NOT assumed tumor-negative)"
        elif reason is None and not used:
            reason = "all_instances_empty_or_duplicate"
        row["usable"], row["reason"] = reason is None and bool(used), reason or "ok"
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(excluded)


def cross_check_ct_header(ct_zip, member, extract_dir):
    """Verify the ZIP-header reader against SimpleITK on one real scan; raises on disagreement."""
    import download_data as D
    with zipfile.ZipFile(ct_zip) as zc:
        mine = ct_header_from_zip(zc, member)
    D.safe_extract(ct_zip, extract_dir, member)
    path = Path(extract_dir) / member
    ref = header(path)
    ok, why = same_grid(mine, ref)
    path.unlink()
    if not ok:
        raise RuntimeError(f"ZIP-header geometry disagrees with SimpleITK for {member}: {why}")
    return True


# ----------------------------------------------------------------------------- splits
def make_splits(patients, ratios=None, seed=None, subset_sizes=None):
    """Patient-level split + nested TRAIN subsets (prefixes of one seeded train ordering)."""
    ratios, seed = ratios or C.SPLIT_RATIOS, C.SEED if seed is None else seed
    subset_sizes = subset_sizes or C.TRAIN_SUBSET_SIZES
    ids = sorted({int(p) for p in patients})
    n = len(ids)
    order = [ids[i] for i in np.random.default_rng(seed).permutation(n)]
    n_test, n_val = round(ratios["test"] * n), round(ratios["val"] * n)
    if n >= 3:
        n_test, n_val = max(n_test, 1), max(n_val, 1)
    test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    notes, subsets = [], {}
    for k in subset_sizes:
        size = len(train) if k == "all" else int(k)
        if size > len(train):
            notes.append(f"requested train subset {k} > {len(train)} available train patients; skipped")
            continue
        subsets["all" if k == "all" else str(k)] = train[:size]
    if "all" in subsets and str(len(train)) in subsets:
        notes.append(f"'all' equals {len(train)}")
    s = {"train": set(train), "val": set(val), "test": set(test)}
    assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"]), "patient leakage"
    assert len(train) + len(val) + len(test) == n
    return {"seed": seed, "ratios": ratios, "n_patients": n, "counts": {k: len(v) for k, v in s.items()},
            "train": train, "val": val, "test": test, "train_subsets": subsets, "notes": notes}


# ----------------------------------------------------------------------------- QC figures
def cmd_manifest(a):
    import pandas as pd
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv, exc_df = build_manifest(patients=set(a.patients) if a.patients else None)
    inv.to_csv(out / "inventory.csv", index=False)
    exc_df.to_csv(out / "excluded.csv", index=False)
    for arc, g in inv.groupby("ct_archive"):
        cross_check_ct_header(C.RAW_DIR / arc, g.iloc[0]["ct_member"], C.EXTRACT_DIR)
        print(f"header cross-check vs SimpleITK OK on {arc}:{g.iloc[0]['ct_member']}")
    print("scans per archive:", dict(inv.ct_archive.value_counts()))
    use = inv[inv.usable]
    print(f"CT scans in archive: {len(inv)}; usable patient-phase-mask tuples: {len(use)}; "
          f"patients: {use.patient_id.nunique()}")
    print("not usable:", dict(inv[~inv.usable].reason.str.split(" ").str[0].value_counts()))
    print("excluded-log reasons:", dict(exc_df.reason.value_counts()) if len(exc_df) else {})
    print("usable by phase:", dict(use.phase_name.value_counts()))
    print("CSV tumor_count == #mask files (annotated phases):",
          dict(inv[inv.n_mask_files > 0].csv_count_matches_files.value_counts(dropna=False)))
    print("CSV slice_thickness == header z-spacing:", dict(inv.csv_thickness_matches_z.value_counts(dropna=False)))
    print("CT dtypes:", dict(inv.ct_dtype.value_counts()), "| scl_slope/inter values:",
          sorted({(r.scl_slope, r.scl_inter) for r in inv.itertuples()}, key=str))
    print(f"wrote {out}/inventory.csv and excluded.csv")


def cmd_splits(a):
    import pandas as pd
    out = Path(a.out_dir)
    inv = pd.read_csv(out / "inventory.csv")
    pats = inv[inv.usable].patient_id.unique()
    sp = make_splits(pats)
    (out / "splits.json").write_text(json.dumps(sp, indent=1))
    rows = []
    for name in ("train", "val", "test"):
        for p in sp[name]:
            first = next((int(k) for k, v in sp["train_subsets"].items() if k != "all" and p in v), None) \
                if name == "train" else None
            rows.append({"patient_id": p, "split": name, "smallest_nested_train_subset": first})
    pd.DataFrame(rows).sort_values("patient_id").to_csv(out / "splits.csv", index=False)
    print(f"seed {sp['seed']}, ratios {sp['ratios']}, patients {sp['n_patients']}: {sp['counts']}")
    print("nested train subsets:", {k: len(v) for k, v in sp["train_subsets"].items()}, sp["notes"])


def cmd_qc(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import SimpleITK as sitk
    import download_data as D
    out = Path(a.out_dir)
    inv = pd.read_csv(out / "inventory.csv")
    use = inv[inv.usable]
    pick = use.sample(n=min(a.n, len(use)), random_state=C.SEED)
    cols = 4
    fig, axes = plt.subplots((len(pick) + cols - 1) // cols, cols, figsize=(4 * cols, 4 * ((len(pick) + cols - 1) // cols)))
    axes = np.atleast_1d(axes).ravel()
    lo, hi = C.HU_WINDOW
    for ax, (_, r) in zip(axes, pick.iterrows()):
        D.safe_extract(C.RAW_DIR / r.ct_archive, C.EXTRACT_DIR, r.ct_member)
        ct_path = C.EXTRACT_DIR / r.ct_member
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path)))
        ct_path.unlink()
        u, _ = union_mask([mask_array(C.EXTRACT_DIR / m) for m in json.loads(r.mask_members)])
        z = int(np.argmax(u.sum((1, 2))))
        ax.imshow(np.clip(ct[z], lo, hi), cmap="gray", vmin=lo, vmax=hi)
        ax.contour(u[z], levels=[0.5], colors="red", linewidths=0.8)
        ys, xs = np.nonzero(u[z])
        pad = 40
        ax.set_xlim(max(xs.min() - pad, 0), min(xs.max() + pad, ct.shape[2]))
        ax.set_ylim(min(ys.max() + pad, ct.shape[1]), max(ys.min() - pad, 0))
        ax.set_title(f"p{r.patient_id} {r.phase_name} z={z}/{ct.shape[0]} n={r.n_instances_used}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"Mask (red) on its own CT phase, window {lo:.0f}..{hi:.0f} HU, array order (not radiological)")
    fig.tight_layout()
    fig.savefig(out / "qc_overlays.png", dpi=110)
    fig2, ax2 = plt.subplots(1, 3, figsize=(13, 3.5))
    ax2[0].hist(use.union_volume_ml.dropna(), bins=np.geomspace(max(use.union_volume_ml.min(), 0.01), use.union_volume_ml.max(), 20))
    ax2[0].set_xscale("log"); ax2[0].set_xlabel("union tumor volume (mL)")
    ax2[1].hist(use.spacing_z, bins=15); ax2[1].set_xlabel("z spacing (mm)")
    ax2[2].hist(use.n_instances_used, bins=np.arange(0.5, use.n_instances_used.max() + 1.5)); ax2[2].set_xlabel("instances per tuple")
    fig2.tight_layout()
    fig2.savefig(out / "qc_summary.png", dpi=110)
    print(f"wrote {out}/qc_overlays.png and qc_summary.png")


# ----------------------------------------------------------------------------- diagnostics (read-only)
def header_from_row(r):
    return {"size_xyz": [int(r.size_x), int(r.size_y), int(r.size_z)],
            "spacing_xyz": [r.spacing_x, r.spacing_y, r.spacing_z],
            "origin_xyz": [r.origin_x, r.origin_y, r.origin_z], "direction": json.loads(r.direction)}


def resample_mask_to_ct(mask_img, ct_img):
    """Nearest-neighbour resample of a mask into the CT grid through physical space (identity transform)."""
    import SimpleITK as sitk
    return sitk.Resample(mask_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask_img.GetPixelID())


def count_components(binary):
    """Number of 26-connected components of a 3D binary array and the fraction of voxels in the largest."""
    from scipy import ndimage
    lab, n = ndimage.label(binary > 0, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(sizes.max() / sizes.sum())


def cmd_diagnose(a):
    """Explain WHY annotated cases were excluded (read-only; the inventory is never modified)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import SimpleITK as sitk
    import download_data as D
    out = Path(a.out_dir)
    inv = pd.read_csv(out / "inventory.csv")
    mask_zip = C.RAW_DIR / "tumor_masks_wawtace_v1_08_05_2024.zip"
    masks, _ = parse_archive_names(mask_zip, parse_mask_name)
    by_key = defaultdict(list)
    for e in masks:
        by_key[(e["patient"], e["phase"])].append(e)

    # 1) connected components of every USABLE instance (matters for per-lesion box prompts)
    comp_rows = []
    for r in inv[inv.usable].itertuples():
        for m in json.loads(r.mask_members):
            n, frac = count_components(mask_array(C.EXTRACT_DIR / m))
            comp_rows.append({"patient_id": r.patient_id, "phase": r.phase, "mask_member": m,
                              "n_components": n, "largest_component_fraction": round(frac, 4)})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "diagnose_components.csv", index=False)
    multi = comp[comp.n_components > 1]
    print(f"usable instances: {len(comp)}; with >1 connected component: {len(multi)}; "
          f"components per instance: {dict(comp.n_components.value_counts().sort_index())}")

    # 2) every annotated-but-unusable case
    bad = inv[(inv.n_mask_files > 0) & (~inv.usable)]
    rows, panels = [], []
    for r in bad.itertuples():
        ct_hdr = header_from_row(r)
        siblings = inv[inv.patient_id == r.patient_id]
        for e in by_key[(r.patient_id, r.phase)]:
            D.safe_extract(mask_zip, C.EXTRACT_DIR, e["member"])
            path = C.EXTRACT_DIR / e["member"]
            mimg = sitk.ReadImage(str(path))
            arr = sitk.GetArrayFromImage(mimg)
            vals, cnts = np.unique(arr, return_counts=True)
            h = header(path)
            same_phase, why = same_grid(h, ct_hdr)
            matches = [int(s.phase) for s in siblings.itertuples() if same_grid(h, header_from_row(s))[0]]
            row = {"patient_id": r.patient_id, "phase": r.phase, "mask_member": e["member"],
                   "labels": json.dumps({int(v): int(c) for v, c in zip(vals, cnts)}),
                   "grid_matches_filename_phase": same_phase, "grid_matches_any_phase_of_patient": json.dumps(matches),
                   "grid_diff": "; ".join(why), "mask_size_xyz": json.dumps(h["size_xyz"]),
                   "ct_size_xyz": json.dumps(ct_hdr["size_xyz"]), "mask_spacing_xyz": json.dumps(np.round(h["spacing_xyz"], 4).tolist()),
                   "ct_spacing_xyz": json.dumps(np.round(ct_hdr["spacing_xyz"], 4).tolist())}
            plot = None
            if not same_phase:
                res = sitk.GetArrayFromImage(resample_mask_to_ct(mimg, _ct_ref(r, D)))
                v0 = float((arr > 0).sum()) * float(np.prod(h["spacing_xyz"])) / 1000
                v1 = float((res > 0).sum()) * float(np.prod(ct_hdr["spacing_xyz"])) / 1000
                row.update(mask_volume_ml=round(v0, 2), resampled_volume_ml=round(v1, 2),
                           volume_ratio=round(v1 / v0, 3) if v0 else None)
                plot = ("resampled", res)
            elif set(np.unique(arr).tolist()) - {0, 1}:
                plot = ("labels", arr)
            rows.append(row)
            if plot:
                panels.append((r, e, plot))
    pd.DataFrame(rows).to_csv(out / "diagnose_unusable.csv", index=False)
    print(pd.DataFrame(rows)[["patient_id", "phase", "labels", "grid_matches_filename_phase",
                              "grid_matches_any_phase_of_patient"]].to_string(index=False))

    # figure: one panel per mask file, on its own phase CT
    cols = 4
    nrows = max((len(panels) + cols - 1) // cols, 1)
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    lo, hi = C.HU_WINDOW
    last_key, ct = None, None
    for ax, (r, e, (kind, arr)) in zip(axes, panels):
        if last_key != r.ct_member:
            ct = sitk.GetArrayFromImage(_ct_ref(r, D))
            last_key = r.ct_member
        pos = arr > 0
        z = int(np.argmax(pos.sum((1, 2)))) if pos.any() else ct.shape[0] // 2
        ax.imshow(np.clip(ct[z], lo, hi), cmap="gray", vmin=lo, vmax=hi)
        if kind == "labels":
            ax.contour(arr[z] == 1, levels=[0.5], colors="red", linewidths=0.8)
            ax.contour(arr[z] == 2, levels=[0.5], colors="cyan", linewidths=0.8)
            ttl = "red=1 cyan=2"
        else:
            ax.contour(arr[z] > 0, levels=[0.5], colors="yellow", linewidths=0.8)
            ttl = "mask resampled to CT grid"
        if pos.any():
            ys, xs = np.nonzero(pos[z])
            ax.set_xlim(max(xs.min() - 40, 0), min(xs.max() + 40, ct.shape[2]))
            ax.set_ylim(min(ys.max() + 40, ct.shape[1]), max(ys.min() - 40, 0))
        ax.set_title(f"p{r.patient_id} {r.phase_name} inst{e['instance']} z={z}\n{ttl}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "diagnose_cases.png", dpi=100)
    print(f"wrote diagnose_components.csv, diagnose_unusable.csv, diagnose_cases.png in {out}")


def _ct_ref(r, D):
    import SimpleITK as sitk
    D.safe_extract(C.RAW_DIR / r.ct_archive, C.EXTRACT_DIR, r.ct_member)
    path = C.EXTRACT_DIR / r.ct_member
    img = sitk.ReadImage(str(path))
    path.unlink()
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("names").set_defaults(fn=cmd_names)
    s = sub.add_parser("inspect")
    s.add_argument("--patients", type=int, nargs="+", default=[10, 13])
    s.add_argument("--out", default=str(C.OUTPUT_DIR / "inspect_report.json"))
    s.set_defaults(fn=cmd_inspect)
    for name, fn in (("manifest", cmd_manifest), ("splits", cmd_splits), ("qc", cmd_qc), ("diagnose", cmd_diagnose)):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", default=str(C.MANIFEST_DIR))
        if name == "manifest":
            s.add_argument("--patients", type=int, nargs="+", help="debug: restrict to these patients")
        if name == "qc":
            s.add_argument("--n", type=int, default=12)
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
