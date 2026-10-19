"""WAW-TACE inspection, inventory, splits and preprocessing.

  python process_data.py names                      # parse archive member names (no pixel data read)
  python process_data.py inspect --patients 10 13   # read real headers/intensities, compare geometry

File-name conventions below were read from the real archive listings:
  CT   <p>/<p>_<phase>_scan.nii.gz
  mask <p>/<p>_<phase>_<instance>_tumor_seg.nrrd
Whether the phase digit in a mask name equals the phase digit in a scan name is a
hypothesis until `inspect` confirms it by geometry; names alone are never trusted.

Axis note: SimpleITK sizes/spacings/origins are (x, y, z); numpy arrays from
GetArrayFromImage are (z, y, x).
"""
import argparse
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
    cts, odd_c = parse_archive_names(C.RAW_DIR / "ct_scans_1_4_wawtace_09_05_24.zip", parse_ct_name)
    print(f"masks: {len(masks)} files, {len({e['patient'] for e in masks})} patients; non-conforming: {odd_m}")
    print(f"CT archive 1/4: {len(cts)} scans, {len({e['patient'] for e in cts})} patients; non-conforming: {odd_c[:5]}")
    print("mask phase counts:", dict(sorted(Counter(e["phase"] for e in masks).items())))
    print("CT phase counts:", dict(sorted(Counter(e["phase"] for e in cts).items())))
    ct_keys = {(e["patient"], e["phase"]) for e in cts}
    mask_keys = {(e["patient"], e["phase"]) for e in masks}
    print(f"(patient,phase) with BOTH a scan in archive 1/4 and a mask (by name only): {len(ct_keys & mask_keys)}")
    print(f"mask (patient,phase) keys whose CT is not in archive 1/4: {len(mask_keys - ct_keys)}")
    per = defaultdict(set)
    for e in masks:
        per[e["patient"]].add(e["phase"])
    print("patients with masks in more than one phase:", sum(len(v) > 1 for v in per.values()))


def cmd_inspect(a):
    out_dir = C.EXTRACT_DIR
    import download_data as D
    ct_zip = C.RAW_DIR / "ct_scans_1_4_wawtace_09_05_24.zip"
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
        D.safe_extract(ct_zip, out_dir, f"{p}/{p}_*")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("names").set_defaults(fn=cmd_names)
    s = sub.add_parser("inspect")
    s.add_argument("--patients", type=int, nargs="+", default=[10, 13])
    s.add_argument("--out", default=str(C.OUTPUT_DIR / "inspect_report.json"))
    s.set_defaults(fn=cmd_inspect)
    a = ap.parse_args()
    a.fn(a)
