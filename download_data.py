"""WAW-TACE downloader (Zenodo record 12741586).

File URLs, sizes and MD5s come from the Zenodo API, not from hard-coded links.
Downloads are streamed, resumable (HTTP Range), retried, atomic (.part -> final)
and MD5-verified. ZIPs are kept intact; members can be listed or selectively extracted.

  python download_data.py --dry-run
  python download_data.py                                    # metadata + masks + CT archive 1/4
  python download_data.py --files ct_hcc_metadata_v2.csv
  python download_data.py --list ct_scans_1_4_wawtace_09_05_24.zip
  python download_data.py --extract tumor_masks_wawtace_v1_08_05_2024.zip
  python download_data.py --extract ct_scans_1_4_wawtace_09_05_24.zip --pattern "*PATIENT*"
"""
import argparse
import fnmatch
import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

import config as C

API = "https://zenodo.org/api/records/{}"
CHUNK = 1 << 20
TIMEOUT = (15, 60)  # connect, read (seconds)
RETRIES = 6


class DownloadError(RuntimeError):
    pass


def get_record_files(record_id=C.ZENODO_RECORD, session=requests):
    """Return {filename: {size, md5, url}} from the Zenodo API."""
    r = session.get(API.format(record_id), timeout=TIMEOUT)
    r.raise_for_status()
    out = {}
    for f in r.json()["files"]:
        key = f.get("key") or f.get("filename")
        checksum = f["checksum"]
        out[key] = {
            "size": int(f["size"]),
            "md5": checksum.split(":", 1)[-1],
            "url": f["links"]["self"],
        }
    return out


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def check_disk(dest_dir, needed_bytes, margin=C.DISK_MARGIN):
    dest_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    if free < needed_bytes * margin:
        raise DownloadError(
            f"not enough disk in {dest_dir}: need {needed_bytes * margin / 1e9:.2f} GB "
            f"(incl. {margin}x margin), have {free / 1e9:.2f} GB"
        )


def _fetch_once(url, part, size):
    """Stream into `part`, resuming from its current size. Raises on network errors."""
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as r:
        if have and r.status_code == 200:  # server ignored Range: restart from scratch
            have = 0
        elif r.status_code not in (200, 206):
            r.raise_for_status()
            raise DownloadError(f"unexpected HTTP status {r.status_code}")
        mode = "ab" if have else "wb"
        with open(part, mode) as f, tqdm(total=size, initial=have, unit="B", unit_scale=True,
                                         desc=part.name.removesuffix(".part")) as bar:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
                bar.update(len(chunk))


def download_file(name, info, dest_dir=C.RAW_DIR, retries=RETRIES, backoff=2.0):
    """Download one file; return its final path. Safe to re-run (skips verified files, resumes .part)."""
    dest = dest_dir / name
    part = dest.with_name(name + ".part")
    if dest.exists():
        if dest.stat().st_size == info["size"] and md5_file(dest) == info["md5"]:
            print(f"[ok] {name} already present and MD5-verified")
            return dest
        raise DownloadError(f"{dest} exists but size/MD5 do not match the record; move it away and re-run")

    have = part.stat().st_size if part.exists() else 0
    if have > info["size"]:
        part.unlink()
        have = 0
    check_disk(dest_dir, info["size"] - have)

    for attempt in range(1, retries + 1):
        try:
            _fetch_once(info["url"], part, info["size"])
            break
        except (requests.RequestException, OSError) as e:
            if attempt == retries:
                raise DownloadError(
                    f"{name}: failed after {retries} attempts ({e}). Partial data kept at {part}; "
                    "re-run the same command to resume."
                ) from e
            wait = backoff ** attempt
            print(f"[retry {attempt}/{retries}] {name}: {e}; resuming in {wait:.0f}s")
            time.sleep(wait)

    size = part.stat().st_size
    if size != info["size"]:
        raise DownloadError(f"{name}: size {size} != expected {info['size']}; .part kept, re-run to resume")
    if md5_file(part) != info["md5"]:
        part.unlink()
        raise DownloadError(f"{name}: MD5 mismatch; corrupt .part deleted, re-run to download again")
    part.replace(dest)
    print(f"[done] {dest} (MD5 verified)")
    return dest


def list_members(archive):
    with zipfile.ZipFile(archive) as z:
        return [(i.filename, i.file_size) for i in z.infolist() if not i.is_dir()]


def safe_extract(archive, out_dir, pattern="*"):
    """Extract only members matching `pattern`; reject any path escaping out_dir. Returns written paths."""
    out_dir = Path(out_dir).resolve()
    written = []
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if not i.is_dir() and fnmatch.fnmatch(i.filename, pattern)]
        check_disk(out_dir, sum(i.file_size for i in members))
        for i in members:
            target = (out_dir / i.filename).resolve()
            if out_dir != target and out_dir not in target.parents:
                raise DownloadError(f"unsafe path in archive, refusing to extract: {i.filename!r}")
        for i in members:
            target = out_dir / i.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK)
            written.append(target)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=C.DEFAULT_FILES, help="record file names to download")
    ap.add_argument("--dry-run", action="store_true", help="show URLs, sizes, MD5s and disk check; download nothing")
    ap.add_argument("--list", metavar="ZIP", help="list members of a downloaded ZIP")
    ap.add_argument("--extract", metavar="ZIP", help="extract members of a downloaded ZIP into EXTRACT_DIR")
    ap.add_argument("--pattern", default="*", help="fnmatch pattern for --extract (default: all members)")
    a = ap.parse_args()

    if a.list or a.extract:
        z = C.RAW_DIR / (a.list or a.extract)
        if a.list:
            for name, size in list_members(z):
                print(f"{size:>14,}  {name}")
        else:
            paths = safe_extract(z, C.EXTRACT_DIR, a.pattern)
            print(f"extracted {len(paths)} files to {C.EXTRACT_DIR}")
        return

    record = get_record_files()
    missing = [f for f in a.files if f not in record]
    if missing:
        sys.exit(f"not in Zenodo record {C.ZENODO_RECORD}: {missing}\navailable: {sorted(record)}")
    total = sum(record[f]["size"] for f in a.files)
    print(f"Zenodo record {C.ZENODO_RECORD}; destination {C.RAW_DIR}")
    for f in a.files:
        r = record[f]
        print(f"  {f}  {r['size'] / 1e9:.3f} GB  md5={r['md5']}\n    {r['url']}")
    free = shutil.disk_usage(C.RAW_DIR if C.RAW_DIR.exists() else C.RAW_DIR.anchor).free
    print(f"total {total / 1e9:.2f} GB (need >= {total * C.DISK_MARGIN / 1e9:.2f} GB free), free {free / 1e9:.2f} GB")
    if a.dry_run:
        print("dry run: nothing downloaded")
        return
    for f in a.files:
        download_file(f, record[f])


if __name__ == "__main__":
    try:
        main()
    except DownloadError as e:
        sys.exit(f"ERROR: {e}")
