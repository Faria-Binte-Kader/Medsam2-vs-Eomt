"""Clone MedSAM2 at the pinned commit and download the verified checkpoint.

Usage: python fetch_medsam2.py [--dry-run]
"""
import argparse
import hashlib
import subprocess
import sys

import requests

import config as C


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clone(dry):
    m = C.MEDSAM2
    print(f"[repo] {m['repo']} @ {m['commit']} -> {C.MEDSAM2_DIR}")
    if dry:
        return
    if not C.MEDSAM2_DIR.exists():
        subprocess.run(["git", "clone", m["repo"], str(C.MEDSAM2_DIR)], check=True)
    subprocess.run(["git", "-C", str(C.MEDSAM2_DIR), "checkout", "--quiet", m["commit"]], check=True)


def checkpoint(dry):
    m = C.MEDSAM2
    url = f"https://huggingface.co/{m['hf_repo']}/resolve/{m['hf_revision']}/{m['checkpoint']}"
    dest = C.MODELS_DIR / m["checkpoint"]
    print(f"[ckpt] {url} -> {dest}")
    if dry:
        return
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == m["size_bytes"] and sha256(dest) == m["sha256"]:
        print("[ckpt] already present and verified")
        return
    part = dest.with_suffix(".part")
    with requests.get(url, stream=True, timeout=(15, 60)) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if part.stat().st_size != m["size_bytes"] or sha256(part) != m["sha256"]:
        part.unlink()
        sys.exit("checkpoint size/SHA-256 mismatch; removed. No fallback to other weights.")
    part.replace(dest)
    print("[ckpt] verified OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    clone(a.dry_run)
    checkpoint(a.dry_run)
