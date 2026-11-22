"""Clone EoMT at the pinned commit and download the verified ADE20K-pretrained checkpoint.

Usage: python fetch_eomt.py [--dry-run] [--no-checkpoint]
"""
import argparse
import subprocess
import sys

import requests

import config as C
from fetch_medsam2 import sha256


def clone(dry):
    e = C.EOMT
    print(f"[repo] {e['repo']} @ {e['commit']} -> {C.EOMT_DIR}")
    if dry:
        return
    if not C.EOMT_DIR.exists():
        subprocess.run(["git", "clone", e["repo"], str(C.EOMT_DIR)], check=True)
    subprocess.run(["git", "-C", str(C.EOMT_DIR), "checkout", "--quiet", e["commit"]], check=True)


def checkpoint(dry):
    m = C.EOMT["ade20k_semantic_L512"]
    url = f"https://huggingface.co/{m['hf_repo']}/resolve/{m['hf_revision']}/{m['filename']}"
    dest = C.MODELS_DIR / "eomt_ade20k_semantic_L512.bin"
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
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args()
    clone(a.dry_run)
    if not a.no_checkpoint:
        checkpoint(a.dry_run)
