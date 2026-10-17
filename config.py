"""Project settings. Edit values here; paths can also be set via environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _p(env, default):
    return Path(os.environ.get(env, ROOT / default)).expanduser().resolve()


RAW_DIR = _p("WAWTACE_RAW_DIR", "data/raw")
EXTRACT_DIR = _p("WAWTACE_EXTRACT_DIR", "data/extracted")
CACHE_DIR = _p("WAWTACE_CACHE_DIR", "data/cache")
MODELS_DIR = _p("WAWTACE_MODELS_DIR", "checkpoints")
OUTPUT_DIR = _p("WAWTACE_OUTPUT_DIR", "results")
MEDSAM2_DIR = _p("MEDSAM2_DIR", "MedSAM2")

SEED = 42
DEVICE = "cuda"  # "cuda" | "cpu"
BATCH_SIZE = 1

ZENODO_RECORD = "12741586"
DEFAULT_FILES = [
    "ct_hcc_metadata_v2.csv",
    "tumor_masks_wawtace_v1_08_05_2024.zip",
    "ct_scans_1_4_wawtace_09_05_24.zip",
]
DISK_MARGIN = 1.25  # require size * margin free before downloading
PHASES = {0: "native", 1: "arterial", 2: "portal", 3: "delayed"}

# Unverified default liver window (centre 60, width 400); check HU units before using.
HU_WINDOW = (-140.0, 260.0)
MASK_INTERPOLATION = "nearest"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRAIN_SUBSET_SIZES = [1, 2, 5, 10, 20, "all"]

MEDSAM2 = {
    "repo": "https://github.com/bowang-lab/MedSAM2",
    "commit": "332f30d420f1d1b08e2a79b3ae6a602458808383",
    "hf_repo": "wanglab/MedSAM2",
    "hf_revision": "e4a6f35edd7e091619cbc0750f462f1574e23955",
    "checkpoint": "MedSAM2_latest.pt",
    "size_bytes": 156040129,
    "sha256": "c92743b99f00d078bf32a3afcc38aaa9faf1c1692dffe3eaa7a90938c1991060",
    "model_cfg": "configs/sam2.1_hiera_t512.yaml",
    "input_size": 512,
}

# Prompt perturbations in in-plane voxels; seeded.
PROMPT_CONDITIONS = {
    "exact": dict(pad=0, translate=0, jitter=0),
    "mild": dict(pad=5, translate=3, jitter=2),
    "strong": dict(pad=15, translate=10, jitter=6),
}
