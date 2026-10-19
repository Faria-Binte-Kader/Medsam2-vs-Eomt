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
MANIFEST_DIR = _p("WAWTACE_MANIFEST_DIR", "manifests")  # small CSV/JSON files, safe to commit
MEDSAM2_DIR = _p("MEDSAM2_DIR", "MedSAM2")

SEED = 42
DEVICE = "cuda"  # "cuda" | "cpu"
BATCH_SIZE = 1

ZENODO_RECORD = "12741586"
CT_ARCHIVES = [
    "ct_scans_1_4_wawtace_09_05_24.zip",
    "ct_scans_2_4_wawtace_09_05_24.zip",
]
DEFAULT_FILES = [
    "ct_hcc_metadata_v2.csv",
    "tumor_masks_wawtace_v1_08_05_2024.zip",
    *CT_ARCHIVES,
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

# SECONDARY evaluation set: annotated cases that fail the strict primary checks but were reviewed
# by eye (manifests/diagnose_cases.png). Reported separately; never used for training, validation
# or model selection, and not part of the frozen split. Flags (joined with "+"):
#   label_remap: any nonzero mask value = tumor (patients whose only tumor label is 2)
#   resample:    mask drawn on another grid; nearest-neighbour resample through physical space onto its
#                own phase's CT grid (volume must stay within +-10 percent)
SECONDARY_RULES = {
    72: "label_remap", 119: "label_remap",
    24: "resample", 26: "resample", 44: "resample", 53: "resample", 86: "resample", 96: "resample",
    99: "resample", 158: "resample", 84: "resample+label_remap", 92: "resample+label_remap",
}
RESAMPLE_VOLUME_TOLERANCE = 0.10

# Preprocessing. The window is applied only to volumes that pass the HU audit (see process_data.py preprocess).
PREPROCESS_VERSION = 1
HU_AUDIT = {  # heuristic plausibility bounds, used to flag (not silently fix) volumes
    "air_percentile_max": -900.0,      # 0.5th percentile of the volume should reach air
    "hu_min_floor": -3100.0, "hu_max_ceiling": 4100.0,
    "tumor_median_range": (-50.0, 250.0),
}

# MedSAM2 oracle-localization prompting (see run_medsam2.py). Fixed before any test-set run.
PROMPT = {
    "lesion_unit": "connected_component_per_instance",  # 26-connected components of each instance mask
    "key_slice": "largest_gt_area_slice",               # axial slice where that lesion's GT area is largest
    "box": "tight in-plane extent of the lesion on the key slice, [x_min, y_min, x_max, y_max], inclusive pixel indices",
    "propagation": "box on the key slice, forward then reverse (as upstream medsam2_infer_3D_CT.py); z range per z_extent",
    "z_extent": "unbounded",  # default z mode, see Z_MODES
    "postprocess": "largest_cc",                        # per-lesion largest 3D component (as upstream); no GT used
    "aggregation": "union of per-lesion predictions",
    "box_units": "native in-plane pixels (x = columns, y = rows)",
}
# z_extent modes. "unbounded": propagate through the whole volume (upstream-literal; the model decides where the lesion ends).
# "gt_extent": the lesion's ground-truth z range (first/last slice containing it) bounds the propagation itself via the
# predictor's max_frame_num_to_track, i.e. the oracle prompt is a 3D extent (key-slice box + z range). Both modes are
# declared up front and reported separately; neither is chosen from test results.
Z_MODES = ("unbounded", "gt_extent")
SIZE_BINS_ML = (10.0, 50.0)   # predeclared tumor-size bins: <10, 10-50, >=50 mL
BOOTSTRAP = {"n": 2000, "seed": SEED}
