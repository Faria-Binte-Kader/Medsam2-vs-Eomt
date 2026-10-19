# CAP 6545: MedSAM2 (oracle-box prompted) vs. EoMT on WAW-TACE liver tumor CT

The MedSAM2 baseline gets a box derived from the ground-truth mask, so it is a *prompted* result,
not a fully automatic one.

## Files
- `config.py` – paths, seed, device, CT window, split ratios, MedSAM2 pins, prompt perturbations
- `fetch_medsam2.py` – clones MedSAM2 at the pinned commit, downloads and SHA-256-checks `MedSAM2_latest.pt`
- `download_data.py` – WAW-TACE downloader: Zenodo API URLs/MD5s, resumable, retried, atomic, disk check, safe selective ZIP extraction
- `process_data.py` – archive inspection, manifest/matching, splits, QC figures (preprocessing to be added)
- `run_medsam2.py` – zero-shot MedSAM2 evaluation (later step)
- `tests/`

## Setup (two conda envs, kept separate)
MedSAM2's `setup.py` needs numpy>=2.0.1 and torch>=2.5.1, so EoMT gets its own env later.

```bash
# MedSAM2 env (versions are the ones MedSAM2 documents; use a CUDA wheel index matching `nvidia-smi`)
conda create -n medsam2 python=3.12 -y && conda activate medsam2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy>=2.0.1 SimpleITK nibabel pandas scipy scikit-image matplotlib requests tqdm pytest hydra-core iopath pillow
python fetch_medsam2.py            # add --dry-run to preview
pip install -e MedSAM2             # prefix SAM2_BUILD_CUDA=0 to skip the optional CUDA extension
python -m pytest -q tests

# EoMT env (later phase; pins to be decided after reading https://github.com/tue-mps/eomt)
conda create -n eomt python=3.10 -y
```

Paths can be changed in `config.py` or with env vars `WAWTACE_RAW_DIR`, `WAWTACE_CACHE_DIR`,
`WAWTACE_MODELS_DIR`, `WAWTACE_OUTPUT_DIR`, `MEDSAM2_DIR`.

## Download data
```bash
python download_data.py --dry-run          # URLs, sizes, MD5s, disk check; downloads nothing
python download_data.py                    # metadata CSV + tumor masks ZIP + CT archives 1/4 and 2/4 (~20.9 GB)
python download_data.py --list ct_scans_1_4_wawtace_09_05_24.zip       # inspect members
python download_data.py --extract tumor_masks_wawtace_v1_08_05_2024.zip
python download_data.py --extract ct_scans_1_4_wawtace_09_05_24.zip --pattern "<fnmatch pattern>"
```

## Inventory, splits, QC
```bash
python process_data.py manifest      # manifests/inventory.csv + excluded.csv
python process_data.py splits        # manifests/splits.json + splits.csv
python process_data.py qc --n 12     # manifests/qc_overlays.png + qc_summary.png
python process_data.py diagnose      # read-only: why annotated cases were excluded + lesion components
```

## Results so far (CT archives 1/4 + 2/4, seed 42, 70/15/15)
- 411 CT scans; **96 usable patient-phase-mask tuples** (96 patients): arterial 46, portal 40, delayed 9, native 1. 173 lesion instances (1 to 6 per tuple).
- Split: **68 train / 14 val / 14 test**. Test: arterial 8, portal 4, delayed 2. Nested train subsets: 1, 2, 5, 10, 20, 68.
- Union tumor volume: median 38 mL, range 1.4 to 942 mL (<10 mL: 11, 10-50 mL: 44, >=50 mL: 41 tuples).
- 17 annotated tuples excluded: 14 mask-grid mismatches (not resampled because alignment cannot be verified; patients 24, 26, 27, 31, 44, 53, 65, 84, 86, 88, 92, 96, 99, 158) and 3 with extra mask label values (patients 72, 119: {0, 2}; 121: {0, 1, 2}). The meaning of label 2 is not documented in what we have, so it is not remapped.
- Also logged: patient 7 duplicate instance dropped, stray `63_1_0.nrrd` (identical bytes) ignored, 120 masks whose CT is in archives 3/4 or 4/4.
- 298 scans have no mask for their phase: unannotated, not treated as tumor-negative.
- 83+ scans store `scl_inter = -1024`; read CTs only through SimpleITK (it applies the scaling).
- 24 QC overlays (two runs) checked by eye: masks sit on lesions.

Not done yet: preprocessing cache, MedSAM2 inference, metrics. Nothing from MedSAM2 has been run.
