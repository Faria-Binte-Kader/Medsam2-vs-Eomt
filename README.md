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
```

## Results so far (archive 1/4 only; to be regenerated with archives 1/4 + 2/4, seed 42, 70/15/15)
- 205 CT scans from 56 patients; **48 usable patient-phase-mask tuples** (48 patients): arterial 22, portal 21, delayed 4, native 1.
- Split: 34 train / 7 val / 7 test. Nested train subsets: 1, 2, 5, 10, 20, 34.
- 8 annotated tuples excluded: 7 mask-grid mismatches (patients 24, 26, 27, 31, 44, 53, 65; not resampled because alignment cannot be verified) and 1 with mask labels {0, 2} (patient 72).
- Also logged: patient 7 duplicate instance dropped, stray `63_1_0.nrrd` (identical bytes) ignored, 177 masks whose CT is in the other archives.
- 149 scans have no mask for their phase: unannotated, not treated as tumor-negative.
- 83 scans store `scl_inter = -1024`; read CTs only through SimpleITK (it applies the scaling). After scaling the intensity range matches the other scans.
- 12 QC overlays checked by eye: masks sit on lesions.

Not done yet: preprocessing cache, MedSAM2 inference, metrics. Nothing from MedSAM2 has been run.
