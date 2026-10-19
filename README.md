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
pip install --no-build-isolation -e MedSAM2   # compiles the optional CUDA extension against YOUR torch; without --no-build-isolation pip builds against a different torch and `_C.so` fails to load
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

## Secondary evaluation set (decided after `diagnose`)
The frozen split uses only the 96 strict (primary) patients. Twelve more annotated patients that failed the strict checks were reviewed by eye
(`manifests/diagnose_cases.png`) and form a **separate secondary set**, reported separately and never used for training, validation or model selection
(rules in `config.SECONDARY_RULES`, output in `manifests/secondary.csv`):
- 72, 119: the tumor mask uses label 2 instead of 1 (single-label files outlining a visible mass), so nonzero = tumor.
- 24, 26, 44, 53, 86, 96, 99, 158: mask drawn on a different grid; resampled (nearest neighbour, through physical space) onto its own phase's CT, volume change within 10%.
- 84, 92: both of the above.
Still excluded: 27, 31, 65, 88 (unverifiable or inconsistent grids) and 121 (one file with labels 1 and 2 of unknown meaning).

Findings worth reporting as limitations: the phase digit in a mask filename is not always the phase the mask was drawn on (in 11 of the 14 grid-mismatch patients the mask grid equals a different phase of the same patient), and 25 of the 96 primary CTs have a grid identical to another phase of the same patient, so geometry alone cannot prove which phase the mask was drawn on.

## Preprocessing
```bash
python process_data.py preprocess --limit 1          # one volume end to end (debug)
python process_data.py preprocess                    # all primary + secondary cases -> $WAWTACE_CACHE_DIR (default data/cache)
```
Per case it audits the HU range after SimpleITK applies the NIfTI scaling (air reaches about -1000, tumor median plausible, integer-valued), windows to uint8
(`config.HU_WINDOW`, default -140..260), caches `case.npz` (windowed CT, union mask, instance masks) and `meta.json` (source files, native geometry,
window, software versions, git commit), and writes `manifests/preprocess_audit.csv`. Volumes failing the audit are flagged and not cached.
The model-side input (gray to RGB, 512 resize, ImageNet normalisation) is built at inference time by `to_medsam2_input`, and predictions are mapped back
to the CT's spacing/origin/direction by `restore_to_native` / `write_native_mask`. A cache built with a different window or version is refused.

## MedSAM2 zero-shot, oracle-localization prompted (`run_medsam2.py`)
This is a **prompted** baseline: the only ground-truth information the model sees is the box prompt. Predictions are never corrected with ground truth.
```bash
python run_medsam2.py --split val --dry-run                 # prompts only, no model (writes results/medsam2/val_dryrun/prompts.jsonl)
python run_medsam2.py --split val --limit 1 --device cuda   # smoke test on one validation volume
python run_medsam2.py --split val --device cuda --hd95      # development on val
python run_medsam2.py --split test --device cuda            # final frozen test run (see guard below)
python run_medsam2.py --split secondary --device cuda       # secondary set, reported separately
```
Prompt policy (`config.PROMPT`, fixed before any test run): each lesion = a 26-connected component of an instance mask; key slice = axial slice
with that lesion's largest GT area; box = tight in-plane extent `[x_min, y_min, x_max, y_max]` (inclusive pixels) on that slice; the box is given on the
key slice and propagated through the whole volume forward, then in reverse (as upstream `medsam2_infer_3D_CT.py`); each lesion's prediction keeps its largest
3D component (as upstream); the volume prediction is the union over lesions. Conditions `exact`, `mild`, `strong` pad, translate and jitter the box in
pixels with a seed derived from (seed, patient, phase, lesion, condition); boxes are clipped to the image, and a box that becomes invalid is logged and
counts as a missed lesion (Dice 0), not dropped. Tumor-size bins are predeclared (`<10`, `10-50`, `>=50` mL).

Outputs in `results/medsam2/<run>/`: `volume_metrics.csv` (Dice, IoU, volumes, runtime, optional HD95), `lesion_metrics.jsonl`, `prompts.jsonl` (every box, units, seed),
`patient_summary.csv`, `summary.csv` and `summary_by_size.csv` (per-patient means, then mean/median/std and a seeded bootstrap 95% CI over patients), native-space masks
(`masks/`), overlays, `config_snapshot.json` (config, versions, git commit). Runs resume; a run directory refuses a different configuration.
**Test guard:** a full `--split test` run registers its configuration in `results/medsam2/test_registry.json`; later test runs with a changed window, prompt policy or model are
refused unless `--allow-test-config-change` is given. Debug runs (`--limit`, `--patients`) do not register. Develop on `val`.

## Verification status of the MedSAM2 pipeline
- Offline unit tests (66): prompt boxes (empty, border-touching), perturbation reproducibility/clipping/invalid handling, lesion components, metrics with known answers, native-space restoration, split disjointness, test-split guard, checkpoint refusal.
- Runner orchestration is tested with a stub predictor defined only in the tests (never used for results).
- The real pinned MedSAM2 code and checkpoint (SHA-256 verified) were run on CPU on a *synthetic* volume to check loading, box prompting, forward/reverse propagation and native-grid output. Its Dice is meaningless and is not a result.
- Not yet run: the pipeline on real WAW-TACE volumes, on GPU (bf16 autocast path), or any reported metric. No real MedSAM2 result exists yet.

### z-extent modes and the CUDA extension
`--z-mode unbounded` (default) propagates the key-slice box through the whole volume, as the upstream scripts do. `--z-mode gt_extent` also gives the lesion's
ground-truth z range, which bounds the propagation itself (`max_frame_num_to_track`), i.e. the oracle prompt is a 3D extent. Both modes are declared up front and
reported separately (run directories `<split>_<mode>`); the in-plane box and key slice are identical in both. `volume_metrics.csv` has `diag_*` columns
(prediction volume outside the GT z range, Dice inside it): these use ground truth for analysis only and are never fed back into predictions.
If a run prints `undefined symbol ... _C.so` and "Skipping the post-processing step", the compiled extension does not match the installed torch and small-hole filling is skipped.
Rebuild it (`cd MedSAM2 && rm -f sam2/_C.so && pip install --no-build-isolation -e .`, needs a CUDA toolkit matching torch's CUDA version). Whichever way, the state is recorded
in `config_snapshot.json` (`sam2_cuda_extension`); use one state for every reported run.

### Known deviation: sam2 compiled extension not built
The lab server has no `nvcc`, so MedSAM2's optional `sam2._C` extension is not built and the predictor skips its small-hole filling post-processing (`fill_hole_area=8` on the
low-resolution masks, a few pixels). Every reported run, validation and test, uses this same state, recorded as `sam2_cuda_extension` in each `config_snapshot.json`. Report it in the methods.

## Analysis plan for the frozen MedSAM2 baseline (written before any test result existed)
Decided after the validation run (`val`, 14 patients) and before touching `test`:
- **Headline protocol:** `--z-mode unbounded` (one GT-derived box on the lesion's key slice, propagated through the volume; least ground-truth information). **Secondary / sensitivity:**
  `--z-mode gt_extent` (adds the GT z range to bound propagation). Both are reported; neither is chosen from test results. No other protocol, window, post-processing or condition will be added.
- **Conditions:** exact, mild, strong (config.PROMPT_CONDITIONS). Mild is the preregistered "small localization error" condition; strong is the "large error" condition.
- **Units and aggregation:** per-volume metrics (Dice, IoU, HD95), then per-patient means, then mean / median / std and a seeded bootstrap 95% CI over patients. Test n = 14 patients, so intervals are wide.
- **Outliers stay in.** Failures (for example a very large, nearly isodense tumor where the tracker drifts onto the whole liver, as seen on val patient 8) are reported, not excluded.
- **Size analysis:** predeclared bins <10, 10-50, >=50 mL. Test has 2 / 5 / 7 patients per bin, so the small bin is descriptive only; lesion-level results (more observations, not independent) are reported separately.
- **Primary vs secondary set:** headline numbers use the 14 primary test patients only. The 12-patient secondary set (relabelled or resampled masks, 1 / 2 / 9 per size bin) is reported separately.
- **What val showed (development only):** mean exact Dice 0.810 (unbounded) and 0.820 (gt_extent); strong perturbation is much worse (0.544 / 0.604); patient 8 (499 mL) over-segmented 2.5 to 3 times; patient 59 under-segmented (volume ratio 0.37).
- **Known deviation:** sam2 compiled extension not built (see above), identical for all runs.
