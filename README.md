# CAP 6545: MedSAM2 (oracle-box prompted) vs. EoMT on WAW-TACE liver tumor CT

The MedSAM2 baseline gets a box derived from the ground-truth mask, so it is a *prompted* result,
not a fully automatic one.

## Files
- `config.py` – paths, seed, device, CT window, split ratios, MedSAM2 pins, prompt perturbations
- `fetch_medsam2.py` – clones MedSAM2 at the pinned commit, downloads and SHA-256-checks `MedSAM2_latest.pt`
- `download_data.py` – WAW-TACE downloader: Zenodo API URLs/MD5s, resumable, retried, atomic, disk check, safe selective ZIP extraction
- `process_data.py` – archive inspection, manifest/matching, splits, QC figures (preprocessing to be added)
- `run_medsam2.py` – zero-shot MedSAM2 evaluation (oracle-prompted baseline)
- `fetch_eomt.py`, `eomt_data.py`, `train_eomt.py`, `eval_eomt.py` – EoMT fine-tuning on slices of the frozen TRAIN patients and 3D evaluation (below)
- `metrics.py`, `make_figures.py` – shared metrics; slide figures from a finished run
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

## Results: MedSAM2 oracle-localization prompted baseline (frozen run, 2026-10-19)
**These are prompted results: each box is derived from the ground-truth mask. They are not fully automatic.** Run from commit `464a1cc16cffaebf592c2493f9025e7d893649dc`, config fingerprint
`4766dae89ab50f63` (identical for all four runs), seed 42, CUDA, python 3.12.15, torch 2.5.1+cu124, numpy 2.5.2, SimpleITK 2.5.6, MedSAM2 pinned at `332f30d` with the pinned `MedSAM2_latest.pt`
(`sam2._C` extension not built, see above). Tables and per-volume CSVs: `reports/medsam2/<split>-<z mode>/` (no scans or masks). Commands:
```bash
python run_medsam2.py --split test      --device cuda --hd95 --z-mode unbounded   # headline
python run_medsam2.py --split test      --device cuda --hd95 --z-mode gt_extent   # sensitivity
python run_medsam2.py --split secondary --device cuda --hd95 --z-mode unbounded
python run_medsam2.py --split secondary --device cuda --hd95 --z-mode gt_extent
```
**Primary test set, 14 patients, mean per-patient Dice [bootstrap 95% CI over patients]** (0 invalid prompts in any run):

| condition | unbounded (headline) | gt_extent (sensitivity) |
|---|---|---|
| exact box | 0.858 [0.832, 0.884] | 0.862 [0.835, 0.887] |
| mild perturbation | 0.806 [0.722, 0.865] | 0.833 [0.789, 0.872] |
| strong perturbation | 0.540 [0.397, 0.670] | 0.596 [0.491, 0.697] |

Mean IoU exact / mild / strong: 0.755 / 0.693 / 0.413 (unbounded), 0.761 / 0.722 / 0.454 (gt_extent). Mean HD95 (mm): 6.1 / 8.1 / 34.9 (unbounded), 5.3 / 6.1 / 17.7 (gt_extent).
Median Dice (exact): 0.861 / 0.871. Realised boxes vs the exact box: mild mean IoU 0.60 (min 0.30), strong mean IoU 0.33 (min 0.07); none were clipped or invalid.
Runtime per volume per condition: median 2.1 s (unbounded), 0.8 s (gt_extent).

**By tumor size (exact box, patient level; bins <10 / 10-50 / >=50 mL contain 2 / 5 / 7 test patients, so descriptive only):** unbounded 0.810 / 0.892 / 0.848; gt_extent 0.810 / 0.901 / 0.849.
Under perturbation the smallest tumors are the most fragile: mild 0.524 (unbounded) and 0.681 (gt_extent), strong 0.201 and 0.414 for the two <10 mL patients.
**Lesion level (32 lesions in 14 patients; lesions of a patient are not independent):** mean Dice exact 0.779 (unbounded) / 0.807 (gt_extent). For the 16 lesions under 10 mL: exact 0.717 / 0.773, mild 0.583 / 0.628,
strong 0.214 / 0.291; the 10-50 mL and >=50 mL lesions score 0.84 with exact boxes.

**Secondary set, 12 patients (relabelled or resampled masks, reported separately):** unbounded exact 0.737 [0.598, 0.841], mild 0.737, strong 0.647; gt_extent exact 0.801 [0.749, 0.853], mild 0.800, strong 0.700.
The unbounded mean is pulled down by patient 84 (17.8 mL lesion, Dice 0.05, prediction 37 times the reference volume, 620 mL outside the lesion's z range); bounding the z range gives 0.66 for that patient.

**Validation (development only):** exact 0.810 (unbounded) and 0.820 (gt_extent); the 499 mL tumor of patient 8 was over-segmented 2.5 to 3 times (the tracker drifted onto the whole liver away from the key slice).

**Reading these numbers**
- With exact boxes MedSAM2 reaches about 0.86 Dice on the test patients; this is the reference EoMT would have to approach, and it is only reachable because the box contains privileged localization information.
- A moderate localization error (mild) costs about 0.03 to 0.05 Dice; a large one (strong) costs about 0.26 to 0.32. For the optional EoMT-to-MedSAM2 hybrid, box quality is the main lever.
- Giving the lesion's z range matters little with exact boxes (+0.004) but helps when boxes are poor and prevents rare whole-organ drift (val patient 8, secondary patient 84); it is more privileged information than a single box.
- Limits: only 14 test patients (wide intervals); 2 patients below 10 mL; one random perturbation draw per lesion and condition (variability across draws is not quantified); mask phase labels are verified by grid geometry only, and 25 of the 96 primary CTs share their grid with another phase;
  the secondary set uses transferred or relabelled masks; the compiled `sam2._C` extension was not available; all numbers use a single global liver window (-140 to 260 HU) instead of upstream's per-lesion windows.
- Nothing was tuned on the test set: the protocol, conditions, window and post-processing were fixed in `464a1cc` (see "Analysis plan") before the test runs.


## EoMT fine-tuning (automatic model, no prompts)
**Setup, in its own env** (EoMT pins torch 2.7.0, lightning 2.5.1, transformers 4.56.1, python 3.13.2 in its README; the MedSAM2 env needs numpy>=2 and torch 2.5.1):
```bash
python fetch_eomt.py                       # clones EoMT at the pinned commit aa017c3 (MIT) and downloads + SHA-256-verifies the ADE20K EoMT-L checkpoint (1.26 GB)
conda create -n eomt python==3.13.2 -y && conda activate eomt
pip install -r EoMT/requirements.txt
pip install SimpleITK pandas scipy pytest requests
python -m pytest -q tests                  # eomt_data / eval_eomt tests run here
```
**Design (decided before any EoMT training run; from reading the pinned EoMT code):**
- One training sample = one axial slice: the same windowed uint8 slice MedSAM2 sees (gray copied to RGB, 512x512 native = EoMT's ADE20K semantic size, no resizing).
- Two classes, background (0) and tumor (1). Every slice gets a background mask plus a tumor mask when tumor is present. EoMT's transforms never finish on an image with no mask, and its semantic read-out is an argmax over classes, so a single tumor class would label every pixel tumor.
- Each epoch uses every tumor slice plus a seeded random sample of tumor-free slices (`neg_per_pos` = 1.0 per tumor slice, redrawn every epoch). No horizontal flip, no colour jitter; EoMT's scale jitter + random crop are kept.
- Initialisation (primary): the ADE20K-pretrained EoMT-L checkpoint with a fresh 2-class head (`--init ade20k`). `--init dinov2` (DINOv2 backbone only, random heads) is available as an ablation.
- Training subsets are the nested TRAIN subsets of the frozen split (1, 2, 5, 10, 20, all = 68 patients); validation patients are for monitoring and checkpoint selection only; test patients are never read by training.
- Weights are saved as plain state dicts every `--ckpt-every` steps (EoMT's own checkpoints would embed the network object). Logging is wandb in offline mode (no account needed).
- Checkpoints are chosen by mean per-patient 3D Dice on the VALIDATION patients (`eval_eomt.py --select`), not by EoMT's slice-level mIoU. The test split only runs with the checkpoint recorded by that selection; `--weights` is refused on test.
- Inference runs on every slice with no prompt and takes EoMT's own semantic read-out (argmax over the two classes at the last block); slices are stacked into a 3D mask, restored to native geometry and scored with `metrics.py` (same CSV columns as the MedSAM2 runs).
```bash
python eomt_data.py --n-patients 5                                 # slice counts for a subset (tumor slices vs tumor-free), no training
python train_eomt.py --n-patients 5 --steps 3000 --dry-run         # shows the exact EoMT config that would run
python train_eomt.py --n-patients 5 --steps 3000 --compile         # trains; run dir results/eomt/n5_ade20k_s3000_seed42/
python eval_eomt.py --run-dir results/eomt/n5_ade20k_s3000_seed42 --select --hd95
python eval_eomt.py --run-dir results/eomt/n5_ade20k_s3000_seed42 --split test --hd95
```
`--steps` has no default on purpose: the number of steps, learning rate and the tumor-free ratio are to be fixed using validation patients only (smallest and largest subset), written down, and then kept identical for every subset size before any test evaluation.

**Verification status:** unit tests (data module against EoMT's real classes, sampler, nested subsets, evaluation, selection and test guard) pass. A 6-step CPU training run with a small backbone on synthetic slices ran through EoMT's own training loop (loss, validation, weights saved, evaluated by `eval_eomt.py`). The pinned ADE20K EoMT-L checkpoint was loaded with a new 2-class head (encoder and query weights identical to the checkpoint; one 512x512 forward pass checked). Not yet run: any real training on WAW-TACE slices, on GPU, with `torch.compile`, or torch 2.7.0 (local checks used torch 2.5.1). No EoMT result exists yet.
