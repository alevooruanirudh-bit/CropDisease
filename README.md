# plant-borb

Reproduction and extension of **Ozcan & Polat, "BorB: A Novel Image Segmentation Technique for
Improving Plant Disease Classification With Deep Learning Models", IEEE Access 13 (2025),
doi:10.1109/ACCESS.2025.3563160**.

| Experiment | What | Command |
|---|---|---|
| Exp 1 | CNN **with** BorB(+) segmentation | `python run.py --config configs/exp1_borb.yaml` |
| Exp 2 | Same recipe on **raw** images | `python run.py --config configs/exp2_noborb.yaml` |
| Exp 3 | Robustness of Exp 1 vs Exp 2 checkpoints | `python run.py --config configs/exp3_robustness.yaml` |

Exp 1 and Exp 2 differ only in `segmentation.method`. They read the **same split CSVs**: a SHA-256
fingerprint of the splits is stored in every `metrics.json`, and paired tests refuse to compare
runs whose fingerprints differ.

## Setup

Works on Windows, macOS and Linux with Python 3.10–3.12. All commands below are run from the
`plant-borb` folder; only creating/activating the virtual environment differs per system.

**Windows (PowerShell or Command Prompt)**

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. With an NVIDIA GPU, install the CUDA build
of PyTorch first (pick the command for your CUDA version at https://pytorch.org/get-started/locally/),
then `pip install -r requirements.txt`. Install Git for Windows (https://git-scm.com) for the
PlantVillage download.

**macOS / Linux**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Then, on any system**

```bash
python -m pytest            # unit tests + end-to-end smoke test
python run.py --smoke       # synthetic 2-class / 50-image / 1-epoch run of Exp 1, 2 and 3
```

The device is chosen automatically: NVIDIA GPU (CUDA), then Apple GPU (MPS), then CPU.

Colab: open `notebooks/run_colab.ipynb` (GPU runtime).

## Data

```bash
python scripts/download_data.py                 # PlantVillage-15, MangoLeafBD, PlantDoc (~1.6 GB on disk)
python scripts/download_data.py --pv-classes 38 # all 38 PlantVillage classes
python scripts/download_data.py --datasets textures   # optional DTD photos for the background test (~0.6 GB)
python scripts/download_data.py --verify-only
```

No account or API key is needed. The script uses public mirrors: a sparse git checkout of `spMohanty/PlantVillage-Dataset`
(`raw/color`, only the needed classes), the HuggingFace mirror `Project-AgML/mango_leaf_disease_classification`
(4,000 original JPEGs), and a sparse checkout of `pratikkayal/PlantDoc-Dataset`. If everything
fails, it prints manual instructions. Expected layout:

```
data/plantvillage/<PlantVillage class>/*.JPG   (any mirror spelling, e.g. "Pepper__bell___..." is matched)
data/mangoleafbd/<class>/*.jpg
data/plantdoc/{train,test}/<PlantDoc class>/*.jpg
data/textures/{soil,grass,clutter}/*.jpg       (optional)
```

**Tomato YLCV class size.** The GitHub/abdallahalidev mirrors contain 5,357 Tomato Yellow Leaf
Curl Virus images; the paper's 20,638-image set (Kaggle `emmarex/plantdisease`) has 3,209. To match
the paper's class prior, `data.class_caps` caps YLCV at 3,209 by seeded random sampling, which gives
20,639 images. Set `data.class_caps: {}` to use all images.

### Data integrity

`run.py` (stage `prepare`) builds the splits before anything else:

1. It removes byte-identical files (md5).
2. It computes a 64-bit pHash per image and links pairs within a class at Hamming distance ≤ 4
   (union-find gives near-duplicate *groups*). Identical-pHash images with different bytes are
   grouped, not deleted.
3. It makes a stratified, **group-aware** 60/20/20 split (`StratifiedGroupKFold` twice, seed 42).
   The result is saved to `splits/<dataset>/{train,val,test}.csv`, together with `dedup_stats.json`
   and `class_counts.csv`.
4. It asserts no path, md5 or group overlap between splits on every run.

PlantDoc class mapping (13 of the 15 classes overlap; PlantDoc has no Potato-healthy or
Tomato-Target-Spot class) is in `src/data/classes.py::PLANTDOC_TO_PV15`.

## Segmentation

`src/segmentation/borb.py`

* **`borb_original`**: mask = (RGB B < 130) OR (Lab b > 135) with OpenCV 8-bit Lab. Background is
  set to black. The polarity (leaf = low B, high b) is *estimated* from training images (centre vs
  border ring) and asserted for PlantVillage in `scripts/segmentation_report.py`.
* **`borb_plus`**:
  * **Thresholds:** from the mean B / b histograms of the **training split only**. The threshold is
    the valley between the two most prominent modes of the Gaussian-smoothed histogram, with Otsu as
    the fallback when the histogram is not bimodal. Thresholds are saved to
    `splits/<dataset>/borb_thresholds.json` with a plot.
  * **Mask:** the B and b masks are OR'ed.
  * **Cleanup:**
    * Opening-by-reconstruction with an elliptical kernel. Whole components that survive the
      opening are kept, so specks vanish but petioles are not cut.
    * Closing, then hole filling.
    * Keep the largest component plus any component larger than 2% of the image.
  * **Lesion protection inside the leaf's convex hull:** dark or brown pixels are kept, and so are
    pixels whose Lab chroma differs from the image's own border background. This recovers pale or
    chlorotic tissue.
  * **Options:** GrabCut refinement (`grabcut: true`). Output is `black` (paper) or `crop` (square
    leaf bounding box plus padding).
* **Baselines:** Otsu on grayscale and K-means (k=2) on Lab (`baselines.py`).

```bash
python scripts/segmentation_report.py --config configs/exp1_borb.yaml        # PlantVillage
python scripts/segmentation_report.py --config configs/exp1_borb_mango.yaml  # MangoLeafBD
```

This writes `results/segmentation/<dataset>/`:
- the step grid `fig_borb_steps.png` (paper Fig. 8/9 style)
- the method comparison `fig_methods.png`
- the mask sanity metric (share of masks covering < 5 % or > 95 % of the image), overall and per
  class, with every failing file listed in `mask_failures.csv`

Segmented images and masks are cached under `cache/<dataset>/<method>_<output>_<hash>/`.

## Training recipe (identical for Exp 1 and Exp 2)

- **Backbones:** timm with ImageNet weights: `vgg16`, `resnet50`, `efficientnet_b3`,
  `mobilenetv3_large` (paper), plus `efficientnetv2_s`, `convnext_tiny`, `swin_t`.
- **Fine-tuning:** two phases. The head is trained alone for `freeze_epochs` at `lr_head`, then the
  whole network at `lr` with AdamW, warmup and a cosine schedule. Label smoothing 0.1, weight decay
  0.05, gradient clipping 1.0.
- **Augmentation (train only):** the paper's Keras augmentation mapped to albumentations (rotation
  40°, zoom 0.115, shear 0.115°, h/v flip, brightness 0.5–1.5, replicate border), plus
  RandomResizedCrop and ColorJitter. MixUp/CutMix sits behind `augment.mixup.enabled`.
- **Selection and stopping:** early stopping on val loss (patience 8, max 30 epochs); the best
  checkpoint is chosen by **val accuracy**. The test set is used once, for reporting.
- **Options:** class-balanced sampling (`train.balanced_sampling`), weighted loss
  (`train.weighted_loss`).
- **HPO:** random search on **val only** (`hpo.enabled: true`). The same seeded candidates and
  budget apply to every experiment.
- **Inference extras:** flip TTA (reported separately as `test_tta`). Top-2/3 backbone ensembles,
  with members ranked by val accuracy, appear in their own table.
- **Seeds:** 3 per backbone (`seeds: [0, 1, 2]`), reported as mean ± std.

Override anything from the CLI:

```bash
python run.py --config configs/exp1_borb.yaml --backbones efficientnet_b3 --seeds 0 --override train.epochs=20 num_workers=8
```

## Reproducing every table

```bash
# main comparison, PlantVillage-15 (paper backbones + stronger ones)
BB=vgg16,resnet50,efficientnet_b3,mobilenetv3_large,efficientnetv2_s,convnext_tiny
python run.py --config configs/exp1_borb.yaml configs/exp2_noborb.yaml --backbones $BB
python run.py --config configs/exp3_robustness.yaml --backbones $BB
# MangoLeafBD
python run.py --config configs/exp1_borb_mango.yaml configs/exp2_noborb_mango.yaml configs/exp3_robustness_mango.yaml
# segmentation comparison (paper Table 11) + ablations, one backbone
python run.py --backbones efficientnet_b3 --config configs/seg_otsu.yaml configs/seg_kmeans.yaml \
    configs/abl_borb_original.yaml configs/abl_crop.yaml configs/abl_noaug.yaml configs/abl_nomorph.yaml
# rebuild tables only
python run.py --config configs/exp1_borb.yaml --stage summary
```

Outputs:

- `results/<exp>/<backbone>/seed<k>/`: `metrics.json`, `classification_report.txt`,
  `cm_abs.png`, `cm_norm.png`, `curves.png`, `gradcam.png`, `predictions.csv`, `history.csv`,
  `best.pt`.
- `results/exp3_robustness/`: `robustness_raw.csv` (accuracy per condition/seed),
  `robustness_summary.csv` (clean, mCA, relative drop, per family, PlantDoc),
  `robustness_mcnemar.csv`, `robustness_plot.png`, `gradcam_background_energy.csv`, and
  `<backbone>/gradcam_compare_seed0.png`.
- `results/summary.md` and `results/summary.csv`:
  - data integrity
  - Exp 1 vs Exp 2 vs the paper's reported numbers
  - McNemar tests and paired-bootstrap Δ CIs
  - ensembles
  - the segmentation comparison
  - robustness
  - model cost
- `results/ablation.md`.

### Exp 3 details

Every perturbation is applied to the raw test image, deterministically. Each image's RNG is seeded
by (seed, family, severity, path). The no-BorB model sees the perturbed image; the BorB model sees
`segment(perturbed image)`. The families are:

- Gaussian noise (σ 5/15/25)
- blur (k 3/7/11)
- brightness (×0.5/0.7/1.3/1.6)
- contrast (×0.5/0.7/1.5)
- JPEG (q 50/20)
- gamma (0.5/2.0)
- rotation (15/30/45°)
- scale (0.7/0.85/1.2)
- background swap: the BorB+-extracted leaf is pasted onto held-out soil, grass or clutter
  textures, which are never used in training; procedural textures are used unless
  `data/textures/` exists
- occlusion: random erasing of 10/20/30% of the image
- PlantDoc zero-shot: accuracy over all 15 logits and restricted to the 13 overlapping classes

## Runtimes and hardware

Measured training throughput at 224 px, batch 16:

| backbone | Apple M4 (MPS) | RTX 3060 12 GB (est.) | Colab T4 (est.) |
|---|---|---|---|
| mobilenetv3_large | 120 img/s | ~450 img/s | ~350 img/s |
| efficientnet_b3 | 30 img/s | ~150 img/s | ~110 img/s |
| convnext_tiny | 31–36 img/s | ~170 img/s | ~120 img/s |
| vgg16 | 15–18 img/s | ~110 img/s | ~70 img/s |

There are 12.4k training images, so one EfficientNet-B3 epoch takes about 7.5 min on an M4, or
1.5–2 min on a 3060/T4. A 30-epoch run takes about 3.5 h on an M4, or about 1 h on a 3060.

**Memory:** batch 16 with AMP (default on CUDA) fits in under 6 GB for every listed backbone. On MPS,
AMP is off by default (`train.amp: true` enables fp16 autocast).

**Other costs:** segmentation QC of all 20.6k images with five methods takes about 1.5 min on 10
CPU cores. Building the BorB+ cache takes under a minute.

## Deviations from the paper (by design or by necessity)

- **Early stopping:** patience 8 with a 30-epoch cap, instead of patience 50, to fit the compute
  budget.
- **Data source:** PlantVillage comes from the spMohanty mirror, with YLCV capped (see above),
  because the exact Kaggle subset needs credentials.
- **Deduplication:** pHash dedup and group-aware splits. The paper uses a plain random split.
- **Augmentation:** Keras `shear_range=0.115` is in degrees and is kept as is (essentially a
  no-op).
- **Recipe:** the training recipe is stronger than the paper's (two-phase, AdamW + cosine, label
  smoothing, RandomResizedCrop/ColorJitter). It is identical for Exp 1 and Exp 2, so their
  comparison stays fair.

---

# Website: Plant Health Check

A web app where anyone can photograph or upload a leaf, fruit, stem, tuber or rhizome and get:

- **Plant:** the likely plant, with probabilities summed over all of that plant's classes.
- **Problem:** the plant part and the disease, pest, nutrient or non-infectious problem.
- **Confidence:** a confidence level, with "not sure" and "not a plant" answers instead of forced guesses.
- **Background:** the cause, symptoms and how it spreads.
- **What to do:** treatment (field practices, organic and biological, chemical at active-ingredient
  level) and prevention.

## Just run the website (no training)

The trained model is included in the repo at `results/web_model_v3b/efficientnet_b0/seed0/` (`best.pt`, `config.yaml`,
`metrics.json`; 658 classes, 89.5% test accuracy).
The datasets are not needed for this.

```bash
python webapp/server.py --run results/web_model_v3b/efficientnet_b0/seed0
```

Open http://localhost:8000. To use it from a phone on the same Wi-Fi, add `--host 0.0.0.0` and open
`http://<this computer's IP>:8000` (on Windows, allow Python through the firewall when asked; the
IP is shown by `ipconfig`, on macOS/Linux by `ifconfig`/`ip addr`). On phones, "Take photo" opens the
phone's camera app and works over plain `http://`. A computer's webcam only opens on `localhost` or an
`https://` address; elsewhere use "Upload photo".

## Rebuild the data and retrain

```bash
python scripts/ingest_catalog.py                  # download + unify every dataset in catalog/datasets.yaml
python scripts/prepare_web_splits.py --out splits/web_v2 --keep-test splits/web/test.csv
python run.py --config configs/web_model_v2.yaml  # train (results/web_model_v2/...)
```

- `--keep-test` keeps an earlier test set (and its near-duplicates) in test, so old and new models
  are compared on the same images. Leave it out for a fresh split.
- Training time depends on hardware: hours on a recent NVIDIA GPU, about a day on an Apple M-series
  laptop, several days on CPU only.

### Datasets that must be downloaded by hand (Kaggle)

Entries with `source: local:<folder>` in `catalog/datasets.yaml` come from Kaggle, which requires
your own free account. The project never contacts Kaggle and stores no credentials. For each one:

1. Open its `url` (listed in the catalog entry) and click **Download**.
2. Unzip it into `kaggle_data/<folder>/`, next to the `plant-borb` folder, keeping the zip's own
   sub-folders (or put the data anywhere and set the environment variable `PLANT_LOCAL_DATA` to
   that location).
3. Run `python scripts/ingest_catalog.py --only <dataset name> ...`.

A missing local dataset is reported and skipped; everything else still builds.

## Catalog and knowledge base

- `catalog/datasets.yaml` lists every source dataset, its licence, and how each source class maps
  to a canonical `Crop | part | Condition` label with a kind (disease, pest, nutrient, abiotic,
  healthy, nonplant).
  - Classes whose meaning is unclear (e.g. "diseased", "dry", "unhealthy fruit") are skipped.
  - Augmented copies are avoided; the few sources that ship only augmented images get small caps.
  - Each source class contributes up to 800 images by default (`cap_per_class`), resized to 288 px
    on the short side.
- `kb/*.yaml` is the treatment knowledge base, one entry per condition, with crop-specific causes
  where the same name means different organisms.
  - Chemical advice is limited to active ingredients; users must follow local registration and
    label rates.
  - The entries were drafted from standard plant-pathology references and **should be reviewed by
    an agronomist before public deployment**.
- **Not a plant:** a dedicated class built from leafless backgrounds and Imagenette photos, so
  random photos are rejected instead of being forced into a disease.
