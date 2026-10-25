# Forecasting Demographic Performance Gaps in Dermatology AI

Can we predict, **before training a model**, how large the skin-tone performance gap will be, using only simple properties of the training set?

This repo runs a controlled pilot study on [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k). We simulate many hypothetical "curator scenarios" (different dataset sizes, skin-tone mixes, and malignancy rates), train a classifier on each, and measure the realized gap between light skin (Fitzpatrick I-II) and dark skin (Fitzpatrick V-VI). We then test whether cheap, pre-release dataset signals can screen for scenarios likely to produce large gaps.

## Overview

For each scenario the pipeline:

1. Samples a training pool with a chosen size, dark-skin share, and malignant fraction.
2. Fine-tunes an EfficientNet-B0 for benign vs. malignant classification.
3. Measures the **realized gaps** on held-out test images:
   - `delta_auc` = AUC(I-II) − AUC(V-VI)
   - `delta_sens` = Sensitivity(I-II) − Sensitivity(V-VI)
4. Computes six **pre-release signals** from the training data alone.

Across scenarios, we then ask how well each signal correlates with, ranks, and forecasts the realized gap.

### Signals

| Signal | Name | Definition |
|---|---|---|
| φ₁ | FST imbalance ratio | n(I-II) / n(V-VI) in the training set |
| φ₂ | Class-conditional ratio | Malignant rate in V-VI / malignant rate in I-II (training set) |
| φ₃ | Empty cells | Fraction of empty cells in the FST (1-6) × nine-partition disease label table |
| φ₄ | Embedding distance | Mean cosine distance from each V-VI training image to its 5 nearest I-II neighbors (ResNet-50 features) |
| φ₅ | LAB color KL divergence | KL(V-VI ‖ I-II) between mean LAB color histograms |
| φ₆ | Probe accuracy gap | Accuracy(I-II) − Accuracy(V-VI) on a small held-out probe set |

φ₁-φ₅ need only the training data. φ₆ additionally requires a trained model and a probe set.

## Repository structure

```
.
├── pipeline.py          # End-to-end pipeline (download → features → scenarios → run → aggregate)
├── aggregate_v2.py      # Refined aggregation: binary AUROC screening, sparse forecast, poster figures
├── data/
│   ├── fitzpatrick17k.csv   # metadata (not included, see Setup)
│   └── images/              # downloaded images (not included)
├── artifacts/
│   ├── features.npz         # ResNet-50 embeddings + LAB histograms
│   ├── scenarios.json       # scenario specifications
│   ├── scenario_results/    # one JSON per scenario
│   └── aggregate/           # tables, headline numbers, figures
```

## Setup

### Requirements

- Python 3.9+
- You definitely need a GPU. 

```bash
pip install numpy pandas scipy scikit-learn scikit-image matplotlib pillow requests torch torchvision
```

### Data

1. Get `fitzpatrick17k.csv` from the [official Fitzpatrick17k repository](https://github.com/mattgroh/fitzpatrick17k) and place it at `data/fitzpatrick17k.csv`.
2. Images are **not** redistributed here. `download_images` fetches them from the source URLs in the metadata.

> **Note on image availability.** Many Fitzpatrick17k image URLs no longer resolve (the dermaamin.com URLs are largely dead). The downloader therefore only attempts the `atlasdermatologico` subset. For the full image set, use the official download link from the dataset authors. The pipeline adapts scenario sizes to however many images are actually available.

## Usage

The pipeline is resumable: re-running any stage skips work that is already done.

```bash
# 1. Download images (atlasdermatologico subset; run somewhere with internet access)
python pipeline.py download_images --workers 4 --timeout 30

# 2. Extract ResNet-50 features + LAB histograms (GPU)
python pipeline.py extract_features

# 3. Generate K scenarios (default K=20 for the pilot)
python pipeline.py generate_scenarios --k 20

# 4. Run each scenario (train, evaluate, compute signals)
for i in $(seq 0 19); do python pipeline.py run_scenario --id $i; done
#   add --force to redo a finished scenario

# 5. Aggregate
python aggregate_v2.py          # poster-ready outputs (recommended)
# or: python pipeline.py aggregate   # simpler multivariate version
```

Scenario runs are independent, so they parallelize naturally across GPUs or a job array on a cluster.

### Scenario design

Scenarios are generated deterministically (`GLOBAL_SEED = 17`):

- **Dark-skin share:** uniform in [0.05, 0.50]
- **Total size N:** chosen from {800, 1500, 3000, 6000}, capped to what the available image pool supports
- **Malignant fraction:** one of {0.20, 0.35, 0.50}
- **Splits:** stratified by skin-tone group; 20% test, a ~300-image probe set, remainder for training
- **Model:** ImageNet-pretrained EfficientNet-B0, 15 epochs, Adam (lr 1e-4), batch size 32, standard augmentation

Scenarios with fewer than 100 training images are recorded as `skipped_small_train` and excluded from aggregation.

## Outputs

Written to `artifacts/aggregate/`:

| File | Description |
|---|---|
| `scenario_results.csv` | Per-scenario signals and realized gaps |
| `univariate_correlations.csv` | Spearman ρ of each signal vs. ΔAUC and ΔSens |
| `binary_aurocs.csv` | Top-quartile AUROC per signal (can a signal flag the highest-gap scenarios?) |
| `headline.json` | Key numbers for the poster |
| `fig_hero_auroc_bars.png` | Discriminative AUROC per signal |
| `fig_signals_scatter.png` | Signals vs. realized ΔAUC |
| `fig_sparse_forecast.png` | Cross-validated forecast vs. observed ΔAUC (φ₂ + φ₅) |

`pipeline.py aggregate` produces a simpler set (`signals_vs_delta_auc.png`, `forecast_vs_observed.png`, `forecast_summary.json`) using a linear model over φ₁-φ₅.

## Results

<!-- Fill in from artifacts/aggregate/headline.json after the run. -->

| | |
|---|---|
| Scenarios | _20_ |
| ΔAUC (mean ± sd) | _0.066±0.123_ |
| Best screening signal (top-quartile AUROC) | _0.8266_ |
| Sparse forecast (φ₂ + φ₅), 5-fold CV | MAE _0.0883_ vs. baseline _0.0936_, R² _0.1979_ |

## Limitations

- This is a **pilot** (K = 20 scenarios by default); correlations and forecast estimates are noisy. Raise `K_PILOT` (or pass `--k`) for a fuller study.
- Scenarios are sampled from a single dataset, so scenarios share images and are not fully independent.
- Only the accessible image subset is used, which may differ in composition from the full Fitzpatrick17k.
- Fitzpatrick skin type labels are an imperfect proxy for skin tone and for the demographic groups that matter clinically.

## Disclaimer

This is research code for studying dataset fairness. It is not a diagnostic tool and must not be used for clinical decision-making.

## Citation

Dataset:

```bibtex
@inproceedings{groh2021evaluating,
  title={Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset},
  author={Groh, Matthew and Harris, Caleb and Soenksen, Luis and Lau, Felix and Han, Rachel and Kim, Aerin and Koochek, Arash and Badri, Omar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
```

## Author

Roya W. Parsa ([royaparsa.dev](https://royaparsa.dev) · [@rparsa49](https://github.com/rparsa49))