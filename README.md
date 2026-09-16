# GOTSurv

GOTSurv is a survival-informed optimal transport framework for learning prognostic risk groups and modelling progression between them.

The framework learns a continuous survival risk score together with survival-informed risk groups, then applies graph-constrained unbalanced optimal transport between neighbouring groups. The resulting transport couplings provide population-level transitions and sample-specific progression vectors.

## Repository

```text
GOTSurv.py
util_file.py
run_GOTSurv.py
run_GOTSurv_baseline_comparison.py
environment.yml
README.md
```

- `GOTSurv.py` — core GOTSurv model and optimal transport implementation.
- `util_file.py` — data loading, bootstrap utilities, survival analysis and plotting functions.
- `run_GOTSurv.py` — main GOTSurv analysis pipeline.
- `run_GOTSurv_baseline_comparison.py` — comparison with CoxPH, GBM, RSF and SurvivalSVM.

## Installation

```bash
conda env create -f environment.yml
conda activate gotsurv
```

## Data

The code expects TCGA data under `Datasets/`:

```text
Datasets/
├── TCGA-CDR-SupplementalTableS1.xlsx
├── All_TITAN_embeddings.csv
└── TCGA_GeneExp/
    ├── TCGA_BRCA_.xlsx
    ├── TCGA_COAD_.xlsx
    └── ...
```

Gene-expression files are expected as `Datasets/TCGA_GeneExp/TCGA_{CANCER}_.xlsx`, and WSI features are read from `Datasets/All_TITAN_embeddings.csv`.

The default survival columns are `DSS.time` and `DSS`.

The data used in the paper are not distributed with this repository.

## Running GOTSurv

Set the cancer types, modality and experiment parameters near the top of `run_GOTSurv.py`, for example:

```python
CANCER_TYPES = ["BRCA"]
modality = "Genes"
K_STATES = 3
N_BOOTSTRAPS = 100
```

Then run:

```bash
python run_GOTSurv.py
```

Results are written to `Results/` and figures to `Plots/Paper_Plots/`.

## Baseline comparison

Configure `run_GOTSurv_baseline_comparison.py` and run:

```bash
python run_GOTSurv_baseline_comparison.py
```

The comparison includes GOTSurv, CoxPH, Gradient Boosting Survival Analysis, Random Survival Forest and Survival SVM, using the same bootstrap splits across models.

## Outputs

The analysis reports C-index, survival-group separation, RMST differences, collapse rate, patient-specific progression vectors, within-risk heterogeneity and sample-level progression profiles.

Progression vectors are inferred from cross-sectional data and should be interpreted as model-derived progression directions rather than observed longitudinal change.

## Citation

Citation will be added following publication.

## Licence

A licence will be added before public release.
