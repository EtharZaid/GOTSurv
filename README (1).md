# GOTSurv

GOTSurv is a survival-informed optimal transport framework for jointly learning prognostic risk groups and modelling progression between them.

The framework first learns a continuous survival risk score together with survival-informed risk groups. These groups are then used to define an ordered progression graph, and graph-constrained unbalanced optimal transport is applied between neighbouring groups. The resulting transport couplings provide both population-level transitions and sample-specific progression vectors.

This repository contains the core GOTSurv implementation, the main TCGA analysis runner, supporting analysis utilities, and the baseline comparison pipeline.

---

## Repository structure

```text
GOTSurv.py
util_file.py
run_GOTSurv.py
run_GOTSurv_baseline_comparison.py
environment.yml
README.md
```

### `GOTSurv.py`

Core implementation of GOTSurv, including:

- survival-informed risk-group learning;
- latent representation learning;
- unbalanced optimal transport between neighbouring risk groups;
- whitening of the latent representation;
- transport-plan estimation;
- sample-specific barycentric progression vectors.

### `util_file.py`

Utility functions used by the analysis pipeline, including:

- TCGA data loading;
- stratified bootstrap sampling;
- aggregation of bootstrap predictions;
- patient-level progression representations;
- risk-group survival analysis;
- Kaplan--Meier plots;
- within-risk progression analysis;
- heatmaps and Cox-based validation;
- representative progression visualisation;
- sample-level progression plots.

### `run_GOTSurv.py`

Main GOTSurv analysis runner.

The script performs bootstrap training and evaluation and generates the main downstream analyses used in the study, including:

- C-index evaluation;
- survival-informed risk-group assignment;
- Kaplan--Meier analysis;
- patient-specific progression vectors;
- within-risk heterogeneity analysis;
- PCA-based progression vector visualisation;
- molecular feature analysis;
- sample-level progression profiles.

### `run_GOTSurv_baseline_comparison.py`

Bootstrap comparison between GOTSurv and established survival models:

- Cox proportional hazards (`CoxPH`);
- Gradient Boosting Survival Analysis (`GBM`);
- Random Survival Forest (`RSF`);
- Survival Support Vector Machine (`SurvivalSVM`).

Baseline risk groups are obtained post hoc from predicted risk scores, while GOTSurv additionally provides its learnt native risk groups.

---

## Installation

We recommend using Conda.

```bash
conda env create -f environment.yml
conda activate gotsurv
```

The main dependencies include:

- Python
- PyTorch
- NumPy
- pandas
- SciPy
- scikit-learn
- scikit-survival
- lifelines
- GeomLoss
- matplotlib
- seaborn
- openpyxl
- tqdm

---

## Data

The code expects TCGA survival data and feature representations to be stored under the `Datasets/` directory.

A typical directory structure is:

```text
Datasets/
├── TCGA-CDR-SupplementalTableS1.xlsx
├── All_TITAN_embeddings.csv
├── TCGA_GeneExp/
│   ├── TCGA_BRCA_.xlsx
│   ├── TCGA_COAD_.xlsx
│   ├── TCGA_LUAD_.xlsx
│   └── ...
```

For gene-expression experiments, each cancer dataset is expected at:

```text
Datasets/TCGA_GeneExp/TCGA_{CANCER}_.xlsx
```

For whole-slide-image experiments, the default feature file is:

```text
Datasets/All_TITAN_embeddings.csv
```

Survival outcomes are read from:

```text
Datasets/TCGA-CDR-SupplementalTableS1.xlsx
```

The expected default survival columns are:

```text
DSS.time
DSS
```

where `DSS.time` is the survival time and `DSS` is the event indicator.

The data used in the paper are not distributed with this repository.

---

## Running GOTSurv

The main configuration options are defined near the top of `run_GOTSurv.py`.

For example:

```python
CANCER_TYPES = ["BRCA"]
modality = "Genes"

K_STATES = 3
N_BOOTSTRAPS = 100
```

Multiple cancer types can be analysed sequentially:

```python
CANCER_TYPES = [
    "BLCA",
    "BRCA",
    "CESC",
    "COAD",
    "LUAD",
    "KIRC",
    "KIRP",
    "LIHC",
    "STAD",
    "UCEC",
]
```

Run:

```bash
python run_GOTSurv.py
```

Results are written to:

```text
Results/
```

and paper figures are written to:

```text
Plots/Paper_Plots/
```

---

## Baseline comparison

Configure the cancer types and modality near the top of:

```text
run_GOTSurv_baseline_comparison.py
```

For example:

```python
CANCER_TYPES = ["BRCA", "COAD"]
modality = "Genes"

models = [
    "GOTSurv",
    "CoxPH",
    "GBM",
    "RSF",
    "SurvivalSVM",
]
```

Run:

```bash
python run_GOTSurv_baseline_comparison.py
```

The comparison uses the same bootstrap splits across models.

For each model, the pipeline records survival ranking performance using the C-index and evaluates risk-group stratification using survival separation and restricted mean survival time (RMST).

Results are written under:

```text
Results/Comparisons results/
```

---

## Risk-group evaluation

GOTSurv learns risk groups jointly with the survival objective.

For comparison, baseline models are converted into discrete risk groups using post-hoc thresholds on their predicted risk scores.

The main evaluation measures include:

- concordance index (C-index);
- multivariate log-rank test;
- adjacent-group RMST differences;
- hazard ratios;
- risk-group collapse rate.

Collapse rate is defined as the proportion of bootstrap runs in which fewer than the requested number of risk groups are recovered.

---

## Progression modelling

After risk-group discovery, GOTSurv defines directed transport only between neighbouring risk groups.

For example, with three risk groups:

```text
Low risk → Intermediate risk → High risk
```

The optimal transport coupling between each pair of neighbouring groups defines a barycentric target for every source sample.

The resulting sample-specific progression vector is:

```text
Δx = barycentric target - source sample
```

This provides an individual description of the feature changes associated with progression towards the neighbouring higher-risk population.

These vectors are inferred from cross-sectional data and should therefore be interpreted as model-derived progression directions rather than observed longitudinal changes within a patient.

---

## Sample-level progression analysis

Sample-level progression profiles compare each patient's progression vector with the group-level mean progression direction.

Within each risk group, the reference direction is calculated as the transport-mass-weighted mean progression vector.

Agreement between an individual patient and the group direction is measured using cosine similarity.

Representative patients are selected at the:

```text
10th, 50th and 90th percentiles
```

of the within-group cosine-similarity distribution.

This provides examples ranging from relatively divergent to strongly group-aligned progression without selecting only extreme cases.

---

## Reproducibility

The analysis uses fixed random seeds for:

- bootstrap sampling;
- survival model fitting;
- GOTSurv training;
- optimal transport optimisation;
- baseline model fitting.

The default seed is:

```python
SEED = 42
```

Bootstrap train/test splits are shared across models in the comparison runner.

---

## Supported modalities

The current data-loading utilities support:

```text
Genes
WSI
Reports
Clinical
```

The experiments reported in the paper focus primarily on gene-expression and whole-slide-image representations.

---

## Output files

Depending on the analysis configuration, the runners produce files containing:

- bootstrap-level C-index values;
- native GOTSurv risk-group assignments;
- baseline post-hoc group assignments;
- survival-group statistics;
- patient-specific progression vectors;
- aggregated bootstrap progression representations;
- within-risk subgroup assignments;
- selected sample-level progression examples;
- paper-ready summary tables.

Output filenames contain the cancer type and modality to allow results from multiple experiments to coexist.

---

## Requirements for scikit-survival

Several baseline models require `scikit-survival`, including:

- Random Survival Forest;
- Gradient Boosting Survival Analysis;
- Survival SVM.

If installation through Conda is preferred, `scikit-survival` should be installed from `conda-forge`.

---

## Citation

If you use GOTSurv in your work, please cite:

```text
Citation will be added following publication.
```

---

## Licence

A licence will be added before public release.
