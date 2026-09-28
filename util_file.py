from __future__ import annotations

import os
import re
import warnings
from typing import Iterable, Optional, Union, Dict

import subprocess
from pathlib import Path

import gseapy as gp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from matplotlib.patheffects import Stroke, Normal
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu
import glob
from matplotlib.patches import Patch

from run_GOTSurv import choose_representative_run_indices


warnings.simplefilter("ignore", category=FutureWarning)

from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

STATE_LABELS = {
    2: {
        0: "Low risk",
        1: "High risk",
    },
    3: {
        0: "Low risk",
        1: "Intermediate risk",
        2: "High risk",
    },
    4: {
        0: "Low risk",
        1: "Low-intermediate risk",
        2: "Intermediate-high risk",
        3: "High risk",
    },
}

GROUP_LABELS = {
    0: "Cluster 1",
    1: "Cluster 2",
    "0": "Cluster 1",
    "1": "Cluster 2",
}


def _state_label(bin_id, k_states=3):
    try:
        return STATE_LABELS.get(int(k_states), {}).get(int(bin_id), f"Bin {bin_id}")
    except Exception:
        return f"State {bin_id}"


def _group_label(cluster_id):
    return GROUP_LABELS.get(cluster_id, GROUP_LABELS.get(str(cluster_id), f"Cluster {cluster_id}"))









# -----------------------------------------------------------
# Dataset + logging helpers
# -----------------------------------------------------------

def fetch_Dataset(cancer, modality, time='DSS.time', event='DSS'):
    df_survival = pd.read_excel("./Datasets/TCGA-CDR-SupplementalTableS1.xlsx")
    df_survival = df_survival.loc[:, ['bcr_patient_barcode', time, event, 'type']]
    df_survival.set_index('bcr_patient_barcode', inplace=True)
    df_survival.index.set_names('SAMPLE_ID', inplace=True)

    if modality == 'WSI':
        data = pd.read_csv("./Datasets/All_TITAN_embeddings.csv")
        data.set_index('SAMPLE_ID', inplace=True)
        data.index = data.index.str[:12]
    elif modality == 'Genes':
        path = rf'./Datasets/TCGA_GeneExp/TCGA_{cancer}_.xlsx'
        data = pd.read_excel(path)
        data['SAMPLE_ID'] = [p[:12] for p in data['SAMPLE_ID']]
        data.set_index('SAMPLE_ID', inplace=True)
    elif modality == 'Reports':
        data = pd.read_excel('./Datasets/Reports_embeddings_All_TCGA.xlsx')
        data['SAMPLE_ID'] = [p[:12] for p in data['SAMPLE_ID']]
        data.set_index('SAMPLE_ID', inplace=True)
    elif modality == 'Clinical':
        data = pd.read_csv("./Datasets/Clinical_data_All_TCGA.tsv", sep='\t')
        drop_keywords = [
            "Survival", "Progress", "Disease Free", "Status",
            "Last Communication", "Last Alive", "Months", "Calculated", "Time"
        ]
        keep_columns = [
            'Diagnosis Age', 'Sex', 'Race Category', 'Neoplasm Disease Stage',
            'Neoplasm Histologic Grade', 'American Joint Committee on Cancer Tumor Stage Code',
            'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code',
            'American Joint Committee on Cancer Metastasis Stage Code',
            'Neoadjuvant Therapy Type Administered Prior To Resection Text', 'Radiation Therapy',
            'Fraction Genome Altered', 'Mutation Count', 'TMB (nonsynonymous)',
            'Buffa Hypoxia Score', 'Ragnum Hypoxia Score', 'Winter Hypoxia Score',
            'MSI MANTIS Score', 'MSIsensor Score', 'Tumor Disease Anatomic Site'
        ]
        df_cancer = data[data['TCGA PanCanAtlas Cancer Type Acronym'] == cancer]
        patient_ids = df_cancer['Patient ID']
        cols_to_drop = [col for col in df_cancer.columns if any(k in col for k in drop_keywords)]
        df_clean = df_cancer.drop(columns=cols_to_drop, errors='ignore')
        available = [col for col in keep_columns if col in df_clean.columns]
        df_model = df_clean[available]
        categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns.tolist()
        df_encoded = pd.get_dummies(df_model, columns=categorical_cols, dummy_na=False)
        selector = VarianceThreshold(threshold=0.05)
        X_var_filtered = selector.fit_transform(df_encoded)
        selected_columns = df_encoded.columns[selector.get_support()]
        data = pd.DataFrame(X_var_filtered, columns=selected_columns)
        data['Patient ID'] = patient_ids.values
        data.set_index('Patient ID', inplace=True)
        data.index.set_names('SAMPLE_ID', inplace=True)
        data.index = data.index.str[:12]
    else:
        raise ValueError(f"Unknown modality: {modality}")

    selected_types = ['COAD', 'READ'] if cancer == 'COAD' else [cancer]
    cancer_clinical = df_survival[df_survival['type'].isin(selected_types)]
    dataset = cancer_clinical.join(data, on='SAMPLE_ID', how='inner')
    if 'type' in dataset.columns:
        dataset = dataset.drop(['type'], axis=1)
    dataset.dropna(inplace=True)
    censoring = 3652
    return dataset, time, event, censoring


def append_experiment_row(master_csv_path: str, row: dict) -> int:
    if os.path.exists(master_csv_path):
        df_old = pd.read_csv(master_csv_path)
        next_num = int(pd.to_numeric(df_old.get("exp_num", pd.Series([0])), errors="coerce").max()) + 1 if len(df_old) else 1
        row["exp_num"] = next_num
        df_out = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    else:
        row["exp_num"] = 1
        df_out = pd.DataFrame([row])
    df_out.to_csv(master_csv_path, index=False)
    return int(row["exp_num"])


def compute_cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index(np.asarray(time, float), -np.asarray(risk, float), np.asarray(event, int)))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, float)
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


# -----------------------------------------------------------
# ID + GSVA helpers
# -----------------------------------------------------------

def _clean_tcga_sample_id(x: str) -> str:
    return re.sub(r"-0[0-9]$", "", str(x))










# -----------------------------------------------------------
# Bootstrap + progression aggregation
# -----------------------------------------------------------
def make_bootstrap_splits(ids, events, n_bootstraps: int, seed: int = 42, min_oob_per_class: int = 1, max_tries: int = 500):
    """
    Stratified bootstrap:
      - sample with replacement within each event class
      - keep original class counts in the bootstrap train set
      - retry until the OOB/test set contains at least `min_oob_per_class`
        samples from each class (when possible)

    Parameters
    ----------
    ids : array-like
        Sample IDs aligned with events.
    events : array-like
        Binary event indicators aligned with ids.
    n_bootstraps : int
        Number of bootstrap replicates.
    seed : int
        Base random seed.
    min_oob_per_class : int
        Minimum number of samples required from each event class in OOB.
    max_tries : int
        Maximum retries per bootstrap replicate.
    """
    ids = np.asarray(list(ids), dtype=object)
    events = np.asarray(events).astype(int)

    if len(ids) != len(events):
        raise ValueError("ids and events must have the same length.")

    idx_event0 = np.where(events == 0)[0]
    idx_event1 = np.where(events == 1)[0]

    if len(idx_event0) == 0 or len(idx_event1) == 0:
        raise ValueError("Both event classes must be present to do stratified bootstrap.")

    splits = []

    for b in range(int(n_bootstraps)):
        success = False

        for attempt in range(int(max_tries)):
            rng = np.random.RandomState(int(seed) + 1000 * b + attempt)

            # bootstrap within each class, preserving class counts
            train0 = rng.choice(idx_event0, size=len(idx_event0), replace=True)
            train1 = rng.choice(idx_event1, size=len(idx_event1), replace=True)
            train_pos = np.concatenate([train0, train1])

            # shuffle combined train positions
            rng.shuffle(train_pos)

            inbag = np.zeros(len(ids), dtype=bool)
            inbag[train_pos] = True

            test_pos = np.where(~inbag)[0]

            if len(test_pos) == 0:
                continue

            test_events = events[test_pos]
            n_oob_0 = int(np.sum(test_events == 0))
            n_oob_1 = int(np.sum(test_events == 1))

            if n_oob_0 >= int(min_oob_per_class) and n_oob_1 >= int(min_oob_per_class):
                splits.append({
                    "bootstrap": b + 1,
                    "train_ids": ids[train_pos].tolist(),
                    "test_ids": ids[test_pos].tolist(),
                })
                success = True
                break

        if not success:
            raise RuntimeError(
                f"Could not create stratified bootstrap split {b+1} with at least "
                f"{min_oob_per_class} OOB samples from each event class after {max_tries} tries."
            )

    return splits

def build_patient_progression_representation(got, feature_df: pd.DataFrame, predicted_risk, *, include_last_bin: bool = True) -> pd.DataFrame:
    """Build patient-level progression representation in the SAME standardized feature space used by OT.

    feature_df must be row-aligned with got.bin_ids and should already be standardized for the
    corresponding run. The resulting delta_* columns are therefore standardized feature drifts.
    """
    if got.bin_ids is None:
        raise ValueError('got.bin_ids missing.')
    bin_ids = np.asarray(got.bin_ids).astype(int)
    uniq = sorted(np.unique(bin_ids).tolist())
    risk_s = pd.Series(np.asarray(predicted_risk, float).reshape(-1), index=feature_df.index, name='predicted_risk')
    feat_cols = feature_df.columns.astype(str).tolist()
    blocks = []
    for sbin in uniq:
        candidate_targets = sorted([int(t) for (s, t) in getattr(got, 'transport_plan_', {}).keys() if int(s) == int(sbin)])
        idx = np.where(bin_ids == int(sbin))[0]
        if idx.size == 0:
            continue
        if len(candidate_targets) == 0:
            if not include_last_bin:
                continue
            src = feature_df.iloc[idx].copy()
            delta = pd.DataFrame(0.0, index=src.index, columns=[f'delta_{c}' for c in feat_cols])
            block = pd.DataFrame(index=src.index)
            block['source_bin'] = int(sbin)
            block['target_bin'] = np.nan
            block['transport_mass'] = 0.0
            block = block.join(delta)
        else:
            tbin = candidate_targets[0]
            P = np.asarray(got.transport_plan_[(int(sbin), int(tbin))], float)
            src_idx = np.where(bin_ids == int(sbin))[0]
            tgt_idx = np.where(bin_ids == int(tbin))[0]
            mass = P.sum(axis=1) + 1e-12
            ybar = (P @ feature_df.iloc[tgt_idx].to_numpy(float)) / mass[:, None]
            delta = ybar - feature_df.iloc[src_idx].to_numpy(float)
            block = pd.DataFrame(index=feature_df.index[src_idx].copy())
            block['source_bin'] = int(sbin)
            block['target_bin'] = int(tbin)
            block['transport_mass'] = mass
            block = block.join(pd.DataFrame(delta, index=block.index, columns=[f'delta_{c}' for c in feat_cols]))
        delta_cols = [c for c in block.columns if c.startswith('delta_')]
        Xd = block[delta_cols].to_numpy(float)
        norms = np.linalg.norm(Xd, axis=1)
        block['drift_norm'] = norms
        block['predicted_risk'] = risk_s.reindex(block.index).to_numpy(float)
        blocks.append(block)
    rep = pd.concat(blocks, axis=0)
    rep.index = pd.Index([_clean_tcga_sample_id(x) for x in rep.index.astype(str)], name='SAMPLE_ID')
    return rep



def build_terminal_incoming_progression_representation(
    got,
    feature_df: pd.DataFrame,
    predicted_risk,
    *,
    terminal_bin: int | None = None,
) -> pd.DataFrame:
    """
    Build non-zero endpoint representations for patients in the terminal risk state.

    This uses the final *forward* OT edge, for example Intermediate risk → High risk.
    For each terminal-state patient j, the incoming drift is:

        delta_in_j = x_terminal_j - barycentric_predecessor_j

    where ``barycentric_predecessor_j`` is computed from source-state patients
    that transport mass into terminal patient j. Thus, high-risk clustering can
    be interpreted as heterogeneity in incoming transition/endpoint patterns,
    not as an outgoing progression beyond the terminal state.

    Notes
    -----
    - The model and state graph are not modified or retrained.
    - ``source_bin`` and ``target_bin`` are set to the terminal state so that
      downstream grouping functions analyse these patients within high risk.
    - ``incoming_from_bin`` records the actual predecessor/source state.
    - Do not replace this with ``include_last_bin=True`` in the outgoing
      representation builder, since that fallback produces all-zero drifts.
    """
    if got.bin_ids is None:
        raise ValueError("got.bin_ids missing.")

    bin_ids = np.asarray(got.bin_ids).astype(int)
    uniq = sorted(np.unique(bin_ids).tolist())
    terminal = int(max(uniq) if terminal_bin is None else terminal_bin)

    incoming_edges = sorted(
        [
            (int(s), int(t))
            for (s, t) in getattr(got, "transport_plan_", {}).keys()
            if int(t) == terminal and int(s) != terminal
        ],
        key=lambda edge: edge[0],
    )
    if len(incoming_edges) == 0:
        raise KeyError(
            f"No incoming cached transport plan found for terminal bin {terminal}. "
            "The terminal analysis requires the final forward OT edge."
        )

    # Adjacent ordered graphs have one final incoming edge; use the closest
    # predecessor when several incoming edges are available.
    source_bin, target_bin = incoming_edges[-1]
    P = np.asarray(got.transport_plan_[(source_bin, target_bin)], dtype=float)

    src_idx = np.where(bin_ids == source_bin)[0]
    tgt_idx = np.where(bin_ids == target_bin)[0]
    if P.shape != (len(src_idx), len(tgt_idx)):
        raise ValueError(
            f"Plan shape mismatch for terminal edge {(source_bin, target_bin)}: "
            f"got {P.shape}, expected ({len(src_idx)}, {len(tgt_idx)})."
        )

    source_X = feature_df.iloc[src_idx].to_numpy(dtype=float)
    target_X = feature_df.iloc[tgt_idx].to_numpy(dtype=float)

    incoming_mass = P.sum(axis=0) + 1e-12
    predecessor_bar = (P.T @ source_X) / incoming_mass[:, None]
    delta = target_X - predecessor_bar

    feat_cols = feature_df.columns.astype(str).tolist()
    block = pd.DataFrame(index=feature_df.index[tgt_idx].copy())
    block["source_bin"] = terminal
    block["target_bin"] = terminal
    block["incoming_from_bin"] = int(source_bin)
    block["transport_mass"] = incoming_mass
    block = block.join(
        pd.DataFrame(
            delta,
            index=block.index,
            columns=[f"delta_{c}" for c in feat_cols],
        )
    )

    delta_cols = [c for c in block.columns if c.startswith("delta_")]
    block["drift_norm"] = np.linalg.norm(block[delta_cols].to_numpy(dtype=float), axis=1)
    risk_s = pd.Series(
        np.asarray(predicted_risk, dtype=float).reshape(-1),
        index=feature_df.index,
        name="predicted_risk",
    )
    block["predicted_risk"] = risk_s.reindex(block.index).to_numpy(dtype=float)
    block["drift_role"] = "terminal_incoming"
    block.index = pd.Index(
        [_clean_tcga_sample_id(x) for x in block.index.astype(str)],
        name="SAMPLE_ID",
    )
    return block


def aggregate_bootstrap_patient_outputs(rep_runs: list[pd.DataFrame], state_runs: list[pd.DataFrame]) -> pd.DataFrame:
    rep_long = []
    for i, rep in enumerate(rep_runs, start=1):
        tmp = rep.copy()
        tmp['run_id'] = int(i)
        tmp.index = pd.Index([_clean_tcga_sample_id(x) for x in tmp.index.astype(str)], name='SAMPLE_ID')
        rep_long.append(tmp.reset_index())
    st_long = []
    for i, st in enumerate(state_runs, start=1):
        tmp = st.copy()
        tmp['run_id'] = int(i)
        tmp['SAMPLE_ID'] = tmp['SAMPLE_ID'].astype(str).map(_clean_tcga_sample_id)
        st_long.append(tmp)
    rep_long = pd.concat(rep_long, axis=0, ignore_index=True)
    st_long = pd.concat(st_long, axis=0, ignore_index=True)

    feat_cols = [c for c in rep_long.columns if c.startswith('delta_')]
    agg_num_cols = [c for c in feat_cols + ['predicted_risk', 'drift_norm', 'transport_mass'] if c in rep_long.columns]
    agg = rep_long.groupby('SAMPLE_ID')[agg_num_cols].mean().add_prefix('mean_')
    counts = rep_long.groupby('SAMPLE_ID').size().rename('n_runs_present')

    def _mode_or_nan(x):
        m = pd.Series(x).dropna().mode()
        return np.nan if len(m) == 0 else m.iloc[0]

    extra = {}
    for col in ['source_bin', 'target_bin']:
        if col in rep_long.columns:
            extra[f'{col}_mode'] = rep_long.groupby('SAMPLE_ID')[col].agg(_mode_or_nan)
            extra[f'{col}_mean'] = rep_long.groupby('SAMPLE_ID')[col].mean()

    state_mode = st_long.groupby('SAMPLE_ID')['state_id'].agg(_mode_or_nan).rename('state_mode')
    state_mean = st_long.groupby('SAMPLE_ID')['state_id'].mean().rename('state_mean')
    state_count = st_long.groupby('SAMPLE_ID').size().rename('state_runs_present')

    out = pd.concat([agg, counts, state_mode, state_mean, state_count] + [v.rename(k) for k,v in extra.items()], axis=1).reset_index()
    for col in ['state_mode','source_bin_mode','target_bin_mode']:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').astype('Int64')
    return out

def _scale_arrows(X2, U2, max_quantile: float = 70, frac_of_range: float = 0.06):
    norms = np.linalg.norm(U2, axis=1) + 1e-12
    dirs = U2 / norms[:, None]
    cap = np.percentile(norms, max_quantile) if norms.size else 1.0
    norms_clip = np.minimum(norms, cap)
    xmin, ymin = np.min(X2, axis=0)
    xmax, ymax = np.max(X2, axis=0)
    data_range = float(min(xmax - xmin, ymax - ymin))
    max_len = frac_of_range * (data_range if data_range > 0 else 1.0)
    lengths = max_len * (norms_clip / (cap + 1e-12))
    return dirs * lengths[:, None]



def plot_km_subplots_per_edge(
    surv_df: pd.DataFrame,
    state_ids,
    *,
    time_col: str,
    event_col: str,
    censor_time: float | None = 3650.0,
    suptitle: str = "Kaplan–Meier by risk group",
    save_path: str | None = None,
    show: bool = True,
    show_at_risk: bool = True,
    state_naming: dict | None = None,
    figsize: tuple[float, float] = (7.2, 7.0),
    linewidth: float = 2.2,
):
    """
    Plot one combined Kaplan–Meier curve figure for all risk groups, with
    a separate number-at-risk table underneath so the KM panel stays square-ish.
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    def _format_p(p):
        if pd.isna(p):
            return "NA"
        if p == 0:
            return "< 1e-300"
        if p < 1e-3:
            exp = int(np.floor(np.log10(p)))
            coeff = p / (10 ** exp)
            return rf"{coeff:.1f} × 10$^{{{exp}}}$"
        return f"{p:.3g}"

    def _state_name(x, state_naming=None):
        if state_naming is not None and isinstance(state_naming, dict):
            d = state_naming.get("states", {})
            if int(x) in d:
                return d[int(x)]
        return f"State {int(x)}"

    def _get_group_palette(group_labels):
        name_palette = {
            "Low risk": "#2ca02c",
            "Low-intermediate risk": "#7fbf3f",
            "Intermediate risk": "#f39c12",
            "Medium risk": "#f39c12",
            "Intermediate-high risk": "#c49c24",
            "High risk": "#d62728",
        }
        fallback = ["#2ca02c", "#7fbf3f", "#f39c12", "#d62728", "#9467bd", "#8c564b"]
        palette = {}
        for i, g in enumerate(group_labels):
            palette[g] = name_palette.get(g, fallback[i % len(fallback)])
        return palette

    def _n_at_risk(sub_df, times):
        t = sub_df["_time_use"].to_numpy(float)
        return [int(np.sum(t >= tt)) for tt in times]

    # -----------------------------
    # prepare data
    # -----------------------------
    d = surv_df.copy()
    d = d[[time_col, event_col]].copy()
    d["state_id"] = np.asarray(state_ids)

    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[event_col] = pd.to_numeric(d[event_col], errors="coerce")
    d["state_id"] = pd.to_numeric(d["state_id"], errors="coerce")

    d = d.dropna(subset=[time_col, event_col, "state_id"]).copy()
    d[event_col] = d[event_col].astype(int)
    d["state_id"] = d["state_id"].astype(int)

    if d.empty:
        raise ValueError("No valid rows available for KM plotting.")

    # -----------------------------
    # administrative censoring
    # -----------------------------
    if censor_time is not None:
        d["_time_use"] = np.minimum(d[time_col].to_numpy(float), float(censor_time))
        d["_event_use"] = np.where(
            d[time_col].to_numpy(float) > float(censor_time),
            0,
            d[event_col].to_numpy(int),
        )
    else:
        d["_time_use"] = d[time_col].to_numpy(float)
        d["_event_use"] = d[event_col].to_numpy(int)

    groups = sorted(d["state_id"].unique().tolist())
    if len(groups) < 2:
        raise ValueError("Need at least two risk groups/states to plot KM curves.")

    group_labels = [_state_name(g, state_naming=state_naming) for g in groups]
    palette = _get_group_palette(group_labels)

    # -----------------------------
    # multivariate log-rank
    # -----------------------------
    mv = multivariate_logrank_test(
        d["_time_use"].to_numpy(float),
        d["state_id"].to_numpy(int),
        d["_event_use"].to_numpy(int),
    )
    pval = float(mv.p_value)

    # -----------------------------
    # x ticks / risk table times
    # -----------------------------


    XMAX_DISPLAY = 1825.0
    if censor_time is not None:
        xticks = np.array([0.0, XMAX_DISPLAY / 2.0, XMAX_DISPLAY])   # 0, 912, 1825
    else:
        xmax = float(np.nanmax(d["_time_use"].to_numpy(float)))
        xticks = np.array([0.0, xmax / 2.0, xmax])

    # -----------------------------
    # plot: with or without risk table
    # -----------------------------


    if show_at_risk:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(
            2, 1,
            height_ratios=[4.8, 1.25],
            hspace=0.08,
            figure=fig,
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_tab = fig.add_subplot(gs[1, 0], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_tab = None


    kmf = KaplanMeierFitter()
    sub_by_group = {}

    for g in groups:
        sub = d[d["state_id"] == g].copy()
        sub_by_group[g] = sub

        label_base = _state_name(g, state_naming=state_naming)
        label = f"{label_base} (n={sub.shape[0]})"

        kmf.fit(
            durations=sub["_time_use"].to_numpy(float),
            event_observed=sub["_event_use"].to_numpy(int),
            label=label,
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            linewidth=linewidth,
            color=palette[label_base],
        )

    # -----------------------------
    # KM axis formatting
    # -----------------------------
    ax.set_xlabel("")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(-0.02, 1.05)


    if censor_time is not None:
        ax.set_xlim(0, XMAX_DISPLAY)

    ax.set_xticks([t for t in xticks if t <= XMAX_DISPLAY])

    if show_at_risk:
        ax.tick_params(axis="x", labelbottom=False)   # table shows the labels
    else:
        ax.set_xlabel("Time (days)")
        ax.tick_params(axis="x", labelbottom=True)

    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", frameon=False, fontsize=9)

    title_line1 = suptitle
    title_line2 = f"Multivariate log-rank p = {_format_p(pval)}"
    ax.set_title(f"{title_line1}\n{title_line2}", fontsize=12, pad=12)

    # keep the KM panel square-ish
    try:
        ax.set_box_aspect(1.0)
    except Exception:
        pass
    
    if show_at_risk:
        # -----------------------------
        # risk table axis
        # -----------------------------
        ax_tab.set_xlim(ax.get_xlim())
        ax_tab.set_xticks(xticks)
        ax_tab.set_xticklabels([f"{int(x)}" for x in xticks])
        ax_tab.set_xlabel("Time (days)")
        ax_tab.set_yticks([])
        for spine in ax_tab.spines.values():
            spine.set_visible(False)
        ax_tab.tick_params(axis="x", length=0)
        ax_tab.grid(False)

        # Leave visual space between group labels and the baseline (time-zero)
        # counts while preserving time-zero for the actual risk calculation.
        x_min, x_max = ax.get_xlim()
        x_range = float(x_max - x_min)
        label_x = -0.12
        display_count_x = [
            float(xticks[0]) + 0.10 * x_range,
            *[float(x) for x in xticks[1:]],
        ]

        # header
        ax_tab.text(
            label_x, 0.92, "Number at risk",
            transform=ax_tab.transAxes,
            ha="right", va="center",
            fontsize=10, fontweight="bold",
            clip_on=False,
        )

        # row positions in axis coordinates
        n_groups = len(groups)
        if n_groups == 1:
            y_positions = [0.45]
        else:
            y_positions = np.linspace(0.68, 0.12, n_groups)

        for y_ax, g in zip(y_positions, groups):
            label_base = _state_name(g, state_naming=state_naming)
            sub = sub_by_group[g]
            counts = _n_at_risk(sub, xticks)

            ax_tab.text(
                label_x, y_ax, label_base,
                transform=ax_tab.transAxes,
                ha="right", va="center",
                fontsize=9,
                color=palette[label_base],
                fontweight="bold",
                clip_on=False,
            )

            for x_display, n in zip(display_count_x, counts):
                ax_tab.text(
                    x_display, y_ax, str(n),
                    transform=ax_tab.transData,
                    ha="center", va="center",
                    fontsize=9,
                    color=palette[label_base],
                )

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, {"p_value": pval, "groups": groups}

def get_progression_naming(k_states: int = 3) -> dict:
    naming = {
        3: {
            "k_states": 3,
            "states": {0: "Low risk", 1: "Intermediate risk", 2: "High risk"},
            "edges": {(0, 1): "Low to intermediate risk", (1, 2): "Intermediate to high risk"},
            "groups": {0: "Cluster 1", 1: "Cluster 2"},
        },
        2: {
            "k_states": 2,
            "states": {0: "Low risk", 1: "High risk"},
            "edges": {(0, 1): "Low to high risk"},
            "groups": {0: "Cluster 1", 1: "Cluster 2"},
        },
        4: {
            "k_states": 4,
            "states": {
                0: "Low risk",
                1: "Low-intermediate risk",
                2: "Intermediate-high risk",
                3: "High risk",
            },
            "edges": {
                (0, 1): "Low to low-intermediate risk",
                (1, 2): "Low-intermediate to intermediate-high risk",
                (2, 3): "Intermediate-high to high risk",
            },
            "groups": {0: "Cluster 1", 1: "Cluster 2"},
        },
    }
    return naming.get(
        int(k_states),
        {
            "k_states": int(k_states),
            "states": {i: f"State {i}" for i in range(int(k_states))},
            "edges": {(i, i + 1): f"State {i} to state {i+1}" for i in range(max(0, int(k_states) - 1))},
            "groups": {0: "Cluster 1", 1: "Cluster 2"},
        },
    )





def _add_significance_stars_to_names(
    feature_names: list[str],
    significance_table: pd.DataFrame,
    *,
    q_thresh: float = 0.05,
):
    if significance_table is None or significance_table.empty:
        return {str(f): str(f) for f in feature_names}

    sig = significance_table[
        np.isfinite(significance_table["q_value"]) &
        (significance_table["q_value"] <= float(q_thresh))
    ].copy()

    if sig.empty:
        return {str(f): str(f) for f in feature_names}

    counts = (
        sig.groupby("feature")["risk_group"]
           .agg(lambda x: len(set(x)))
           .to_dict()
    )

    out = {}
    for f in feature_names:
        c = int(counts.get(str(f), 0))
        if c >= 2:
            out[str(f)] = f"** {f}"
        elif c == 1:
            out[str(f)] = f"* {f}"
        else:
            out[str(f)] = str(f)
    return out





def _cluster_row_order(df: pd.DataFrame) -> list[str]:
    """
    Return a row order from hierarchical clustering.
    """
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    X = df.copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if X.shape[0] <= 2:
        return X.index.astype(str).tolist()

    try:
        # correlation distance usually works well for heatmap patterns
        D = pdist(X.to_numpy(dtype=float), metric="correlation")
        if not np.isfinite(D).all():
            D = pdist(X.to_numpy(dtype=float), metric="euclidean")
        Z = linkage(D, method="average")
        order = leaves_list(Z)
        return X.index[order].astype(str).tolist()
    except Exception:
        return X.index.astype(str).tolist()




def _bh_fdr_simple(pvals):
    pvals = np.asarray(pvals, float)
    ok = np.isfinite(pvals)
    out = np.full_like(pvals, np.nan, dtype=float)
    if ok.sum() == 0:
        return out
    pv = pvals[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[np.where(ok)[0]] = tmp
    return out











def cox_univariate_top_features(df: pd.DataFrame, genes: list[str], time_col: str, event_col: str, censor_time: float | None = 3652.0, standardize: bool = True, min_n: int = 25):
    d = df.copy()
    d[time_col] = pd.to_numeric(d[time_col], errors='coerce')
    d[event_col] = pd.to_numeric(d[event_col], errors='coerce')
    d = d.dropna(subset=[time_col, event_col])
    d[event_col] = d[event_col].astype(int)
    if censor_time is not None:
        ct = float(censor_time)
        t0 = d[time_col].to_numpy(float)
        e0 = d[event_col].to_numpy(int)
        d[time_col] = np.minimum(t0, ct)
        d[event_col] = ((e0 == 1) & (t0 <= ct)).astype(int)
    rows = []
    for g in genes:
        if g not in d.columns:
            continue
        x = pd.to_numeric(d[g], errors='coerce')
        dd = d[[time_col, event_col]].copy(); dd[g] = x; dd = dd.dropna(subset=[g])
        if dd.shape[0] < min_n:
            continue
        if standardize:
            mu = dd[g].mean(); sd = dd[g].std(ddof=0)
            if sd == 0 or not np.isfinite(sd):
                continue
            dd[g] = (dd[g] - mu) / sd
        try:
            cph = CoxPHFitter()
            cph.fit(dd, duration_col=time_col, event_col=event_col, robust=True)
            s = cph.summary.loc[g]
            rows.append({'gene': g, 'HR': float(np.exp(s['coef'])), 'CI95_low': float(np.exp(s['coef lower 95%'])), 'CI95_high': float(np.exp(s['coef upper 95%'])), 'p': float(s['p'])})
        except Exception:
            continue
    res = pd.DataFrame(rows)
    if not res.empty:
        res['q'] = _bh_fdr(res['p'].to_numpy())
        res = res.sort_values('p').reset_index(drop=True)
    return res




# -----------------------------------------------------------
# External targets / pathways from aggregated representation
# -----------------------------------------------------------



















# -----------------------------------------------------------
# Added from extended util for clustering / residualization / pie charts
# -----------------------------------------------------------

def progression_vectors_from_cached_plan_latent(
    got,
    feature_df: pd.DataFrame,
    source_bin: int,
    target_bin: int | None = None,
    use_whitened: bool = False,
):
    """
    Build patient-specific barycentric progression vectors in latent space.

    Returns a DataFrame indexed by source-bin samples with:
      - delta_latent_* : progression vector in latent space
      - src_latent_*   : source latent coordinates
      - ybar_latent_*  : barycentric mapped latent coordinates
      - transport_mass
      - source_bin / target_bin
    """
    if target_bin is None:
        target_bin = int(source_bin) + 1

    if got.bin_ids is None:
        raise ValueError("got.bin_ids missing.")

    if use_whitened:
        Z = getattr(got, "zw_", None)
        prefix = "latentw"
    else:
        Z = getattr(got, "z_", None)
        prefix = "latent"

    if Z is None:
        raise ValueError(f"got.{ 'zw_' if use_whitened else 'z_' } is missing.")

    Z = np.asarray(Z, float)
    n_dim = Z.shape[1]
    latent_cols = [f"{prefix}_{i}" for i in range(n_dim)]

    key = (int(source_bin), int(target_bin))
    if not hasattr(got, "transport_plan_") or key not in got.transport_plan_:
        raise KeyError(f"Missing cached transport plan for edge {key}.")

    P = np.asarray(got.transport_plan_[key], float)

    src_idx = np.where(np.asarray(got.bin_ids) == int(source_bin))[0]
    tgt_idx = np.where(np.asarray(got.bin_ids) == int(target_bin))[0]

    if P.shape != (len(src_idx), len(tgt_idx)):
        raise ValueError(
            f"Plan shape mismatch for edge {key}: got {P.shape}, "
            f"expected ({len(src_idx)}, {len(tgt_idx)})"
        )

    mass = P.sum(axis=1) + 1e-12
    ybar = (P @ Z[tgt_idx]) / mass[:, None]
    delta = ybar - Z[src_idx]

    out = pd.DataFrame(index=feature_df.index[src_idx].copy())
    out["source_bin"] = int(source_bin)
    out["target_bin"] = int(target_bin)
    out["transport_mass"] = mass

    src_df = pd.DataFrame(Z[src_idx], index=out.index, columns=[f"src_{c}" for c in latent_cols])
    ybar_df = pd.DataFrame(ybar, index=out.index, columns=[f"ybar_{c}" for c in latent_cols])
    delta_df = pd.DataFrame(delta, index=out.index, columns=[f"delta_{c}" for c in latent_cols])

    out = out.join(src_df).join(ybar_df).join(delta_df)
    return out

def build_patient_progression_representation_latent(
    got,
    feature_df: pd.DataFrame,
    predicted_risk,
    *,
    include_last_bin: bool = False,
    use_whitened: bool = False,
) -> pd.DataFrame:
    """
    Build patient-level OT progression representation for all source bins with outgoing edges,
    using LATENT-SPACE drift instead of original-space drift.
    """
    if got.bin_ids is None:
        raise ValueError("got.bin_ids missing.")

    bin_ids = np.asarray(got.bin_ids).astype(int)
    uniq = sorted(np.unique(bin_ids).tolist())
    out_blocks = []

    risk_s = pd.Series(np.asarray(predicted_risk, float).reshape(-1), index=feature_df.index, name="predicted_risk")

    for sbin in uniq:
        candidate_targets = sorted([int(t) for (s, t) in getattr(got, "transport_plan_", {}).keys() if int(s) == int(sbin)])

        if len(candidate_targets) == 0:
            if include_last_bin:
                idx = np.where(bin_ids == int(sbin))[0]
                if idx.size == 0:
                    continue

                if use_whitened:
                    Z = np.asarray(getattr(got, "zw_", None), float)
                    prefix = "latentw"
                else:
                    Z = np.asarray(getattr(got, "z_", None), float)
                    prefix = "latent"

                if Z is None:
                    raise ValueError(f"got.{ 'zw_' if use_whitened else 'z_' } is missing.")

                src = pd.DataFrame(
                    Z[idx],
                    index=feature_df.index[idx],
                    columns=[f"src_{prefix}_{j}" for j in range(Z.shape[1])]
                )
                ybar = pd.DataFrame(
                    Z[idx],
                    index=feature_df.index[idx],
                    columns=[f"ybar_{prefix}_{j}" for j in range(Z.shape[1])]
                )
                delta = pd.DataFrame(
                    0.0,
                    index=feature_df.index[idx],
                    columns=[f"delta_{prefix}_{j}" for j in range(Z.shape[1])]
                )

                block = pd.DataFrame(index=src.index)
                block["source_bin"] = int(sbin)
                block["target_bin"] = np.nan
                block["transport_mass"] = 0.0
                block = block.join(src).join(ybar).join(delta)
            else:
                continue
        else:
            tbin = candidate_targets[0]
            block = progression_vectors_from_cached_plan_latent(
                got=got,
                feature_df=feature_df,
                source_bin=int(sbin),
                target_bin=int(tbin),
                use_whitened=use_whitened,
            )

        delta_cols = [c for c in block.columns if str(c).startswith("delta_")]
        Xd = block[delta_cols].to_numpy(float)
        norms = np.linalg.norm(Xd, axis=1)
        block["drift_norm"] = norms

        denom = norms[:, None] + 1e-12
        dir_arr = Xd / denom
        dir_df = pd.DataFrame(
            dir_arr,
            index=block.index,
            columns=[c.replace("delta_", "drift_dir_") for c in delta_cols],
        )
        block = block.join(dir_df)
        block["predicted_risk"] = risk_s.reindex(block.index).to_numpy(float)
        out_blocks.append(block)

    if not out_blocks:
        raise ValueError("No latent progression representations could be built.")

    rep = pd.concat(out_blocks, axis=0)
    rep = rep[~rep.index.duplicated(keep="first")]
    return rep


def residualize_progression_features_against_risk(
    progression_df: pd.DataFrame,
    *,
    risk_col: str = "pred_risk",
    delta_prefix: str = "delta_",
) -> pd.DataFrame:
    """
    Residualize progression features against a continuous risk score.

    For each delta_* column:
        delta_resid = delta - E[delta | risk]
    using simple linear regression.
    """
    if risk_col not in progression_df.columns:
        raise ValueError(f"Missing risk column '{risk_col}' in progression_df.")

    out = progression_df.copy()
    delta_cols = [c for c in out.columns if str(c).startswith(delta_prefix)]
    if len(delta_cols) == 0:
        raise ValueError(f"No columns found with prefix '{delta_prefix}'.")

    risk = pd.to_numeric(out[risk_col], errors="coerce")
    keep = risk.notna()

    if keep.sum() < 5:
        raise ValueError("Too few non-missing risk values to residualize features.")

    Xr = risk.loc[keep].to_numpy(dtype=float).reshape(-1, 1)

    for col in delta_cols:
        y = pd.to_numeric(out[col], errors="coerce")
        both = keep & y.notna()
        if both.sum() < 5:
            continue

        model = LinearRegression()
        model.fit(risk.loc[both].to_numpy(dtype=float).reshape(-1, 1), y.loc[both].to_numpy(dtype=float))

        pred = pd.Series(np.nan, index=out.index, dtype=float)
        pred.loc[both] = model.predict(risk.loc[both].to_numpy(dtype=float).reshape(-1, 1))
        out[col] = y - pred

    return out










def assign_within_state_clusters_from_agg_residualized(
    agg_df: pd.DataFrame,
    *,
    state_col: str = 'state_mode',
    delta_prefix: str = 'mean_delta_',
    risk_col: str = 'mean_predicted_risk',
    n_clusters: int = 2,
    random_state: int = 42,
    min_samples: int = 8,
    k_states: int | None = None,
    include_terminal_state: bool = False,
) -> pd.DataFrame:
    out = agg_df.copy()
    if 'SAMPLE_ID' not in out.columns:
        out = out.reset_index().rename(columns={'index': 'SAMPLE_ID'})
    out['within_state_cluster'] = pd.Series([pd.NA] * out.shape[0], dtype='Int64')
    delta_cols = [c for c in out.columns if c.startswith(delta_prefix)]
    states_present = sorted(pd.Series(out[state_col]).dropna().astype(int).unique().tolist())
    if k_states is not None and not include_terminal_state:
        states_present = [s for s in states_present if s < int(k_states) - 1]
    for s in states_present:
        mask = pd.to_numeric(out[state_col], errors='coerce').astype('Int64') == int(s)
        sub = out.loc[mask, [risk_col] + delta_cols].apply(pd.to_numeric, errors='coerce')
        keep = sub.notna().all(axis=1)
        idx = sub.index[keep]
        if len(idx) < max(int(n_clusters), int(min_samples)):
            continue
        prog = sub.loc[idx].copy().rename(columns={risk_col: 'pred_risk', **{c:c.replace(delta_prefix,'delta_') for c in delta_cols}})
        prog = residualize_progression_features_against_risk(prog, risk_col='pred_risk', delta_prefix='delta_')
        use_cols = [c for c in prog.columns if c.startswith('delta_')]
        X = StandardScaler().fit_transform(prog[use_cols].to_numpy(float))
        labels = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=20).fit_predict(X)
        out.loc[idx, 'within_state_cluster'] = labels.astype(int)
    return out



















# -----------------------------------------------------------
# Concise summary tables for pies / grouped heatmaps
# -----------------------------------------------------------






# -----------------------------------------------------------
# Oncology-facing within-risk subgroup summary figures
# -----------------------------------------------------------



def _pretty_pathway_name_for_plot(name, name_mapper: dict | None = None, strip_hallmark: bool = False):
    """
    Format pathway/signature names for plotting.

    Current policy:
      - do NOT apply external name mapping;
      - do NOT strip the 'HALLMARK' prefix;
      - do NOT change capitalisation;
      - only replace underscores with spaces and collapse repeated whitespace.
    """
    s = str(name)
    # name_mapper is intentionally ignored so the file-provided names are used as-is.
    if strip_hallmark:
        s = s.replace("HALLMARK_", "")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s























# =============================================================================
# Compact consensusTME + MMC2 immune-landscape plotting utilities
# =============================================================================




























# NEW signature — drop-in replacement:

















# =============================================================================
# CATEGORIZED PATHWAY FUNCTIONS
# Drop these into util_functions_survival_ot_progression_bootstrap.py
#
# Category assignment is loaded from a pre-built CSV mapping file produced
# by build_pathway_category_map.py — run that script once before plotting.
# =============================================================================













# -----------------------------------------------------------------------------
# Categorized version of plot_risk_group_pathway_activity_and_km
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------
# Signature-only helpers: shared filtering logic
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Signature-only: risk-group bar chart + KM
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Signature-only: oncology within-risk cluster plot + KM
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Risk score vs drift magnitude scatter plot
# -----------------------------------------------------------------------------




import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore")


# =============================================================================
# Colour palettes and labels
# =============================================================================

_RISK_PALETTE = {
    "Low risk":               "#2CA02C",
    "Intermediate risk":      "#F39C12",
    "Intermediate-high risk": "#E67E22",
    "High risk":              "#D62728",
    "Low-intermediate risk":  "#A5C93D",
}

# Paired colours within each risk group.
# K=2 example:
#   Low risk  Cluster 1 = green, Cluster 2 = blue
#   High risk Cluster 1 = red,   Cluster 2 = purple
_RISK_CLUSTER_PALETTE = {
    "Low risk": {
        0: "#2CA02C",  # green
        1: "#1F77B4",  # blue
        2: "#17BECF",
    },
    "Intermediate risk": {
        0: "#E67E22",
        1: "#FFD700",  # bright yellow
        2: "#FFD166",
    },
    "Low-intermediate risk": {
        0: "#A5C93D",
        1: "#4DAF4A",
        2: "#80B1D3",
    },
    "Intermediate-high risk": {
        0: "#E67E22",
        1: "#C77CFF",
        2: "#FDB462",
    },
    "High risk": {
        0: "#D62728",  # red
        1: "#7B3294",  # purple
        2: "#E7298A",
    },
}

_CLUSTER_LABELS = {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3"}




def _state_label(state: int, k_states: int = 3) -> str:
    labels = {
        2: {0: "Low risk", 1: "High risk"},
        3: {0: "Low risk", 1: "Intermediate risk", 2: "High risk"},
        4: {0: "Low risk", 1: "Low-intermediate risk",
            2: "Intermediate-high risk", 3: "High risk"},
    }
    return labels.get(int(k_states), {}).get(int(state), f"State {state}")




# =============================================================================
# 1. Scatter: risk score vs drift magnitude
# =============================================================================



# =============================================================================
# 2. Boxplot: drift magnitude distribution per risk group
# =============================================================================



# =============================================================================
# 3. C-index of drift magnitude as survival predictor, per risk group
# =============================================================================









# =============================================================================
# Drift edge-case report: highest/lowest drift samples per OOB risk group
# =============================================================================




# =============================================================================
# Residualized-only gene/pathway heatmaps with top-risk samples per risk/cluster
# =============================================================================




import os
import re
import textwrap
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Small project-compatible helpers
# -----------------------------------------------------------------------------


def _gt_clean_id_local(x: Any) -> str:
    """Use the project TCGA cleaner if available; otherwise keep first 12 chars."""
    try:
        return _clean_tcga_sample_id(x)  # type: ignore[name-defined]
    except Exception:
        s = str(x).strip().replace(".", "-")
        m = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", s, flags=re.I)
        if m:
            return m.group(1).upper()
        return s[:12] if len(s) >= 12 else s


def _gt_state_label_local(x: int, k_states: int = 3) -> str:
    try:
        return _state_label(int(x), k_states=k_states)  # type: ignore[name-defined]
    except Exception:
        labels = {
            2: {0: "Low risk", 1: "High risk"},
            3: {0: "Low risk", 1: "Intermediate risk", 2: "High risk"},
            4: {
                0: "Low risk",
                1: "Low-intermediate risk",
                2: "Intermediate-high risk",
                3: "High risk",
            },
        }
        return labels.get(int(k_states), {}).get(int(x), f"State {x}")


def _gt_group_label_local(x: int) -> str:
    try:
        return _group_label(int(x))  # type: ignore[name-defined]
    except Exception:
        try:
            return f"Cluster {int(x) + 1}"
        except Exception:
            return f"Cluster {x}"


def _gt_bh_fdr(pvals: np.ndarray | list[float]) -> np.ndarray:
    try:
        return _bh_fdr_simple(pvals)  # type: ignore[name-defined]
    except Exception:
        pvals = np.asarray(pvals, dtype=float)
        out = np.full_like(pvals, np.nan, dtype=float)
        ok = np.isfinite(pvals)
        if ok.sum() == 0:
            return out
        pv = pvals[ok]
        n = len(pv)
        order = np.argsort(pv)
        ranked = pv[order]
        q = ranked * n / (np.arange(1, n + 1))
        q = np.minimum.accumulate(q[::-1])[::-1]
        q = np.clip(q, 0, 1)
        tmp = np.empty_like(q)
        tmp[order] = q
        out[np.where(ok)[0]] = tmp
        return out


def _gt_wrap(labels: list[str], width: int = 30) -> list[str]:
    out = []
    for x in labels:
        s = str(x).replace("_", " ")
        out.append("\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s)
    return out


def _gt_infer_risk_score_col(meta: pd.DataFrame, preferred: str | None = None) -> str | None:
    if preferred is not None and preferred in meta.columns:
        return preferred
    candidates = [
        "mean_predicted_risk",
        "predicted_risk",
        "pred_risk",
        "risk",
        "risk_score",
        "mean_pre_ot_risk",
        "pre_ot_risk",
        "mean_risk",
        "state_mean",
    ]
    for c in candidates:
        if c in meta.columns:
            return c
    return None


def _add_suffix_before_ext(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}" if ext else f"{path}{suffix}"


def _cluster_row_order(H: pd.DataFrame) -> list[str]:
    """Return row order from hierarchical clustering; falls back to current order."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        arr = H.to_numpy(dtype=float)
        if arr.shape[0] <= 2:
            return H.index.tolist()
        d = pdist(arr, metric="correlation")
        if not np.isfinite(d).all():
            d = pdist(arr, metric="euclidean")
        z = linkage(d, method="average")
        return H.index[leaves_list(z)].tolist()
    except Exception:
        return H.index.tolist()


def _pretty_feature_name_for_plot(g: str, name_mapper: dict | None = None) -> str:
    try:
        return _pretty_pathway_name_for_plot(g, name_mapper=name_mapper)  # type: ignore[name-defined]
    except Exception:
        if name_mapper is not None:
            return str(name_mapper.get(g, g))
        return str(g)


# -----------------------------------------------------------------------------
# Sample ordering
# -----------------------------------------------------------------------------


def prepare_residualized_top_risk_sample_order(
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    cluster_col: str = "within_state_cluster",
    risk_col: str | None = None,
    n_per_cluster: int = 40,
    k_states: int = 3,
    include_terminal_state: bool = True,
    ascending_risk: bool = False,  # kept for backward compatibility; robust state-aware sorting is used below
) -> tuple[list[str], pd.DataFrame, list[tuple[int, int, list[str]]]]:
    """Select samples from each risk/cluster block, with risk-score sign anchored to state order."""
    meta = patient_meta_res.copy()
    if sample_id_col in meta.columns:
        meta[sample_id_col] = meta[sample_id_col].astype(str).map(_gt_clean_id_local)
        meta = meta.set_index(sample_id_col)
    else:
        meta.index = pd.Index([_gt_clean_id_local(x) for x in meta.index.astype(str)], name="SAMPLE_ID")

    for col in [state_col, cluster_col]:
        if col not in meta.columns:
            raise ValueError(f"Missing required column {col!r} in patient_meta_res.")
        meta[col] = pd.to_numeric(meta[col], errors="coerce")

    meta = meta.dropna(subset=[state_col, cluster_col]).copy()
    meta[state_col] = meta[state_col].astype(int)
    meta[cluster_col] = meta[cluster_col].astype(int)

    risk_col_use = _gt_infer_risk_score_col(meta, preferred=risk_col)
    if risk_col_use is not None:
        meta[risk_col_use] = pd.to_numeric(meta[risk_col_use], errors="coerce")
        if risk_col is not None and risk_col not in meta.columns:
            print(
                f"[within-risk heatmap] requested risk_col={risk_col!r} not found; "
                f"falling back to {risk_col_use!r}."
            )

    states = sorted(meta[state_col].dropna().astype(int).unique().tolist())
    if not include_terminal_state:
        states = [s for s in states if int(s) < int(k_states) - 1]

    risk_sign = 1.0
    if risk_col_use is not None and len(states) >= 2:
        state_means = meta.groupby(state_col)[risk_col_use].mean().reindex(states).dropna()
        if state_means.shape[0] >= 2:
            lo_mean = float(state_means.iloc[0])
            hi_mean = float(state_means.iloc[-1])
            if np.isfinite(lo_mean) and np.isfinite(hi_mean) and hi_mean < lo_mean:
                risk_sign = -1.0
                print(
                    f"[within-risk heatmap] risk column {risk_col_use!r} runs opposite "
                    "to state order; flipping sorting direction."
                )

    if risk_col_use is not None:
        meta["_risk_sort_key"] = risk_sign * pd.to_numeric(meta[risk_col_use], errors="coerce")

    selected_ids: list[str] = []
    ordered_blocks: list[tuple[int, int, list[str]]] = []

    for state in states:
        ms = meta[meta[state_col] == int(state)].copy()
        clusters = sorted(ms[cluster_col].dropna().astype(int).unique().tolist())
        state_centre = None
        if risk_col_use is not None and int(state) not in [0, int(k_states) - 1]:
            state_centre = float(pd.to_numeric(ms["_risk_sort_key"], errors="coerce").median())

        for cl in clusters:
            block = ms[ms[cluster_col] == int(cl)].copy()
            if block.empty:
                continue

            if risk_col_use is not None:
                block = block.dropna(subset=["_risk_sort_key"]).copy()
                if int(state) == 0:
                    block = block.sort_values("_risk_sort_key", ascending=True, na_position="last")
                elif int(state) == int(k_states) - 1:
                    block = block.sort_values("_risk_sort_key", ascending=False, na_position="last")
                else:
                    block["_risk_distance_to_middle"] = (block["_risk_sort_key"] - state_centre).abs()
                    block = block.sort_values("_risk_distance_to_middle", ascending=True, na_position="last")
            else:
                block = block.sort_index()

            block = block.head(int(n_per_cluster)).copy()
            ids = block.index.astype(str).tolist()
            selected_ids.extend(ids)
            ordered_blocks.append((int(state), int(cl), ids))

    meta["_risk_label"] = meta[state_col].map(lambda s: _gt_state_label_local(int(s), k_states=k_states))
    meta["_cluster_label"] = meta[cluster_col].map(lambda c: _gt_group_label_local(int(c)))
    if risk_col_use is not None:
        meta["_risk_score_for_sort"] = meta[risk_col_use]

    seen = set()
    selected_ids = [x for x in selected_ids if not (x in seen or seen.add(x))]
    return selected_ids, meta, ordered_blocks


def prepare_top_risk_sample_order_by_risk_group(
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    risk_col: str | None = None,
    n_per_risk_group: int = 40,
    k_states: int = 3,
    include_terminal_state: bool = True,
) -> tuple[list[str], pd.DataFrame, list[tuple[int, list[str]]]]:
    """Select samples from each main risk group, with risk-score sign anchored to state order."""
    meta = patient_meta_res.copy()
    if sample_id_col in meta.columns:
        meta[sample_id_col] = meta[sample_id_col].astype(str).map(_gt_clean_id_local)
        meta = meta.set_index(sample_id_col)
    else:
        meta.index = pd.Index([_gt_clean_id_local(x) for x in meta.index.astype(str)], name="SAMPLE_ID")

    if state_col not in meta.columns:
        raise ValueError(f"Missing required column {state_col!r} in patient_meta_res.")

    meta[state_col] = pd.to_numeric(meta[state_col], errors="coerce")
    meta = meta.dropna(subset=[state_col]).copy()
    meta[state_col] = meta[state_col].astype(int)

    states = sorted(meta[state_col].dropna().astype(int).unique().tolist())
    if not include_terminal_state:
        states = [s for s in states if int(s) < int(k_states) - 1]

    risk_col_use = _gt_infer_risk_score_col(meta, preferred=risk_col)
    if risk_col_use is not None:
        meta[risk_col_use] = pd.to_numeric(meta[risk_col_use], errors="coerce")
        if risk_col is not None and risk_col not in meta.columns:
            print(f"[risk-group heatmap] requested risk_col={risk_col!r} not found; falling back to {risk_col_use!r}.")

    risk_sign = 1.0
    if risk_col_use is not None and len(states) >= 2:
        state_means = meta.groupby(state_col)[risk_col_use].mean().reindex(states).dropna()
        if state_means.shape[0] >= 2:
            lo_mean = float(state_means.iloc[0])
            hi_mean = float(state_means.iloc[-1])
            if np.isfinite(lo_mean) and np.isfinite(hi_mean) and hi_mean < lo_mean:
                risk_sign = -1.0
                print("[risk-group heatmap] risk score runs opposite to state order; flipping sorting direction.")

    if risk_col_use is not None:
        meta["_risk_sort_key"] = risk_sign * pd.to_numeric(meta[risk_col_use], errors="coerce")

    selected_ids: list[str] = []
    ordered_blocks: list[tuple[int, list[str]]] = []

    for state in states:
        block = meta[meta[state_col] == int(state)].copy()
        if block.empty:
            continue

        if risk_col_use is not None:
            block = block.dropna(subset=["_risk_sort_key"]).copy()
            if int(state) == 0:
                block = block.sort_values("_risk_sort_key", ascending=True, na_position="last")
            elif int(state) == int(k_states) - 1:
                block = block.sort_values("_risk_sort_key", ascending=False, na_position="last")
            else:
                centre = float(block["_risk_sort_key"].median())
                block["_risk_distance_to_middle"] = (block["_risk_sort_key"] - centre).abs()
                block = block.sort_values("_risk_distance_to_middle", ascending=True, na_position="last")
        else:
            block = block.sort_index()

        block = block.head(int(n_per_risk_group)).copy()
        ids = block.index.astype(str).tolist()
        selected_ids.extend(ids)
        ordered_blocks.append((int(state), ids))

    seen = set()
    selected_ids = [x for x in selected_ids if not (x in seen or seen.add(x))]

    meta["_risk_label"] = meta[state_col].map(lambda s: _gt_state_label_local(int(s), k_states=k_states))
    if risk_col_use is not None:
        meta["_risk_score_for_sort"] = meta[risk_col_use]

    return selected_ids, meta, ordered_blocks


# -----------------------------------------------------------------------------
# Statistical feature selection
# -----------------------------------------------------------------------------


def test_features_between_residualized_clusters_by_risk(
    feature_df: pd.DataFrame,
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    cluster_col: str = "within_state_cluster",
    k_states: int = 3,
    min_group_n: int = 5,
    q_thresh: float = 0.05,
) -> pd.DataFrame:
    """Mann-Whitney test between clusters inside each risk group."""
    from scipy.stats import mannwhitneyu

    X = feature_df.copy()
    X.index = pd.Index([_gt_clean_id_local(x) for x in X.index.astype(str)], name="SAMPLE_ID")
    X = X.loc[:, ~X.columns.duplicated(keep="first")].copy()

    meta = patient_meta_res.copy()
    if sample_id_col in meta.columns:
        meta[sample_id_col] = meta[sample_id_col].astype(str).map(_gt_clean_id_local)
        meta = meta.set_index(sample_id_col)
    else:
        meta.index = pd.Index([_gt_clean_id_local(x) for x in meta.index.astype(str)], name="SAMPLE_ID")

    for c in [state_col, cluster_col]:
        if c not in meta.columns:
            raise ValueError(f"Missing required column {c!r} in patient_meta_res.")
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    meta = meta.dropna(subset=[state_col, cluster_col]).copy()
    meta[state_col] = meta[state_col].astype(int)
    meta[cluster_col] = meta[cluster_col].astype(int)

    common = X.index.intersection(meta.index)
    X = X.loc[common].apply(pd.to_numeric, errors="coerce")
    meta = meta.loc[common]

    rows: list[dict[str, Any]] = []
    for state in sorted(meta[state_col].unique().tolist()):
        ms = meta[meta[state_col] == int(state)]
        clusters = sorted(ms[cluster_col].dropna().astype(int).unique().tolist())
        if len(clusters) < 2:
            continue
        c1, c2 = clusters[:2]
        idx1 = ms.index[ms[cluster_col] == c1]
        idx2 = ms.index[ms[cluster_col] == c2]
        pvals = []
        row_ix = []

        for feat in X.columns:
            x1 = pd.to_numeric(X.loc[idx1, feat], errors="coerce").dropna().to_numpy(float)
            x2 = pd.to_numeric(X.loc[idx2, feat], errors="coerce").dropna().to_numpy(float)
            if len(x1) < int(min_group_n) or len(x2) < int(min_group_n):
                continue
            try:
                p = float(mannwhitneyu(x1, x2, alternative="two-sided").pvalue)
            except Exception:
                p = np.nan
            rows.append({
                "risk_state": int(state),
                "risk_group": _gt_state_label_local(int(state), k_states=k_states),
                "feature": str(feat),
                "p_value": p,
                "median_cluster1": float(np.nanmedian(x1)),
                "median_cluster2": float(np.nanmedian(x2)),
                "effect_median_cluster2_minus_cluster1": float(np.nanmedian(x2) - np.nanmedian(x1)),
                "n_cluster1": int(len(x1)),
                "n_cluster2": int(len(x2)),
            })
            pvals.append(p)
            row_ix.append(len(rows) - 1)

        if row_ix:
            qvals = _gt_bh_fdr(np.asarray(pvals, dtype=float))
            for ix, q in zip(row_ix, qvals):
                rows[ix]["q_value"] = float(q) if np.isfinite(q) else np.nan

    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_effect"] = pd.to_numeric(out["effect_median_cluster2_minus_cluster1"], errors="coerce").abs()
        out["is_significant"] = pd.to_numeric(out["q_value"], errors="coerce") <= float(q_thresh)
        out = out.sort_values(["risk_state", "q_value", "p_value", "abs_effect"], ascending=[True, True, True, False]).reset_index(drop=True)
    return out


def test_features_between_risk_groups_kruskal(
    feature_df: pd.DataFrame,
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    k_states: int = 3,
    min_group_n: int = 5,
    q_thresh: float = 0.05,
) -> pd.DataFrame:
    """Kruskal-Wallis test for each feature across main risk groups."""
    from scipy.stats import kruskal

    X = feature_df.copy()
    X.index = pd.Index([_gt_clean_id_local(x) for x in X.index.astype(str)], name="SAMPLE_ID")
    X = X.loc[:, ~X.columns.duplicated(keep="first")].copy()

    meta = patient_meta_res.copy()
    if sample_id_col in meta.columns:
        meta[sample_id_col] = meta[sample_id_col].astype(str).map(_gt_clean_id_local)
        meta = meta.set_index(sample_id_col)
    else:
        meta.index = pd.Index([_gt_clean_id_local(x) for x in meta.index.astype(str)], name="SAMPLE_ID")

    if state_col not in meta.columns:
        raise ValueError(f"Missing required column {state_col!r} in patient_meta_res.")

    meta[state_col] = pd.to_numeric(meta[state_col], errors="coerce")
    meta = meta.dropna(subset=[state_col]).copy()
    meta[state_col] = meta[state_col].astype(int)

    common = X.index.intersection(meta.index)
    X = X.loc[common].apply(pd.to_numeric, errors="coerce")
    meta = meta.loc[common]
    states = sorted(meta[state_col].dropna().astype(int).unique().tolist())

    rows: list[dict[str, Any]] = []
    pvals: list[float] = []

    for feat in X.columns:
        values_by_state = []
        medians: dict[str, float] = {}
        ns: dict[str, int] = {}

        for state in states:
            idx = meta.index[meta[state_col] == int(state)]
            vals = pd.to_numeric(X.loc[idx, feat], errors="coerce").dropna().to_numpy(float)
            label = _gt_state_label_local(int(state), k_states=k_states)
            safe = str(label).replace(" ", "_").replace("-", "_").replace("/", "_").lower()
            ns[f"n_{safe}"] = int(len(vals))
            medians[f"median_{safe}"] = float(np.nanmedian(vals)) if len(vals) else np.nan
            if len(vals) >= int(min_group_n):
                values_by_state.append(vals)

        if len(values_by_state) < 2:
            continue

        try:
            p = float(kruskal(*values_by_state).pvalue)
        except Exception:
            p = np.nan

        median_vals = [v for v in medians.values() if np.isfinite(pd.to_numeric(v, errors="coerce"))]
        max_median_diff = float(np.nanmax(median_vals) - np.nanmin(median_vals)) if len(median_vals) >= 2 else np.nan

        rows.append({
            "feature": str(feat),
            "p_value": p,
            "max_median_difference": max_median_diff,
            **ns,
            **medians,
        })
        pvals.append(p)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = _gt_bh_fdr(np.asarray(pvals, dtype=float))
        out["abs_effect"] = pd.to_numeric(out["max_median_difference"], errors="coerce").abs()
        out["is_significant"] = pd.to_numeric(out["q_value"], errors="coerce") <= float(q_thresh)
        out = out.sort_values(["q_value", "p_value", "abs_effect"], ascending=[True, True, False]).reset_index(drop=True)
    return out


def select_features_for_residualized_heatmap(
    feature_df: pd.DataFrame,
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    cluster_col: str = "within_state_cluster",
    k_states: int = 3,
    top_n_features: int = 40,
    q_thresh: float = 0.05,
    min_group_n: int = 5,
    fallback_by_variance: bool = True,
    exclude_feature_regex: str | None = None,
    include_feature_regex: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select top features significant between clusters within any risk group."""
    X = feature_df.copy()
    if include_feature_regex is not None:
        X = X[[c for c in X.columns if re.search(include_feature_regex, str(c), flags=re.I)]].copy()
    if exclude_feature_regex is not None:
        X = X[[c for c in X.columns if not re.search(exclude_feature_regex, str(c), flags=re.I)]].copy()

    sig = test_features_between_residualized_clusters_by_risk(
        X,
        patient_meta_res,
        sample_id_col=sample_id_col,
        state_col=state_col,
        cluster_col=cluster_col,
        k_states=k_states,
        min_group_n=min_group_n,
        q_thresh=q_thresh,
    )

    chosen: list[str] = []
    if not sig.empty:
        sig2 = sig[np.isfinite(pd.to_numeric(sig["q_value"], errors="coerce")) & (pd.to_numeric(sig["q_value"], errors="coerce") <= float(q_thresh))].copy()
        if not sig2.empty:
            summary = (
                sig2.groupby("feature")
                .agg(
                    n_sig_risk_groups=("risk_state", lambda x: len(set(x))),
                    best_q=("q_value", "min"),
                    best_p=("p_value", "min"),
                    max_abs_effect=("abs_effect", "max"),
                )
                .reset_index()
                .sort_values(["n_sig_risk_groups", "best_q", "best_p", "max_abs_effect"], ascending=[False, True, True, False])
            )
            chosen = summary.head(int(top_n_features))["feature"].astype(str).tolist()

    if len(chosen) == 0 and fallback_by_variance:
        print(f"[within-risk heatmap] No cluster-significant features at q <= {q_thresh}; falling back to top variance features.")
        Xnum = X.apply(pd.to_numeric, errors="coerce")
        chosen = Xnum.var(axis=0).sort_values(ascending=False).head(int(top_n_features)).index.astype(str).tolist()

    return chosen, sig


def select_features_for_risk_group_heatmap(
    feature_df: pd.DataFrame,
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    k_states: int = 3,
    top_n_features: int = 40,
    q_thresh: float = 0.05,
    min_group_n: int = 5,
    fallback_by_variance: bool = True,
    exclude_feature_regex: str | None = None,
    include_feature_regex: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select top features significant across main risk groups using Kruskal-Wallis."""
    X = feature_df.copy()
    if include_feature_regex is not None:
        X = X[[c for c in X.columns if re.search(include_feature_regex, str(c), flags=re.I)]].copy()
    if exclude_feature_regex is not None:
        X = X[[c for c in X.columns if not re.search(exclude_feature_regex, str(c), flags=re.I)]].copy()

    sig = test_features_between_risk_groups_kruskal(
        X,
        patient_meta_res,
        sample_id_col=sample_id_col,
        state_col=state_col,
        k_states=k_states,
        min_group_n=min_group_n,
        q_thresh=q_thresh,
    )

    chosen: list[str] = []
    if not sig.empty:
        sig2 = sig[np.isfinite(pd.to_numeric(sig["q_value"], errors="coerce")) & (pd.to_numeric(sig["q_value"], errors="coerce") <= float(q_thresh))].copy()
        if not sig2.empty:
            chosen = sig2.sort_values(["q_value", "p_value", "abs_effect"], ascending=[True, True, False]).head(int(top_n_features))["feature"].astype(str).tolist()

    if len(chosen) == 0 and fallback_by_variance:
        print(f"[risk-group heatmap] No Kruskal-significant features at q <= {q_thresh}; falling back to top variance features.")
        Xnum = X.apply(pd.to_numeric, errors="coerce")
        chosen = Xnum.var(axis=0).sort_values(ascending=False).head(int(top_n_features)).index.astype(str).tolist()

    return chosen, sig


def _p_to_stars(p: float) -> str:
    """Consistent display stars used on heatmap row labels.

    This is deliberately based on p-values, not q-values, because these stars are
    used as a visual annotation. Feature selection can still use q_thresh/FDR.
    """
    try:
        p = float(p)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return ""


def _star_prefix(label: str) -> tuple[str, str]:
    """Split a row label into ('*** ', base), ('** ', base), ('* ', base), or ('', label)."""
    label = str(label)
    for pref in ["*** ", "** ", "* "]:
        if label.startswith(pref):
            return pref, label[len(pref):]
    return "", label


def _build_feature_star_map_from_pvalues(
    features: list[str],
    pmap: dict[str, float],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in features:
        p = pmap.get(str(f), np.nan)
        stars = _p_to_stars(p)
        out[str(f)] = (stars + " " if stars else "") + str(f)
    return out


def _build_kruskal_feature_star_map(
    features: list[str],
    sig_tbl: pd.DataFrame,
    *,
    p_col: str = "p_value",
) -> tuple[dict[str, str], pd.Series]:
    """Stars for the main risk-group heatmap.

    One gene has one Kruskal-Wallis p-value across the risk groups. The star
    level is based on that p-value:
      * p < 0.05, ** p < 0.01, *** p < 0.001.
    """
    if sig_tbl is None or sig_tbl.empty or p_col not in sig_tbl.columns:
        pvals = pd.Series(np.nan, index=[str(f) for f in features], dtype=float)
        return {str(f): str(f) for f in features}, pvals

    tmp = sig_tbl.copy()
    tmp["feature"] = tmp["feature"].astype(str)
    tmp[p_col] = pd.to_numeric(tmp[p_col], errors="coerce")
    pvals = tmp.groupby("feature")[p_col].min().reindex([str(f) for f in features])
    return _build_feature_star_map_from_pvalues(features, pvals.to_dict()), pvals


def _combine_within_risk_feature_pvalues(
    features: list[str],
    sig_tbl: pd.DataFrame,
    *,
    method: str = "min",
) -> pd.Series:
    """Collapse risk-specific cluster p-values to one display p-value per gene.

    method='min' uses the strongest within-risk cluster separation, i.e.
    min p across Low/Intermediate/High. This matches the heatmap selection logic
    because a gene can be selected if it separates clusters in any risk group.

    method='fisher' combines the available risk-specific p-values using Fisher's
    method. Use this if you want an overall across-risk cluster-separation signal.
    """
    idx = [str(f) for f in features]
    if sig_tbl is None or sig_tbl.empty or "p_value" not in sig_tbl.columns:
        return pd.Series(np.nan, index=idx, dtype=float)

    tmp = sig_tbl.copy()
    tmp["feature"] = tmp["feature"].astype(str)
    tmp["p_value"] = pd.to_numeric(tmp["p_value"], errors="coerce")
    tmp = tmp[np.isfinite(tmp["p_value"])].copy()

    method = str(method).lower()
    if method in ["min", "best", "minimum"]:
        return tmp.groupby("feature")["p_value"].min().reindex(idx)

    if method in ["fisher", "fisher_combined", "combined"]:
        try:
            from scipy.stats import combine_pvalues
        except Exception as e:
            raise ImportError("within_star_p_method='fisher' requires scipy.") from e
        vals: dict[str, float] = {}
        for feat, g in tmp.groupby("feature"):
            p = g["p_value"].dropna().clip(lower=np.nextafter(0, 1), upper=1).to_numpy(float)
            if len(p) == 0:
                vals[str(feat)] = np.nan
            else:
                vals[str(feat)] = float(combine_pvalues(p, method="fisher").pvalue)
        return pd.Series(vals).reindex(idx)

    raise ValueError("within_star_p_method must be 'min' or 'fisher'.")


def _build_within_feature_star_map(
    features: list[str],
    sig_tbl: pd.DataFrame,
    *,
    method: str = "min",
) -> tuple[dict[str, str], pd.Series]:
    """Stars for within-risk heatmap.

    Each gene has one p-value per risk group because cluster 1 vs cluster 2 is
    tested separately inside Low/Intermediate/High. For a single row label, the
    default display p-value is min p across the available risk-specific tests.
    """
    pvals = _combine_within_risk_feature_pvalues(features, sig_tbl, method=method)
    return _build_feature_star_map_from_pvalues(features, pvals.to_dict()), pvals


def _add_heatmap_star_legend(
    fig: plt.Figure,
    *,
    text: str = "Feature stars: * p<0.05   ** p<0.01   *** p<0.001",
    y: float = 0.935,
    fontsize: int = 8,
) -> None:
    fig.text(0.5, y, text, ha="center", va="top", fontsize=fontsize)


# -----------------------------------------------------------------------------
# Heatmap plotting internals
# -----------------------------------------------------------------------------


def _prepare_feature_matrix_for_heatmap(
    feature_df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    include_feature_regex: str | None = None,
    exclude_feature_regex: str | None = None,
) -> pd.DataFrame:
    X = feature_df.copy()
    X.index = pd.Index([_gt_clean_id_local(x) for x in X.index.astype(str)], name="SAMPLE_ID")
    X = X.loc[:, ~X.columns.duplicated(keep="first")].copy()

    if feature_cols is not None:
        X = X[[c for c in feature_cols if c in X.columns]].copy()
    if include_feature_regex is not None:
        X = X[[c for c in X.columns if re.search(include_feature_regex, str(c), flags=re.I)]].copy()
    if exclude_feature_regex is not None:
        X = X[[c for c in X.columns if not re.search(exclude_feature_regex, str(c), flags=re.I)]].copy()
    if X.shape[1] == 0:
        raise ValueError("No features remain after include/exclude filtering.")
    return X


def _zscore_for_heatmap(X: pd.DataFrame, selected_ids: list[str], features: list[str], value_mode: str) -> pd.DataFrame:
    Xnum = X[features].apply(pd.to_numeric, errors="coerce")
    if value_mode == "zscore_all":
        mu = Xnum.mean(axis=0)
        sd = Xnum.std(axis=0).replace(0, np.nan)
        Xz = Xnum.sub(mu, axis=1).div(sd, axis=1).replace([np.inf, -np.inf], np.nan)
    elif value_mode == "zscore_selected":
        tmp = Xnum.loc[selected_ids]
        mu = tmp.mean(axis=0)
        sd = tmp.std(axis=0).replace(0, np.nan)
        Xz = Xnum.sub(mu, axis=1).div(sd, axis=1).replace([np.inf, -np.inf], np.nan)
    elif value_mode == "raw":
        Xz = Xnum.copy()
    else:
        raise ValueError("value_mode must be 'zscore_all', 'zscore_selected', or 'raw'.")
    return Xz.loc[selected_ids, features].T.copy().fillna(0.0)


def _apply_feature_labels(
    H: pd.DataFrame,
    features: list[str],
    star_map: dict[str, str],
    *,
    name_mapper: dict | None = None,
) -> pd.DataFrame:
    labels = []
    for f in features:
        label = star_map.get(str(f), str(f))
        if name_mapper is not None:
            prefix, base = _star_prefix(label)
            label = prefix + str(name_mapper.get(base, base))
        labels.append(label)
    H = H.copy()
    H.index = labels
    return H


RISK_PALETTE = {
    "Low risk": (0.18, 0.70, 0.35),
    "Intermediate risk": (0.95, 0.55, 0.10),
    "Medium risk": (0.95, 0.55, 0.10),
    "High risk": (0.85, 0.15, 0.15),
    "Low-intermediate risk": (0.90, 0.70, 0.15),
    "Intermediate-high risk": (0.90, 0.35, 0.10),
}

CLUSTER_PALETTE = {
    "Cluster 1": (0.92, 0.40, 0.32),
    "Cluster 2": (0.24, 0.52, 0.88),
    "Cluster 3": (0.55, 0.40, 0.75),
    "Cluster 4": (0.35, 0.70, 0.70),
}


def _reorder_within_risk_cluster_columns(
    H: pd.DataFrame,
    meta_sel: pd.DataFrame,
    *,
    mode: str | None = "risk_gradient",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reorder cluster blocks within each risk group for display only.

    The cluster assignments are not changed. Only the left-to-right order of
    cluster blocks within each risk group is changed.

    Modes
    -----
    cluster_id / none
        Keep the existing cluster-label order.
    risk_gradient
        Compute a low-to-high expression direction from the selected heatmap rows
        and sort cluster blocks by their projection on that direction. This tends
        to make the heatmap visually progress from low-like to high-like.
    mean_expression
        Sort blocks by mean expression over the selected heatmap rows.
    """
    if mode is None:
        mode = "cluster_id"
    mode = str(mode).lower()
    if mode in ["none", "cluster_id", "cluster", "original", "id"]:
        return H, meta_sel, pd.DataFrame()

    required = {"_risk_label", "_cluster_label"}
    if not required.issubset(set(meta_sel.columns)):
        return H, meta_sel, pd.DataFrame()

    risk_labels = list(pd.unique(meta_sel["_risk_label"].astype(str)))
    if len(risk_labels) == 0:
        return H, meta_sel, pd.DataFrame()

    # Direction in feature space: high-risk mean minus low-risk mean.
    direction = None
    if mode == "risk_gradient" and len(risk_labels) >= 2:
        low_ids = meta_sel.index[meta_sel["_risk_label"].astype(str) == risk_labels[0]]
        high_ids = meta_sel.index[meta_sel["_risk_label"].astype(str) == risk_labels[-1]]
        low_ids = [x for x in low_ids if x in H.columns]
        high_ids = [x for x in high_ids if x in H.columns]
        if len(low_ids) > 0 and len(high_ids) > 0:
            direction = H[high_ids].mean(axis=1) - H[low_ids].mean(axis=1)
            direction = direction.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if not np.isfinite(direction.to_numpy(float)).any() or float(np.linalg.norm(direction.to_numpy(float))) <= 0:
                direction = None

    rows = []
    new_cols: list[str] = []

    for rg in risk_labels:
        rg_ids_all = meta_sel.index[meta_sel["_risk_label"].astype(str) == rg]
        cluster_labels = list(pd.unique(meta_sel.loc[rg_ids_all, "_cluster_label"].astype(str)))
        block_rows = []

        for cl in cluster_labels:
            ids = meta_sel.index[
                (meta_sel["_risk_label"].astype(str) == rg)
                & (meta_sel["_cluster_label"].astype(str) == cl)
            ].astype(str).tolist()
            ids = [x for x in ids if x in H.columns]
            if not ids:
                continue

            profile = H[ids].mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if mode == "risk_gradient" and direction is not None:
                score = float(np.dot(profile.to_numpy(float), direction.to_numpy(float)))
            elif mode in ["mean", "mean_expression", "average"]:
                score = float(np.nanmean(H[ids].to_numpy(float)))
            else:
                raise ValueError("cluster_order_mode must be 'cluster_id', 'risk_gradient', or 'mean_expression'.")

            block_rows.append({
                "risk_label": rg,
                "cluster_label": cl,
                "display_score": score,
                "n_samples": len(ids),
                "sample_ids": ids,
            })

        block_rows = sorted(block_rows, key=lambda r: (r["display_score"], r["cluster_label"]))
        for r in block_rows:
            new_cols.extend(r["sample_ids"])
            rows.append(r)

    if not new_cols:
        return H, meta_sel, pd.DataFrame()

    H2 = H.loc[:, new_cols].copy()
    meta2 = meta_sel.loc[new_cols].copy()
    order_tbl = pd.DataFrame(rows)
    return H2, meta2, order_tbl


def plot_risk_group_top_risk_feature_heatmap(
    feature_df: pd.DataFrame,
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    risk_col: str | None = "mean_predicted_risk",
    n_per_risk_group: int = 40,
    top_n_features: int = 40,
    q_thresh: float = 0.05,
    min_group_n: int = 5,
    star_p_col: str = "p_value",
    show_star_legend: bool = True,
    star_legend_y: float = 0.935,
    k_states: int = 3,
    include_terminal_state: bool = True,
    feature_cols: list[str] | None = None,
    include_feature_regex: str | None = None,
    exclude_feature_regex: str | None = None,
    selected_features: list[str] | None = None,
    name_mapper: dict | None = None,
    row_cluster: bool = True,
    value_mode: str = "zscore_all",
    cmap: str = "RdBu_r",
    vmax: float | None = 2.0,
    title: str = "Top significant genes — risk groups only",
    cbar_label: str = "Feature z-score",
    figsize: tuple[float, float] | None = None,
    ytick_fontsize: int = 7,
    wrap_width: int = 34,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Main risk-group heatmap: no cluster annotation; features selected by Kruskal-Wallis."""
    X = _prepare_feature_matrix_for_heatmap(
        feature_df,
        feature_cols=feature_cols,
        include_feature_regex=include_feature_regex,
        exclude_feature_regex=exclude_feature_regex,
    )

    selected_ids, meta, _ = prepare_top_risk_sample_order_by_risk_group(
        patient_meta_res,
        sample_id_col=sample_id_col,
        state_col=state_col,
        risk_col=risk_col,
        n_per_risk_group=n_per_risk_group,
        k_states=k_states,
        include_terminal_state=include_terminal_state,
    )
    selected_ids = [sid for sid in selected_ids if sid in X.index]
    if not selected_ids:
        raise ValueError("No selected samples overlap feature_df.")

    if selected_features is None:
        features, sig_tbl = select_features_for_risk_group_heatmap(
            X,
            meta.reset_index(),
            sample_id_col="SAMPLE_ID",
            state_col=state_col,
            k_states=k_states,
            top_n_features=top_n_features,
            q_thresh=q_thresh,
            min_group_n=min_group_n,
            fallback_by_variance=True,
        )
    else:
        sig_tbl = test_features_between_risk_groups_kruskal(
            X,
            meta.reset_index(),
            sample_id_col="SAMPLE_ID",
            state_col=state_col,
            k_states=k_states,
            min_group_n=min_group_n,
            q_thresh=q_thresh,
        )
        features = [f for f in selected_features if f in X.columns]

    features = [f for f in features if f in X.columns][: int(top_n_features)]
    if not features:
        raise ValueError("No selected features are present in feature_df.")

    H = _zscore_for_heatmap(X, selected_ids, features, value_mode)
    star_map, feature_star_p = _build_kruskal_feature_star_map(
        features,
        sig_tbl,
        p_col=star_p_col,
    )
    H = _apply_feature_labels(H, features, star_map, name_mapper=name_mapper)

    if row_cluster and H.shape[0] > 2:
        H = H.loc[_cluster_row_order(H)].copy()

    meta_sel = meta.loc[selected_ids].copy()

    if figsize is None:
        figsize = (max(10.5, 0.050 * H.shape[1] + 5.0), max(7.0, 0.24 * H.shape[0] + 2.2))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.16, 8.0], width_ratios=[24, 1.25], hspace=0.035, wspace=0.045)
    ax_risk_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_risk_bar)
    cax = fig.add_subplot(gs[1, 1])

    ax_risk_bar.imshow(np.array([[RISK_PALETTE.get(str(v), (0.8, 0.8, 0.8)) for v in meta_sel["_risk_label"].astype(str)]]), aspect="auto")
    ax_risk_bar.set_xticks([])
    ax_risk_bar.set_yticks([])
    ax_risk_bar.set_ylabel("Risk", rotation=0, ha="right", va="center", fontsize=10, labelpad=28)
    for sp in ax_risk_bar.spines.values():
        sp.set_visible(False)

    vmax_use = 2.0 if vmax is None else float(vmax)
    sns.heatmap(H, ax=ax, cmap=cmap, center=0.0, vmin=-vmax_use, vmax=vmax_use, cbar=True, cbar_ax=cax, cbar_kws={"label": cbar_label}, xticklabels=False, yticklabels=True)
    ax.set_xlabel(f"Top {int(n_per_risk_group)} samples by risk within each risk group")
    ax.set_ylabel("Features")
    ax.set_yticklabels(_gt_wrap(H.index.tolist(), width=wrap_width), rotation=0, fontsize=ytick_fontsize)
    ax.tick_params(axis="y", length=0)

    col_meta = meta_sel[["_risk_label"]].copy()
    n_cols = col_meta.shape[0]
    risk_blocks = []
    risk_bounds = []
    start = 0
    current = str(col_meta.iloc[0]["_risk_label"])
    for i in range(1, n_cols):
        here = str(col_meta.iloc[i]["_risk_label"])
        if here != current:
            risk_bounds.append(i)
            risk_blocks.append((start, i, current))
            start = i
            current = here
    risk_blocks.append((start, n_cols, current))

    for b in risk_bounds:
        ax.axvline(b, color="black", linewidth=1.7)
        ax_risk_bar.axvline(b, color="black", linewidth=1.7)
    for s, e, rg in risk_blocks:
        ax_risk_bar.text((s + e) / 2.0 - 0.5, -0.55, rg, ha="center", va="bottom", fontsize=10.5, fontweight="bold", transform=ax_risk_bar.transData, clip_on=False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.992)
    if show_star_legend:
        _add_heatmap_star_legend(fig, y=star_legend_y)
    fig.subplots_adjust(left=0.30, right=0.965, top=0.865, bottom=0.055, hspace=0.035, wspace=0.045)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, {
        "kind": "risk_groups",
        "heatmap_matrix": H,
        "ordered_meta": meta_sel,
        "selected_features": features,
        "significance": sig_tbl,
        "feature_star_pvalues": feature_star_p,
        "star_legend": "* p<0.05; ** p<0.01; *** p<0.001",
        "risk_order_used": [b[2] for b in risk_blocks],
        "vmax": vmax_use,
        "save_path": save_path,
    }


def plot_residualized_top_risk_feature_heatmap(
    feature_df: pd.DataFrame,
    patient_meta_res: pd.DataFrame,
    *,
    sample_id_col: str = "SAMPLE_ID",
    state_col: str = "state_mode",
    cluster_col: str = "within_state_cluster",
    risk_col: str | None = "mean_predicted_risk",
    n_per_cluster: int = 40,
    top_n_features: int = 40,
    q_thresh: float = 0.05,
    min_group_n: int = 5,
    within_star_p_method: str = "min",
    cluster_order_mode: str | None = "risk_gradient",
    show_star_legend: bool = True,
    star_legend_y: float = 0.935,
    k_states: int = 3,
    include_terminal_state: bool = True,
    ascending_risk: bool = False,
    feature_cols: list[str] | None = None,
    include_feature_regex: str | None = None,
    exclude_feature_regex: str | None = None,
    selected_features: list[str] | None = None,
    name_mapper: dict | None = None,
    row_cluster: bool = True,
    value_mode: str = "zscore_all",
    cmap: str = "RdBu_r",
    vmax: float | None = 2.0,
    title: str = "Top significant genes — residualized clusters only",
    cbar_label: str = "Feature z-score",
    figsize: tuple[float, float] | None = None,
    ytick_fontsize: int = 7,
    wrap_width: int = 34,
    save_path: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Within-risk heatmap: Risk + Cluster bars; features selected by within-risk cluster tests."""
    X = _prepare_feature_matrix_for_heatmap(
        feature_df,
        feature_cols=feature_cols,
        include_feature_regex=include_feature_regex,
        exclude_feature_regex=exclude_feature_regex,
    )

    selected_ids, meta, _ = prepare_residualized_top_risk_sample_order(
        patient_meta_res,
        sample_id_col=sample_id_col,
        state_col=state_col,
        cluster_col=cluster_col,
        risk_col=risk_col,
        n_per_cluster=n_per_cluster,
        k_states=k_states,
        include_terminal_state=include_terminal_state,
        ascending_risk=ascending_risk,
    )
    selected_ids = [sid for sid in selected_ids if sid in X.index]
    if not selected_ids:
        raise ValueError("No selected samples overlap feature_df.")

    if selected_features is None:
        features, sig_tbl = select_features_for_residualized_heatmap(
            X,
            meta.reset_index(),
            sample_id_col="SAMPLE_ID",
            state_col=state_col,
            cluster_col=cluster_col,
            k_states=k_states,
            top_n_features=top_n_features,
            q_thresh=q_thresh,
            min_group_n=min_group_n,
            fallback_by_variance=True,
        )
    else:
        sig_tbl = test_features_between_residualized_clusters_by_risk(
            X,
            meta.reset_index(),
            sample_id_col="SAMPLE_ID",
            state_col=state_col,
            cluster_col=cluster_col,
            k_states=k_states,
            min_group_n=min_group_n,
            q_thresh=q_thresh,
        )
        features = [f for f in selected_features if f in X.columns]

    features = [f for f in features if f in X.columns][: int(top_n_features)]
    if not features:
        raise ValueError("No selected features are present in feature_df.")

    H = _zscore_for_heatmap(X, selected_ids, features, value_mode)
    meta_sel = meta.loc[selected_ids].copy()

    # Reorder cluster blocks within each risk group for display only.
    # This does not change cluster assignments or labels.
    H, meta_sel, cluster_display_order = _reorder_within_risk_cluster_columns(
        H,
        meta_sel,
        mode=cluster_order_mode,
    )

    star_map, feature_star_p = _build_within_feature_star_map(
        features,
        sig_tbl,
        method=within_star_p_method,
    )
    H = _apply_feature_labels(H, features, star_map, name_mapper=name_mapper)

    if row_cluster and H.shape[0] > 2:
        H = H.loc[_cluster_row_order(H)].copy()

    if figsize is None:
        figsize = (max(10.5, 0.038 * H.shape[1] + 5.5), max(7.0, 0.24 * H.shape[0] + 2.4))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.16, 0.16, 8.0], width_ratios=[24, 1.25], hspace=0.035, wspace=0.045)
    ax_risk_bar = fig.add_subplot(gs[0, 0])
    ax_sub_bar = fig.add_subplot(gs[1, 0], sharex=ax_risk_bar)
    ax = fig.add_subplot(gs[2, 0], sharex=ax_risk_bar)
    cax = fig.add_subplot(gs[2, 1])

    ax_risk_bar.imshow(np.array([[RISK_PALETTE.get(str(v), (0.8, 0.8, 0.8)) for v in meta_sel["_risk_label"].astype(str)]]), aspect="auto")
    ax_sub_bar.imshow(np.array([[CLUSTER_PALETTE.get(str(v), (0.9, 0.9, 0.9)) for v in meta_sel["_cluster_label"].astype(str)]]), aspect="auto")

    for a, lab in [(ax_risk_bar, "Risk"), (ax_sub_bar, "Cluster")]:
        a.set_xticks([])
        a.set_yticks([])
        a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10, labelpad=28)
        for sp in a.spines.values():
            sp.set_visible(False)

    vmax_use = 2.0 if vmax is None else float(vmax)
    sns.heatmap(H, ax=ax, cmap=cmap, center=0.0, vmin=-vmax_use, vmax=vmax_use, cbar=True, cbar_ax=cax, cbar_kws={"label": cbar_label}, xticklabels=False, yticklabels=True)
    ax.set_xlabel(f"Top {int(n_per_cluster)} samples by risk within each residualized cluster")
    ax.set_ylabel("Features")
    ax.set_yticklabels(_gt_wrap(H.index.tolist(), width=wrap_width), rotation=0, fontsize=ytick_fontsize)
    ax.tick_params(axis="y", length=0)

    col_meta = meta_sel[["_risk_label", "_cluster_label"]].copy()
    n_cols = col_meta.shape[0]
    blocks = []
    subgroup_bounds = []
    risk_blocks = []
    risk_bounds = []

    start = 0
    current = (str(col_meta.iloc[0]["_risk_label"]), str(col_meta.iloc[0]["_cluster_label"]))
    for i in range(1, n_cols):
        here = (str(col_meta.iloc[i]["_risk_label"]), str(col_meta.iloc[i]["_cluster_label"]))
        if here != current:
            subgroup_bounds.append(i)
            blocks.append((start, i, current[0], current[1]))
            start = i
            current = here
    blocks.append((start, n_cols, current[0], current[1]))

    start = 0
    current = str(col_meta.iloc[0]["_risk_label"])
    for i in range(1, n_cols):
        here = str(col_meta.iloc[i]["_risk_label"])
        if here != current:
            risk_bounds.append(i)
            risk_blocks.append((start, i, current))
            start = i
            current = here
    risk_blocks.append((start, n_cols, current))

    for b in subgroup_bounds:
        ax.axvline(b, color="black", linewidth=0.65)
        ax_sub_bar.axvline(b, color="black", linewidth=0.65)
    for b in risk_bounds:
        ax.axvline(b, color="black", linewidth=1.7)
        ax_risk_bar.axvline(b, color="black", linewidth=1.7)
        ax_sub_bar.axvline(b, color="black", linewidth=1.7)

    for s, e, rg in risk_blocks:
        ax_risk_bar.text((s + e) / 2.0 - 0.5, -0.55, rg, ha="center", va="bottom", fontsize=10.5, fontweight="bold", transform=ax_risk_bar.transData, clip_on=False)
    for s, e, _, sg in blocks:
        ax_sub_bar.text((s + e) / 2.0 - 0.5, -0.55, sg, ha="center", va="bottom", fontsize=8.5, transform=ax_sub_bar.transData, clip_on=False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.992)
    if show_star_legend:
        _add_heatmap_star_legend(fig, y=star_legend_y)
    fig.subplots_adjust(left=0.30, right=0.965, top=0.865, bottom=0.055, hspace=0.035, wspace=0.045)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, {
        "kind": "within_risk_clusters",
        "heatmap_matrix": H,
        "ordered_meta": meta_sel,
        "selected_features": features,
        "significance": sig_tbl,
        "feature_star_pvalues": feature_star_p,
        "star_legend": "* p<0.05; ** p<0.01; *** p<0.001",
        "within_star_p_method": within_star_p_method,
        "cluster_order_mode": cluster_order_mode,
        "cluster_display_order": cluster_display_order,
        "risk_order_used": [b[2] for b in risk_blocks],
        "vmax": vmax_use,
        "save_path": save_path,
    }


# -----------------------------------------------------------------------------
# Main heatmap wrapper
# -----------------------------------------------------------------------------


def plot_residualized_gene_heatmap_top_risk_samples(
    gene_expression_df: pd.DataFrame,
    patient_meta_res: pd.DataFrame,
    *,
    save_both_heatmaps: bool = True,
    save_path: str | None = None,
    risk_group_save_path: str | None = None,
    within_cluster_save_path: str | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Gene heatmap wrapper.

    save_both_heatmaps=True returns:
      {
        "risk_groups": {"fig", "info", "save_path"},
        "within_risk_clusters": {"fig", "info", "save_path"},
      }

    save_both_heatmaps=False returns the old single (fig, info) tuple for within-risk clusters.
    """
    kwargs.setdefault("cbar_label", "Gene expression z-score")
    kwargs.setdefault("top_n_features", 40)
    kwargs.setdefault("n_per_cluster", 40)
    kwargs.setdefault("ytick_fontsize", 7)

    if not save_both_heatmaps:
        kwargs.setdefault("title", "Top significant genes — residualized clusters only")
        return plot_residualized_top_risk_feature_heatmap(
            gene_expression_df,
            patient_meta_res,
            save_path=save_path or within_cluster_save_path,
            show=show,
            **kwargs,
        )

    if save_path is not None:
        if risk_group_save_path is None:
            risk_group_save_path = _add_suffix_before_ext(save_path, "_risk_groups")
        if within_cluster_save_path is None:
            within_cluster_save_path = _add_suffix_before_ext(save_path, "_within_risk_clusters")

    common_kwargs = dict(kwargs)

    # Risk-only call: remove cluster-only options and translate n_per_cluster -> n_per_risk_group.
    risk_kwargs = dict(common_kwargs)
    risk_kwargs.pop("cluster_col", None)
    risk_kwargs.pop("ascending_risk", None)
    risk_kwargs.pop("cluster_order_mode", None)
    risk_kwargs.pop("within_star_p_method", None)
    n_per_risk_group = int(risk_kwargs.pop("n_per_risk_group", risk_kwargs.pop("n_per_cluster", kwargs.get("n_per_cluster", 40))))
    if "title" not in risk_kwargs:
        risk_kwargs["title"] = "Top significant genes — risk groups only"

    risk_fig, risk_info = plot_risk_group_top_risk_feature_heatmap(
        gene_expression_df,
        patient_meta_res,
        n_per_risk_group=n_per_risk_group,
        save_path=risk_group_save_path,
        show=show,
        **risk_kwargs,
    )

    # Within-risk call: remove risk-only options.
    cluster_kwargs = dict(common_kwargs)
    cluster_kwargs.pop("n_per_risk_group", None)
    cluster_kwargs.pop("star_p_col", None)
    if "title" not in cluster_kwargs:
        cluster_kwargs["title"] = "Top significant genes — residualized clusters only"

    cluster_fig, cluster_info = plot_residualized_top_risk_feature_heatmap(
        gene_expression_df,
        patient_meta_res,
        save_path=within_cluster_save_path,
        show=show,
        **cluster_kwargs,
    )

    return {
        "risk_groups": {"fig": risk_fig, "info": risk_info, "save_path": risk_group_save_path},
        "within_risk_clusters": {"fig": cluster_fig, "info": cluster_info, "save_path": within_cluster_save_path},
    }


# -----------------------------------------------------------------------------
# Cox helper + forest plots
# -----------------------------------------------------------------------------


def cox_univariate_top_features(
    data: pd.DataFrame,
    *,
    genes: list[str],
    time_col: str,
    event_col: str,
    standardize: bool = True,
    min_n: int = 10,
    min_events: int = 3,
) -> pd.DataFrame:
    """
    Robust univariate Cox helper used by the orientation forest plots.

    Returns columns:
      gene, HR, CI95_low, CI95_high, p, q, coef, se, n, events
    """
    try:
        from lifelines import CoxPHFitter
    except Exception as e:
        raise ImportError("This function requires lifelines. Install with: pip install lifelines") from e

    rows: list[dict[str, Any]] = []
    use_cols = [time_col, event_col]
    for g in genes:
        if g not in data.columns:
            continue
        df = data[[g] + use_cols].copy()
        df[g] = pd.to_numeric(df[g], errors="coerce")
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df[df[time_col] > 0].copy()
        if df.shape[0] < int(min_n):
            continue
        events = int(df[event_col].sum())
        if events < int(min_events) or events >= df.shape[0]:
            continue
        if float(df[g].std(ddof=0)) <= 0 or not np.isfinite(float(df[g].std(ddof=0))):
            continue

        if standardize:
            sd = float(df[g].std(ddof=0))
            df[g] = (df[g] - float(df[g].mean())) / sd

        fit_df = df.rename(columns={g: "x", time_col: "_time", event_col: "_event"})[["x", "_time", "_event"]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph = CoxPHFitter(penalizer=1e-6)
                cph.fit(fit_df, duration_col="_time", event_col="_event", show_progress=False)
            summ = cph.summary.loc["x"]
            coef = float(summ["coef"])
            hr = float(np.exp(coef))
            ci_low = float(np.exp(summ["coef lower 95%"]))
            ci_high = float(np.exp(summ["coef upper 95%"]))
            p = float(summ["p"])
            se = float(summ.get("se(coef)", np.nan))
        except Exception:
            continue

        rows.append({
            "gene": str(g),
            "HR": hr,
            "CI95_low": ci_low,
            "CI95_high": ci_high,
            "p": p,
            "coef": coef,
            "se": se,
            "n": int(fit_df.shape[0]),
            "events": events,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["gene", "HR", "CI95_low", "CI95_high", "p", "q", "coef", "se", "n", "events"])
    out["q"] = _gt_bh_fdr(out["p"].to_numpy(float))
    out = out.sort_values("p", ascending=True).reset_index(drop=True)
    return out


def _extract_heatmap_info_for_cox(x: Any) -> dict[str, Any]:
    """Accept info dict, (fig, info), or {fig, info, save_path}."""
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], dict):
        return x[1]
    if isinstance(x, dict) and "info" in x and isinstance(x["info"], dict):
        return x["info"]
    if isinstance(x, dict):
        return x
    raise ValueError("Could not extract heatmap info dictionary from heatmap_out.")


def _validate_single_heatmap_orientation_via_cox(
    heatmap_info: dict,
    feature_matrix: pd.DataFrame,
    survival_df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    low_label: str = "Low risk",
    high_label: str = "High risk",
    standardize: bool = True,
    title: str = "Univariate Cox for heatmap features",
    save_path: str | None = None,
    show: bool = True,
) -> dict[str, Any]:
    """Orientation check for one heatmap."""
    heatmap_info = _extract_heatmap_info_for_cox(heatmap_info)
    feats = list(heatmap_info.get("selected_features", []))
    if not feats:
        raise ValueError("heatmap_info['selected_features'] is empty.")

    X = feature_matrix.copy()
    X.index = pd.Index([_gt_clean_id_local(x) for x in X.index.astype(str)], name="SAMPLE_ID")
    X = X.loc[:, ~X.columns.duplicated(keep="first")].copy()

    surv = survival_df[[time_col, event_col]].copy()
    surv.index = pd.Index([_gt_clean_id_local(x) for x in surv.index.astype(str)], name="SAMPLE_ID")

    data = X.join(surv, how="inner")
    cox_tbl = cox_univariate_top_features(data, genes=feats, time_col=time_col, event_col=event_col, standardize=standardize)

    present = [f for f in feats if f in set(cox_tbl["gene"].astype(str))]
    missing = [f for f in feats if f not in present]
    if missing:
        print(f"[orientation check] {len(missing)} feature(s) skipped: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if not present:
        raise ValueError("No heatmap features could be tested against survival.")

    t = cox_tbl.copy()
    t["gene"] = t["gene"].astype(str)
    t = t.set_index("gene").reindex(present)

    if "ordered_meta" not in heatmap_info:
        raise ValueError("heatmap_info must contain 'ordered_meta'.")
    meta = heatmap_info["ordered_meta"].copy()
    meta.index = pd.Index([_gt_clean_id_local(x) for x in meta.index.astype(str)], name="SAMPLE_ID")
    if "_risk_label" not in meta.columns:
        raise ValueError("heatmap_info['ordered_meta'] must contain '_risk_label'.")

    low_ids = meta.index[meta["_risk_label"].astype(str) == low_label]
    high_ids = meta.index[meta["_risk_label"].astype(str) == high_label]
    if len(low_ids) == 0:
        raise ValueError(f"No samples found with low_label={low_label!r}.")
    if len(high_ids) == 0:
        raise ValueError(f"No samples found with high_label={high_label!r}.")

    X_side = X.reindex(meta.index)
    side = {}
    for f in present:
        if f not in X_side.columns:
            side[f] = np.nan
            continue
        col = pd.to_numeric(X_side[f], errors="coerce")
        side[f] = float(np.nanmedian(col.reindex(low_ids))) - float(np.nanmedian(col.reindex(high_ids)))
    t["heatmap_low_minus_high"] = pd.Series(side).reindex(present).values

    for c in ["HR", "CI95_low", "CI95_high", "q"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=["HR", "CI95_low", "CI95_high"]).copy()
    if t.empty:
        raise ValueError("No valid Cox rows remained after removing NaNs/infs.")
    t = t.sort_values("HR")

    hr = t["HR"].values
    lo = t["CI95_low"].values
    hi = t["CI95_high"].values
    q = t["q"].values
    y = np.arange(len(t))

    fig, ax = plt.subplots(figsize=(9, 0.34 * len(t) + 1.8))
    colors = ["#B2182B" if h > 1 else "#2166AC" for h in hr]
    for yi, (h, l, u, c) in enumerate(zip(hr, lo, hi, colors)):
        ax.plot([l, u], [yi, yi], color=c, lw=1.6, zorder=1)
        ax.plot(h, yi, "o", color=c, ms=5, zorder=2)

    ax.axvline(1.0, ls="--", lw=1, color="0.4")
    ax.set_xscale("log")
    ax.set_yticks(y)

    def _star(qq: float) -> str:
        if not np.isfinite(qq):
            return ""
        return "***" if qq <= 1e-3 else "**" if qq <= 1e-2 else "*" if qq <= 5e-2 else ""
    # {_star(qi)} (HR={h:.2f}, q={qi:.1e})
    ax.set_yticklabels([f"{_pretty_feature_name_for_plot(g)}" for g, h, qi in zip(t.index, hr, q)], fontsize=8)
    ax.set_xlabel(f"Hazard ratio per SD, 95% CI, log scale — {event_col}")
    ax.set_title(title + "\nred = higher feature value → worse survival; blue = protective")
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#B2182B", lw=3, label="HR > 1, risky"),
            Line2D([0], [0], color="#2166AC", lw=3, label="HR < 1, protective"),
        ],
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

    valid = t.dropna(subset=["heatmap_low_minus_high", "HR"]).copy()
    valid["agrees"] = ((valid["heatmap_low_minus_high"] > 0) & (valid["HR"] < 1)) | ((valid["heatmap_low_minus_high"] < 0) & (valid["HR"] > 1))
    n = int(valid.shape[0])
    n_agree = int(valid["agrees"].sum())
    frac = n_agree / n if n else np.nan

    rho = np.nan
    p_rho = np.nan
    try:
        from scipy.stats import spearmanr
        if n >= 3:
            rho, p_rho = spearmanr(valid["heatmap_low_minus_high"], np.log(valid["HR"]), nan_policy="omit")
    except Exception:
        pass

    verdict = "ORIENTED CORRECTLY" if (np.isfinite(frac) and frac >= 0.5 and (not np.isfinite(rho) or rho < 0)) else "POSSIBLE INVERSION — investigate"

    print("\n================ heatmap orientation check ================")
    print(title)
    print(f"features tested: {n}")
    if np.isfinite(frac):
        print(f"direction agreement: {n_agree}/{n} ({100 * frac:.0f}%) [Low-enriched -> protective / High-enriched -> risky]")
    else:
        print("direction agreement: NA")
    print(f"Spearman(heatmap side, logHR) = {rho:.3f} (p={p_rho:.2g}) [expect NEGATIVE if oriented correctly]")
    print(f"VERDICT: {verdict}")
    print("==========================================================")

    return {
        "cox_table": cox_tbl,
        "merged": t.reset_index().rename(columns={"gene": "feature", "index": "feature"}),
        "n_tested": n,
        "frac_agree": frac,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "verdict": verdict,
        "figure": fig,
        "save_path": save_path,
    }


def validate_heatmap_orientation_via_cox(
    heatmap_out: dict,
    feature_matrix: pd.DataFrame,
    survival_df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    low_label: str = "Low risk",
    high_label: str = "High risk",
    standardize: bool = True,
    title: str | dict = "Univariate Cox for heatmap features",
    save_path: str | dict | None = None,
    show: bool = True,
):
    """Run Cox orientation check for either a single heatmap or the two-heatmap wrapper output."""
    is_two = isinstance(heatmap_out, dict) and ("risk_groups" in heatmap_out or "within_risk_clusters" in heatmap_out)

    if is_two:
        if isinstance(save_path, dict):
            save_risk = save_path.get("risk_groups")
        elif isinstance(save_path, str):
            save_risk = _add_suffix_before_ext(save_path, "_risk_groups")
        else:
            save_risk = None

        if isinstance(title, dict):
            title_risk = title.get("risk_groups", "Main risk groups — univariate Cox for heatmap features")
        else:
            title_risk = "Main risk groups — " + str(title)

        results: dict[str, Any] = {}
        if "risk_groups" in heatmap_out:
            results["risk_groups"] = _validate_single_heatmap_orientation_via_cox(
                heatmap_out["risk_groups"],
                feature_matrix,
                survival_df,
                time_col=time_col,
                event_col=event_col,
                low_label=low_label,
                high_label=high_label,
                standardize=standardize,
                title=title_risk,
                save_path=save_risk,
                show=show,
            )

        return results

    return _validate_single_heatmap_orientation_via_cox(
        heatmap_out,
        feature_matrix,
        survival_df,
        time_col=time_col,
        event_col=event_col,
        low_label=low_label,
        high_label=high_label,
        standardize=standardize,
        title=str(title),
        save_path=save_path if isinstance(save_path, str) else None,
        show=show,
    )






def plot_representative_quiver_runs_pca_z_and_pls_risk(
    got_runs,
    risk_runs,
    score_runs,
    *,
    state_runs=None,
    run_indices=None,
    run_info=None,

    # run selection
    n_medians=3,
    include_best=True,
    higher_is_better=True,

    # only these two methods
    methods=("pca_z", "pls_risk_zw"),

    # quiver settings
    n_clusters=50,
    min_points=10,
    random_state=0,

    # terminal arrows
    include_terminal_incoming=True,
    terminal_arrow_mode="continuation",  # "entry", "continuation", or "reverse"
    terminal_boundary_quantile=1.0,
    terminal_max_arrows=30,
    terminal_min_points=1,

    # visual settings
    ncols=2,
    subplot_size=(6.2, 5.0),
    gridsize=60,
    cmap="viridis",
    hex_alpha=0.60,

    arrow_color="blue",
    terminal_arrow_color="blue",
    arrow_outline_color="white",
    arrow_outline_width=3.5,
    head_width=0.13,
    head_length=0.13,
    width=0.005,

    # arrow scaling
    arrow_max_quantile=70,
    arrow_frac_of_range=0.06,
    terminal_arrow_frac_of_range=0.06,

    # axis crop
    point_region_quantile=0.0,
    axis_pad_fraction=0.03,

    # output
    title_prefix="GOTFlow quiver representative runs",
    colorbar_label="Post-OT predicted risk",
    save_dir=None,
    save_prefix="gotflow_quiver_projection",
    show=False,
):
    """
    Plot representative GOTFlow runs for two projection methods only:
        1) pca_z
        2) pls_risk_zw

    One figure is created per projection method.

    Arrow interpretation:
        - red arrows:
            normal outgoing OT drift arrows from non-terminal states.

        - black arrows:
            terminal-state arrows inferred from the final incoming transition.

    terminal_arrow_mode:
        "entry":
            draw predecessor/intermediate position -> high-risk sample.
            This is the most faithful incoming-transition view.

        "continuation":
            draw from high-risk sample and continue along the incoming direction.
            This is visually more intuitive for showing terminal-state flow.

        "reverse":
            draw high-risk sample -> predecessor/intermediate position.
            This is useful later for reverse-risk / counterfactual analysis.
    """
    import os
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.cross_decomposition import PLSRegression
    from matplotlib.colors import LinearSegmentedColormap, Normalize


    # ------------------------------------------------------------
    # Run selection
    # ------------------------------------------------------------
    if run_indices is None:
        run_indices, run_info = choose_representative_run_indices(
            score_runs,
            n_medians=n_medians,
            include_best=include_best,
            higher_is_better=higher_is_better,
        )
    else:
        run_indices = [int(i) for i in run_indices]

    n_runs = len(run_indices)

    # ------------------------------------------------------------
    # Previous risk colour scheme: green -> orange -> red
    # Use one shared risk scale across all selected runs.
    # ------------------------------------------------------------
    risk_cmap = LinearSegmentedColormap.from_list(
        "risk_gradient",
        ["#2E8B57", "#F4A261", "#D62828"],
        N=256,
    )

    all_risk_vals = []
    for _idx in run_indices:
        r = np.asarray(risk_runs[_idx], float).reshape(-1)
        all_risk_vals.append(r[np.isfinite(r)])

    if len(all_risk_vals) > 0 and sum(len(x) for x in all_risk_vals) > 0:
        all_risk_vals = np.concatenate(all_risk_vals)
        risk_vmin = float(np.nanmin(all_risk_vals))
        risk_vmax = float(np.nanmax(all_risk_vals))
    else:
        risk_vmin, risk_vmax = 0.0, 1.0

    if not np.isfinite(risk_vmin) or not np.isfinite(risk_vmax) or risk_vmax <= risk_vmin:
        risk_vmin, risk_vmax = 0.0, 1.0

    risk_norm = Normalize(vmin=risk_vmin, vmax=risk_vmax)



    # Keep only allowed methods.
    allowed_methods = {"pca_z", "pls_risk_zw"}
    methods = [m for m in methods if m in allowed_methods]

    if len(methods) == 0:
        raise ValueError("No valid methods supplied. Use 'pca_z' and/or 'pls_risk_zw'.")

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _axis_limits(X2):
        q = float(point_region_quantile)

        if q < 0 or q >= 0.5:
            raise ValueError("point_region_quantile must be in [0, 0.5).")

        if q == 0:
            x0, x1 = np.nanmin(X2[:, 0]), np.nanmax(X2[:, 0])
            y0, y1 = np.nanmin(X2[:, 1]), np.nanmax(X2[:, 1])
        else:
            x0, x1 = np.nanquantile(X2[:, 0], [q, 1.0 - q])
            y0, y1 = np.nanquantile(X2[:, 1], [q, 1.0 - q])

        xr = max(x1 - x0, 1e-12)
        yr = max(y1 - y0, 1e-12)

        return (
            x0 - axis_pad_fraction * xr,
            x1 + axis_pad_fraction * xr,
            y0 - axis_pad_fraction * yr,
            y1 + axis_pad_fraction * yr,
        )

    def _scale_arrows(X2, U, frac_of_range=None):
        U = np.asarray(U, float)

        if U.size == 0:
            return U

        if frac_of_range is None:
            frac_of_range = arrow_frac_of_range

        norms = np.linalg.norm(U, axis=1) + 1e-12
        dirs = U / norms[:, None]

        cap = np.percentile(norms, float(arrow_max_quantile))
        norms_clip = np.minimum(norms, cap)

        xmin, ymin = np.min(X2, axis=0)
        xmax, ymax = np.max(X2, axis=0)

        data_range = float(min(xmax - xmin, ymax - ymin))
        max_len = float(frac_of_range) * (data_range if data_range > 0 else 1.0)

        lengths = max_len * (norms_clip / (cap + 1e-12))
        return dirs * lengths[:, None]

    def _safe_state_ids(got, state_runs, run_idx, n):
        """
        Resolve state IDs.

        state_runs[run_idx] can be:
            - DataFrame with state_id column
            - numeric array
            - None

        If state_runs is invalid, falls back to got.bin_ids.
        """
        state_ids = None

        if state_runs is not None:
            candidate = state_runs[run_idx]

            if hasattr(candidate, "columns"):
                cols = list(candidate.columns)

                if "state_id" in cols:
                    state_ids = candidate["state_id"].to_numpy(dtype=int)
                elif "state" in cols:
                    state_ids = candidate["state"].to_numpy(dtype=int)
                elif "bin_id" in cols:
                    state_ids = candidate["bin_id"].to_numpy(dtype=int)
                else:
                    try:
                        state_ids = candidate.iloc[:, -1].to_numpy(dtype=int)
                        print(
                            f"[Quiver] Using last column of state_runs[{run_idx}] "
                            f"as state IDs: {candidate.columns[-1]!r}"
                        )
                    except Exception:
                        state_ids = None
            else:
                try:
                    state_ids = np.asarray(candidate).astype(int).reshape(-1)
                except Exception:
                    state_ids = None

        if state_ids is None:
            state_ids = getattr(got, "bin_ids", None)

        if state_ids is None:
            raise ValueError(f"No valid state IDs for run {run_idx}.")

        state_ids = np.asarray(state_ids).astype(int).reshape(-1)

        if state_ids.shape[0] != n:
            raise ValueError(
                f"state_ids length {state_ids.shape[0]} does not match sample count {n} "
                f"for run {run_idx}."
            )

        return state_ids

    def _project_run(got, risk, state_ids, method):
        """
        Return Z2, V2, title for one projection method.
        """
        z = getattr(got, "z_", None)
        zw = getattr(got, "zw_", None)
        vw = getattr(got, "Vw_", None)

        if z is None:
            raise ValueError("got.z_ is missing.")
        if zw is None:
            raise ValueError("got.zw_ is missing.")
        if vw is None:
            raise ValueError("got.Vw_ is missing.")

        z = np.asarray(z, float)
        zw = np.asarray(zw, float)
        vw = np.asarray(vw, float)

        method = str(method)

        if method == "pca_z":
            scaler = StandardScaler().fit(z)
            Zs = scaler.transform(z)

            pca = PCA(n_components=2, random_state=random_state)
            Z2 = pca.fit_transform(Zs)

            # Drift is stored in whitened latent coordinates as Vw_.
            # This is approximate for pca_z, but it matches the projection style
            # that looked most useful in your earlier plots.
            V_scaled = vw / (scaler.scale_[None, :] + 1e-12)
            V2 = V_scaled @ pca.components_.T

            return Z2, V2, "PCA on z"

        if method == "pls_risk_zw":
            scaler = StandardScaler().fit(zw)
            Zs = scaler.transform(zw)

            y = np.asarray(risk, float).reshape(-1, 1)

            pls = PLSRegression(n_components=2, scale=False)
            pls.fit(Zs, y)

            Z2 = pls.transform(Zs)

            # Project vectors through the same x-space PLS rotation.
            V_scaled = vw / (scaler.scale_[None, :] + 1e-12)
            V2 = V_scaled @ pls.x_rotations_[:, :2]

            return Z2, V2, "PLS on zw supervised by risk"

        raise ValueError(f"Unknown projection method: {method}")

    def _terminal_arrows_display_space(got, Z2, risk, state_ids):
        """
        Compute terminal arrows directly in the displayed 2D projection.

        Uses the final incoming edge S -> T:
            pred_2d = barycentric predecessor position in displayed space
            end_2d  = terminal/high-risk sample position in displayed space

        Then draws according to terminal_arrow_mode.
        """
        state_ids = np.asarray(state_ids).astype(int).reshape(-1)
        risk = np.asarray(risk, float).reshape(-1)

        if getattr(got, "transport_plan_", None) is None:
            return np.empty((0, 2)), np.empty((0, 2))

        terminal_bin = int(np.nanmax(state_ids))

        incoming_edges = sorted(
            [
                (int(s), int(t))
                for (s, t) in got.transport_plan_.keys()
                if int(t) == terminal_bin and int(s) != terminal_bin
            ],
            key=lambda e: e[0],
        )

        if len(incoming_edges) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        source_bin, target_bin = incoming_edges[-1]
        P_plan = np.asarray(got.transport_plan_[(source_bin, target_bin)], float)

        src_idx = np.where(state_ids == source_bin)[0]
        tgt_idx = np.where(state_ids == target_bin)[0]

        if len(src_idx) == 0 or len(tgt_idx) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        if P_plan.shape != (len(src_idx), len(tgt_idx)):
            print(
                f"[Quiver] Terminal plan shape mismatch: plan={P_plan.shape}, "
                f"expected=({len(src_idx)}, {len(tgt_idx)}). Skipping terminal arrows."
            )
            return np.empty((0, 2)), np.empty((0, 2))

        incoming_mass = P_plan.sum(axis=0) + 1e-12

        # Barycentric predecessor in displayed space.
        pred_2d = (P_plan.T @ Z2[src_idx]) / incoming_mass[:, None]

        # Terminal endpoints in displayed space.
        end_2d = Z2[tgt_idx]

        # Select terminal samples.
        r_tgt = risk[tgt_idx]
        finite = np.isfinite(r_tgt)

        if finite.sum() == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        # terminal_boundary_quantile=1.0 means include all finite terminal samples.
        cutoff = np.nanquantile(r_tgt[finite], float(terminal_boundary_quantile))
        candidate_local = np.where(finite & (r_tgt <= cutoff))[0]

        if candidate_local.size == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        # Sort by risk so selected arrows are spread from entry to extreme terminal.
        candidate_local = candidate_local[np.argsort(r_tgt[candidate_local])]

        if candidate_local.size > int(terminal_max_arrows):
            pick = np.linspace(
                0,
                candidate_local.size - 1,
                int(terminal_max_arrows),
            ).astype(int)
            candidate_local = candidate_local[pick]

        pred_sel = pred_2d[candidate_local]
        end_sel = end_2d[candidate_local]
        incoming_vec = end_sel - pred_sel

        mode = str(terminal_arrow_mode).lower()

        if mode == "entry":
            # Faithful incoming transition:
            # predecessor/intermediate position -> high-risk sample.
            start = pred_sel
            vec = incoming_vec

        elif mode == "continuation":
            # Visual terminal continuation:
            # start at high-risk sample and continue along incoming direction.
            start = end_sel
            vec = incoming_vec

        elif mode == "reverse":
            # Counterfactual reverse-risk direction:
            # high-risk sample -> predecessor/intermediate-like position.
            start = end_sel
            vec = -incoming_vec

        else:
            raise ValueError(
                "terminal_arrow_mode must be one of "
                "{'entry', 'continuation', 'reverse'}."
            )

        # Optional clustering/averaging of terminal arrows.
        # Default terminal_min_points=1 means no arrows are discarded.
        if terminal_min_points is not None and int(terminal_min_points) > 1:
            if start.shape[0] >= int(terminal_min_points):
                k_term = min(
                    start.shape[0],
                    max(1, int(start.shape[0] // int(terminal_min_points))),
                )

                if k_term >= 1:
                    cid = KMeans(
                        n_clusters=k_term,
                        n_init="auto",
                        random_state=random_state,
                    ).fit_predict(start)

                    P_list = []
                    U_list = []

                    for c in np.unique(cid):
                        idx = np.where(cid == c)[0]
                        if idx.size < int(terminal_min_points):
                            continue
                        P_list.append(start[idx].mean(axis=0))
                        U_list.append(vec[idx].mean(axis=0))

                    if len(P_list) > 0:
                        start = np.vstack(P_list)
                        vec = np.vstack(U_list)

        return start, vec

    # ------------------------------------------------------------
    # Main loop: one figure per projection method
    # ------------------------------------------------------------
    figs = {}
    outs = {}

    for method in methods:
        ncols_use = int(max(1, ncols))
        nrows = int(math.ceil(n_runs / ncols_use))

        fig_w = subplot_size[0] * ncols_use
        fig_h = subplot_size[1] * nrows

        fig, axes = plt.subplots(nrows, ncols_use, figsize=(fig_w, fig_h))
        axes = np.atleast_1d(axes).ravel()

        last_hb = None
        method_out = {}

        for panel_i, run_idx in enumerate(run_indices):
            ax = axes[panel_i]

            got = got_runs[run_idx]
            risk = np.asarray(risk_runs[run_idx], float).reshape(-1)

            n_samples = np.asarray(got.z_).shape[0]
            state_ids = _safe_state_ids(got, state_runs, run_idx, n_samples)

            try:
                Z2, V2, method_title = _project_run(got, risk, state_ids, method)
            except Exception as e:
                ax.axis("off")
                ax.set_title(
                    f"{method} failed\n{type(e).__name__}: {e}",
                    fontsize=9,
                )
                method_out[int(run_idx)] = {"error": str(e)}
                continue

            # Cluster normal outgoing arrows.
            km = KMeans(
                n_clusters=int(min(n_clusters, Z2.shape[0])),
                n_init="auto",
                random_state=random_state,
            )
            cid = km.fit_predict(Z2)

            P, U, counts, labs = got.__class__.cluster_average(
                Z2,
                V2,
                cid,
                min_points=min_points,
            )

            Uv = _scale_arrows(Z2, U, frac_of_range=arrow_frac_of_range)

            # Terminal arrows are separate and do not affect base P/U.
            if include_terminal_incoming:
                P_term, U_term = _terminal_arrows_display_space(
                    got,
                    Z2,
                    risk,
                    state_ids,
                )
                Uv_term = _scale_arrows(
                    Z2,
                    U_term,
                    frac_of_range=terminal_arrow_frac_of_range,
                )
            else:
                P_term = np.empty((0, 2))
                U_term = np.empty((0, 2))
                Uv_term = np.empty((0, 2))

            hb = ax.hexbin(
            Z2[:, 0],
            Z2[:, 1],
            C=risk,
            reduce_C_function=np.mean,
            gridsize=gridsize,
            cmap=risk_cmap,
            norm=risk_norm,
            mincnt=1,
            alpha=hex_alpha,
            )

            # Separate colourbar per subplot to avoid overlap.
            cbar = fig.colorbar(
                hb,
                ax=ax,
                fraction=0.046,
                pad=0.04,
            )
            cbar.set_label(colorbar_label)

            # Normal outgoing arrows.
            for (x0, y0), (ux, uy) in zip(P, Uv):
                vals = np.array([x0, y0, ux, uy], float)
                if not np.isfinite(vals).all():
                    continue
                if abs(ux) + abs(uy) < 1e-12:
                    continue

                a = ax.arrow(
                    x0,
                    y0,
                    ux,
                    uy,
                    length_includes_head=False,
                    head_width=head_width,
                    head_length=head_length,
                    width=width,
                    color=arrow_color,
                    zorder=10,
                )
                a.set_path_effects([
                    pe.Stroke(
                        linewidth=arrow_outline_width,
                        foreground=arrow_outline_color,
                    ),
                    pe.Normal(),
                ])

            # Terminal arrows.
            for (x0, y0), (ux, uy) in zip(P_term, Uv_term):
                vals = np.array([x0, y0, ux, uy], float)
                if not np.isfinite(vals).all():
                    continue
                if abs(ux) + abs(uy) < 1e-12:
                    continue

                a = ax.arrow(
                    x0,
                    y0,
                    ux,
                    uy,
                    length_includes_head=False,
                    head_width=head_width,
                    head_length=head_length,
                    width=width,
                    color=terminal_arrow_color,
                    zorder=11,
                )
                a.set_path_effects([
                    pe.Stroke(
                        linewidth=arrow_outline_width,
                        foreground=arrow_outline_color,
                    ),
                    pe.Normal(),
                ])

            x0, x1, y0, y1 = _axis_limits(Z2)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

            if run_info is not None and "run_index" in run_info.columns:
                row = run_info[run_info["run_index"] == run_idx]
                if len(row):
                    label = str(row.iloc[0].get("label", f"run {run_idx}"))
                    score = row.iloc[0].get("score", np.nan)
                    panel_title = f"{label}: run {run_idx}, c={score:.3f}"
                else:
                    panel_title = f"run {run_idx}"
            else:
                panel_title = f"run {run_idx}"

            ax.set_title(panel_title)
            ax.set_xlabel("Axis 1")
            ax.set_ylabel("Axis 2")
            ax.grid(alpha=0.2)
            ax.set_aspect(1)
            ax.set_aspect('auto')

            method_out[int(run_idx)] = {
                "Z2": Z2,
                "V2": V2,
                "P": P,
                "U": U,
                "Uv": Uv,
                "P_term": P_term,
                "U_term": U_term,
                "Uv_term": Uv_term,
                "counts": counts,
                "cluster_labels": labs,
                "method": method,
                "terminal_arrow_mode": terminal_arrow_mode,
            }

        for ax in axes[n_runs:]:
            ax.axis("off")

        fig.suptitle(f"{title_prefix} — {method}", fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{save_prefix}_{method}.pdf")
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        figs[method] = fig
        outs[method] = method_out

    return figs, outs



def plot_sample_level_progression(
    progression_df: pd.DataFrame,
    *,
    delta_prefix: str = "mean_delta_",
    state_col: str = "state_mode",
    mass_col: str | None = "mean_transport_mass",
    risk_label_col: str = "risk_group",
    id_col: str | None = "SAMPLE_ID",
    id_truncate: int | None = 12,
    top_genes: int = 8,
    quantiles=(0.10, 0.50, 0.90),
    gene_mode: str = "per_patient",   # "per_patient" or "shared"
    figsize_per_panel=(3.6, 2.8),
    color_up: str = "#D62828",
    color_down: str = "#2E5E8A",
    color_ref: str = "#CCCCCC",
    title: str = "Sample-level progression profiles",
    save_path: str | None = None,
    show: bool = False,
):
    """
    Plot representative sample-level feature-space progression profiles.

    Within each risk group:
      1. compute the transport-mass-weighted mean progression vector;
      2. measure cosine similarity between each sample vector and this mean;
      3. select samples at fixed quantiles of cosine agreement;
      4. plot either each sample's strongest-changing features or a shared
         group-level feature set.

    Returns
    -------
    fig
        Matplotlib figure.
    summary
        Table describing the selected samples.
    """

    df = progression_df.copy()

    delta_cols = [
        c for c in df.columns
        if str(c).startswith(delta_prefix)
    ]
    if not delta_cols:
        raise ValueError(
            f"No progression columns found with prefix '{delta_prefix}'."
        )

    if state_col not in df.columns:
        raise ValueError(f"Missing state column '{state_col}'.")

    if gene_mode not in {"per_patient", "shared"}:
        raise ValueError(
            "gene_mode must be 'per_patient' or 'shared'."
        )

    gene_names = [
        str(c)[len(delta_prefix):]
        for c in delta_cols
    ]

    def _cosine(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na == 0 or nb == 0:
            return np.nan

        return float(np.dot(a, b) / (na * nb))

    def _group_mean_shift(D, mass):
        if mass is None:
            return np.nanmean(D, axis=0)

        w = np.asarray(mass, dtype=float)
        w = np.where(
            np.isfinite(w) & (w > 0),
            w,
            0.0,
        )

        if w.sum() <= 0:
            return np.nanmean(D, axis=0)

        w = w / w.sum()
        return (w[:, None] * D).sum(axis=0)

    def _patient_id(sub, pos):
        if id_col is not None and id_col in sub.columns:
            pid = str(sub[id_col].iloc[pos])
        else:
            pid = str(sub.index[pos])

        if id_truncate is not None:
            pid = pid[:int(id_truncate)]

        return pid

    states = sorted(
        pd.to_numeric(
            df[state_col],
            errors="coerce",
        ).dropna().unique()
    )

    per_state = {}

    for state in states:

        sub = df[
            pd.to_numeric(df[state_col], errors="coerce")
            == state
        ].copy()

        D = sub[delta_cols].to_numpy(dtype=float)

        keep = np.isfinite(D).all(axis=1)
        D = D[keep]
        sub = sub.loc[keep].copy()

        if D.shape[0] < max(5, len(quantiles)):
            continue

        if mass_col is not None and mass_col in sub.columns:
            mass = sub[mass_col].to_numpy(dtype=float)
        else:
            mass = None

        ref = _group_mean_shift(D, mass)

        cosine = np.asarray(
            [_cosine(D[i], ref) for i in range(D.shape[0])],
            dtype=float,
        )

        valid = np.where(np.isfinite(cosine))[0]

        if valid.size == 0:
            continue

        order = valid[np.argsort(cosine[valid])]

        picks = []

        for q in quantiles:
            pos = int(
                np.clip(
                    round(float(q) * (len(order) - 1)),
                    0,
                    len(order) - 1,
                )
            )
            picks.append(int(order[pos]))

        if (
            risk_label_col in sub.columns
            and sub[risk_label_col].notna().any()
        ):
            label = str(
                sub[risk_label_col].mode().iloc[0]
            )
        else:
            label = _state_label(
                int(state),
                k_states=len(states),
            )

        per_state[int(state)] = {
            "D": D,
            "ref": ref,
            "picks": picks,
            "cosine": cosine,
            "label": label,
            "sub": sub,
        }

    if not per_state:
        raise ValueError(
            "No risk groups contained enough valid progression vectors."
        )

    nrows = len(per_state)
    ncols = len(quantiles)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_panel[0] * ncols,
            figsize_per_panel[1] * nrows,
        ),
        squeeze=False,
    )

    summary_rows = []

    for row, (state, values) in enumerate(per_state.items()):

        if gene_mode == "shared":

            shared_idx = np.argsort(
                np.abs(values["ref"])
            )[::-1][:top_genes]

            gene_idx_per_panel = [
                shared_idx
            ] * len(values["picks"])

            scale_values = np.concatenate([
                values["D"][values["picks"]][:, shared_idx].ravel(),
                values["ref"][shared_idx],
            ])

        else:

            gene_idx_per_panel = [
                np.argsort(
                    np.abs(values["D"][patient_idx])
                )[::-1][:top_genes]
                for patient_idx in values["picks"]
            ]

            scale_values = np.concatenate([
                np.r_[
                    values["D"][patient_idx][gene_idx],
                    values["ref"][gene_idx],
                ]
                for patient_idx, gene_idx
                in zip(
                    values["picks"],
                    gene_idx_per_panel,
                )
            ])

        max_abs = (
            float(np.nanmax(np.abs(scale_values)))
            if scale_values.size
            else 1.0
        )

        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0

        xlim = (
            -1.15 * max_abs,
            1.15 * max_abs,
        )

        for col, (q, patient_idx, gene_idx) in enumerate(
            zip(
                quantiles,
                values["picks"],
                gene_idx_per_panel,
            )
        ):

            ax = axes[row, col]

            patient_values = values["D"][patient_idx][gene_idx]
            reference_values = values["ref"][gene_idx]
            genes = [
                gene_names[j]
                for j in gene_idx
            ]

            y = np.arange(len(genes))[::-1]

            ax.barh(
                y,
                reference_values,
                color=color_ref,
                height=0.78,
                zorder=1,
            )

            colors = [
                color_up if value >= 0 else color_down
                for value in patient_values
            ]

            ax.barh(
                y,
                patient_values,
                color=colors,
                height=0.52,
                zorder=2,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.axvline(
                0.0,
                color="0.35",
                linestyle="--",
                linewidth=0.9,
                zorder=3,
            )

            ax.set_yticks(y)

            if gene_mode == "per_patient" or col == 0:
                ax.set_yticklabels(
                    genes,
                    fontsize=7,
                )
            else:
                ax.set_yticklabels([])

            ax.set_xlim(xlim)
            ax.tick_params(
                axis="x",
                labelsize=7,
            )

            pid = _patient_id(
                values["sub"],
                patient_idx,
            )

            cosine_value = values["cosine"][patient_idx]

            ax.set_title(
                f"{pid}\n"
                f"q{int(float(q) * 100)} "
                f"(cos = {cosine_value:+.2f})",
                fontsize=8,
            )

            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            if col == 0:
                ax.set_ylabel(
                    values["label"],
                    fontsize=9,
                    fontweight="bold",
                    labelpad=8,
                )

            if row == nrows - 1:
                ax.set_xlabel(
                    r"$\Delta x$ (progression)",
                    fontsize=8,
                )

            patient_top = set(
                np.argsort(
                    np.abs(values["D"][patient_idx])
                )[::-1][:top_genes]
            )

            group_top = set(
                np.argsort(
                    np.abs(values["ref"])
                )[::-1][:top_genes]
            )

            summary_rows.append({
                "state": state,
                "risk_group": values["label"],
                "quantile": float(q),
                "patient_id": pid,
                "cosine_to_group_mean": cosine_value,
                "top_gene_overlap_with_group":
                    len(patient_top & group_top),
            })

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_up,
            edgecolor="black",
            linewidth=0.5,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_down,
            edgecolor="black",
            linewidth=0.5,
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_ref,
        ),
    ]

    fig.legend(
        handles,
        [
            "increases towards target",
            "decreases towards target",
            "group-level mean shift",
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        title,
        fontsize=12,
        y=1.01,
    )

    fig.tight_layout(
        rect=[0, 0.03, 1, 0.97]
    )

    if save_path is not None:
        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, pd.DataFrame(summary_rows)