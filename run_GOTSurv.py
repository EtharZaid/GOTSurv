                      
                       
"""
Multi-cancer GOTSurv post-analysis runner for paper plots.


  1. choose_representative_run_indices
  2. plot_representative_quiver_runs_pca_z_and_pls_risk
  3. plot_km_subplots_per_edge
  4. plot_residualized_gene_heatmap_top_risk_samples
  5. validate_heatmap_orientation_via_cox
  6. edge_shift_heatmap

All paper plots are saved to:
    Plots/Paper_Plots

Plot filenames follow:
    {cancer}_{modality}_{plot_name}.pdf
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from GOTSurv import GOTFlow, ClusterAwareSurvivalTeacher
import util_file as util


                                                                               
                           
                                                                               

SEED = 42
                                             
# You can choose a set of caner types to run, or just one for testing. The full set of cancer types is:
# ["BLCA","BRCA","CESC", "COAD", "KIRC", "KIRP", "LIHC", "LUAD", "STAD", "UCEC"]                                                               
CANCER_TYPES = ["BRCA"]

#Choose one of the following modalities: "Genes", "WSI"                    
modality = "Genes"

TIME_COL = "DSS.time"
EVENT_COL = "DSS"

MASTER_LOG_CSV = "Results/experiments_log_clustering_bootstrap.csv"
EXP_DETAILS = "multi-cancer paper-only added prog analysis"

RESULTS_DIR = "Results"
PLOTS_DIR = "Plots"

                                                                          
N_BOOTSTRAPS = 10
BOOTSTRAP_SEED = 42

                                                             
K_STATES = 3
BLUR = 0.5
SCALING = 0.9
REACH = 0.01
OT_EPOCHS = 200
OT_LR = 1e-4
WEIGHT_DECAY = 1e-2
DISTILL_TEACHER = False
DISTILL_LOSS = "mse"
ETA_SUP = 1.0
INNER_VAL_FRAC = 0.15
CLUSTER_LOSS_TERMS = ["ranking", "proportion"]

WITHIN_STATE_CLUSTERS = 2
INCLUDE_TERMINAL_ANALYSIS = True

                                                       
COAD_CLINICAL_FILE = "Datasets/TCGA_clinical_NTP-iCMSlabels_updated20211020 1.csv"


                                                                               
                                                                 
                                                                               

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_stage_seed(run_id: int, stage: str) -> int:
    offsets = {"teacher": 0, "ot": 100_000}
    if stage not in offsets:
        raise ValueError(f"Unknown stage: {stage!r}")
    return int(SEED + offsets[stage] + int(run_id))


def train_val_split_indices(n: int, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]


def teacher_default_params() -> dict:
    if K_STATES > 2:
        return {
            "hidden": 64,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "epochs": 500,
            "patience": 40,
            "lambda_cluster": 0.65,
            "lambda_sep": 0.5,
            "lambda_w": 0.1,
            "lambda_ent": 0.01,
            "min_ratio": 0.30, 
        }

    if K_STATES == 2:
        return {
            "hidden": 64,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "epochs": 400,
            "patience": 40,
            "lambda_cluster": 0.5,
            "lambda_sep": 0.005,
            "lambda_w": 0.1,
            "lambda_ent": 0.01,
            "min_ratio": 0.4,
        }

    raise ValueError("K_STATES must be 2, 3, or 4.")


TEACHER_DEFAULT_PARAMS = teacher_default_params()


def pretrain_teacher_with_validation(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    params: dict,
    seed: int,
) -> GOTFlow:
    tr_idx, va_idx = train_val_split_indices(
        len(X_tr),
        val_frac=INNER_VAL_FRAC,
        seed=seed,
    )

    X_sub, X_val = X_tr[tr_idx], X_tr[va_idx]
    t_sub, t_val = t_tr[tr_idx], t_tr[va_idx]
    e_sub, e_val = e_tr[tr_idx], e_tr[va_idx]

    teacher = ClusterAwareSurvivalTeacher(
        input_dim=X_tr.shape[1],
        hidden_dim=int(params.get("hidden", 64)),
        num_clusters=int(K_STATES),
        dropout=float(params.get("dropout", 0.2)),
        lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-3)),
        epochs=int(params.get("epochs", 300)),
        patience=int(params.get("patience", 40)),
        eval_every=5,
        lambda_w=float(params.get("lambda_w", 0.1)),
        lambda_cluster=float(params.get("lambda_cluster", 0.1)),
        lambda_sep=float(params.get("lambda_sep", 0.1)),
        lambda_ent=float(params.get("lambda_ent", 0.01)),
        min_ratio=float(params.get("min_ratio", 0.05)),
        loss_terms=list(params.get("loss_terms", CLUSTER_LOSS_TERMS)),
        random_state=int(seed),
    )

    teacher.fit(
        X_sub,
        t_sub,
        e_sub,
        X_val=X_val,
        time_val=t_val,
        event_val=e_val,
        X_transductive=X_tr,
        verbose=False,
    )

    got = teacher.to_gotflow(blur=BLUR, scaling=SCALING, reach=REACH, debias=True)
    got.pretrain_history_df_ = getattr(teacher, "history_df_", None)
    return got


def choose_representative_run_indices(
    scores,
    *,
    n_medians=3,
    include_best=True,
    higher_is_better=True,
):
    """
    Select representative runs from a list of run scores.

    This is copied from run_gotsurv_v1 so run selection stays unchanged.
    """
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)

    if ok.sum() == 0:
        raise ValueError("No finite run scores available.")

    valid_idx = np.where(ok)[0]
    valid_scores = scores[ok]
    order = np.argsort(valid_scores)

    n_valid = len(valid_scores)
    center = n_valid // 2

    n_medians = int(max(1, n_medians))
    half = n_medians // 2

    positions = list(range(center - half, center - half + n_medians))
    positions = [min(max(p, 0), n_valid - 1) for p in positions]

    median_indices = [int(valid_idx[order[p]]) for p in positions]

    selected = []
    labels = []

    for j, idx in enumerate(median_indices):
        if idx not in selected:
            selected.append(idx)
            labels.append("median" if n_medians == 1 else f"median_{j + 1}")

    if include_best:
        best_local = int(np.nanargmax(valid_scores)) if higher_is_better else int(np.nanargmin(valid_scores))
        best_idx = int(valid_idx[best_local])

        if best_idx not in selected:
            selected.append(best_idx)
            labels.append("best")
        else:
            labels[selected.index(best_idx)] += "+best"

    info = pd.DataFrame({
        "label": labels,
        "run_index": selected,
        "score": [float(scores[i]) for i in selected],
    })

    return selected, info


                                                                               
                                 
                                                                               

def subset_by_ids(values: np.ndarray, all_ids: pd.Index, ids: List[str]) -> np.ndarray:
    s = pd.Series(np.asarray(values), index=all_ids.astype(str))
    return s.reindex(pd.Index(ids).astype(str)).to_numpy()


def _safe_json(x) -> str:
    import json
    try:
        return json.dumps(x, sort_keys=True)
    except Exception:
        return str(x)


def _plot_name(cancer_name: str, modality_name: str, proper_name: str,exp_num: str) -> str:
    proper = str(proper_name).strip().replace(" ", "_")
    return str(Path(PLOTS_DIR) / f"{exp_num}_{cancer_name}_{modality_name}_{proper}.pdf")


def load_filtered_dataset(cancer_name: str, modality_name: str):
    df, _, _, censoring = util.fetch_Dataset(
        cancer_name,
        modality_name,
        time=TIME_COL,
        event=EVENT_COL,
    )

    if cancer_name == "COAD":
        #Anomolous samples that are known to be mislabeled or have other issues
        df = df.drop(
                [x for x in ["TCGA-A6-2679", "TCGA-AA-A004"] if x in df.index],
                axis=0,
            ).copy()

    return df, censoring


def grouping_survival_metrics(time, event, groups, *, rmst_tau=None, run_id=0, phase="native_state"):
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import restricted_mean_survival_time

    t = np.asarray(time, float).reshape(-1)
    e = np.asarray(event, int).reshape(-1)
    g = np.asarray(groups).astype(int).reshape(-1)

    keep = np.isfinite(t) & np.isfinite(e) & np.isfinite(g)
    t, e, g = t[keep], e[keep], g[keep]
    uniq = sorted(np.unique(g).tolist())
    tau_use = float(rmst_tau) if rmst_tau is not None else float(np.nanmax(t))

    if len(uniq) < 2:
        return {
            "run_id": int(run_id),
            "phase": phase,
            "n_test": int(len(t)),
            "groups_present": _safe_json(uniq),
            "n_groups_present": int(len(uniq)),
            "full_k_groups": int(len(uniq) == int(K_STATES)),
            "multivariate_p": np.nan,
            "median_adjacent_pairwise_p": np.nan,
            "median_adjacent_pairwise_q": np.nan,
            "mean_adjacent_abs_rmst_diff": np.nan,
            "extreme_groups_rmst_diff_signed": np.nan,
            "extreme_groups_abs_rmst_diff": np.nan,
            "extreme_groups_hr_high_vs_low": np.nan,
        }, pd.DataFrame()

    try:
        mv = multivariate_logrank_test(t, g, e)
        multivariate_p = float(mv.p_value)
    except Exception:
        multivariate_p = np.nan

    pair_rows = []
    adjacent_abs_rmst = []

    for i in range(len(uniq) - 1):
        s1, s2 = uniq[i], uniq[i + 1]
        m1, m2 = g == s1, g == s2
        p = np.nan
        rmst_diff = np.nan

        if m1.sum() >= 3 and m2.sum() >= 3:
            try:
                lr = logrank_test(
                    t[m1],
                    t[m2],
                    event_observed_A=e[m1],
                    event_observed_B=e[m2],
                )
                p = float(lr.p_value)
            except Exception:
                pass

            try:
                km1 = KaplanMeierFitter().fit(t[m1], e[m1])
                km2 = KaplanMeierFitter().fit(t[m2], e[m2])
                rmst1 = float(restricted_mean_survival_time(km1, t=tau_use))
                rmst2 = float(restricted_mean_survival_time(km2, t=tau_use))
                rmst_diff = float(rmst2 - rmst1)
                adjacent_abs_rmst.append(abs(rmst_diff))
            except Exception:
                pass

        pair_rows.append({
            "run_id": int(run_id),
            "phase": phase,
            "group_1": int(s1),
            "group_2": int(s2),
            "pair_type": "adjacent",
            "p_value": p,
            "rmst_diff_group2_minus_group1": rmst_diff,
        })

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        try:
            pair_df["q_value"] = util._bh_fdr(pair_df["p_value"].to_numpy(float))
        except Exception:
            pair_df["q_value"] = np.nan

    low_g, high_g = uniq[0], uniq[-1]
    m_low, m_high = g == low_g, g == high_g

    try:
        km_low = KaplanMeierFitter().fit(t[m_low], e[m_low])
        km_high = KaplanMeierFitter().fit(t[m_high], e[m_high])
        rmst_low = float(restricted_mean_survival_time(km_low, t=tau_use))
        rmst_high = float(restricted_mean_survival_time(km_high, t=tau_use))
        extreme_signed = float(rmst_low - rmst_high)
        extreme_abs = abs(extreme_signed)
    except Exception:
        extreme_signed, extreme_abs = np.nan, np.nan

    try:
        d_hr = pd.DataFrame({
            "time": np.r_[t[m_low], t[m_high]],
            "event": np.r_[e[m_low], e[m_high]],
            "high_group": np.r_[
                np.zeros(m_low.sum(), dtype=int),
                np.ones(m_high.sum(), dtype=int),
            ],
        })
        cph = CoxPHFitter()
        cph.fit(d_hr, duration_col="time", event_col="event", robust=True)
        hr = float(np.exp(cph.summary.loc["high_group", "coef"]))
    except Exception:
        hr = np.nan

    return {
        "run_id": int(run_id),
        "phase": phase,
        "n_test": int(len(t)),
        "groups_present": _safe_json(uniq),
        "n_groups_present": int(len(uniq)),
        "full_k_groups": int(len(uniq) == int(K_STATES)),
        "multivariate_p": multivariate_p,
        "median_adjacent_pairwise_p": float(np.nanmedian(pair_df["p_value"])) if not pair_df.empty else np.nan,
        "median_adjacent_pairwise_q": (
            float(np.nanmedian(pair_df["q_value"]))
            if not pair_df.empty and "q_value" in pair_df.columns
            else np.nan
        ),
        "mean_adjacent_abs_rmst_diff": float(np.nanmean(adjacent_abs_rmst)) if len(adjacent_abs_rmst) else np.nan,
        "extreme_groups_rmst_diff_signed": extreme_signed,
        "extreme_groups_abs_rmst_diff": extreme_abs,
        "extreme_groups_hr_high_vs_low": hr,
    }, pair_df


def reorder_consensus_states_by_pre_ot_risk(
    agg_df: pd.DataFrame,
    *,
    state_col="state_mode",
    risk_col="mean_pre_ot_risk",
):
    out = agg_df.copy()
    valid = out.dropna(subset=[state_col, risk_col]).copy()

    if valid.empty:
        raise ValueError("No rows with valid consensus states and pre-OT risk are available.")

    valid[state_col] = pd.to_numeric(valid[state_col], errors="coerce").astype(int)
    valid[risk_col] = pd.to_numeric(valid[risk_col], errors="coerce")

    state_risk = (
        valid.groupby(state_col)[risk_col]
        .agg(
            n_samples="size",
            mean_pre_ot_risk="mean",
            median_pre_ot_risk="median",
        )
        .sort_values("mean_pre_ot_risk", ascending=True)
    )

    old_states = state_risk.index.astype(int).tolist()
    remap = {old_state: new_state for new_state, old_state in enumerate(old_states)}

    out[state_col] = pd.to_numeric(out[state_col], errors="coerce").map(remap).astype("Int64")

    for col in ["source_bin_mode", "target_bin_mode"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(remap).astype("Int64")

    ordering_table = state_risk.reset_index().rename(columns={state_col: "old_state"})
    ordering_table["new_state"] = ordering_table["old_state"].map(remap).astype(int)
    ordering_table["old_state_label"] = ordering_table["old_state"].map(
        lambda x: util._state_label(int(x), K_STATES)
    )
    ordering_table["new_state_label"] = ordering_table["new_state"].map(
        lambda x: util._state_label(int(x), K_STATES)
    )
    ordering_table = ordering_table.sort_values("new_state").reset_index(drop=True)

    return out, remap, ordering_table


def apply_consensus_state_remap(
    agg_df: pd.DataFrame,
    remap: dict,
    *,
    state_cols=("state_mode", "source_bin_mode", "target_bin_mode"),
):
    out = agg_df.copy()
    for col in state_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(remap).astype("Int64")
    return out


def aggregate_oob_native_state_assignments(oob_runs: list[pd.DataFrame], all_ids: pd.Index) -> pd.DataFrame:
    if len(oob_runs) == 0:
        raise ValueError("No OOB native state assignments were collected.")

    long = pd.concat(oob_runs, axis=0, ignore_index=True).copy()
    long["SAMPLE_ID"] = long["SAMPLE_ID"].astype(str).map(util._clean_tcga_sample_id)
    long["state_id"] = pd.to_numeric(long["state_id"], errors="coerce")
    long["pre_ot_risk"] = pd.to_numeric(long["pre_ot_risk"], errors="coerce")
    long = long.dropna(subset=["SAMPLE_ID", "state_id", "pre_ot_risk"]).copy()
    long["state_id"] = long["state_id"].astype(int)

    def _mode_or_nan(x):
        m = pd.Series(x).dropna().mode()
        return np.nan if len(m) == 0 else m.iloc[0]

    grouped = long.groupby("SAMPLE_ID")
    out = pd.concat(
        [
            grouped["state_id"].agg(_mode_or_nan).rename("state_mode"),
            grouped["state_id"].mean().rename("state_mean"),
            grouped["pre_ot_risk"].mean().rename("mean_pre_ot_risk"),
            grouped["pre_ot_risk"].median().rename("median_pre_ot_risk"),
            grouped.size().rename("n_oob_predictions"),
        ],
        axis=1,
    )

    out["state_mode"] = pd.to_numeric(out["state_mode"], errors="coerce").astype("Int64")
    return out.reindex(pd.Index(all_ids.astype(str), name="SAMPLE_ID"))


def build_oob_grouped_progression_table(
    agg_std: pd.DataFrame,
    agg_terminal_incoming_std: pd.DataFrame | None,
    oob_state_table: pd.DataFrame,
    *,
    k_states: int,
    include_terminal_analysis: bool,
) -> pd.DataFrame:
    out = agg_std.copy()
    out["drift_role"] = "outgoing_progression"

    terminal_state = int(k_states) - 1

    if include_terminal_analysis and agg_terminal_incoming_std is not None:
        oob_terminal_ids = oob_state_table.index[
            pd.to_numeric(oob_state_table["state_mode"], errors="coerce") == terminal_state
        ]

        replace_ids = (
            out.index
            .intersection(agg_terminal_incoming_std.index)
            .intersection(oob_terminal_ids)
        )

        drift_like_cols = [
            c for c in agg_terminal_incoming_std.columns
            if c.startswith("mean_delta_") or c in [
                "mean_drift_norm",
                "mean_transport_mass",
                "source_bin_mode",
                "target_bin_mode",
                "source_bin_mean",
                "target_bin_mean",
            ]
        ]
        drift_like_cols = [c for c in drift_like_cols if c in out.columns]

        out.loc[replace_ids, drift_like_cols] = agg_terminal_incoming_std.loc[
            replace_ids,
            drift_like_cols,
        ]
        out.loc[replace_ids, "drift_role"] = "terminal_incoming"

    out["state_mode"] = oob_state_table["state_mode"]
    out["mean_pre_ot_risk"] = oob_state_table["mean_pre_ot_risk"]

    if "median_pre_ot_risk" in oob_state_table.columns:
        out["median_pre_ot_risk"] = oob_state_table["median_pre_ot_risk"]

    out = out.dropna(subset=["state_mode", "mean_pre_ot_risk"]).copy()
    out["state_mode"] = pd.to_numeric(out["state_mode"], errors="coerce").astype("Int64")
    return out


def add_readable_group_labels(cluster_assign: pd.DataFrame, naming: dict) -> pd.DataFrame:
    out = cluster_assign.copy()
    out["risk_group"] = pd.to_numeric(out["state_mode"], errors="coerce").map(naming["states"])
    out["within_state_cluster_name"] = pd.to_numeric(out["within_state_cluster"], errors="coerce").map(naming["groups"])
    return out


                                                                               
             
                                                                               

def run_one_cancer(cancer_name: str, modality_name: str) -> dict:
    cancer_name = str(cancer_name).upper()
    modality_name = str(modality_name)
    dataset_name = f"{cancer_name}_{modality_name}"

    print("\n" + "=" * 80)
    print(f"Running {dataset_name}")
    print("=" * 80)

    set_seed(SEED)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    df, censoring = load_filtered_dataset(cancer_name, modality_name)
    rmst_tau = float(np.nanquantile(df[TIME_COL].to_numpy(dtype=float), 0.90))

    all_ids = pd.Index(df.index.astype(str), name="SAMPLE_ID")
    common_ids = pd.Index(df.index.astype(str), name="SAMPLE_ID")
    naming = util.get_progression_naming(K_STATES)

    feature_cols = [c for c in df.columns if c not in [TIME_COL, EVENT_COL]]

    bootstrap_info = util.make_bootstrap_splits(
        df.index.tolist(),
        df[EVENT_COL].to_numpy(),
        n_bootstraps=N_BOOTSTRAPS,
        seed=BOOTSTRAP_SEED,
        min_oob_per_class=1,
    )

    rows = []
    grouping_rows = []
    pairwise_rows = []

    rep_runs_std = []
    rep_runs_latent = []
    rep_runs_latentw = []
    terminal_incoming_rep_runs_std = []

    state_runs = []
    oob_native_state_runs = []

    got_runs = []
    risk_post_runs_for_quiver = []
    post_cindex_runs_for_quiver = []


    for split in tqdm(bootstrap_info, desc=f"{dataset_name} bootstrap runs"):
        run_id = int(split["bootstrap"])

        teacher_seed = get_stage_seed(run_id, "teacher")
        set_seed(teacher_seed)

        train_ids = [str(x) for x in split["train_ids"]]
        test_ids = [str(x) for x in split["test_ids"]]

        tr_df = df.reindex(train_ids).dropna(subset=feature_cols + [TIME_COL, EVENT_COL])
        te_df = df.reindex(test_ids).dropna(subset=feature_cols + [TIME_COL, EVENT_COL])

        X_tr_raw = tr_df[feature_cols].to_numpy(dtype=float)
        X_te_raw = te_df[feature_cols].to_numpy(dtype=float)

        t_tr = tr_df[TIME_COL].to_numpy(dtype=float)
        t_te = te_df[TIME_COL].to_numpy(dtype=float)

        e_tr = tr_df[EVENT_COL].to_numpy(dtype=int)
        e_te = te_df[EVENT_COL].to_numpy(dtype=int)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = (
            scaler.transform(X_te_raw)
            if X_te_raw.shape[0]
            else np.empty((0, X_tr.shape[1]), dtype=float)
        )
        X_all = scaler.transform(df[feature_cols].to_numpy(dtype=float))

        try:
            got = pretrain_teacher_with_validation(
                X_tr,
                t_tr,
                e_tr,
                TEACHER_DEFAULT_PARAMS,
                seed=teacher_seed,
            )
        except Exception as e:
            print(f"[{dataset_name} bootstrap {run_id}] Teacher pretraining failed: {type(e).__name__}: {e}")
            continue

        risk_teacher_tr = got.predict_target(X_tr)
        risk_teacher_te = got.predict_target(X_te) if X_te.shape[0] else np.array([])

        teacher_c_tr = util.compute_cindex(t_tr, e_tr, risk_teacher_tr)
        teacher_c_te = (
            util.compute_cindex(t_te, e_te, risk_teacher_te)
            if X_te.shape[0] and len(np.unique(e_te)) > 1
            else np.nan
        )

        risk_teacher_all = got.predict_target(X_all)
        state_ids_all = got.predict_cluster_states(X_all, order_by="risk")

        ot_seed = get_stage_seed(run_id, "ot")
        set_seed(ot_seed)

        try:
            got.fit(
                X_all,
                state_ids_all,
                learn_phi=True,
                learn_target=True,
                target=risk_teacher_all if DISTILL_TEACHER else None,
                target_type="regression",
                target_loss_name=DISTILL_LOSS,
                use_regression_loss=bool(DISTILL_TEACHER),
                time=None,
                event=None,
                use_survival_loss=False,
                eta=ETA_SUP,
                epochs=OT_EPOCHS,
                lr=OT_LR,
                weight_decay=WEIGHT_DECAY,
                drift_mode="barycentric",
                verbose=False,
                update_states=False,
                sinkhorn_max_iter=1000,
                sinkhorn_tol=1e-6,
                state_kl_weight=0.0,
                use_virtual_drop=False,
            )
        except Exception as e:
            print(f"[{dataset_name} bootstrap {run_id}] OT training failed: {type(e).__name__}: {e}")
            continue

        state_ids_te = subset_by_ids(
            state_ids_all,
            all_ids,
            te_df.index.astype(str).tolist(),
        ).astype(int)

        oob_native_state_runs.append(pd.DataFrame({
            "SAMPLE_ID": te_df.index.astype(str),
            "state_id": state_ids_te,
            "pre_ot_risk": np.asarray(risk_teacher_te, dtype=float).reshape(-1),
            "run_id": int(run_id),
        }))

        native_summary, native_pair_df = grouping_survival_metrics(
            t_te,
            e_te,
            state_ids_te,
            rmst_tau=rmst_tau,
            run_id=run_id,
            phase="native_state_pre_ot",
        )

        grouping_rows.append(native_summary)
        if not native_pair_df.empty:
            pairwise_rows.append(native_pair_df)

        risk_post_tr = got.predict_target(X_tr)
        risk_post_te = got.predict_target(X_te) if X_te.shape[0] else np.array([])

        post_c_tr = util.compute_cindex(t_tr, e_tr, risk_post_tr)
        post_c_te = (
            util.compute_cindex(t_te, e_te, risk_post_te)
            if X_te.shape[0] and len(np.unique(e_te)) > 1
            else np.nan
        )

        feature_df_std = pd.DataFrame(
            X_all,
            index=df.index.astype(str),
            columns=feature_cols,
        )

        rep_std = util.build_patient_progression_representation(
            got,
            feature_df_std,
            risk_teacher_all,
            include_last_bin=False,
        )
        rep_lat = util.build_patient_progression_representation_latent(
            got,
            feature_df_std,
            risk_teacher_all,
            include_last_bin=False,
            use_whitened=False,
        )
        rep_latw = util.build_patient_progression_representation_latent(
            got,
            feature_df_std,
            risk_teacher_all,
            include_last_bin=False,
            use_whitened=True,
        )

        rep_runs_std.append(rep_std)
        rep_runs_latent.append(rep_lat)
        rep_runs_latentw.append(rep_latw)

        if INCLUDE_TERMINAL_ANALYSIS:
            try:
                terminal_incoming_rep_runs_std.append(
                    util.build_terminal_incoming_progression_representation(
                        got,
                        feature_df_std,
                        risk_teacher_all,
                        terminal_bin=K_STATES - 1,
                    )
                )
            except Exception as e:
                print(
                    f"[{dataset_name} bootstrap {run_id}] "
                    f"Terminal incoming analysis failed: {type(e).__name__}: {e}"
                )

        state_runs.append(pd.DataFrame({
            "SAMPLE_ID": df.index.astype(str),
            "state_id": state_ids_all,
        }))

        rows.append({
            "run_id": run_id,
            "teacher_seed": int(teacher_seed),
            "ot_seed": int(ot_seed),
            "n_train": int(tr_df.shape[0]),
            "n_test": int(te_df.shape[0]),
            "events_train": int(e_tr.sum()),
            "events_test": int(e_te.sum()) if len(e_te) else 0,
            "cindex_train_pre_ot": teacher_c_tr,
            "cindex_test_pre_ot": teacher_c_te,
            "cindex_train_post_ot": post_c_tr,
            "cindex_test_post_ot": post_c_te,
            "native_full_k_groups": native_summary.get("full_k_groups", np.nan),
            "native_n_groups_present": native_summary.get("n_groups_present", np.nan),
            "native_multivariate_p": native_summary.get("multivariate_p", np.nan),
            "native_median_adjacent_pairwise_q": native_summary.get("median_adjacent_pairwise_q", np.nan),
            "native_mean_adjacent_abs_rmst_diff": native_summary.get("mean_adjacent_abs_rmst_diff", np.nan),
            "native_extreme_groups_rmst_diff_signed": native_summary.get("extreme_groups_rmst_diff_signed", np.nan),
            "native_extreme_groups_abs_rmst_diff": native_summary.get("extreme_groups_abs_rmst_diff", np.nan),
        })

        risk_post_all = got.predict_target(X_all)

        got_runs.append(got)
        risk_post_runs_for_quiver.append(np.asarray(risk_post_all).reshape(-1))
        post_cindex_runs_for_quiver.append(float(post_c_te))


        print(
            f"[{dataset_name} bootstrap {run_id}] "
            f"PRE train={teacher_c_tr:.4f} test={teacher_c_te:.4f} | "
            f"POST train={post_c_tr:.4f} test={post_c_te:.4f}"
        )

    res = pd.DataFrame(rows)
    grouping_df = pd.DataFrame(grouping_rows)
    pairs_df = (
        pd.concat(pairwise_rows, axis=0, ignore_index=True)
        if pairwise_rows
        else pd.DataFrame()
    )

    if res.empty:
        raise RuntimeError(f"No successful bootstrap runs were completed for {dataset_name}.")

                                                                               
                                                                   
                                                                               
    agg_std = util.aggregate_bootstrap_patient_outputs(
        rep_runs_std,
        state_runs,
    ).set_index("SAMPLE_ID").reindex(common_ids)

    agg_lat = util.aggregate_bootstrap_patient_outputs(
        rep_runs_latent,
        state_runs,
    ).set_index("SAMPLE_ID").reindex(common_ids)

    agg_latw = util.aggregate_bootstrap_patient_outputs(
        rep_runs_latentw,
        state_runs,
    ).set_index("SAMPLE_ID").reindex(common_ids)

    agg_terminal_incoming_std = None
    if INCLUDE_TERMINAL_ANALYSIS and len(terminal_incoming_rep_runs_std) > 0:
        agg_terminal_incoming_std = util.aggregate_bootstrap_patient_outputs(
            terminal_incoming_rep_runs_std,
            state_runs,
        ).set_index("SAMPLE_ID").reindex(common_ids)

    for _agg in [agg_std, agg_lat, agg_latw] + (
        [agg_terminal_incoming_std] if agg_terminal_incoming_std is not None else []
    ):
        if "mean_predicted_risk" in _agg.columns:
            _agg.rename(columns={"mean_predicted_risk": "mean_pre_ot_risk"}, inplace=True)

    agg_std, full_consensus_remap, full_consensus_ordering_table = reorder_consensus_states_by_pre_ot_risk(
        agg_std,
        state_col="state_mode",
        risk_col="mean_pre_ot_risk",
    )
    agg_lat = apply_consensus_state_remap(agg_lat, full_consensus_remap)
    agg_latw = apply_consensus_state_remap(agg_latw, full_consensus_remap)

    if agg_terminal_incoming_std is not None:
        agg_terminal_incoming_std = apply_consensus_state_remap(
            agg_terminal_incoming_std,
            full_consensus_remap,
        )

    oob_native_state_plot_table = aggregate_oob_native_state_assignments(
        oob_native_state_runs,
        common_ids,
    )
    oob_native_state_plot_table, _, oob_consensus_ordering_table = reorder_consensus_states_by_pre_ot_risk(
        oob_native_state_plot_table,
        state_col="state_mode",
        risk_col="mean_pre_ot_risk",
    )

    oob_coverage_n = int(oob_native_state_plot_table["state_mode"].notna().sum())
    oob_coverage_fraction = float(oob_coverage_n / max(len(common_ids), 1))

    print(
        f"\n{dataset_name} OOB consensus native-state coverage: "
        f"{oob_coverage_n}/{len(common_ids)} patients "
        f"({100.0 * oob_coverage_fraction:.1f}%)."
    )

    agg_oob_groups_for_subgroups = build_oob_grouped_progression_table(
        agg_std=agg_std,
        agg_terminal_incoming_std=agg_terminal_incoming_std,
        oob_state_table=oob_native_state_plot_table,
        k_states=K_STATES,
        include_terminal_analysis=INCLUDE_TERMINAL_ANALYSIS,
    )

    cluster_assign_resid = util.assign_within_state_clusters_from_agg_residualized(
        agg_oob_groups_for_subgroups.reset_index(),
        state_col="state_mode",
        delta_prefix="mean_delta_",
        risk_col="mean_pre_ot_risk",
        n_clusters=WITHIN_STATE_CLUSTERS,
        random_state=SEED,
        k_states=K_STATES,
        include_terminal_state=INCLUDE_TERMINAL_ANALYSIS,
    )
    cluster_assign_resid = add_readable_group_labels(cluster_assign_resid, naming)


                                                                               
                                       
                                                                               
    mean_pre = float(np.nanmean(res["cindex_test_pre_ot"]))
    std_pre = float(np.nanstd(res["cindex_test_pre_ot"], ddof=1)) if len(res) > 1 else 0.0
    mean_post = float(np.nanmean(res["cindex_test_post_ot"]))
    std_post = float(np.nanstd(res["cindex_test_post_ot"], ddof=1)) if len(res) > 1 else 0.0

    valid_grouping = (
        grouping_df[grouping_df["full_k_groups"] == 1].copy()
        if not grouping_df.empty
        else pd.DataFrame()
    )
    collapse_rate = (
        1.0 - (len(valid_grouping) / max(len(grouping_df), 1))
        if not grouping_df.empty
        else np.nan
    )

    log_row = {
        "dataset": dataset_name,
        "cancer": cancer_name,
        "modality": modality_name,
        "exp_details": EXP_DETAILS,
        "n_bootstraps": N_BOOTSTRAPS,
        "K_STATES": K_STATES,
        "mean_cindex_pre_ot": mean_pre,
        "std_cindex_pre_ot": std_pre,
        "mean_cindex_post_ot": mean_post,
        "std_cindex_post_ot": std_post,
        "rmst_tau_90pct": float(rmst_tau),
        "native_n_valid_group_runs": int(len(valid_grouping)) if not grouping_df.empty else 0,
        "native_collapse_rate": collapse_rate,
        "native_multivariate_p_median": (
            float(np.nanmedian(valid_grouping["multivariate_p"]))
            if not valid_grouping.empty
            else np.nan
        ),
        "native_median_adjacent_pairwise_q_median": (
            float(np.nanmedian(valid_grouping["median_adjacent_pairwise_q"]))
            if not valid_grouping.empty
            else np.nan
        ),
        "native_mean_adjacent_abs_rmst_diff": (
            float(np.nanmean(valid_grouping["mean_adjacent_abs_rmst_diff"]))
            if not valid_grouping.empty
            else np.nan
        ),
        "native_extreme_groups_rmst_diff_signed_mean": (
            float(np.nanmean(valid_grouping["extreme_groups_rmst_diff_signed"]))
            if not valid_grouping.empty
            else np.nan
        ),
        "native_extreme_groups_abs_rmst_diff_mean": (
            float(np.nanmean(valid_grouping["extreme_groups_abs_rmst_diff"]))
            if not valid_grouping.empty
            else np.nan
        ),
        "native_extreme_groups_abs_rmst_diff_std": (
            float(np.nanstd(valid_grouping["extreme_groups_abs_rmst_diff"]))
            if not valid_grouping.empty
            else np.nan
        ),
        "lambda_cluster": TEACHER_DEFAULT_PARAMS["lambda_cluster"],
        "lambda_sep": TEACHER_DEFAULT_PARAMS["lambda_sep"],
        "lambda_w": TEACHER_DEFAULT_PARAMS["lambda_w"],
        "lambda_ent": TEACHER_DEFAULT_PARAMS["lambda_ent"],
        "min_ratio": TEACHER_DEFAULT_PARAMS["min_ratio"],
        "risk_group_definition": "OOB consensus native states ordered by aggregated pre-OT risk",
        "within_risk_cluster_definition": "Residualized aggregated OT progression drifts within OOB risk groups",
        "oob_consensus_n_patients_covered": int(oob_coverage_n),
        "oob_consensus_coverage_fraction": float(oob_coverage_fraction),
        "include_terminal_analysis": bool(INCLUDE_TERMINAL_ANALYSIS),
        "paper_plots_dir": PLOTS_DIR,
    }

    print("\n================ Bootstrap summary ================")
    print(f"{dataset_name} Mean PRE-OT  test c-index: {mean_pre:.4f} ± {std_pre:.4f}")
    print(f"{dataset_name} Mean POST-OT test c-index: {mean_post:.4f} ± {std_post:.4f}")

    exp_num = util.append_experiment_row(MASTER_LOG_CSV, log_row)

    result_prefix = f"{RESULTS_DIR}/{exp_num}_{dataset_name}"

    res.to_csv(f"{result_prefix}_bootstrap_metrics_by_run.csv", index=False)
    grouping_df.to_csv(f"{result_prefix}_bootstrap_grouping_summary_by_run.csv", index=False)
    pairs_df.to_csv(f"{result_prefix}_bootstrap_state_survival_pairs.csv", index=False)

    full_consensus_ordering_table.to_csv(
        f"{result_prefix}_full_consensus_state_ordering_by_pre_ot_risk.csv",
        index=False,
    )
    oob_consensus_ordering_table.to_csv(
        f"{result_prefix}_oob_consensus_state_ordering_by_pre_ot_risk.csv",
        index=False,
    )
    oob_native_state_plot_table.reset_index().to_csv(
        f"{result_prefix}_oob_consensus_native_state_assignments.csv",
        index=False,
    )

    agg_std.reset_index().to_csv(
        f"{result_prefix}_aggregated_patient_progression_standardized_full_consensus.csv",
        index=False,
    )
    agg_lat.reset_index().to_csv(
        f"{result_prefix}_aggregated_patient_progression_latent_full_consensus.csv",
        index=False,
    )
    agg_latw.reset_index().to_csv(
        f"{result_prefix}_aggregated_patient_progression_latent_whitened_full_consensus.csv",
        index=False,
    )
    agg_oob_groups_for_subgroups.reset_index().to_csv(
        f"{result_prefix}_aggregated_patient_progression_oob_groups_for_subgroups.csv",
        index=False,
    )
    cluster_assign_resid.to_csv(
        f"{result_prefix}_oob_risk_residualized_cluster_assignments.csv",
        index=False,
    )

    if agg_terminal_incoming_std is not None:
        agg_terminal_incoming_std.reset_index().to_csv(
            f"{result_prefix}_aggregated_terminal_incoming_progression_standardized.csv",
            index=False,
        )

                                                                               
                                    
                                                                               
    post_plot_rows = []

    def _record_plot(name: str, path: str | None, status: str, error: str | None = None):
        post_plot_rows.append({
            "cancer": cancer_name,
            "modality": modality_name,
            "dataset": dataset_name,
            "exp_num": exp_num,
            "plot_name": name,
            "save_path": path,
            "status": status,
            "error": error,
        })

                                          
    rep_indices, rep_info = choose_representative_run_indices(
        post_cindex_runs_for_quiver,
        n_medians=6,
        include_best=True,
        higher_is_better=True,
    )
    rep_info.to_csv(f"{result_prefix}_representative_run_indices.csv", index=False)

                                                           
    try:
        util.plot_representative_quiver_runs_pca_z_and_pls_risk(
            got_runs,
            risk_post_runs_for_quiver,
            post_cindex_runs_for_quiver,
            state_runs=state_runs,
            run_indices=rep_indices,
            run_info=rep_info,

            n_clusters=50,
            min_points=10,

            include_terminal_incoming=True,
            terminal_arrow_mode="continuation",
            terminal_boundary_quantile=1.0,
            terminal_max_arrows=30,
            terminal_min_points=1,

            ncols=2,
            subplot_size=(7, 7),
            gridsize=60,

            arrow_color="blue",
            terminal_arrow_color="blue",

            title_prefix=f"{dataset_name} GOTSurv quiver representative runs",
            colorbar_label="Post-OT predicted risk",

            save_dir=PLOTS_DIR,
            save_prefix=f"{exp_num}_{cancer_name}_{modality_name}_gotsurv_quiver_projection",
            show=False,
        )
        _record_plot(
            "gotsurv_quiver_projection",
            str(Path(PLOTS_DIR) / f"{exp_num}_{cancer_name}_{modality_name}_gotsurv_quiver_projection_<method>.pdf"),
            "ok",
        )
    except Exception as e:
        _record_plot("gotsurv_quiver_projection", None, "failed", f"{type(e).__name__}: {e}")
        print(f"[{dataset_name}] Quiver plotting failed: {type(e).__name__}: {e}")


    survival_df = df[[TIME_COL, EVENT_COL]].copy()

                                  
    km_state_ids = (
        oob_native_state_plot_table
        .reindex(df.index)["state_mode"]
        .to_numpy()
    )

    km_save_path = _plot_name(cancer_name, modality_name, "aggregated_km",exp_num)
    try:
        util.plot_km_subplots_per_edge(
            survival_df,
            km_state_ids,
            time_col=TIME_COL,
            event_col=EVENT_COL,
            censor_time=3652,
            suptitle=f"{dataset_name} OOB consensus KM by risk group",
            save_path=km_save_path,
            show=False,
            state_naming=naming,
            show_at_risk=False,
        )
        _record_plot("aggregated_km", km_save_path, "ok")
    except Exception as e:
        _record_plot("aggregated_km", km_save_path, "failed", f"{type(e).__name__}: {e}")
        print(f"[{dataset_name}] KM plotting failed: {type(e).__name__}: {e}")

                                                        
    feature_df_for_heatmap = df[feature_cols].copy()
    feature_df_for_heatmap.index = feature_df_for_heatmap.index.astype(str)

    heatmap_save_path = _plot_name(cancer_name, modality_name, "gene_heatmap_top20", exp_num)
    heatmap_out = None

    try:
        heatmap_out = util.plot_residualized_gene_heatmap_top_risk_samples(
            gene_expression_df=feature_df_for_heatmap,
            patient_meta_res=cluster_assign_resid,
            risk_col="mean_pre_ot_risk",
            n_per_cluster=40,
            top_n_features=15,
            q_thresh=0.05,
            min_group_n=5,
            k_states=K_STATES,
            cluster_order_mode="risk_gradient",
            within_star_p_method="min",
            save_path=heatmap_save_path,
            show=False,
        )
        _record_plot(
            "gene_heatmap_top15",
            heatmap_save_path,
            "ok",
        )
    except TypeError:
                                                                            
                                                            
        heatmap_out = util.plot_residualized_gene_heatmap_top_risk_samples(
            gene_expression_df=feature_df_for_heatmap,
            patient_meta_res=cluster_assign_resid,
            risk_col="mean_pre_ot_risk",
            n_per_cluster=40,
            top_n_features=15,
            q_thresh=0.05,
            min_group_n=5,
            k_states=K_STATES,
            save_path=heatmap_save_path,
            show=False,
        )
        _record_plot("gene_heatmap_top15", heatmap_save_path, "ok_with_legacy_signature")
    except Exception as e:
        _record_plot("gene_heatmap_top15", heatmap_save_path, "failed", f"{type(e).__name__}: {e}")
        print(f"[{dataset_name}] Gene heatmap failed: {type(e).__name__}: {e}")

                                             
    cox_save_path = _plot_name(cancer_name, modality_name, "cox_forest_heatmap_features", exp_num)
    if heatmap_out is not None:
        try:
            util.validate_heatmap_orientation_via_cox(
                heatmap_out,
                feature_matrix=df[feature_cols],
                survival_df=df,
                time_col=TIME_COL,
                event_col=EVENT_COL,
                save_path=cox_save_path,
                show=False,
            )
            _record_plot("cox_forest_heatmap_features", cox_save_path, "ok")
        except Exception as e:
            _record_plot("cox_forest_heatmap_features", cox_save_path, "failed", f"{type(e).__name__}: {e}")
            print(f"[{dataset_name}] Cox orientation forest failed: {type(e).__name__}: {e}")
    else:
        _record_plot("cox_forest_heatmap_features", cox_save_path, "skipped", "heatmap_out unavailable")



    sample_progression_save_path = _plot_name(
        cancer_name,
        modality_name,
        "sample_level_progression",
        exp_num,
    )

    try:
        _, sample_selection_df = util.plot_sample_level_progression(
            cluster_assign_resid,
            delta_prefix="mean_delta_",
            state_col="state_mode",
            mass_col="mean_transport_mass",
            risk_label_col="risk_group",
            id_col="SAMPLE_ID",
            id_truncate=12,
            top_genes=8,
            quantiles=(0.10, 0.50, 0.90),
            gene_mode="per_patient",
            title=f"{dataset_name} sample-level progression profiles",
            save_path=sample_progression_save_path,
            show=False,
        )

        sample_selection_df.to_csv(
            f"{result_prefix}_sample_level_selected_patients.csv",
            index=False,
        )

        _record_plot(
            "sample_level_progression",
            sample_progression_save_path,
            "ok",
        )

    except Exception as e:
        _record_plot(
            "sample_level_progression",
            sample_progression_save_path,
            "failed",
            f"{type(e).__name__}: {e}",
        )
        print(
            f"[{dataset_name}] Sample-level progression plot failed: "
            f"{type(e).__name__}: {e}"
        )                      

    post_plot_df = pd.DataFrame(post_plot_rows)
    post_plot_df.to_csv(f"{result_prefix}_post_plot_status.csv", index=False)

    summary = {
        "cancer": cancer_name,
        "modality": modality_name,
        "dataset": dataset_name,
        "exp_num": int(exp_num),
        "n_samples": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "n_successful_bootstraps": int(res.shape[0]),
        "mean_cindex_pre_ot": mean_pre,
        "std_cindex_pre_ot": std_pre,
        "mean_cindex_post_ot": mean_post,
        "std_cindex_post_ot": std_post,
        "native_n_valid_group_runs": int(len(valid_grouping)) if not grouping_df.empty else 0,
        "native_collapse_rate": collapse_rate,
        "oob_consensus_coverage_fraction": oob_coverage_fraction,
        "exp_num": int(exp_num),
    }

    print(f"\nDone with {dataset_name}. Experiment number: {exp_num}")
    return summary


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    set_seed(SEED)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for cancer_name in CANCER_TYPES:
        try:
            summary = run_one_cancer(cancer_name, modality)
            summary["status"] = "ok"
            summary["error"] = ""
        except Exception as e:
            summary = {
                "cancer": str(cancer_name).upper(),
                "modality": modality,
                "dataset": f"{str(cancer_name).upper()}_{modality}",
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            }
            print(f"[{cancer_name}_{modality}] FAILED: {type(e).__name__}: {e}")

        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = Path(RESULTS_DIR) / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nAll requested cancers finished.")
    print(f"Summary saved to: {summary_path}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
