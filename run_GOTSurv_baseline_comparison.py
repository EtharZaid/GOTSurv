                      
                       
from __future__ import annotations

"""
Bootstrap comparison runner for survival baselines and GOTSurv.

What it does
------------
Models:
- CoxPH
- RSF
- GBM
- Survival SVM
- GOTSurv

Per bootstrap run:
1) fit each model on the bootstrap train split
2) compute test-set c-index
3) create post-hoc risk groups for baseline models using:
   - quantile binning (fit thresholds on train, apply to test)
   - k-means on 1D risk scores (fit on train, apply to test)
4) for GOTSurv, compute native state-group metrics on the test set
5) for every grouping result, compute:
   - multivariate log-rank p-value
   - pairwise log-rank p-values
   - FDR-corrected pairwise p-values
   - mean adjacent absolute RMST difference
   - extreme-group signed RMST difference
   - extreme-group absolute RMST difference
   - extreme-group hazard ratio

Outputs
-------
Results/<exp_num>_baseline_cindex_by_run.csv
Results/<exp_num>_baseline_grouping_summary_by_run.csv
Results/<exp_num>_baseline_grouping_pairwise_by_run.csv
Results/<exp_num>_baseline_failures.csv
"""

import json
import os
import random
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import util_file as util
from GOTSurv import ClusterAwareSurvivalTeacher, GOTFlow

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
from lifelines.utils import concordance_index, restricted_mean_survival_time

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

                                                                    
                    
                                                                    
                                                                 
                                                   
CANCER_TYPES = ["BLCA","BRCA", "CESC","COAD", "LUAD", "KIRC","KIRP","LIHC", "STAD","UCEC"]

modality = "WSI"
TIME_COL = "DSS.time"
EVENT_COL = "DSS"
stage_23 = False
use_KMeans = False

MASTER_LOG_CSV = "Results/experiments_log_baselines_bootstrap.csv"
EXP_DETAILS = "multi-cancer TCGA baseline comparison | no entropy"

                                                                                           
models = ["GOTSurv", "CoxPH", "GBM", "RSF", "SurvivalSVM"]

N_BOOTSTRAPS = 100
BOOTSTRAP_SEED = 42

                                                         
K_STATES = 3

                                                                                 
cancer = CANCER_TYPES[0]
DATASET_NAME = f"{cancer}_{modality}"

                                                                         
drop_anom = True

                                                            
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
CLUSTER_LOSS_TERMS = ["ranking", "proportion","separation"]


def get_teacher_default_params(cancer_name: str, modality_name: str = "Genes") -> dict:
    """Teacher parameters with K fixed at 3, matching run_gotsurv_v1 defaults."""
    cancer_name = str(cancer_name).upper()
    modality_name = str(modality_name)


                                                       
    return {
        "hidden": 64,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "epochs": 600,
        "patience": 40,
        "lambda_cluster": 0.65,
        "lambda_sep": 0.1,
        "lambda_w": 0.3,
        "lambda_ent": 0.01,
        "min_ratio": 0.30,
    }


TEACHER_DEFAULT_PARAMS = get_teacher_default_params(cancer, modality)


def set_active_cancer(cancer_name: str) -> None:
    """Update dataset-dependent globals before running one cancer type."""
    global cancer, DATASET_NAME, TEACHER_DEFAULT_PARAMS
    cancer = str(cancer_name).upper()
    DATASET_NAME = f"{cancer}_{modality}"
    TEACHER_DEFAULT_PARAMS = get_teacher_default_params(cancer, modality)

                                                                  
EVAL_POSTHOC_ON_GOTSURV = True

                                                                    
                 
                                                                    
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
    """Independent deterministic seeds for each bootstrap/model stage."""
    offsets = {
        "teacher": 0,
        "ot": 100_000,
        "rsf": 400_000,
        "gbm": 500_000,
        "svm": 600_000,
        "kmeans": 700_000,
    }
    if stage not in offsets:
        raise ValueError(f"Unknown seeding stage: {stage!r}")
    return int(SEED + offsets[stage] + int(run_id))

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, float)
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

def compute_cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time = np.asarray(time, float)
    event = np.asarray(event, int)
    risk = np.asarray(risk, float)
    return float(concordance_index(time, -risk, event))

def train_val_split_indices(n: int, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_frac)))
    return idx[n_val:], idx[:n_val]

def pretrain_teacher_with_validation(X_tr: np.ndarray, t_tr: np.ndarray, e_tr: np.ndarray, params: dict, seed: int) -> GOTFlow:
    tr_idx, va_idx = train_val_split_indices(len(X_tr), val_frac=INNER_VAL_FRAC, seed=seed)
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
    teacher.fit(X_sub, t_sub, e_sub, X_val=X_val, time_val=t_val, event_val=e_val, X_transductive=X_tr, verbose=False)
    got = teacher.to_gotflow(blur=BLUR, scaling=SCALING, reach=REACH, debias=True)
    return got


def _safe_json(x) -> str:
    try:
        return json.dumps(x, sort_keys=True)
    except Exception:
        return str(x)


                                                                    
                  
                                                                    
def quantile_groups_from_train(train_risk: np.ndarray, test_risk: np.ndarray, k_states: int) -> np.ndarray:
    tr = np.asarray(train_risk, float).reshape(-1)
    te = np.asarray(test_risk, float).reshape(-1)
    probs = np.linspace(0, 1, k_states + 1)[1:-1]
    cuts = np.quantile(tr, probs)
    cuts = np.unique(cuts)
    groups = np.digitize(te, bins=cuts, right=False).astype(int)
    return groups


def kmeans_groups_from_train(train_risk: np.ndarray, test_risk: np.ndarray, k_states: int, seed: int) -> np.ndarray:
    tr = np.asarray(train_risk, float).reshape(-1, 1)
    te = np.asarray(test_risk, float).reshape(-1, 1)
    km = KMeans(n_clusters=int(k_states), random_state=seed, n_init=20)
    km.fit(tr)
    tr_lab = km.predict(tr)
    te_lab = km.predict(te)

                                                                      
    means = {lab: float(np.mean(tr[tr_lab == lab])) for lab in np.unique(tr_lab)}
    ordered = {old: new for new, old in enumerate(sorted(means, key=lambda z: means[z]))}
    te_ord = np.vectorize(ordered.get)(te_lab).astype(int)
    return te_ord


                                                                    
                           
                                                                    
def grouping_survival_metrics(
    time: np.ndarray,
    event: np.ndarray,
    groups: np.ndarray,
    *,
    rmst_tau: float | None = None,
    model_name: str,
    grouping_method: str,
    run_id: int,
) -> Tuple[dict, pd.DataFrame]:
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    g = np.asarray(groups).astype(int)

    keep = np.isfinite(t) & np.isfinite(e) & np.isfinite(g)
    t, e, g = t[keep], e[keep], g[keep]

    uniq = sorted(np.unique(g).tolist())
    if len(uniq) < 2:
        return {
            "run_id": run_id,
            "model": model_name,
            "grouping_method": grouping_method,
            "n_test": int(len(t)),
            "groups_present": _safe_json(uniq),
            "n_groups_present": int(len(uniq)),
            "full_k_groups": int(len(uniq) == int(K_STATES)),
            "multivariate_p": np.nan,
            "mean_adjacent_pairwise_p": np.nan,
            "mean_adjacent_pairwise_q": np.nan,
            "mean_adjacent_abs_rmst_diff": np.nan,
            "extreme_groups_rmst_diff_signed": np.nan,
            "extreme_groups_abs_rmst_diff": np.nan,
            "extreme_groups_hr_high_vs_low": np.nan,
            "extreme_groups_hr_ci_low": np.nan,
            "extreme_groups_hr_ci_high": np.nan,
            "extreme_groups_hr_p": np.nan,
        }, pd.DataFrame()

    try:
        mv = multivariate_logrank_test(t, g, e)
        multivariate_p = float(mv.p_value)
    except Exception:
        multivariate_p = np.nan

    pair_rows = []
    tau_use = float(rmst_tau) if rmst_tau is not None else float(np.nanmax(t))

                    
    adjacent_rmst = []
    for i in range(len(uniq) - 1):
        s1, s2 = uniq[i], uniq[i + 1]
        m1, m2 = g == s1, g == s2
        if m1.sum() < 3 or m2.sum() < 3:
            pair_rows.append({
                "run_id": run_id,
                "model": model_name,
                "grouping_method": grouping_method,
                "group_1": int(s1),
                "group_2": int(s2),
                "pair_type": "adjacent",
                "p_value": np.nan,
                "rmst_diff_group2_minus_group1": np.nan,
            })
            continue

        try:
            lr = logrank_test(t[m1], t[m2], event_observed_A=e[m1], event_observed_B=e[m2])
            p = float(lr.p_value)
        except Exception:
            p = np.nan

        try:
            km1 = KaplanMeierFitter().fit(t[m1], e[m1])
            km2 = KaplanMeierFitter().fit(t[m2], e[m2])
            rmst1 = float(restricted_mean_survival_time(km1, t=tau_use))
            rmst2 = float(restricted_mean_survival_time(km2, t=tau_use))
            rmst_diff = rmst2 - rmst1
            adjacent_rmst.append(abs(rmst_diff))
        except Exception:
            rmst_diff = np.nan

        pair_rows.append({
            "run_id": run_id,
            "model": model_name,
            "grouping_method": grouping_method,
            "group_1": int(s1),
            "group_2": int(s2),
            "pair_type": "adjacent",
            "p_value": p,
            "rmst_diff_group2_minus_group1": rmst_diff,
        })

    pair_df = pd.DataFrame(pair_rows)
    if not pair_df.empty:
        pair_df["q_value"] = _bh_fdr(pair_df["p_value"].to_numpy(float))
    else:
        pair_df["q_value"] = []

                            
    low_g, high_g = uniq[0], uniq[-1]
    m_low, m_high = g == low_g, g == high_g
    try:
        km_low = KaplanMeierFitter().fit(t[m_low], e[m_low])
        km_high = KaplanMeierFitter().fit(t[m_high], e[m_high])
        rmst_low = float(restricted_mean_survival_time(km_low, t=tau_use))
        rmst_high = float(restricted_mean_survival_time(km_high, t=tau_use))
                                                                                              
        extreme_rmst_signed = rmst_low - rmst_high
        extreme_rmst_abs = abs(extreme_rmst_signed)
    except Exception:
        extreme_rmst_signed = np.nan
        extreme_rmst_abs = np.nan

    try:
        d_hr = pd.DataFrame({
            "time": np.r_[t[m_low], t[m_high]],
            "event": np.r_[e[m_low], e[m_high]],
            "high_group": np.r_[np.zeros(m_low.sum(), dtype=int), np.ones(m_high.sum(), dtype=int)],
        })
        cph = CoxPHFitter()
        cph.fit(d_hr, duration_col="time", event_col="event", robust=True)
        s = cph.summary.loc["high_group"]
        hr = float(np.exp(s["coef"]))
        hr_lo = float(np.exp(s["coef lower 95%"]))
        hr_hi = float(np.exp(s["coef upper 95%"]))
        hr_p = float(s["p"])
    except Exception:
        hr, hr_lo, hr_hi, hr_p = np.nan, np.nan, np.nan, np.nan

    summary = {
        "run_id": run_id,
        "model": model_name,
        "grouping_method": grouping_method,
        "n_test": int(len(t)),
        "groups_present": _safe_json(uniq),
        "group_counts": _safe_json({int(s): int(np.sum(g == s)) for s in range(int(K_STATES))}),
        "n_groups_present": int(len(uniq)),
        "full_k_groups": int(len(uniq) == int(K_STATES)),
        "multivariate_p": multivariate_p,
        "median_adjacent_pairwise_p": float(np.nanmedian(pair_df["p_value"])) if not pair_df.empty else np.nan,
        "median_adjacent_pairwise_q": float(np.nanmedian(pair_df["q_value"])) if not pair_df.empty else np.nan,
        "mean_adjacent_abs_rmst_diff": float(np.nanmean(adjacent_rmst)) if len(adjacent_rmst) > 0 else np.nan,
        "extreme_groups_rmst_diff_signed": extreme_rmst_signed,
        "extreme_groups_abs_rmst_diff": extreme_rmst_abs,
        "extreme_groups_hr_high_vs_low": hr,
        "extreme_groups_hr_ci_low": hr_lo,
        "extreme_groups_hr_ci_high": hr_hi,
        "extreme_groups_hr_p": hr_p,
    }
    return summary, pair_df


                                                                    
                         
                                                                    
def fit_predict_coxph(X_tr, t_tr, e_tr, X_te, feature_names):
    df_tr = pd.DataFrame(X_tr, columns=feature_names)
    df_tr["time"] = t_tr
    df_tr["event"] = e_tr
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_tr, duration_col="time", event_col="event", robust=True)
    df_te = pd.DataFrame(X_te, columns=feature_names)
    risk_te = cph.predict_partial_hazard(df_te).to_numpy().reshape(-1)
    risk_tr = cph.predict_partial_hazard(pd.DataFrame(X_tr, columns=feature_names)).to_numpy().reshape(-1)
    return risk_tr, risk_te

def _to_sksurv_struct(time, event):
    from sksurv.util import Surv
    t = np.asarray(time, float).copy()
    t[t <= 0] = 1e-6
    return Surv.from_arrays(event=np.asarray(event, bool), time=t)

def fit_predict_rsf(X_tr, t_tr, e_tr, X_te, seed):
    from sksurv.ensemble import RandomSurvivalForest
    y_tr = _to_sksurv_struct(t_tr, e_tr)
    model = RandomSurvivalForest(
        n_estimators=300,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_tr), model.predict(X_te)

def fit_predict_gbm(X_tr, t_tr, e_tr, X_te, seed):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    y_tr = _to_sksurv_struct(t_tr, e_tr)
    model = GradientBoostingSurvivalAnalysis(
        learning_rate=0.05,
        n_estimators=300,
        max_depth=2,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_tr), model.predict(X_te)

def fit_predict_surv_svm(X_tr, t_tr, e_tr, X_te, seed):
    from sksurv.svm import FastSurvivalSVM
    y_tr = _to_sksurv_struct(t_tr, e_tr)
    model = FastSurvivalSVM(rank_ratio=1.0, max_iter=2000, random_state=seed)
    model.fit(X_tr, y_tr)
    return model.predict(X_tr), model.predict(X_te)




def fit_predict_gotsurv(X_tr, t_tr, e_tr, X_all, X_te, seed, *, all_ids=None, test_ids=None, run_id=None):
    """Fit GOTSurv and return both pre-OT and post-OT risks.

    Native states used for grouping are the pre-OT model states computed on the
    full filtered cohort and then subset to the test IDs, matching the original
    and ablation runners.
    """
    teacher_seed = int(seed if run_id is None else get_stage_seed(int(run_id), "teacher"))
    set_seed(teacher_seed)
    teacher = pretrain_teacher_with_validation(X_tr, t_tr, e_tr, TEACHER_DEFAULT_PARAMS, seed=teacher_seed)

    risk_pre_tr = teacher.predict_target(X_tr)
    risk_pre_te = teacher.predict_target(X_te)
    risk_teacher_all = teacher.predict_target(X_all)
    state_ids_all_pre = teacher.predict_cluster_states(X_all, order_by="risk")

    if all_ids is not None and test_ids is not None:
        state_te_pre = subset_by_ids(state_ids_all_pre, pd.Index(all_ids), list(test_ids)).astype(int)
    else:
        state_te_pre = teacher.predict_cluster_states(X_te, order_by="risk").astype(int)

    got = teacher
    ot_seed = int(seed if run_id is None else get_stage_seed(int(run_id), "ot"))
    set_seed(ot_seed)
    got.fit(
        X_all,
        state_ids_all_pre,
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
    risk_post_tr = got.predict_target(X_tr)
    risk_post_te = got.predict_target(X_te)
    return (
        np.asarray(risk_pre_tr, float),
        np.asarray(risk_pre_te, float),
        np.asarray(risk_post_tr, float),
        np.asarray(risk_post_te, float),
        np.asarray(state_te_pre, int),
    )


                                                                    
                
                                                                    
def run_model(model_name: str, X_tr, t_tr, e_tr, X_all, X_te, feature_names, seed, *, all_ids=None, test_ids=None, run_id=None):
    if model_name == "CoxPH":
        return fit_predict_coxph(X_tr, t_tr, e_tr, X_te, feature_names)
    if model_name == "RSF":
        return fit_predict_rsf(X_tr, t_tr, e_tr, X_te, seed)
    if model_name == "GBM":
        return fit_predict_gbm(X_tr, t_tr, e_tr, X_te, seed)
    if model_name == "SurvivalSVM":
        return fit_predict_surv_svm(X_tr, t_tr, e_tr, X_te, seed)
    if model_name == "GOTSurv":
        return fit_predict_gotsurv(X_tr, t_tr, e_tr, X_all, X_te, seed, all_ids=all_ids, test_ids=test_ids, run_id=run_id)
    raise ValueError(model_name)
def make_paper_ready_summary_table(cindex_df: pd.DataFrame, grouping_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per model + grouping method.

    - For c-index-only rows, grouping_method is 'risk_prediction'.
    - For grouping rows:
        * p-values are summarised by median across runs
        * RMST / HR are summarised by mean ± std across runs
        * only runs with the full requested K groups are used
    """

                                   
                     
                                   
    ctab = (
        cindex_df.groupby("model")["cindex_test"]
        .agg(
            cindex_mean="mean",
            cindex_median="median",
            cindex_std="std",
            n_runs="count",
        )
        .reset_index()
    )

                                   
                               
                                   
    pred_rows = ctab.copy()
    pred_rows["grouping_method"] = "risk_prediction"

    for col in [
        "n_valid_group_runs",
        "collapse_rate",
        "multivariate_p_median",
        "median_adjacent_pairwise_q_median",
        "mean_adjacent_abs_rmst_diff_mean",
        "mean_adjacent_abs_rmst_diff_std",
        "extreme_groups_abs_rmst_diff_mean",
        "extreme_groups_abs_rmst_diff_std",
        "extreme_groups_rmst_diff_signed_mean",
        "extreme_groups_rmst_diff_signed_std",
        "extreme_groups_hr_high_vs_low_mean",
        "extreme_groups_hr_high_vs_low_std",
    ]:
        pred_rows[col] = np.nan

                                   
                            
                                   
    if grouping_df.empty:
        cols = [
            "model",
            "grouping_method",
            "cindex_mean",
            "cindex_median",
            "cindex_std",
            "n_runs",
            "n_valid_group_runs",
            "collapse_rate",
            "multivariate_p_median",
            "median_adjacent_pairwise_q_median",
            "mean_adjacent_abs_rmst_diff_mean",
            "mean_adjacent_abs_rmst_diff_std",
            "extreme_groups_abs_rmst_diff_mean",
            "extreme_groups_abs_rmst_diff_std",
            "extreme_groups_rmst_diff_signed_mean",
            "extreme_groups_rmst_diff_signed_std",
            "extreme_groups_hr_high_vs_low_mean",
            "extreme_groups_hr_high_vs_low_std",
        ]
        return pred_rows[cols].copy()

    g = grouping_df.copy()

                           
    if "full_k_groups" not in g.columns:
        if "n_groups_present" in g.columns:
            g["full_k_groups"] = (pd.to_numeric(g["n_groups_present"], errors="coerce") == int(K_STATES)).astype(int)
        else:
            g["full_k_groups"] = 1

                                   
                               
                                   
    g_counts = (
        g.groupby(["model", "grouping_method"])
        .agg(
            n_group_runs=("run_id", "count"),
            n_valid_group_runs=("full_k_groups", "sum"),
        )
        .reset_index()
    )
    g_counts["collapse_rate"] = 1.0 - (
        g_counts["n_valid_group_runs"] / g_counts["n_group_runs"].clip(lower=1)
    )

                                   
                              
                                   
    g_valid = g[g["full_k_groups"] == 1].copy()

    if g_valid.empty:
        gtab = g_counts.copy()
        for col in [
            "multivariate_p_median",
            "median_adjacent_pairwise_q_median",
            "mean_adjacent_abs_rmst_diff_mean",
            "mean_adjacent_abs_rmst_diff_std",
            "extreme_groups_abs_rmst_diff_mean",
            "extreme_groups_abs_rmst_diff_std",
            "extreme_groups_rmst_diff_signed_mean",
            "extreme_groups_rmst_diff_signed_std",
            "extreme_groups_hr_high_vs_low_mean",
            "extreme_groups_hr_high_vs_low_std",
        ]:
            gtab[col] = np.nan
    else:
                                                            
        pairwise_q_col = (
            "median_adjacent_pairwise_q"
            if "median_adjacent_pairwise_q" in g_valid.columns
            else "mean_adjacent_pairwise_q"
        )

        gtab = (
            g_valid.groupby(["model", "grouping_method"])
            .agg(
                multivariate_p_median=("multivariate_p", "median"),
                median_adjacent_pairwise_q_median=(pairwise_q_col, "median"),
                mean_adjacent_abs_rmst_diff_mean=("mean_adjacent_abs_rmst_diff", "mean"),
                mean_adjacent_abs_rmst_diff_std=("mean_adjacent_abs_rmst_diff", "std"),
                extreme_groups_abs_rmst_diff_mean=("extreme_groups_abs_rmst_diff", "mean"),
                extreme_groups_abs_rmst_diff_std=("extreme_groups_abs_rmst_diff", "std"),
                extreme_groups_rmst_diff_signed_mean=("extreme_groups_rmst_diff_signed", "mean"),
                extreme_groups_rmst_diff_signed_std=("extreme_groups_rmst_diff_signed", "std"),
                extreme_groups_hr_high_vs_low_mean=("extreme_groups_hr_high_vs_low", "mean"),
                extreme_groups_hr_high_vs_low_std=("extreme_groups_hr_high_vs_low", "std"),
            )
            .reset_index()
        )

        gtab = g_counts.merge(gtab, on=["model", "grouping_method"], how="left")

                                   
                            
                                   
    gtab = gtab.merge(ctab, on="model", how="left")

                                   
                                        
                                   
    out = pd.concat([pred_rows, gtab], axis=0, ignore_index=True, sort=False)

    desired = [
        "model",
        "grouping_method",
        "cindex_mean",
        "cindex_median",
        "cindex_std",
        "n_runs",
        "n_valid_group_runs",
        "collapse_rate",
        "multivariate_p_median",
        "median_adjacent_pairwise_q_median",
        "mean_adjacent_abs_rmst_diff_mean",
        "mean_adjacent_abs_rmst_diff_std",
        "extreme_groups_abs_rmst_diff_mean",
        "extreme_groups_abs_rmst_diff_std",
        "extreme_groups_rmst_diff_signed_mean",
        "extreme_groups_rmst_diff_signed_std",
        "extreme_groups_hr_high_vs_low_mean",
        "extreme_groups_hr_high_vs_low_std",
    ]

    for c in desired:
        if c not in out.columns:
            out[c] = np.nan

    out = out[desired].copy()

    method_order = {
        "risk_prediction": 0,
        "native_state": 1,
        "quantile": 2,
        "kmeans": 3,
    }
    out["_mord"] = out["grouping_method"].map(method_order).fillna(99)
    out = out.sort_values(["model", "_mord"]).drop(columns="_mord").reset_index(drop=True)

    return out


def subset_by_ids(values: np.ndarray, all_ids: pd.Index, ids: List[str]) -> np.ndarray:
    s = pd.Series(np.asarray(values), index=all_ids.astype(str))
    return s.reindex(pd.Index(ids).astype(str)).to_numpy()


                                                                    
                     
                                                                    
def prepare_dataset_for_comparison() -> Tuple[pd.DataFrame, List[str], float]:
    """Match the analysis runners' cohort filtering before all models are fit."""
    df, _, _, censoring = util.fetch_Dataset(cancer, modality, time=TIME_COL, event=EVENT_COL)


                                                    
    if drop_anom and cancer == "COAD":
        df = df.drop([x for x in ["TCGA-A6-2679", "TCGA-AA-A004"] if x in df.index], axis=0).copy()

                                                                    
    if stage_23 and cancer == "COAD":
        stages = [
            "Stage II", "Stage IIA", "Stage IIB", "Stage IIC",
            "Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC",
        ]
        df_survival = pd.read_excel("./Datasets/TCGA-CDR-SupplementalTableS1.xlsx")
        df_survival = df_survival.loc[:, ["bcr_patient_barcode", "ajcc_pathologic_tumor_stage"]]
        df_survival.set_index("bcr_patient_barcode", inplace=True)
        df_survival.index.set_names("SAMPLE_ID", inplace=True)
        df = df.join(df_survival, on="SAMPLE_ID", how="inner")
        df = df[df["ajcc_pathologic_tumor_stage"].isin(stages)]
        df.drop(["ajcc_pathologic_tumor_stage"], axis=1, inplace=True)

    feature_names = [c for c in df.columns if c not in [TIME_COL, EVENT_COL]]
    rmst_tau = float(np.nanquantile(df[TIME_COL].to_numpy(dtype=float), 0.90))
    return df, feature_names, rmst_tau


                                                                    
                
                                                                    
def _fmt_mean_std(mean_value, std_value, digits: int = 3) -> str:
    if not np.isfinite(mean_value):
        return ""
    if not np.isfinite(std_value):
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


def _fmt_p(p_value) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 1e-3:
        return f"{p_value:.2e}"
    return f"{p_value:.3f}"


def add_paper_ready_formatted_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add string columns that can be pasted directly into a manuscript table."""
    out = summary_df.copy()
    if out.empty:
        return out

    out["cindex_mean_std"] = [
        _fmt_mean_std(m, s, digits=3)
        for m, s in zip(out.get("cindex_mean", np.nan), out.get("cindex_std", np.nan))
    ]
    out["adjacent_rmst_mean_std"] = [
        _fmt_mean_std(m, s, digits=1)
        for m, s in zip(out.get("mean_adjacent_abs_rmst_diff_mean", np.nan), out.get("mean_adjacent_abs_rmst_diff_std", np.nan))
    ]
    out["extreme_rmst_mean_std"] = [
        _fmt_mean_std(m, s, digits=1)
        for m, s in zip(out.get("extreme_groups_abs_rmst_diff_mean", np.nan), out.get("extreme_groups_abs_rmst_diff_std", np.nan))
    ]
    out["hr_high_vs_low_mean_std"] = [
        _fmt_mean_std(m, s, digits=2)
        for m, s in zip(out.get("extreme_groups_hr_high_vs_low_mean", np.nan), out.get("extreme_groups_hr_high_vs_low_std", np.nan))
    ]
    out["collapse_rate_percent"] = [
        "" if not np.isfinite(x) else f"{100.0 * x:.1f}%"
        for x in out.get("collapse_rate", pd.Series(np.nan, index=out.index))
    ]
    out["multivariate_p_median_fmt"] = [_fmt_p(x) for x in out.get("multivariate_p_median", pd.Series(np.nan, index=out.index))]
    out["adjacent_q_median_fmt"] = [_fmt_p(x) for x in out.get("median_adjacent_pairwise_q_median", pd.Series(np.nan, index=out.index))]
    return out


def make_one_cancer_by_run_table(
    cindex_df: pd.DataFrame,
    grouping_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    *,
    cancer_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """One easy-to-inspect long CSV per cancer containing c-index, grouping and failure rows."""
    blocks = []

    if not cindex_df.empty:
        c = cindex_df.copy()
        if "cancer" not in c.columns:
            c.insert(0, "cancer", cancer_name)
        if "dataset" not in c.columns:
            c.insert(1, "dataset", dataset_name)
        c["result_type"] = "cindex"
        c["grouping_method"] = "risk_prediction"
        blocks.append(c)

    if not grouping_df.empty:
        g = grouping_df.copy()
        if "cancer" not in g.columns:
            g.insert(0, "cancer", cancer_name)
        if "dataset" not in g.columns:
            g.insert(1, "dataset", dataset_name)
        g["result_type"] = "grouping"
        blocks.append(g)

    if not failures_df.empty:
        f = failures_df.copy()
        if "cancer" not in f.columns:
            f.insert(0, "cancer", cancer_name)
        if "dataset" not in f.columns:
            f.insert(1, "dataset", dataset_name)
        f["result_type"] = "failure"
        blocks.append(f)

    if not blocks:
        return pd.DataFrame(columns=["cancer", "dataset", "result_type"])

    return pd.concat(blocks, axis=0, ignore_index=True, sort=False)


def make_cindex_summary(cindex_df: pd.DataFrame) -> pd.DataFrame:
    if cindex_df.empty:
        return pd.DataFrame()
    return (
        cindex_df.groupby("model")
        .agg(
            cindex_mean=("cindex_test", "mean"),
            cindex_median=("cindex_test", "median"),
            cindex_std=("cindex_test", "std"),
            cindex_test_pre_ot_mean=("cindex_test_pre_ot", "mean"),
            cindex_test_pre_ot_std=("cindex_test_pre_ot", "std"),
            cindex_test_post_ot_mean=("cindex_test_post_ot", "mean"),
            cindex_test_post_ot_std=("cindex_test_post_ot", "std"),
            n_runs=("cindex_test", "count"),
        )
        .reset_index()
        .sort_values("cindex_mean", ascending=False)
    )


def make_grouping_summary(grouping_df: pd.DataFrame) -> pd.DataFrame:
    if grouping_df.empty:
        return pd.DataFrame()

    grouping_counts = (
        grouping_df.groupby(["model", "grouping_method"])
        .agg(
            n_group_runs=("run_id", "count"),
            n_valid_group_runs=("full_k_groups", "sum"),
        )
        .reset_index()
    )
    grouping_counts["collapse_rate"] = 1.0 - (
        grouping_counts["n_valid_group_runs"] / grouping_counts["n_group_runs"].clip(lower=1)
    )

    grouping_valid = grouping_df[grouping_df["full_k_groups"] == 1].copy()
    if grouping_valid.empty:
        return grouping_counts.copy()

    grouping_summary = (
        grouping_valid.groupby(["model", "grouping_method"])
        .agg(
            multivariate_p_median=("multivariate_p", "median"),
            median_adjacent_pairwise_q_median=("median_adjacent_pairwise_q", "median"),
            mean_adjacent_abs_rmst_diff_mean=("mean_adjacent_abs_rmst_diff", "mean"),
            mean_adjacent_abs_rmst_diff_std=("mean_adjacent_abs_rmst_diff", "std"),
            extreme_groups_rmst_diff_mean=("extreme_groups_abs_rmst_diff", "mean"),
            extreme_groups_rmst_diff_std=("extreme_groups_abs_rmst_diff", "std"),
            extreme_groups_hr_high_vs_low_mean=("extreme_groups_hr_high_vs_low", "mean"),
            extreme_groups_hr_high_vs_low_std=("extreme_groups_hr_high_vs_low", "std"),
        )
        .reset_index()
    )
    return grouping_counts.merge(grouping_summary, on=["model", "grouping_method"], how="left")


def compare_gotsurv_to_best_model(
    cindex_df: pd.DataFrame,
    *,
    cancer_name: str,
    dataset_name: str,
    my_model: str = "GOTSurv",
    save_path: str | None = None,
) -> pd.DataFrame:
    """Paired bootstrap comparison of GOTSurv against the best available non-GOTSurv model."""
    from scipy.stats import ttest_rel, wilcoxon

    if cindex_df.empty or my_model not in set(cindex_df.get("model", [])):
        out = pd.DataFrame()
        if save_path is not None:
            out.to_csv(save_path, index=False)
        return out

    d = cindex_df.copy()
    d["cindex_test"] = pd.to_numeric(d["cindex_test"], errors="coerce")
    mean_perf = d.groupby("model")["cindex_test"].mean().sort_values(ascending=False)
    other_models = mean_perf.drop(index=my_model, errors="ignore")
    if other_models.empty:
        out = pd.DataFrame([{
            "cancer": cancer_name,
            "dataset": dataset_name,
            "my_model": my_model,
            "best_other_model": np.nan,
            "n_paired_bootstraps": 0,
            "mean_cindex_my_model": float(mean_perf.get(my_model, np.nan)),
            "mean_cindex_best_other": np.nan,
            "mean_difference_my_minus_best": np.nan,
            "median_difference_my_minus_best": np.nan,
            "diff_95ci_low": np.nan,
            "diff_95ci_high": np.nan,
            "wilcoxon_p": np.nan,
            "paired_ttest_p": np.nan,
        }])
        if save_path is not None:
            out.to_csv(save_path, index=False)
        return out

    best_other = other_models.index[0]
    wide = (
        d.pivot_table(index="run_id", columns="model", values="cindex_test", aggfunc="mean")
        .reindex(columns=[my_model, best_other])
        .dropna()
    )
    if wide.empty:
        out = pd.DataFrame()
        if save_path is not None:
            out.to_csv(save_path, index=False)
        return out

    diff = wide[my_model] - wide[best_other]
    try:
        wilcoxon_p = float(wilcoxon(diff, alternative="two-sided").pvalue)
    except Exception:
        wilcoxon_p = np.nan
    try:
        paired_t_p = float(ttest_rel(wide[my_model], wide[best_other]).pvalue)
    except Exception:
        paired_t_p = np.nan

    out = pd.DataFrame([{
        "cancer": cancer_name,
        "dataset": dataset_name,
        "my_model": my_model,
        "best_other_model": best_other,
        "n_paired_bootstraps": int(wide.shape[0]),
        "mean_cindex_my_model": float(wide[my_model].mean()),
        "mean_cindex_best_other": float(wide[best_other].mean()),
        "mean_difference_my_minus_best": float(diff.mean()),
        "median_difference_my_minus_best": float(diff.median()),
        "diff_95ci_low": float(np.percentile(diff, 2.5)),
        "diff_95ci_high": float(np.percentile(diff, 97.5)),
        "wilcoxon_p": wilcoxon_p,
        "paired_ttest_p": paired_t_p,
    }])
    if save_path is not None:
        out.to_csv(save_path, index=False)
    return out


                                                                    
                  
                                                                    
def run_one_cancer(cancer_name: str) -> dict:
    set_active_cancer(cancer_name)
    set_seed(SEED)
    os.makedirs("Results", exist_ok=True)
    os.makedirs("Results/Comparisons results", exist_ok=True)

    df, feature_names, rmst_tau = prepare_dataset_for_comparison()
    all_ids = pd.Index(df.index.astype(str), name="SAMPLE_ID")

    bootstrap_info = util.make_bootstrap_splits(
        df.index.tolist(),
        df[EVENT_COL].to_numpy(),
        n_bootstraps=N_BOOTSTRAPS,
        seed=BOOTSTRAP_SEED,
        min_oob_per_class=1,
    )

    cindex_rows = []
    grouping_rows = []
    pairwise_rows = []
    failure_rows = []

    for split in tqdm(bootstrap_info, desc=f"{cancer} bootstrap comparison"):
        run_id = int(split["bootstrap"])
        train_ids = [str(x) for x in split["train_ids"]]
        test_ids = [str(x) for x in split["test_ids"]]

        tr_df = df.reindex(train_ids).dropna(subset=feature_names + [TIME_COL, EVENT_COL])
        te_df = df.reindex(test_ids).dropna(subset=feature_names + [TIME_COL, EVENT_COL])

        X_tr_raw = tr_df[feature_names].to_numpy(dtype=float)
        X_te_raw = te_df[feature_names].to_numpy(dtype=float)
        t_tr = tr_df[TIME_COL].to_numpy(dtype=float)
        t_te = te_df[TIME_COL].to_numpy(dtype=float)
        e_tr = tr_df[EVENT_COL].to_numpy(dtype=int)
        e_te = te_df[EVENT_COL].to_numpy(dtype=int)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw) if X_te_raw.shape[0] else np.empty((0, X_tr.shape[1]), dtype=float)
        X_all = scaler.transform(df[feature_names].to_numpy(dtype=float))

        for model_name in models:
            model_seed = np.nan
            try:
                if model_name == "GOTSurv":
                                                                                     
                    model_seed = get_stage_seed(run_id, "teacher")
                    risk_pre_tr, risk_pre_te, risk_tr, risk_te, state_te = run_model(
                        model_name,
                        X_tr,
                        t_tr,
                        e_tr,
                        X_all,
                        X_te,
                        feature_names,
                        model_seed,
                        all_ids=all_ids,
                        test_ids=te_df.index.astype(str).tolist(),
                        run_id=run_id,
                    )
                else:
                    if model_name == "RSF":
                        model_seed = get_stage_seed(run_id, "rsf")
                    elif model_name == "GBM":
                        model_seed = get_stage_seed(run_id, "gbm")
                    elif model_name == "SurvivalSVM":
                        model_seed = get_stage_seed(run_id, "svm")
                    else:
                        model_seed = int(SEED + run_id)

                    set_seed(int(model_seed))
                    risk_tr, risk_te = run_model(
                        model_name,
                        X_tr,
                        t_tr,
                        e_tr,
                        X_all,
                        X_te,
                        feature_names,
                        int(model_seed),
                        run_id=run_id,
                    )
                    risk_tr = np.asarray(risk_tr, float).reshape(-1)
                    risk_te = np.asarray(risk_te, float).reshape(-1)
                    risk_pre_tr, risk_pre_te = risk_tr, risk_te

                    if not np.isfinite(risk_tr).all() or not np.isfinite(risk_te).all():
                        raise ValueError(f"{model_name} produced non-finite risk scores.")
                    state_te = None

                c_pre_tr = compute_cindex(t_tr, e_tr, risk_pre_tr) if X_tr.shape[0] and np.isfinite(risk_pre_tr).all() else np.nan
                c_pre_te = compute_cindex(t_te, e_te, risk_pre_te) if X_te.shape[0] and np.isfinite(risk_pre_te).all() else np.nan
                c_post_tr = compute_cindex(t_tr, e_tr, risk_tr) if X_tr.shape[0] and np.isfinite(risk_tr).all() else np.nan
                c_te = compute_cindex(t_te, e_te, risk_te) if X_te.shape[0] and np.isfinite(risk_te).all() else np.nan

                cindex_rows.append({
                    "cancer": cancer,
                    "dataset": DATASET_NAME,
                    "run_id": run_id,
                    "model": model_name,
                    "model_seed": int(model_seed),
                    "teacher_seed": int(get_stage_seed(run_id, "teacher")) if model_name == "GOTSurv" else np.nan,
                    "ot_seed": int(get_stage_seed(run_id, "ot")) if model_name == "GOTSurv" else np.nan,
                    "n_train": int(tr_df.shape[0]),
                    "n_test": int(te_df.shape[0]),
                    "events_train": int(e_tr.sum()),
                    "events_test": int(e_te.sum()) if len(e_te) else 0,
                    "cindex_train_pre_ot": c_pre_tr,
                    "cindex_test_pre_ot": c_pre_te,
                    "cindex_train_post_ot": c_post_tr,
                    "cindex_test_post_ot": c_te,
                    "cindex_test": c_te,
                })

                if model_name != "GOTSurv":
                    q_groups = quantile_groups_from_train(risk_tr, risk_te, K_STATES)
                    q_summary, q_pair_df = grouping_survival_metrics(
                        t_te,
                        e_te,
                        q_groups,
                        rmst_tau=rmst_tau,
                        model_name=model_name,
                        grouping_method="quantile",
                        run_id=run_id,
                    )
                    q_summary.update({"cancer": cancer, "dataset": DATASET_NAME})
                    grouping_rows.append(q_summary)
                    if not q_pair_df.empty:
                        q_pair_df.insert(0, "cancer", cancer)
                        q_pair_df.insert(1, "dataset", DATASET_NAME)
                        pairwise_rows.append(q_pair_df)

                    if use_KMeans:
                        km_groups = kmeans_groups_from_train(risk_tr, risk_te, K_STATES, seed=get_stage_seed(run_id, "kmeans"))
                        km_summary, km_pair_df = grouping_survival_metrics(
                            t_te,
                            e_te,
                            km_groups,
                            rmst_tau=rmst_tau,
                            model_name=model_name,
                            grouping_method="kmeans",
                            run_id=run_id,
                        )
                        km_summary.update({"cancer": cancer, "dataset": DATASET_NAME})
                        grouping_rows.append(km_summary)
                        if not km_pair_df.empty:
                            km_pair_df.insert(0, "cancer", cancer)
                            km_pair_df.insert(1, "dataset", DATASET_NAME)
                            pairwise_rows.append(km_pair_df)

                else:
                    native_summary, native_pair_df = grouping_survival_metrics(
                        t_te,
                        e_te,
                        state_te,
                        rmst_tau=rmst_tau,
                        model_name=model_name,
                        grouping_method="native_state",
                        run_id=run_id,
                    )
                    native_summary.update({"cancer": cancer, "dataset": DATASET_NAME})
                    grouping_rows.append(native_summary)
                    if not native_pair_df.empty:
                        native_pair_df.insert(0, "cancer", cancer)
                        native_pair_df.insert(1, "dataset", DATASET_NAME)
                        pairwise_rows.append(native_pair_df)

                    if EVAL_POSTHOC_ON_GOTSURV:
                        q_groups = quantile_groups_from_train(risk_tr, risk_te, K_STATES)
                        q_summary, q_pair_df = grouping_survival_metrics(
                            t_te,
                            e_te,
                            q_groups,
                            rmst_tau=rmst_tau,
                            model_name=model_name,
                            grouping_method="quantile",
                            run_id=run_id,
                        )
                        q_summary.update({"cancer": cancer, "dataset": DATASET_NAME})
                        grouping_rows.append(q_summary)
                        if not q_pair_df.empty:
                            q_pair_df.insert(0, "cancer", cancer)
                            q_pair_df.insert(1, "dataset", DATASET_NAME)
                            pairwise_rows.append(q_pair_df)

                        if use_KMeans:
                            km_groups = kmeans_groups_from_train(risk_tr, risk_te, K_STATES, seed=get_stage_seed(run_id, "kmeans"))
                            km_summary, km_pair_df = grouping_survival_metrics(
                                t_te,
                                e_te,
                                km_groups,
                                rmst_tau=rmst_tau,
                                model_name=model_name,
                                grouping_method="kmeans",
                                run_id=run_id,
                            )
                            km_summary.update({"cancer": cancer, "dataset": DATASET_NAME})
                            grouping_rows.append(km_summary)
                            if not km_pair_df.empty:
                                km_pair_df.insert(0, "cancer", cancer)
                                km_pair_df.insert(1, "dataset", DATASET_NAME)
                                pairwise_rows.append(km_pair_df)

                print(f"[{cancer} run {run_id}] {model_name}: c-index={c_te:.4f}")

            except Exception as e:
                failure_rows.append({
                    "cancer": cancer,
                    "dataset": DATASET_NAME,
                    "run_id": run_id,
                    "model": model_name,
                    "model_seed": int(model_seed) if np.isfinite(model_seed) else np.nan,
                    "teacher_seed": int(get_stage_seed(run_id, "teacher")) if model_name == "GOTSurv" else np.nan,
                    "ot_seed": int(get_stage_seed(run_id, "ot")) if model_name == "GOTSurv" else np.nan,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })
                print(f"[{cancer} run {run_id}] {model_name} failed: {type(e).__name__}: {e}")

    cindex_df = pd.DataFrame(cindex_rows)
    grouping_df = pd.DataFrame(grouping_rows)
    pairwise_df = pd.concat(pairwise_rows, axis=0, ignore_index=True) if len(pairwise_rows) > 0 else pd.DataFrame()
    failures_df = pd.DataFrame(failure_rows)

    row = {
        "dataset": DATASET_NAME,
        "cancer": cancer,
        "exp_details": EXP_DETAILS,
        "n_bootstraps": N_BOOTSTRAPS,
        "K_STATES": K_STATES,
        "rmst_tau_90pct": float(rmst_tau),
        "models": _safe_json(models),
        "mean_cindex_GOTSurv": float(cindex_df.loc[cindex_df["model"] == "GOTSurv", "cindex_test"].mean()) if not cindex_df.empty else np.nan,
        "mean_cindex_CoxPH": float(cindex_df.loc[cindex_df["model"] == "CoxPH", "cindex_test"].mean()) if not cindex_df.empty else np.nan,
        "mean_cindex_RSF": float(cindex_df.loc[cindex_df["model"] == "RSF", "cindex_test"].mean()) if not cindex_df.empty else np.nan,
        "mean_cindex_GBM": float(cindex_df.loc[cindex_df["model"] == "GBM", "cindex_test"].mean()) if not cindex_df.empty else np.nan,
        "mean_cindex_SurvivalSVM": float(cindex_df.loc[cindex_df["model"] == "SurvivalSVM", "cindex_test"].mean()) if not cindex_df.empty else np.nan,
        "distill_teacher": bool(DISTILL_TEACHER),
        "seed_rule": "teacher=SEED+run_id; ot=SEED+100000+run_id; bootstrap seed fixed at 42",
    }
    exp_num = util.append_experiment_row(MASTER_LOG_CSV, row)

    prefix = f"Results/Comparisons results/{exp_num}_{cancer}_{modality}_K{K_STATES}"

                                                              
    cindex_df.to_csv(f"{prefix}_baseline_cindex_by_run.csv", index=False)
    grouping_df.to_csv(f"{prefix}_baseline_grouping_summary_by_run.csv", index=False)
    pairwise_df.to_csv(f"{prefix}_baseline_grouping_pairwise_by_run.csv", index=False)
    failures_df.to_csv(f"{prefix}_baseline_failures.csv", index=False)

    cidx_summary = make_cindex_summary(cindex_df)
    if not cidx_summary.empty:
        cidx_summary.insert(0, "cancer", cancer)
        cidx_summary.insert(1, "dataset", DATASET_NAME)
        cidx_summary.to_csv(f"{prefix}_baseline_cindex_summary.csv", index=False)

    grouping_summary = make_grouping_summary(grouping_df)
    if not grouping_summary.empty:
        grouping_summary.insert(0, "cancer", cancer)
        grouping_summary.insert(1, "dataset", DATASET_NAME)
        grouping_summary.to_csv(f"{prefix}_baseline_grouping_summary_aggregated.csv", index=False)

                                           
    one_cancer_by_run = make_one_cancer_by_run_table(
        cindex_df,
        grouping_df,
        failures_df,
        cancer_name=cancer,
        dataset_name=DATASET_NAME,
    )
    one_cancer_csv = f"{prefix}_ONE_CANCER_all_results_by_run.csv"
    one_cancer_by_run.to_csv(one_cancer_csv, index=False)

    paper_ready = make_paper_ready_summary_table(cindex_df, grouping_df)
    paper_ready.insert(0, "cancer", cancer)
    paper_ready.insert(1, "dataset", DATASET_NAME)
    paper_ready.insert(2, "K_STATES", K_STATES)
    paper_ready = add_paper_ready_formatted_columns(paper_ready)
    paper_ready_csv = f"{prefix}_paper_ready_summary_table.csv"
    paper_ready.to_csv(paper_ready_csv, index=False)

    paired_stats = compare_gotsurv_to_best_model(
        cindex_df,
        cancer_name=cancer,
        dataset_name=DATASET_NAME,
        my_model="GOTSurv",
        save_path=f"{prefix}_gotsurv_vs_best_model_paired_test.csv",
    )

    print("\nSaved for", cancer)
    print(f"  {one_cancer_csv}")
    print(f"  {paper_ready_csv}")
    print(f"  {prefix}_baseline_cindex_by_run.csv")
    print(f"  {prefix}_baseline_grouping_summary_by_run.csv")
    print(f"  {prefix}_baseline_failures.csv")

    return {
        "cancer": cancer,
        "dataset": DATASET_NAME,
        "exp_num": exp_num,
        "one_cancer_csv": one_cancer_csv,
        "paper_ready_csv": paper_ready_csv,
        "paper_ready": paper_ready,
        "paired_stats": paired_stats,
        "cindex_df": cindex_df,
        "grouping_df": grouping_df,
        "failures_df": failures_df,
    }


def main():
    set_seed(SEED)
    os.makedirs("Results", exist_ok=True)
    os.makedirs("Results/Comparisons results TCGA", exist_ok=True)

    all_paper_ready = []
    all_paired_stats = []
    cancer_run_manifest = []

    for cancer_name in CANCER_TYPES:
        print("\n" + "=" * 80)
        print(f"Running cancer type: {str(cancer_name).upper()} with K_STATES={K_STATES}")
        print("=" * 80)
        try:
            out = run_one_cancer(str(cancer_name).upper())
            all_paper_ready.append(out["paper_ready"])
            if out["paired_stats"] is not None and not out["paired_stats"].empty:
                all_paired_stats.append(out["paired_stats"])
            cancer_run_manifest.append({
                "cancer": out["cancer"],
                "dataset": out["dataset"],
                "exp_num": out["exp_num"],
                "one_cancer_csv": out["one_cancer_csv"],
                "paper_ready_csv": out["paper_ready_csv"],
            })
        except Exception as e:
            cancer_run_manifest.append({
                "cancer": str(cancer_name).upper(),
                "dataset": f"{str(cancer_name).upper()}_{modality}",
                "exp_num": np.nan,
                "one_cancer_csv": "",
                "paper_ready_csv": "",
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            print(f"[multi-cancer runner] {cancer_name} failed: {type(e).__name__}: {e}")

    manifest_df = pd.DataFrame(cancer_run_manifest)
    manifest_path = f"Results/Comparisons results/multi_cancer_K{K_STATES}_run_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    if all_paper_ready:
        all_paper_ready_df = pd.concat(all_paper_ready, axis=0, ignore_index=True, sort=False)
        all_paper_ready_path = f"Results/Comparisons results/multi_cancer_K{K_STATES}_ALL_CANCERS_ALL_MODELS_paper_ready_summary.csv"
        all_paper_ready_df.to_csv(all_paper_ready_path, index=False)
        print("\nSaved combined paper-ready table:")
        print(f"  {all_paper_ready_path}")
    else:
        all_paper_ready_path = ""
        print("\nNo successful cancer runs; combined paper-ready table was not created.")

    if all_paired_stats:
        all_paired_df = pd.concat(all_paired_stats, axis=0, ignore_index=True, sort=False)
        all_paired_path = f"Results/Comparisons results/multi_cancer_K{K_STATES}_gotsurv_vs_best_model_paired_tests.csv"
        all_paired_df.to_csv(all_paired_path, index=False)
        print("Saved combined paired tests:")
        print(f"  {all_paired_path}")

    print("Saved run manifest:")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
