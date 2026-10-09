
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from geomloss import SamplesLoss
from typing import Optional, Iterable, Dict, Tuple, List, Any

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression


                                                                     

def _is_scalar(x) -> bool:
    try:
        return np.isscalar(x)
    except Exception:
        return isinstance(x, (int, float))


def _expand_param(param, n_edges: int, name: str) -> List[float]:
    if _is_scalar(param):
        return [float(param)] * n_edges
    arr = list(param)
    if len(arr) != n_edges:
        raise ValueError(f"{name} must be scalar or length {n_edges}. Got {len(arr)}.")
    return [float(v) for v in arr]


def _safe_normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float).reshape(-1)
    s = float(w.sum())
    return w / (s if s > 0 else 1.0)


def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    D2 = x2 + y2 - 2.0 * (X @ Y.T)
    return np.maximum(D2, 0.0)


def _whiten_fit(Z: np.ndarray, lambda_w: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Z = np.asarray(Z, float)
    mu = Z.mean(axis=0, keepdims=True)
    Xc = Z - mu
    Sigma = (Xc.T @ Xc) / max(1, Z.shape[0])
    d = Sigma.shape[0]
    evals, evecs = np.linalg.eigh(Sigma + lambda_w * np.eye(d))
    evals = np.maximum(evals, 1e-12)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mu.squeeze(0), inv_sqrt, Sigma


def _whiten_apply(Z: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (Z - mu.reshape(1, -1)) @ W.T


                                                                          

def generalized_sinkhorn_uv(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    eps: float,
    alpha: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_src = np.asarray(X_src, float)
    X_tgt = np.asarray(X_tgt, float)
    a = np.maximum(np.asarray(a, float).reshape(-1), 1e-16)
    b = np.maximum(np.asarray(b, float).reshape(-1), 1e-16)

    C = _pairwise_sq_dists(X_src, X_tgt)
    eps = float(max(eps, 1e-12))
    K = np.exp(-C / eps)

    u = np.ones(X_src.shape[0], dtype=np.float64)
    v = np.ones(X_tgt.shape[0], dtype=np.float64)

    KT = K.T
    for _ in range(int(max_iter)):
        Kv = K @ v
        Kv = np.maximum(Kv, 1e-300)
        u_new = (a / Kv) ** alpha

        KTu = KT @ u_new
        KTu = np.maximum(KTu, 1e-300)
        v_new = (b / KTu) ** alpha

        du = np.max(np.abs(np.log(u_new + 1e-300) - np.log(u + 1e-300)))
        dv = np.max(np.abs(np.log(v_new + 1e-300) - np.log(v + 1e-300)))
        u, v = u_new, v_new
        if max(du, dv) < tol:
            break

    return u, v, K


def uv_to_plan(u: np.ndarray, v: np.ndarray, K: np.ndarray) -> np.ndarray:
    return (u[:, None] * K) * v[None, :]


                                                                       
                           
                                                                       

class GOTFlow:
    """
    GOTFlow with OTSurv-inspired stabilizers:
      - Partial-mass schedule via epoch-dependent reach scaling (rho ramp).
      - Global long-tail (or uniform) marginal regularizer on state occupancy
        computed from *soft* bin assignments based on predicted risk.
      - Optional state updates during training (quantile or rank-based).
      - Optional Cox loss for survival supervision (besides pairwise ranking).
      - Optional "virtual-drop" adjustment when computing barycentric drift
        to treat unmatched mass as staying in place (no drift).
    """

    def __init__(
        self,
        *,
        blur: float | List[float] | np.ndarray = 0.05,
        scaling: float | List[float] | np.ndarray = 0.9,
        reach: float | List[float] | np.ndarray = 0.5,
        phi: Optional[nn.Module] = None,
        debias: bool = True,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        lambda_w: float = 1e-3,
    ):
        self.blur = blur
        self.scaling = scaling
        self.reach = reach
        self.debias = bool(debias)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.lambda_w = float(lambda_w)

        if phi is None:
            raise ValueError("phi must be provided.")
        self.phi = phi.to(self.device)

                   
        self.mu_z_: Optional[np.ndarray] = None
        self.Wwhiten_: Optional[np.ndarray] = None

                       
        self.target_head: Optional[nn.Module] = None
        self.target_type: Optional[str] = None

                          
        self.X: Optional[np.ndarray] = None
        self.bin_ids: Optional[np.ndarray] = None
        self.edges_: List[Tuple[int, int]] = []
        self.edge_wts_: List[float] = []
        self.loss_log_: List[Dict[str, float]] = []

        self.z_: Optional[np.ndarray] = None
        self.zw_: Optional[np.ndarray] = None

                       
        self.edge_energy_: Dict[Tuple[int, int], float] = {}
        self.edge_drift_: Dict[Tuple[int, int], np.ndarray] = {}
        self.transport_plan_: Dict[Tuple[int, int], np.ndarray] = {}
        self.edge_log_: List[Dict[str, Any]] = []

        self.Vw_: Optional[np.ndarray] = None

                                                                          


    @staticmethod
    def make_target_head(
        z_dim: int,
        *,
        target_type: str = "regression",
        n_classes: int = 2,
        hidden: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
        head_style: str = "mlp",                                             
    ) -> nn.Module:
        target_type = str(target_type).lower()
        head_style = str(head_style).lower()

        if target_type == "classification" and head_style.startswith("tanh"):
            head_style = "mlp"

        din = int(z_dim)
        layers: List[nn.Module] = []

        if head_style == "linear":
            if target_type == "regression":
                return nn.Sequential(nn.Linear(din, 1))
            elif target_type == "classification":
                return nn.Sequential(nn.Linear(din, int(n_classes)))
            else:
                raise ValueError("target_type must be 'regression' or 'classification'.")

        if head_style == "tanh_linear":
            layers = [nn.Linear(din, 1, bias=True), nn.Tanh()]
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            return nn.Sequential(*layers)

        elif head_style == "tanh_mlp":
            layers = [nn.Linear(din, int(hidden), bias=True), nn.Tanh()]
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(int(hidden), 1, bias=True))
            return nn.Sequential(*layers)

        elif head_style == "mlp":
            for _ in range(max(0, int(depth))):
                layers += [nn.Linear(din, int(hidden)), nn.ReLU()]
                if dropout and float(dropout) > 0:
                    layers += [nn.Dropout(float(dropout))]
                din = int(hidden)
        else:
            raise ValueError("head_style must be one of {'linear','mlp','tanh_linear','tanh_mlp'}.")

        if target_type == "regression":
            layers += [nn.Linear(din, 1)]
        elif target_type == "classification":
            layers += [nn.Linear(din, int(n_classes))]
        else:
            raise ValueError("target_type must be 'regression' or 'classification'.")

        return nn.Sequential(*layers)
                                                                         

    @staticmethod
    def quantile_bin(t: np.ndarray, K: int, right: bool = False):
        t = np.asarray(t, dtype=float).reshape(-1)
        assert K >= 1
        qs = np.quantile(t, np.linspace(0.0, 1.0, K + 1))
        bin_ids = np.clip(np.digitize(t, qs[1:-1], right=right), 0, K - 1)
        return bin_ids.astype(int), qs

    @staticmethod
    def rank_bin(t: np.ndarray, K: int):
        t = np.asarray(t, dtype=float).reshape(-1)
        order = np.argsort(t)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0.0, 1.0, len(t), endpoint=False)
        return GOTFlow.quantile_bin(ranks, K=K)

                                                                                 

    def _embed_torch(self, X: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        Xt = torch.tensor(X, dtype=self.dtype, device=self.device, requires_grad=requires_grad)
        return self.phi(Xt)

    def _embed_np(self, X: np.ndarray, batch: int = 20000) -> np.ndarray:
        self.phi.eval()
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                z = self._embed_torch(X[i:i + batch], requires_grad=False)
                out.append(z.detach().cpu().numpy())
        return np.vstack(out)

    def _fit_whitening(self, Z: np.ndarray):
        mu, W, _ = _whiten_fit(Z, lambda_w=self.lambda_w)
        self.mu_z_ = mu.astype(np.float32, copy=False)
        self.Wwhiten_ = W.astype(np.float32, copy=False)

    def _whiten_np(self, Z: np.ndarray) -> np.ndarray:
        assert self.mu_z_ is not None and self.Wwhiten_ is not None
        return _whiten_apply(Z, self.mu_z_, self.Wwhiten_).astype(np.float32, copy=False)

    def _whiten_torch(self, z: torch.Tensor) -> torch.Tensor:
        assert self.mu_z_ is not None and self.Wwhiten_ is not None
        mu = torch.tensor(self.mu_z_, dtype=z.dtype, device=z.device).view(1, -1)
        W = torch.tensor(self.Wwhiten_, dtype=z.dtype, device=z.device)
        return (z - mu) @ W.T

                                                                            

    def _get_loss(self, blur: float, scaling: float, reach: float) -> SamplesLoss:
        if self.debias:
            return SamplesLoss("sinkhorn", p=2, blur=float(blur), scaling=float(scaling), reach=float(reach), debias=True)
        return SamplesLoss("sinkhorn", p=2, blur=float(blur), scaling=float(scaling), reach=float(reach))

    @staticmethod
    def _loss_with_weights(loss_fn: SamplesLoss, x: torch.Tensor, y: torch.Tensor, a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> torch.Tensor:
        if a is None and b is None:
            return loss_fn(x, y)
        if a is None:
            a = torch.ones((x.shape[0],), dtype=x.dtype, device=x.device) / max(1, x.shape[0])
        if b is None:
            b = torch.ones((y.shape[0],), dtype=y.dtype, device=y.device) / max(1, y.shape[0])
        return loss_fn(a, x, b, y)

                                                                       

    def _ensure_target_head(
        self,
        z_dim: int,
        *,
        target: Optional[np.ndarray],
        target_type: str,
        head_hidden: int,
        head_depth: int,
        head_dropout: float,
        head_style: str,
    ):
        if self.target_head is not None:
            return

        target_type = str(target_type).lower()
        n_classes = 2
        if target_type == "classification":
            if target is None:
                raise ValueError("classification target requires target array to infer n_classes.")
            n_classes = int(np.max(np.asarray(target))) + 1

        self.target_head = GOTFlow.make_target_head(
            z_dim,
            target_type=target_type,
            n_classes=n_classes,
            hidden=head_hidden,
            depth=head_depth,
            dropout=head_dropout,
            head_style=head_style,
        ).to(self.device)
        self.target_type = target_type

    def predict_target(self, X: np.ndarray, batch: int = 20000) -> np.ndarray:
        assert self.target_head is not None, "No target_head. Train with learn_target=True."
        self.phi.eval()
        self.target_head.eval()
        X = np.asarray(X, float)
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                z = self._embed_torch(X[i:i+batch], requires_grad=False)
                y = self.target_head(z)
                out.append(y.detach().cpu().numpy())
        y = np.vstack(out)
        if self.target_type == "classification":
            return y.argmax(axis=1)
        return y.reshape(-1)

                                                                                 

    @staticmethod
    def _cox_ph_loss(risk: torch.Tensor, time: np.ndarray, event: np.ndarray, supervised_idx: Optional[np.ndarray] = None) -> torch.Tensor:
        if risk.ndim == 2 and risk.shape[1] == 1:
            r = risk.reshape(-1)
        else:
            r = risk.reshape(-1)

        t = np.asarray(time).reshape(-1)
        e = np.asarray(event).reshape(-1).astype(int)
        if supervised_idx is not None:
            t = t[supervised_idx]
            e = e[supervised_idx]
            r = r[supervised_idx]

        t_t = torch.tensor(t, dtype=r.dtype, device=r.device)
        e_t = torch.tensor(e, dtype=torch.float32, device=r.device)

        order = torch.argsort(t_t)
        r_s = r[order]
        e_s = e_t[order]
        log_cum = torch.logcumsumexp(r_s.flip(0), dim=0).flip(0)

        num_events = torch.clamp(e_s.sum(), min=1.0)
        loss = -((r_s - log_cum) * e_s).sum() / num_events
        return loss

    @staticmethod
    def _survival_pairwise_loss(
        risk: torch.Tensor,
        time: np.ndarray,
        event: np.ndarray,
        *,
        supervised_idx: Optional[np.ndarray] = None,
        num_pairs: Optional[int] = 10000,
        margin: float = 0.0,
        mode: str = "logistic",
        temperature: float = 1.0,
        rng_seed: int = 0,
    ) -> torch.Tensor:
        if risk.ndim == 2 and risk.shape[1] == 1:
            r = risk.reshape(-1)
        else:
            r = risk.reshape(-1)

        t = np.asarray(time).reshape(-1)
        e = np.asarray(event).reshape(-1).astype(int)

        if supervised_idx is not None:
            t = t[supervised_idx]
            e = e[supervised_idx]
            r = r[supervised_idx]

        rng = np.random.default_rng(int(rng_seed))
        idx_event = np.where(e == 1)[0]
        if idx_event.size == 0:
            return torch.tensor(0.0, dtype=r.dtype, device=r.device)

        if num_pairs is None:
            ii = []
            jj = []
            for i in idx_event:
                js = np.where(t > t[i])[0]
                for j in js:
                    ii.append(i); jj.append(j)
            if len(ii) == 0:
                return torch.tensor(0.0, dtype=r.dtype, device=r.device)
            ii = np.asarray(ii, int)
            jj = np.asarray(jj, int)
        else:
            ii = rng.choice(idx_event, size=int(num_pairs), replace=True)
            jj = rng.integers(0, len(t), size=int(num_pairs))
            valid = (e[ii] == 1) & (t[ii] < t[jj])
            ii = ii[valid]
            jj = jj[valid]
            if ii.size == 0:
                return torch.tensor(0.0, dtype=r.dtype, device=r.device)

        ri = r[ii]
        rj = r[jj]
        diff = (ri - rj) - float(margin)

        mode = str(mode).lower()
        T = float(max(temperature, 1e-6))
        if mode == "hinge":
            loss = torch.relu(float(margin) - (ri - rj)).mean()
        else:
            loss = F.softplus(-diff / T).mean()
        return loss

                                                                                            

    @staticmethod
    def _state_prior(K: int, prior: str = "longtail", alpha: float = 1.5, reverse: bool = False) -> np.ndarray:
        prior = str(prior).lower()
        if prior == "uniform":
            p = np.ones(K, float)
        else:
            k = np.arange(1, K + 1, dtype=float)
            p = 1.0 / (k ** float(alpha))
        if reverse:
            p = p[::-1].copy()
        p = p / max(p.sum(), 1e-12)
        return p

    @staticmethod
    def _soft_bin_probs_from_edges(risk: torch.Tensor, edges: np.ndarray, temp: float = 0.05) -> torch.Tensor:
        r = risk.reshape(-1, 1)
        edges = np.asarray(edges, float).reshape(-1)
        lo = float(edges[0] - 1e6)
        hi = float(edges[-1] + 1e6)
        e_ext = np.concatenate([[lo], edges[1:-1], [hi]])
        e_t = torch.tensor(e_ext, dtype=r.dtype, device=r.device).view(1, -1)

        T = float(max(temp, 1e-6))
        cdf = torch.sigmoid((r - e_t) / T)
        probs = cdf[:, :-1] - cdf[:, 1:]
        probs = torch.clamp(probs, min=1e-12, max=1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def _state_kl_loss(
        self,
        risk_pred: torch.Tensor,
        K: int,
        *,
        prior: str = "longtail",
        prior_alpha: float = 1.5,
        prior_reverse: bool = False,
        soft_temp: float = 0.05,
    ) -> torch.Tensor:
        with torch.no_grad():
            risk_np = risk_pred.detach().cpu().numpy().reshape(-1)
            _, edges = GOTFlow.quantile_bin(risk_np, K=K)

        probs = self._soft_bin_probs_from_edges(risk_pred.reshape(-1), edges, temp=soft_temp)
        q = torch.clamp(probs.mean(dim=0), min=1e-12, max=1.0)

        p = self._state_prior(K, prior=prior, alpha=prior_alpha, reverse=prior_reverse)
        p_t = torch.tensor(p, dtype=q.dtype, device=q.device)
        kl = (q * (torch.log(q) - torch.log(torch.clamp(p_t, min=1e-12)))).sum()
        return kl

                                                                     

    def fit(
        self,
        X: np.ndarray,
        bin_ids: np.ndarray,
        *,
        edges: Optional[Iterable[Tuple[int, int]]] = None,
        edge_weights: Optional[Any] = None,
        sample_weights: Optional[np.ndarray] = None,
        normalize_weights: bool = True,
        learn_phi: bool = True,
        epochs: int = 200,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        rng_seed: int = 0,
        max_src_per_edge: int = 2000,
        max_tgt_per_edge: int = 2000,
        whiten_update_every: int = 5,
        whiten_max_points: int = 50000,
                                           
        learn_target: bool = False,
        target: Optional[np.ndarray] = None,
        target_type: str = "regression",
        target_loss_name: str = "mse",
        use_regression_loss: bool = False,
        optimizer_name: str = "adam",
                           
        time: Optional[np.ndarray] = None,
        event: Optional[np.ndarray] = None,
        use_survival_loss: bool = False,
        surv_loss_type: str = "pairwise",
        surv_ranking_mode: str = "logistic",
        surv_margin: float = 0.0,
        surv_temperature: float = 1.0,
        surv_num_pairs: Optional[int] = 10000,
        eta_surv: float = 1.0,
                         
        eta: float = 1.0,
        verbose: bool = True,
                      
        drift_mode: str = "barycentric",
        sinkhorn_max_iter: int = 1000,
        sinkhorn_tol: float = 1e-6,
        aggregate: str = "weighted_mean",
                               
        use_partial_mass_schedule: bool = False,
        rho_start: float = 0.3,
        rho_end: float = 1.0,
        rho_warmup_epochs: int = 100,
        rho_schedule: str = "sigmoid",
        state_kl_weight: float = 0.0,
        state_prior: str = "longtail",
        state_prior_alpha: float = 1.5,
        state_prior_reverse: bool = False,
        state_soft_temp: float = 0.05,
        update_states: bool = False,
        update_states_every: int = 0,
        update_states_after: int = 0,
        update_states_method: str = "rank",
        update_states_K: Optional[int] = None,
        use_virtual_drop: bool = False,
    ):
        X = np.asarray(X, float)
        bin_ids = np.asarray(bin_ids, int).reshape(-1)
        if X.ndim != 2 or bin_ids.shape[0] != X.shape[0]:
            raise ValueError("X must be 2D and bin_ids length must equal N.")

        if (time is None) ^ (event is None):
            raise ValueError("Provide both time and event (or neither).")
        if time is not None:
            time = np.asarray(time).reshape(-1)
            event = np.asarray(event).reshape(-1).astype(int)
            if time.shape[0] != X.shape[0] or event.shape[0] != X.shape[0]:
                raise ValueError("time/event must match X length.")

                                         
        Z0 = self._embed_np(X[:min(1024, X.shape[0])], batch=20000)
        self._fit_whitening(Z0)

        if learn_target:
            self._ensure_target_head(
                z_dim=int(Z0.shape[1]),
                target=target,
                target_type=target_type,
                head_hidden=64,
                head_depth=1,
                head_dropout=0.0,
                head_style="linear",
            )

        self.X = X
        self.bin_ids = bin_ids

        uniq_sorted = np.sort(np.unique(bin_ids))
        if edges is None:
            edges_list = [(int(uniq_sorted[i]), int(uniq_sorted[i + 1])) for i in range(len(uniq_sorted) - 1)]
        else:
            edges_list = [(int(a), int(b)) for (a, b) in list(edges)]
        if len(edges_list) == 0:
            raise ValueError("No edges to fit (need at least 2 states).")
        self.edges_ = edges_list
        L = len(edges_list)

        blur_list = _expand_param(self.blur, L, "blur")
        scaling_list = _expand_param(self.scaling, L, "scaling")
        reach_list = _expand_param(self.reach, L, "reach")

        if edge_weights is None:
            w_list = [1.0] * L
        elif isinstance(edge_weights, dict):
            w_list = [float(edge_weights.get((S, T), 1.0)) for (S, T) in edges_list]
        else:
            w_list = list(np.asarray(edge_weights, float).reshape(-1))
            if len(w_list) != L:
                raise ValueError("edge_weights must be dict or length == len(edges).")
        self.edge_wts_ = w_list

        sw = None
        if sample_weights is not None:
            sw = np.asarray(sample_weights, float).reshape(-1)
            if sw.shape[0] != X.shape[0]:
                raise ValueError("sample_weights must be length N.")

        rng = np.random.default_rng(int(rng_seed))

        def rebuild_bin_index(current_bin_ids: np.ndarray) -> Dict[int, np.ndarray]:
            u = np.sort(np.unique(current_bin_ids))
            return {int(b): np.where(current_bin_ids == int(b))[0] for b in u}

        bin_to_idx = rebuild_bin_index(bin_ids)

        K_states = int(update_states_K) if update_states_K is not None else int(len(np.unique(bin_ids)))

        params: List[torch.nn.Parameter] = []
        if learn_phi:
            params += list(self.phi.parameters())
        if learn_target and self.target_head is not None:
            params += list(self.target_head.parameters())
        if len(params) == 0:
            raise ValueError("No trainable parameters.")
        optimizer_name = str(optimizer_name).lower()
        if optimizer_name == "adam":
            opt = optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
        elif optimizer_name == "adamw":
            opt = optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
        else:
            raise ValueError("optimizer_name must be {'adam','adamw'}")

        loss_cache: Dict[Tuple[float, float, float], SamplesLoss] = {}

        def get_loss_params(blur: float, scaling: float, reach: float) -> SamplesLoss:
            key = (float(blur), float(scaling), float(reach))
            if key not in loss_cache:
                loss_cache[key] = self._get_loss(*key)
            return loss_cache[key]

        def subsample(idxs: np.ndarray, max_n: int) -> np.ndarray:
            if max_n is None or max_n <= 0 or idxs.size <= max_n:
                return idxs
            return rng.choice(idxs, size=max_n, replace=False)

        def rho_at_epoch(ep: int) -> float:
            if not use_partial_mass_schedule:
                return 1.0
            ep = int(ep)
            T = max(1, int(rho_warmup_epochs))
            if ep >= T:
                return float(rho_end)
            if str(rho_schedule).lower() == "linear":
                a = ep / float(T)
                return float(rho_start + a * (rho_end - rho_start))
            a = ep / float(T)
            s = 1.0 / (1.0 + np.exp(-10.0 * (a - 0.5)))
            return float(rho_start + s * (rho_end - rho_start))

        self.loss_log_ = []
        self.phi.train()
        if self.target_head is not None:
            self.target_head.train()

        last_reach_scale = 1.0

        for ep in range(int(epochs)):
            opt.zero_grad(set_to_none=True)

            if (ep % max(1, int(whiten_update_every))) == 0:
                with torch.no_grad():
                    if X.shape[0] > int(whiten_max_points):
                        idxw = rng.choice(X.shape[0], size=int(whiten_max_points), replace=False)
                        Zw = self._embed_np(X[idxw], batch=20000)
                    else:
                        Zw = self._embed_np(X, batch=20000)
                    self._fit_whitening(Zw)

            if update_states and learn_target and update_states_every and (ep >= int(update_states_after)) and (ep % int(update_states_every) == 0):
                with torch.no_grad():
                    r_now = self.predict_target(X)
                if str(update_states_method).lower() == "quantile":
                    bin_ids_new, _ = GOTFlow.quantile_bin(r_now, K=K_states)
                else:
                    bin_ids_new, _ = GOTFlow.rank_bin(r_now, K=K_states)

                bin_ids = np.asarray(bin_ids_new, int).reshape(-1)
                self.bin_ids = bin_ids
                bin_to_idx = rebuild_bin_index(bin_ids)
                if verbose:
                    print(f"[GOTFlow] update_states @ep={ep}, method={update_states_method}, K={K_states}")

            rho_ep = rho_at_epoch(ep)
            reach_scale = 1.0 / max(rho_ep, 1e-6)
            last_reach_scale = reach_scale

            total_ot = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            for ei, (S, T) in enumerate(edges_list):
                src_idx = bin_to_idx.get(int(S), np.array([], int))
                tgt_idx = bin_to_idx.get(int(T), np.array([], int))
                if src_idx.size == 0 or tgt_idx.size == 0:
                    continue

                si = subsample(src_idx, max_src_per_edge)
                tj = subsample(tgt_idx, max_tgt_per_edge)

                z_src = self._embed_torch(X[si], requires_grad=True)
                z_tgt = self._embed_torch(X[tj], requires_grad=False).detach()
                zw_src = self._whiten_torch(z_src)
                zw_tgt = self._whiten_torch(z_tgt)

                if sw is None:
                    a = b = None
                else:
                    a_np = sw[si].copy()
                    b_np = sw[tj].copy()
                    if normalize_weights:
                        a_np = _safe_normalize_weights(a_np)
                        b_np = _safe_normalize_weights(b_np)
                    a = torch.tensor(a_np, dtype=self.dtype, device=self.device)
                    b = torch.tensor(b_np, dtype=self.dtype, device=self.device)

                reach_eff = float(reach_list[ei]) * float(reach_scale)
                loss_fn = get_loss_params(blur_list[ei], scaling_list[ei], reach_eff)
                E_pos = self._loss_with_weights(loss_fn, zw_src, zw_tgt, a, b)
                total_ot = total_ot + float(w_list[ei]) * E_pos

            total_sup = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            reg_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            surv_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            kl_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)

            if learn_target and self.target_head is not None:
                z_all = self._embed_torch(X, requires_grad=True)
                pred_all = self.target_head(z_all)

                if use_regression_loss and target is not None:
                    if str(target_type).lower() == "regression":
                        y_true = torch.tensor(np.asarray(target).reshape(-1, 1), dtype=self.dtype, device=self.device)
                        nm = str(target_loss_name).lower()
                        if nm == "mse":
                            reg_loss = torch.mean((pred_all - y_true) ** 2)
                        elif nm == "mae":
                            reg_loss = torch.mean(torch.abs(pred_all - y_true))
                        elif nm == "huber":
                            reg_loss = nn.SmoothL1Loss()(pred_all, y_true)
                        else:
                            raise ValueError("target_loss_name must be mse/mae/huber for regression.")
                    else:
                        y_true = torch.tensor(np.asarray(target).reshape(-1), dtype=torch.long, device=self.device)
                        reg_loss = nn.CrossEntropyLoss()(pred_all, y_true)
                    total_sup = total_sup + reg_loss

                if use_survival_loss and (time is not None) and (event is not None):
                    if str(surv_loss_type).lower() == "cox":
                        surv_loss = self._cox_ph_loss(pred_all, time=time, event=event)
                    else:
                        surv_loss = self._survival_pairwise_loss(
                            pred_all,
                            time=time,
                            event=event,
                            num_pairs=surv_num_pairs,
                            margin=surv_margin,
                            mode=surv_ranking_mode,
                            temperature=surv_temperature,
                            rng_seed=int(rng_seed) + int(ep),
                        )
                    total_sup = total_sup + float(eta_surv) * surv_loss

                if state_kl_weight and float(state_kl_weight) > 0:
                    kl_loss = self._state_kl_loss(
                        pred_all.reshape(-1),
                        K=K_states,
                        prior=state_prior,
                        prior_alpha=state_prior_alpha,
                        prior_reverse=state_prior_reverse,
                        soft_temp=state_soft_temp,
                    )
                    total_sup = total_sup + float(state_kl_weight) * kl_loss

            total = total_ot + float(eta) * total_sup
            total.backward()
            opt.step()

            rec = {
                "epoch": int(ep),
                "ot": float(total_ot.detach().cpu().numpy()),
                "sup": float(total_sup.detach().cpu().numpy()),
                "reg": float(reg_loss.detach().cpu().numpy()),
                "surv": float(surv_loss.detach().cpu().numpy()),
                "kl": float(kl_loss.detach().cpu().numpy()),
                "rho": float(rho_ep),
                "total": float(total.detach().cpu().numpy()),
            }
            self.loss_log_.append(rec)

            if verbose and ((ep % 25) == 0 or ep == epochs - 1):
                msg = f"[GOTFlow] ep {ep:4d}/{epochs}  OT={rec['ot']:.6g}  SUP={rec['sup']:.6g}  REG={rec['reg']:.6g}  SURV={rec['surv']:.6g}  KL={rec['kl']:.6g}  rho={rec['rho']:.3f}  total={rec['total']:.6g}"
                print(msg)

        self.phi.eval()
        if self.target_head is not None:
            self.target_head.eval()

        Z = self._embed_np(X, batch=20000)
        self._fit_whitening(Z)
        self.z_ = Z.astype(np.float32, copy=False)
        self.zw_ = self._whiten_np(Z)

        self.edge_energy_.clear()
        self.edge_drift_.clear()
        self.transport_plan_.clear()
        self.edge_log_ = []

        zw_all = self.zw_
        assert zw_all is not None
        d = zw_all.shape[1]
        Vw = np.zeros((X.shape[0], d), dtype=np.float64)
        denom = np.zeros((X.shape[0],), dtype=np.float64)

        for ei, (S, T) in enumerate(edges_list):
            src_idx = bin_to_idx.get(int(S), np.array([], int))
            tgt_idx = bin_to_idx.get(int(T), np.array([], int))
            if src_idx.size == 0 or tgt_idx.size == 0:
                self.edge_log_.append({"edge": (int(S), int(T)), "skipped": True, "n_src": int(src_idx.size), "n_tgt": int(tgt_idx.size)})
                continue

            blur_e = float(blur_list[ei])
            scaling_e = float(scaling_list[ei])
            reach_e = float(reach_list[ei]) * float(last_reach_scale)
            w_e = float(w_list[ei])

            if sw is None:
                a_plan = np.ones(src_idx.size, dtype=float)
                b_plan = np.ones(tgt_idx.size, dtype=float)
            else:
                a_np = sw[src_idx].copy()
                b_np = sw[tgt_idx].copy()
                if normalize_weights:
                    a_np = _safe_normalize_weights(a_np)
                    b_np = _safe_normalize_weights(b_np)
                a_plan = a_np
                b_plan = b_np

            loss_fn = self._get_loss(blur_e, scaling_e, reach_e)
            x_t = torch.tensor(zw_all[src_idx], dtype=self.dtype, device=self.device, requires_grad=False)
            y_t = torch.tensor(zw_all[tgt_idx], dtype=self.dtype, device=self.device, requires_grad=False)
            E = self._loss_with_weights(loss_fn, x_t, y_t, None, None)
            self.edge_energy_[(int(S), int(T))] = float(E.detach().cpu().numpy())

            eps_val = float(max(blur_e ** 2, 1e-12))
            lam = 1.0 / max(reach_e, 1e-12)
            alpha = float(np.clip(lam / (lam + eps_val), 1e-6, 1.0 - 1e-6))

            u, v, Kmat = generalized_sinkhorn_uv(
                X_src=zw_all[src_idx],
                X_tgt=zw_all[tgt_idx],
                a=a_plan,
                b=b_plan,
                eps=eps_val,
                alpha=alpha,
                max_iter=sinkhorn_max_iter,
                tol=sinkhorn_tol,
            )

            if drift_mode == "barycentric":
                P = uv_to_plan(u, v, Kmat)
                self.transport_plan_[(int(S), int(T))] = P.astype(np.float32, copy=False)

                row_sum = P.sum(axis=1, keepdims=True) + 1e-12
                ybar = (P @ zw_all[tgt_idx]) / row_sum

                if use_virtual_drop:
                    a_i = a_plan.reshape(-1, 1)
                    dropped = np.maximum(a_i - row_sum, 0.0)
                    denom2 = row_sum + dropped + 1e-12
                    ybar = (P @ zw_all[tgt_idx] + dropped * zw_all[src_idx]) / denom2

                drift = ybar - zw_all[src_idx]

            elif drift_mode == "grad":
                xg = torch.tensor(zw_all[src_idx], dtype=self.dtype, device=self.device, requires_grad=True)
                yg = torch.tensor(zw_all[tgt_idx], dtype=self.dtype, device=self.device, requires_grad=False)
                Eg = self._loss_with_weights(loss_fn, xg, yg, None, None)
                Eg.backward()
                drift = (-xg.grad.detach().cpu().numpy()).astype(np.float64, copy=False)
            else:
                raise ValueError("drift_mode must be {'barycentric','grad'}")

            self.edge_drift_[(int(S), int(T))] = drift.astype(np.float32, copy=False)
            Vw[src_idx] += (w_e * drift)
            denom[src_idx] += w_e

            self.edge_log_.append({
                "edge": (int(S), int(T)),
                "skipped": False,
                "n_src": int(src_idx.size),
                "n_tgt": int(tgt_idx.size),
                "blur": blur_e,
                "reach": reach_e,
                "scaling": scaling_e,
                "alpha": alpha,
                "w": w_e,
                "mode": drift_mode,
                "energy": self.edge_energy_[(int(S), int(T))],
            })

        if aggregate == "weighted_mean":
            mask = denom > 0
            Vw[mask] = Vw[mask] / denom[mask, None]
        elif aggregate != "sum":
            raise ValueError("aggregate must be {'weighted_mean','sum'}")

        self.Vw_ = Vw.astype(np.float32, copy=False)
        return self

                                                                           
                                       
                                                                           

                                                                                               

                                                                                 
                                                                 
                                
                                 
                             
                                                       
                                                       
                                                                 
                           
                                   
                                 
                           
        

    @staticmethod
    def cluster_average(Z2: np.ndarray, V2: np.ndarray, cid: np.ndarray, min_points: int = 10):
        Z2 = np.asarray(Z2, float)
        V2 = np.asarray(V2, float)
        cid = np.asarray(cid, int)
        P = []
        U = []
        counts = []
        labs = []
        for k in sorted(np.unique(cid)):
            idx = np.where(cid == k)[0]
            if idx.size < int(min_points):
                continue
            P.append(Z2[idx].mean(axis=0))
            U.append(V2[idx].mean(axis=0))
            counts.append(int(idx.size))
            labs.append(int(k))
        return np.vstack(P), np.vstack(U), np.asarray(counts), np.asarray(labs)


    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator
    from sklearn.neighbors import NearestNeighbors


                                                                           
                            
                                                                           


                                                                           
                            
                                                                           


                                                                           
                                      
                                                                           


                                                                       
                                                      
                                                                       

class ClusterAwareSurvivalTeacher:
    """
    Lightweight version of the ClusterAwareSurvivalRanker adapted for GOTFlow.

    It jointly learns:
      - a survival risk score
      - soft cluster memberships
      - a latent encoder that can be reused as GOTFlow.phi

    The learned hard clusters are ordered by mean predicted risk and used as
    OT states, avoiding quantile binning of the risk score.
    """

    class _Net(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, num_clusters: int, dropout: float):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            self.risk_head = nn.Sequential(
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.Tanh(),
                nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
                nn.Linear(int(hidden_dim), 1),
                nn.Tanh(),
            )
            self.cluster_head = nn.Linear(int(hidden_dim), int(num_clusters))

        def forward(self, x):
            z = self.encoder(x)
            risk = self.risk_head(z).reshape(-1)
            memberships = torch.softmax(self.cluster_head(z), dim=1)
            return risk, memberships, z

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 64,
        num_clusters: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        epochs: int = 300,
        patience: int = 40,
        eval_every: int = 5,
        lambda_w: float = 0.1,
        lambda_cluster: float = 0.1,
        lambda_sep: float = 0.1,
        lambda_ent: float = 0.01,
        lambda_logrank: float = 0.0,
        min_ratio: float = 0.05,
        p: int = 2,
        loss_terms: Optional[List[str]] = None,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_clusters = int(num_clusters)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.eval_every = int(eval_every)
        self.lambda_w = float(lambda_w)
        self.lambda_cluster = float(lambda_cluster)
        self.lambda_sep = float(lambda_sep)
        self.lambda_ent = float(lambda_ent)
        self.lambda_logrank = float(lambda_logrank)
        self.min_ratio = float(min_ratio)
        self.p = int(p)
        self.loss_terms = loss_terms or ["ranking", "proportion", "separation", "entropy"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = int(random_state)
        self.model = self._Net(input_dim, hidden_dim, num_clusters, dropout).to(self.device)
        self.history_: List[Dict[str, float]] = []
        self.best_epoch_ = -1
        self.best_score_ = np.nan
        self.cluster_order_: Optional[Dict[int, int]] = None

    @staticmethod
    def _to_tensor(v, dtype=torch.float32, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return torch.as_tensor(v, dtype=dtype, device=device)

    @staticmethod
    def _valid_pairs(T: torch.Tensor, E: torch.Tensor):
        dT = T[:, None] - T[None, :]
        dP = (dT > 0) & (E[None, :] > 0)                                  
        return torch.where(dP)

    @staticmethod
    def _entropy_loss(memberships: torch.Tensor):
        return torch.mean(torch.sum(memberships * torch.log(memberships + 1e-8), dim=1))



    @staticmethod
    def _cluster_risk_separation_loss(risk_scores: torch.Tensor, memberships: torch.Tensor, eps: float = 1e-8):
        w = memberships
        wsum = w.sum(dim=0).clamp_min(eps)
        mu = (w.t() @ risk_scores.reshape(-1)) / wsum
        diffs = torch.abs(mu[:, None] - mu[None, :])
        triu = torch.triu(diffs, diagonal=1)
        num_pairs = (mu.numel() * (mu.numel() - 1)) / 2
        return -(triu.sum() / (num_pairs + eps))

    @staticmethod
    def _min_cluster_proportion_loss(memberships: torch.Tensor, min_ratio: float = 0.05):
        avg_probs = memberships.mean(dim=0)
        return F.relu(float(min_ratio) - avg_probs).sum()

    

    def _cluster_weighted_pairwise_loss(self, risk: torch.Tensor, T: torch.Tensor, E: torch.Tensor, M: torch.Tensor, i_idx, j_idx):
        if i_idx.numel() == 0:
            return risk.new_tensor(0.0)
                                                                                              
        margin = F.relu(1.0 - (risk[j_idx] - risk[i_idx]))
        Mi, Mj = M[i_idx], M[j_idx]
                                                                                   
        # Weight each comparable pair by the probability that the earlier-event
        # sample j is assigned to the same or a higher-risk group than sample i:
        # w_ij = sum_k sum_{ell >= k} m_ik m_jell.
        suffix = torch.flip(torch.cumsum(torch.flip(M, (1,)), dim=1), (1,))
        weights = (Mi * suffix[j_idx]).sum(dim=1)
        return (weights * margin).sum() / weights.sum().clamp_min(1e-12)
 
 



        
    @staticmethod
    def _cindex_numpy(time, event, risk):
        from lifelines.utils import concordance_index
        return float(concordance_index(np.asarray(time, float), -np.asarray(risk, float), np.asarray(event, int)))

    def _predict_arrays(self, X: np.ndarray):
        self.model.eval()
        X_t = self._to_tensor(np.asarray(X, float), device=self.device)
        with torch.no_grad():
            risk, memberships, z = self.model(X_t)
        return (
            risk.detach().cpu().numpy().reshape(-1),
            memberships.detach().cpu().numpy(),
            z.detach().cpu().numpy(),
        )

    def fit(self, X_train, time_train, event_train, *, X_val=None, time_val=None, event_val=None, X_transductive=None, verbose=False):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_train = np.asarray(X_train, float)
        time_train = np.asarray(time_train, float).reshape(-1)
        event_train = np.asarray(event_train, int).reshape(-1)

        X_t = self._to_tensor(X_train, device=self.device)
        T_t = self._to_tensor(time_train, device=self.device)
        E_t = self._to_tensor(event_train, device=self.device)
        i_idx, j_idx = self._valid_pairs(T_t, E_t)

        if X_transductive is not None:
            X_trans = np.asarray(X_transductive, float)
        elif X_val is not None:
            X_trans = np.vstack([X_train, np.asarray(X_val, float)])
        else:
            X_trans = X_train
        X_trans_t = self._to_tensor(X_trans, device=self.device)

        use_val = X_val is not None and time_val is not None and event_val is not None
        if use_val:
            X_val = np.asarray(X_val, float)
            time_val = np.asarray(time_val, float).reshape(-1)
            event_val = np.asarray(event_val, int).reshape(-1)

        opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_state = None
        best_score = -np.inf
        bad = 0
        self.history_ = []

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad(set_to_none=True)
            risk, memberships, _ = self.model(X_t)
            loss = risk.new_tensor(0.0)
            rank_loss = risk.new_tensor(0.0)
            prop_loss = risk.new_tensor(0.0)
            sep_loss = risk.new_tensor(0.0)
            ent_loss = risk.new_tensor(0.0)

            if "ranking" in self.loss_terms:
                rank_loss = self._cluster_weighted_pairwise_loss(risk, T_t, E_t, memberships, i_idx, j_idx)
                loss = loss + rank_loss

            if any(term in self.loss_terms for term in ["proportion", "separation", "entropy"]):
                risk_all, M_all, _ = self.model(X_trans_t)
                if "proportion" in self.loss_terms:
                    prop_loss = self._min_cluster_proportion_loss(M_all, self.min_ratio)
                    loss = loss + self.lambda_cluster * prop_loss
                if "separation" in self.loss_terms:
                    sep_loss = self._cluster_risk_separation_loss(risk_all, M_all)
                    loss = loss + self.lambda_sep * sep_loss
                if "entropy" in self.loss_terms:
                    ent_loss = self._entropy_loss(M_all)
                    loss = loss + self.lambda_ent * ent_loss

                                                  
            w_pen = risk.new_tensor(0.0)
            for m in self.model.risk_head.modules():
                if isinstance(m, nn.Linear):
                    w_pen = w_pen + m.weight.norm(self.p)
            loss = loss + self.lambda_w * w_pen

            loss.backward()
            opt.step()

            row = {
                "epoch": int(epoch),
                "loss": float(loss.detach().cpu().item()),
                "ranking": float(rank_loss.detach().cpu().item()),
                "proportion": float(prop_loss.detach().cpu().item()),
                "separation": float(sep_loss.detach().cpu().item()),
                "entropy": float(ent_loss.detach().cpu().item()),
                "weight_penalty": float(w_pen.detach().cpu().item()),
            }

            if use_val and ((epoch % self.eval_every) == 0 or epoch == self.epochs - 1):
                tr_risk, _, _ = self._predict_arrays(X_train)
                va_risk, _, _ = self._predict_arrays(X_val)
                tr_c = self._cindex_numpy(time_train, event_train, tr_risk)
                va_c = self._cindex_numpy(time_val, event_val, va_risk)
                row["train_cindex"] = tr_c
                row["val_cindex"] = va_c
                if np.isfinite(va_c) and va_c > best_score + 1e-6:
                    best_score = va_c
                    self.best_epoch_ = int(epoch)
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    bad = 0
                    row["improved"] = True
                else:
                    bad += 1
                    row["improved"] = False

            self.history_.append(row)
            if verbose and ((epoch % 25) == 0 or epoch == self.epochs - 1):
                msg = f"[ClusterAwareTeacher] ep {epoch:4d}/{self.epochs} loss={row['loss']:.6g} rank={row['ranking']:.6g}"
                if "val_cindex" in row:
                    msg += f" train_c={row['train_cindex']:.4f} val_c={row['val_cindex']:.4f}"
                print(msg)

            if use_val and self.patience is not None and bad >= int(self.patience):
                if verbose:
                    print(f"[ClusterAwareTeacher] early stopping at epoch={epoch}, best_epoch={self.best_epoch_}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_score_ = float(best_score) if np.isfinite(best_score) else np.nan
        self.history_df_ = __import__("pandas").DataFrame(self.history_)
        return self

    def decision_function(self, X):
        return self._predict_arrays(X)[0]

    def get_cluster_memberships(self, X):
        return self._predict_arrays(X)[1]


    def get_ordered_cluster_states(self, X, risk: Optional[np.ndarray] = None):
        M = self.get_cluster_memberships(X)
        hard = M.argmax(axis=1).astype(int)
        if risk is None:
            risk = self.decision_function(X)
        risk = np.asarray(risk, float).reshape(-1)
        means = []
        for k in range(self.num_clusters):
            mask = hard == k
            means.append(np.nanmean(risk[mask]) if mask.any() else np.inf)
        order = np.argsort(means)
        remap = {int(old): int(new) for new, old in enumerate(order)}
        self.cluster_order_ = remap
        return np.asarray([remap[int(k)] for k in hard], dtype=int)

    def to_gotflow(self, **got_kwargs):
        """Return a GOTFlow object using this model's encoder and risk head."""
        got = GOTFlow(phi=self.model.encoder, **got_kwargs)
        got.target_head = self.model.risk_head.to(got.device)
        got.target_type = "regression"
        got.cluster_teacher_ = self
        got.state_head = self.model.cluster_head.to(got.device)

        def _predict_cluster_memberships(X, batch=20000):
            return self.get_cluster_memberships(X)

        def _predict_cluster_states(X, order_by="risk"):
            risk = got.predict_target(X) if order_by == "risk" else None
            return self.get_ordered_cluster_states(X, risk=risk)

        got.predict_cluster_memberships = _predict_cluster_memberships
        got.predict_cluster_states = _predict_cluster_states
        return got
