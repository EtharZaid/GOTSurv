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
    def _min_cluster_proportion_loss(memberships: torch.Tensor, min_ratio: float = 0.05):
        avg_probs = memberships.mean(dim=0)
        return F.relu(float(min_ratio) - avg_probs).sum()

    @staticmethod
    def _cluster_risk_separation_loss(risk_scores: torch.Tensor, memberships: torch.Tensor, eps: float = 1e-8):
        w = memberships
        wsum = w.sum(dim=0).clamp_min(eps)
        mu = (w.t() @ risk_scores.reshape(-1)) / wsum
        diffs = torch.abs(mu[:, None] - mu[None, :])
        triu = torch.triu(diffs, diagonal=1)
        num_pairs = (mu.numel() * (mu.numel() - 1)) / 2
        return -(triu.sum() / (num_pairs + eps))

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
