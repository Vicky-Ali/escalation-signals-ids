"""Tier 2: MLP encoder trained with supervised contrastive learning, linear head, embedding novelty
(paper Sec. "Cascade and tiers").

* `Preproc`: quantile-normal transform of numeric columns (fitted on D_tr), NaN -> 0 plus missing indicators,
  binary (one-hot) columns passed through.
* `FlowNet` / `T2Flow`: MLP 256-256-128 (GELU, dropout 0.1) + projection head, SupCon loss (Khosla et al. 2020,
  temperature 0.1) on two augmented views (Gaussian noise + 20% feature dropout) with class-balanced batches;
  then the encoder is frozen and a linear cross-entropy head is fitted (L-BFGS), its class-weight exponent chosen
  by D_sel class-balanced log-loss.
* `tune_t2`: grid `T2_GRID` (SupCon steps x noise) chosen once per fold by D_sel macro-F1.
* `embedding_novelty`: raw Tier-2 novelty = Euclidean distance of the L2-normalised embedding h to the q-th
  nearest D_tr embedding of the predicted class (and the class-agnostic variant).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer

from . import novelty as NV
from .tier1 import Q, class_weights, seed_all

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEAD_GAMMAS = [0.0, 0.5, 1.0]   # head class-weight exponent, chosen on D_sel class-balanced log-loss
T2_GRID = dict(steps=[1000, 3000], noise=[0.1, 0.3])
T2_CFG = dict(hidden=(256, 256), emb=128, proj_dim=128, dropout=0.1, supcon_temp=0.1, noise=0.1, feat_drop=0.2,
              batch=512, steps=1000, lr=1e-3, wd=1e-4, head_steps=100, head_wd=1e-4)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020, L_out). z: (N, d) L2-normalised, labels: (N,)."""
    sim = z @ z.T / temperature
    n = z.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    valid = n_pos > 0
    mean_log_pos = (log_prob * pos).sum(1)[valid] / n_pos[valid]
    return -mean_log_pos.mean()


class Preproc:
    """Quantile-normal transform of numeric columns (fitted on D_tr), NaN -> 0 plus missing indicators for
    columns with missing values in D_tr; binary (one-hot) columns passed through."""

    def fit(self, X):
        X = np.asarray(X, np.float64)
        fin = np.where(np.isnan(X), 0, X)
        self.binary = np.all(np.isin(fin, [0.0, 1.0]), axis=0) & ~np.isnan(X).any(0)
        self.num = ~self.binary
        self.miss = np.isnan(X).any(0)
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                                      random_state=0).fit(X[:, self.num])
        return self

    def transform(self, X):
        X = np.asarray(X, np.float64)
        Z = self.qt.transform(X[:, self.num])
        Z = np.nan_to_num(Z, nan=0.0)
        return np.hstack([Z, X[:, self.binary], np.isnan(X[:, self.miss]).astype(float)]).astype(np.float32)


class FlowNet(nn.Module):
    def __init__(self, d_in, hidden=(256, 256), emb=128, proj_dim=128, dropout=0.1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, emb)]
        self.enc = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, proj_dim))

    def encode(self, x):
        return self.enc(x)

    def forward(self, x):
        h = self.enc(x)
        return h, F.normalize(self.proj(h), dim=1)


class T2Flow:
    def __init__(self, n_classes: int, seed: int, cfg: dict | None = None):
        self.K = n_classes
        self.seed = seed
        self.cfg = dict(T2_CFG, **(cfg or {}))

    def _view(self, x):
        c = self.cfg
        x = x + c["noise"] * torch.randn_like(x)
        keep = (torch.rand_like(x) >= c["feat_drop"]).float()
        return x * keep

    def fit(self, X, y, Xsel=None, ysel=None, log=print):
        c = self.cfg
        seed_all(self.seed)
        self.pre = Preproc().fit(X)
        Z = torch.tensor(self.pre.transform(X), device=DEVICE)
        self.net = FlowNet(Z.shape[1], c["hidden"], c["emb"], c["proj_dim"], c["dropout"]).to(DEVICE)
        opt = torch.optim.AdamW(self.net.parameters(), lr=c["lr"], weight_decay=c["wd"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=c["steps"], pct_start=0.1)
        by_cls = [torch.tensor(np.flatnonzero(y == k), device=DEVICE) for k in range(self.K)]
        present = [k for k in range(self.K) if len(by_cls[k])]
        per = max(2, c["batch"] // len(present))
        self.net.train()
        t0 = time.time()
        for step in range(c["steps"]):
            idx = torch.cat([by_cls[k][torch.randint(len(by_cls[k]), (per,), device=DEVICE)] for k in present])
            lab = torch.cat([torch.full((per,), k, device=DEVICE) for k in present])
            xb = Z[idx]
            _, proj = self.net(torch.cat([self._view(xb), self._view(xb)]))
            loss = supcon_loss(proj, torch.cat([lab, lab]), c["supcon_temp"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if (step + 1) % 1000 == 0:
                log(f"   [T2 SupCon] step {step + 1}/{c['steps']} loss={loss.item():.4f} ({time.time() - t0:.0f}s)")
        self.net.eval()
        for p_ in self.net.parameters():
            p_.requires_grad_(False)
        H = self._h(Z)
        yt = torch.tensor(y, device=DEVICE)
        if Xsel is not None and len(Xsel):
            Hs = self._h(torch.tensor(self.pre.transform(Xsel), device=DEVICE))
        best = None
        for gamma in (HEAD_GAMMAS if Xsel is not None else [1.0]):
            head = self._fit_head(H, yt, class_weights(y, self.K, gamma)[y])
            if Xsel is None:
                best = (head, gamma, np.nan)
                break
            with torch.no_grad():
                lp = F.log_softmax(head(Hs), 1).cpu().numpy()
            nll = -lp[np.arange(len(ysel)), ysel]
            ll = float(np.mean([nll[ysel == k].mean() for k in np.unique(ysel)]))
            log(f"   [T2 head] gamma={gamma} D_sel class-balanced log-loss={ll:.5f}")
            if best is None or ll < best[2]:
                best = (head, gamma, ll)
        self.head, self.head_gamma = best[0], best[1]
        self.head.eval()
        return self

    def _fit_head(self, H, yt, w):
        """Weighted CE + small L2, full-batch L-BFGS."""
        torch.manual_seed(self.seed)
        w = torch.tensor(w, dtype=torch.float32, device=DEVICE)
        w = w / w.sum()
        head = nn.Linear(H.shape[1], self.K).to(DEVICE)
        opt = torch.optim.LBFGS(head.parameters(), lr=1.0, max_iter=self.cfg["head_steps"],
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            l = (F.cross_entropy(head(H), yt, reduction="none") * w).sum() \
                + self.cfg["head_wd"] * head.weight.pow(2).sum()
            l.backward()
            return l
        opt.step(closure)
        return head.eval()

    @torch.no_grad()
    def _h(self, Z):
        return torch.cat([self.net.encode(Z[i:i + 65536]) for i in range(0, len(Z), 65536)])

    @torch.no_grad()
    def predict(self, X):
        """(logits, L2-normalised embedding h)."""
        H = self._h(torch.tensor(self.pre.transform(X), device=DEVICE))
        return self.head(H).cpu().numpy(), F.normalize(H, dim=1).cpu().numpy()


def tune_t2(Xtr, ytr, Xsel, ysel, K, seed=0, log=print):
    from .metrics import macro_prf
    best, rows = None, []
    keys = list(T2_GRID)
    for vals in itertools.product(*T2_GRID.values()):
        cfg = dict(zip(keys, vals))
        m = T2Flow(K, seed, cfg).fit(Xtr, ytr, Xsel, ysel, log=lambda *a: None)
        lg, _ = m.predict(Xsel)
        f1 = macro_prf(ysel, lg.argmax(1), K)["macro_F1"]
        rows.append(dict(**cfg, head_gamma=m.head_gamma, sel_macro_F1=f1))
        log(f"   [T2 tune] {cfg} D_sel macro-F1={f1:.4f}")
        if best is None or f1 > best[1]:
            best = (cfg, f1)
    return best[0], rows


def embedding_novelty(emb_q, pred_q, emb_tr, y_tr, q=Q, backend="numpy"):
    """Raw Tier-2 novelty (class-conditional N_2, class-agnostic N^a_2) of query embeddings vs D_tr embeddings."""
    one = np.ones(len(emb_tr), int)
    if backend == "torch":
        return (NV.class_conditional_qnn(emb_q, pred_q, emb_tr, y_tr, one, q, "euclidean"),
                NV.qnn(emb_q, emb_tr, one, q, "euclidean"))
    return (NV.class_conditional_novelty(emb_q, pred_q, emb_tr, np.asarray(y_tr), one, q, "euclidean"),
            NV.agnostic_novelty(emb_q, emb_tr, one, q, "euclidean"))
