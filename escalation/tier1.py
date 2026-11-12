"""Tier 1: XGBoost classifier, temperature scaling and leaf-path novelty (paper Sec. "Cascade and tiers").

* `T1Flow`: multi-class XGBoost (hist) on numeric features; class-weight exponent gamma in sample weights.
* `tune_t1`: grid `XGB_GRID` chosen once per fold by D_sel log-loss.
* `fit_temperature` / `confidence`: C_k = temperature-scaled maximum softmax, T fitted by NLL on D_sel
  (used for both tiers).
* `leaf_novelty`: raw Tier-1 novelty = normalised Hamming distance between leaf-index vectors (fraction of trees
  with a different leaf) to the q-th nearest D_tr sample of the predicted class (and the class-agnostic variant).
"""
from __future__ import annotations

import itertools
import random

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from . import novelty as NV

Q = 10                     # q of the q-th nearest neighbour
XGB_GRID = dict(max_depth=[3, 6], learning_rate=[0.1, 0.3], n_estimators=[100, 300], class_weight_gamma=[0.0, 0.5, 1.0])


def _device_type():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:          # pragma: no cover
        return "cpu"


def seed_all(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(y: np.ndarray, n_classes: int, gamma: float = 1.0) -> np.ndarray:
    """Per-class weight (n / (K n_c))^gamma: gamma=1 class-balanced, 0 natural prior, 0.5 in between."""
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    return np.where(cnt > 0, (len(y) / (n_classes * np.maximum(cnt, 1))) ** gamma, 0.0)


# ------------------------------------------------------------------------------------------------
# Confidence (both tiers)
# ------------------------------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar temperature T minimising NLL of softmax(logits / T) on (logits, y)."""
    logits = np.asarray(logits, float)
    y = np.asarray(y, int)
    if len(y) == 0:
        return 1.0

    def nll(logt):
        lp = log_softmax(logits / np.exp(logt), axis=1)
        return -lp[np.arange(len(y)), y].mean()
    r = minimize_scalar(nll, bounds=(np.log(0.01), np.log(100.0)), method="bounded")
    return float(np.exp(r.x))


def confidence(logits: np.ndarray, T: float = 1.0):
    """(predicted class, temperature-scaled max softmax)."""
    p = softmax(np.asarray(logits, float) / T, axis=1)
    return p.argmax(1), p.max(1)


# ------------------------------------------------------------------------------------------------
# XGBoost Tier 1
# ------------------------------------------------------------------------------------------------
class T1Flow:
    def __init__(self, n_classes: int, params: dict, seed: int, device: str | None = None):
        self.K = n_classes
        self.params = params
        self.seed = seed
        self.device = device or _device_type()

    def fit(self, X, y):
        import xgboost as xgb
        p = dict(self.params)
        gamma = p.pop("class_weight_gamma", 1.0)
        w = class_weights(y, self.K, gamma)[y]
        self.clf = xgb.XGBClassifier(objective="multi:softprob", num_class=self.K, tree_method="hist",
                                     device=self.device, random_state=self.seed, eval_metric="mlogloss",
                                     n_jobs=4, **p)
        self.clf.fit(X, y, sample_weight=w)
        self.booster = self.clf.get_booster()
        self.booster.set_param({"device": self.device})
        return self

    def _dm(self, X):
        import xgboost as xgb
        return xgb.DMatrix(np.asarray(X, np.float32), missing=np.nan)

    def logits(self, X):
        return self.booster.predict(self._dm(X), output_margin=True).reshape(len(X), self.K)

    def leaves(self, X, chunk=50_000):
        """Leaf index of every sample in every tree (the leaf-path representation phi_1)."""
        out = []
        for s in range(0, len(X), chunk):
            lv = self.booster.predict(self._dm(X[s:s + chunk]), pred_leaf=True)
            out.append(lv.reshape(min(chunk, len(X) - s), -1).astype(np.int16))
        return np.concatenate(out) if out else np.zeros((0, 0), np.int16)


def tune_t1(Xtr, ytr, Xsel, ysel, K, seed=0, log=print):
    best, rows = None, []
    keys = list(XGB_GRID)
    for vals in itertools.product(*XGB_GRID.values()):
        p = dict(zip(keys, vals))
        lg = T1Flow(K, p, seed).fit(Xtr, ytr).logits(Xsel)
        lp = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
        ll = float(-lp[np.arange(len(ysel)), ysel].mean())
        rows.append(dict(**p, sel_logloss=ll))
        log(f"   [T1 tune] {p} D_sel log-loss={ll:.5f}")
        if best is None or ll < best[1]:
            best = (p, ll)
    return best[0], rows


# ------------------------------------------------------------------------------------------------
# Leaf-path novelty
# ------------------------------------------------------------------------------------------------
def leaf_novelty(leaves_q, pred_q, leaves_tr, y_tr, q=Q, backend="numpy"):
    """Raw Tier-1 novelty (class-conditional N_1, class-agnostic N^a_1) of query leaf vectors vs the D_tr leaf vectors.
    backend 'numpy' (brute force) or 'torch' (exact chunked implementation for large data)."""
    one = np.ones(len(leaves_tr), int)
    if backend == "torch":
        return (NV.class_conditional_qnn(leaves_q, pred_q, leaves_tr, y_tr, one, q, "hamming"),
                NV.qnn(leaves_q, leaves_tr, one, q, "hamming"))
    return (NV.class_conditional_novelty(leaves_q, pred_q, leaves_tr, np.asarray(y_tr), one, q, "hamming"),
            NV.agnostic_novelty(leaves_q, leaves_tr, one, q, "hamming"))
