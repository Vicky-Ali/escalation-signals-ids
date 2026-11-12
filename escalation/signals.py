"""The nine escalation signals (paper Sec. "Escalation signals"), computed per tier k in {1, 2}.

Order and names as in the paper (higher value = more reason to escalate):
  1 confidence     1 - C_k, C_k = temperature-scaled maximum softmax (T fitted on D_sel)
  2 novelty_cc     class-conditional novelty N~_k: q-th NN distance to D_tr samples of the predicted class
                   (Tier 1: leaf-path Hamming; Tier 2: embedding Euclidean), class-wise D_sel ECDF
  3 knn_agnostic   class-agnostic kNN novelty N^a_k (Sun et al. 2022): q-th NN over all D_tr samples, pooled D_sel ECDF
  4 kappa          confidence-novelty conflict kappa_k = C_k * N~_k
  5 mahalanobis    Tier 2: minimum class-conditional Mahalanobis distance of the embedding (shared covariance,
                   fitted on D_tr); Tier 1 has no embedding and uses 1 - C_1
  6 scod           SCOD-style w (1 - C_k) + (1 - w) N^a_k, w in {0.25, 0.5, 0.75} chosen by LOFO pseudo-UAL
                   (see cascade.choose_scod_w)
  7 energy         E_k = -T logsumexp(z / T)
  8 entropy        predictive entropy of softmax(z / T)
  9 disagreement   Tier 2: 1[argmax T1 != argmax T2] + 0.5 (1 - C_2); Tier 1 uses 1 - C_1
For budget-matched routing every raw signal is mapped through its pooled D_sel ECDF (keys prefixed `e_`).

`ScoreBundle` holds one run's per-sample scores for all partitions (D_tr, D_sel, D_crc, Test, open set, shift and,
for LOFO models, the pseudo-open family) and provides `get(part, key)` and `tiers(...)`.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, softmax

from .calibration import ClassECDF, jitter_signal
from .tier1 import Q, confidence, fit_temperature

W_GRID = [0.25, 0.5, 0.75]
# signal name -> (tier-1 score key, tier-2 score key); SCOD keys are completed with the chosen w
SIGNALS = {
    "confidence": ("conf1", "conf2"),
    "novelty_cc": ("Nc1", "Nc2"),
    "knn_agnostic": ("Na1", "Na2"),
    "kappa": ("k1", "k2"),
    "mahalanobis": ("conf1", "M2"),
    "scod": ("scod1", "scod2"),
    "energy": ("energy1", "energy2"),
    "entropy": ("entropy1", "entropy2"),
    "disagreement": ("conf1", "dis2"),
}
LABEL = {"confidence": "1 - C", "novelty_cc": "class-cond. novelty N~", "knn_agnostic": "kNN agnostic Na",
         "kappa": "kappa = C x N~", "mahalanobis": "Mahalanobis (T2; 1-C at T1)",
         "scod": "SCOD-style w(1-C)+(1-w)Na", "energy": "energy", "entropy": "predictive entropy",
         "disagreement": "T1/T2 disagreement (1-C at T1)"}


class MahalanobisOOD:
    """Class-conditional Mahalanobis distance with a shared (shrunk) covariance.
    Score = minimum squared distance to any known class centroid. Higher = more novel."""

    def __init__(self, shrinkage=1e-3):
        self.means = {}
        self.precision = None
        self.shrinkage = shrinkage

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            self.means[int(c)] = X[y == c].mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov += np.eye(cov.shape[0]) * self.shrinkage
        self.precision = np.linalg.pinv(cov)
        return self

    def score(self, X):
        all_d = []
        for c, mu in self.means.items():
            d = X - mu
            d2 = np.einsum("bi,ij,bj->b", d, self.precision, d)
            all_d.append(d2)
        return np.min(np.stack(all_d, axis=1), axis=1)


def signal_keys(name, w=None):
    """(tier-1 key, tier-2 key) of a signal; for 'scod' the weight w selects the precomputed combination."""
    k1, k2 = SIGNALS[name]
    if name == "scod":
        if w is None:
            raise ValueError("scod needs w")
        k1, k2 = f"scod1_{w}", f"scod2_{w}"
    return k1, k2


def compute_raw_scores(leaves, emb, t1_logits, t2_logits, u_tr, y_tr, query, q=Q, backend="numpy"):
    """Raw novelty and Mahalanobis scores for every row in `query` (indices into the row arrays):
    rawNc{k} / rawNa{k} (class-conditional / class-agnostic q-th NN distance to the D_tr rows `u_tr`),
    rawM2 (minimum class-conditional Mahalanobis distance of the Tier-2 embedding, fitted on D_tr).
    Rows not in `query` get NaN (they are never routed, e.g. D_tr itself)."""
    from . import tier1, tier2
    n = len(t1_logits)
    out = {}
    qi = np.asarray(query)
    for k, feat, lg, fn in ((1, leaves, t1_logits, tier1.leaf_novelty), (2, emb, t2_logits, tier2.embedding_novelty)):
        pred = np.asarray(lg).argmax(1)
        nc, na = np.full(n, np.nan), np.full(n, np.nan)
        nc[qi], na[qi] = fn(feat[qi], pred[qi], feat[u_tr], y_tr, q, backend)
        out[f"rawNc{k}"], out[f"rawNa{k}"] = nc, na
    e64 = np.asarray(emb, np.float64)
    maha = MahalanobisOOD().fit(e64[u_tr], np.asarray(y_tr))
    m2 = np.full(n, np.nan)
    m2[qi] = maha.score(e64[qi])
    out["rawM2"] = m2
    return out


class ScoreBundle:
    """Per-run scores. Inputs are per ROW (unique sample) arrays; `u[part]` maps each sample of a partition to its
    row and `y[part]` gives its label. Required partitions: 'tr', 'sel', 'crc', 'test', 'open' (+ optional 'shift',
    'pseudo')."""

    def __init__(self, classes, u, y, t1_logits, t2_logits, rawNc1, rawNa1, rawNc2, rawNa2, rawM2,
                 c1=0.0, c2=0.0, cnt=None):
        self.classes = list(classes)
        self.K = len(self.classes)
        self.parts = list(u)
        self.u = {p: np.asarray(v) for p, v in u.items()}
        self.y = {p: np.asarray(v) for p, v in y.items()}
        self.cnt = cnt or {p: np.ones(len(v), int) for p, v in self.u.items()}
        self.c1, self.c2 = float(c1), float(c2)
        n_u = len(t1_logits)
        sel_u, sel_y = self.u["sel"], self.y["sel"]
        raw = {1: (rawNc1, rawNa1), 2: (rawNc2, rawNa2)}
        U, self.temps = {}, {}
        for k, logits in ((1, t1_logits), (2, t2_logits)):
            T = fit_temperature(logits[sel_u], sel_y)
            self.temps[k] = T
            pred, C = confidence(logits, T)
            Ncr, Nar = np.asarray(raw[k][0], float), np.asarray(raw[k][1], float)
            Nc = ClassECDF(Ncr[sel_u], pred[sel_u]).transform(Ncr, pred)
            Na = ClassECDF(Nar[sel_u]).transform(Nar)
            U.update({f"pred{k}": pred, f"C{k}": C, f"Nc{k}": Nc, f"Na{k}": Na, f"k{k}": C * Nc, f"ka{k}": C * Na,
                      f"conf{k}": 1.0 - C, f"rawNc{k}": Ncr, f"rawNa{k}": Nar})
        Mr = np.asarray(rawM2, float)
        U["M2"] = ClassECDF(Mr[sel_u]).transform(Mr)
        U["rawM2"] = Mr
        U["zero"] = np.zeros(n_u)
        # energy, entropy (same temperature as C), disagreement, SCOD combinations
        for k, logits in ((1, t1_logits), (2, t2_logits)):
            T = self.temps[k]
            z = np.asarray(logits).astype(np.float64) / T
            U[f"energy{k}"] = -T * logsumexp(z, axis=1)
            p = softmax(z, axis=1)
            U[f"entropy{k}"] = -(p * np.log(np.clip(p, 1e-300, None))).sum(1)
        U["dis2"] = (U["pred1"] != U["pred2"]).astype(float) + 0.5 * U["conf2"]
        for w in W_GRID:
            for k in (1, 2):
                U[f"scod{k}_{w}"] = w * U[f"conf{k}"] + (1.0 - w) * U[f"Na{k}"]
        self.U = U
        keys = {k for name, pair in SIGNALS.items() if name != "scod" for k in pair}
        keys |= {f"scod{k}_{w}" for w in W_GRID for k in (1, 2)}
        for key in keys:                       # pooled D_sel ECDF of every signal (NaN -> 1)
            U[f"e_{key}"] = ClassECDF(U[key][sel_u]).transform(U[key])
        self.signal_keys = sorted(keys)

    @classmethod
    def from_npz(cls, path):
        """Load a score bundle (.npz) with keys classes, parts, u_<part>, y_<part>, t1_logits, t2_logits,
        rawNc1, rawNa1, rawNc2, rawNa2, rawM2 and optionally c1, c2, cnt_<part>."""
        b = np.load(path, allow_pickle=False)
        parts = list(b["parts"])
        return cls(b["classes"], {p: b[f"u_{p}"] for p in parts}, {p: b[f"y_{p}"] for p in parts},
                   b["t1_logits"], b["t2_logits"], b["rawNc1"], b["rawNa1"], b["rawNc2"], b["rawNa2"], b["rawM2"],
                   float(b["c1"]) if "c1" in b else 0.0, float(b["c2"]) if "c2" in b else 0.0,
                   {p: b[f"cnt_{p}"] for p in parts} if all(f"cnt_{p}" in b for p in parts) else None)

    def add_jitter(self, seed: int):
        """Jittered (randomised tie-breaking) versions `j_<key>` of every signal."""
        rng = np.random.default_rng(seed)
        for key in self.signal_keys:
            self.U[f"j_{key}"] = jitter_signal(self.U[key], self.u["sel"], rng)

    def get(self, part, key):
        return self.U[key][self.u[part]]

    def tiers(self, part, s1, s2, tau, lam):
        from .cascade import cascade_route
        return cascade_route(self.get(part, "C1"), self.get(part, s1), self.get(part, "C2"), self.get(part, s2),
                             tau[0], tau[1], lam[0], lam[1])
