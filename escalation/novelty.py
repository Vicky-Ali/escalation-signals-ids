"""q-th nearest-neighbour novelty (paper Sec. "Escalation signals": class-conditional and class-agnostic novelty).

Novelty of a query x at tier k = distance to the q-th nearest reference sample of D_tr
  * class-conditional: references restricted to the class predicted by tier k (N_k);
  * class-agnostic (kNN baseline, Sun et al. 2022): references from all classes (N^a_k).
Distances:
  * Tier 1 ("hamming"): normalised Hamming distance between XGBoost leaf-index vectors
    (fraction of trees in which the two samples fall into different leaves);
  * Tier 2 ("euclidean"): Euclidean distance between L2-normalised encoder embeddings.
All functions take UNIQUE reference rows with multiplicities (`ref_counts`), which is exact for the q-th nearest
neighbour when the reference set contains duplicates; q_eff = min(q, total reference count).

Two equivalent implementations are provided:
  * NumPy brute force (`pairwise_distance`, `qth_nn_distance`, `class_conditional_novelty`, `agnostic_novelty`);
  * an exact, chunked PyTorch implementation (`qnn`, `class_conditional_qnn`) used for large flow datasets
    (int8 one-hot matrix products for the leaf Hamming distance, exact int32 accumulation). It returns the same
    values as the NumPy version (Euclidean in float32).
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------------------------
# NumPy reference implementation
# ------------------------------------------------------------------------------------------------
def pairwise_distance(Q: np.ndarray, R: np.ndarray, metric: str) -> np.ndarray:
    if metric == "hamming":          # normalised Hamming: fraction of trees with a different leaf
        out = np.empty((len(Q), len(R)))
        for i in range(0, len(Q), 256):
            out[i:i + 256] = (Q[i:i + 256, None, :] != R[None, :, :]).mean(2)
        return out
    if metric == "euclidean":
        Q = np.asarray(Q, float)
        R = np.asarray(R, float)
        d2 = (Q ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None, :] - 2 * Q @ R.T
        return np.sqrt(np.clip(d2, 0, None))
    raise ValueError(metric)


def qth_nn_distance(D: np.ndarray, ref_counts: np.ndarray, q: int) -> np.ndarray:
    """D: (n_query, n_ref_unique) distances; ref_counts: multiplicity of each unique reference row.
    Returns the distance to the q_eff-th nearest reference sample, q_eff = min(q, total count)."""
    ref_counts = np.asarray(ref_counts, int)
    q_eff = int(min(q, ref_counts.sum()))
    if q_eff == 0:
        return np.full(len(D), np.nan)
    order = np.argsort(D, axis=1, kind="stable")
    cum = np.cumsum(ref_counts[order], axis=1)
    pos = (cum >= q_eff).argmax(1)
    return D[np.arange(len(D)), order[np.arange(len(D)), pos]]


def class_conditional_novelty(Q, pred, R, r_cls, r_counts, q, metric) -> np.ndarray:
    """Distance of each query to the q-th nearest reference sample of its PREDICTED class."""
    pred = np.asarray(pred)
    out = np.full(len(Q), np.nan)
    for c in np.unique(pred):
        qi = np.flatnonzero(pred == c)
        ri = np.flatnonzero(r_cls == c)
        if len(ri) == 0:
            continue                      # no reference for this class -> NaN, treated as maximally novel
        out[qi] = qth_nn_distance(pairwise_distance(Q[qi], R[ri], metric), r_counts[ri], q)
    return out


def agnostic_novelty(Q, R, r_counts, q, metric) -> np.ndarray:
    """Class-agnostic kNN (Sun et al. 2022): q-th nearest reference sample over ALL classes."""
    return qth_nn_distance(pairwise_distance(Q, R, metric), r_counts, q)


# ------------------------------------------------------------------------------------------------
# Exact chunked PyTorch implementation (GPU if available)
# ------------------------------------------------------------------------------------------------
REF_BYTES = 2.5e9        # int8 one-hot reference chunk budget
QRY_BYTES = 0.8e9        # int8 one-hot query batch budget
EUC_REF_CHUNK = 30_000
OUT_ELEMS = 6e7          # (query batch x reference chunk) distance block


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _qth_from_candidates(dist, cnt, q_eff: int):
    """dist, cnt: (b, k) candidate distances (any order) and multiplicities -> q_eff-th sample distance."""
    import torch
    d, o = torch.sort(dist, dim=1, stable=True)
    c = torch.gather(cnt, 1, o).cumsum(1)
    pos = (c >= q_eff).int().argmax(1)
    return d.gather(1, pos[:, None])[:, 0]


def _merge(best_d, best_c, d, c, k):
    import torch
    if best_d is None:
        v, i = torch.topk(d, min(k, d.shape[1]), dim=1, largest=False)
        return v, torch.gather(c, 1, i)
    d = torch.cat([best_d, d], 1)
    c = torch.cat([best_c, c], 1)
    v, i = torch.topk(d, min(k, d.shape[1]), dim=1, largest=False)
    return v, torch.gather(c, 1, i)


def _r8(n):
    return int(-(-n // 8) * 8)


def _hamming_qnn(Q, R, counts, q_eff, k):
    import torch
    DEV = _device()
    with torch.no_grad():
        Q, R = np.asarray(Q), np.asarray(R)
        T = R.shape[1]
        M = int(max(R.max(), Q.max() if len(Q) else 0)) + 1
        Rg = torch.from_numpy(R.astype(np.int16)).to(DEV)
        Qg = torch.from_numpy(Q.astype(np.int16)).to(DEV)
        off = torch.arange(T, device=DEV, dtype=torch.int64)[None, :] * M
        # compact column id per (tree, leaf) pair present in the reference (presence table, T x M)
        used = torch.zeros(T * M, dtype=torch.bool, device=DEV)
        for rs in range(0, len(R), 20000):
            used[(Rg[rs:rs + 20000].long() + off).ravel()] = True
        cmap = torch.cumsum(used.long(), 0) - 1
        L = int(used.sum())
        Lp = _r8(L)
        n, nr = len(Q), len(R)
        RC = max(8, min(_r8(nr), int(REF_BYTES // Lp) // 8 * 8, 40_000))
        QB = max(24, min(_r8(max(n, 24)), int(QRY_BYTES // Lp) // 8 * 8, int(OUT_ELEMS // RC) // 8 * 8))
        out_d = [None] * ((n + QB - 1) // QB)
        out_c = [None] * len(out_d)
        cnt_all = torch.tensor(np.asarray(counts, np.int64), device=DEV)
        for rs in range(0, nr, RC):
            rc = cmap[Rg[rs:rs + RC].long() + off]
            m = rc.shape[0]
            mp = _r8(m)
            B = torch.zeros(Lp, mp, dtype=torch.int8, device=DEV)
            B[rc, torch.arange(m, device=DEV)[:, None].expand(-1, T)] = 1
            rcnt = torch.zeros(mp, dtype=torch.int64, device=DEV)
            rcnt[:m] = cnt_all[rs:rs + m]
            pad = torch.arange(mp, device=DEV) >= m
            for bi, qs in enumerate(range(0, n, QB)):
                code = Qg[qs:qs + QB].long() + off
                qv = used[code]                                     # leaf never reached by a reference row: no match
                qc = cmap[code]
                b = qc.shape[0]
                A = torch.zeros(max(b, 24), Lp, dtype=torch.int8, device=DEV)
                A[torch.arange(b, device=DEV)[:, None].expand(-1, T)[qv], qc[qv]] = 1
                match = torch._int_mm(A, B)[:b]                            # (b, mp) int32, exact
                d = 1.0 - match.float() / T
                d[:, pad] = float("inf")
                out_d[bi], out_c[bi] = _merge(out_d[bi], out_c[bi], d, rcnt[None, :].expand(b, -1), k)
            del B
        res = np.empty(n)
        for bi, qs in enumerate(range(0, n, QB)):
            res[qs:qs + QB] = _qth_from_candidates(out_d[bi], out_c[bi], q_eff).double().cpu().numpy()
        return res


def _euclid_qnn(Q, R, counts, q_eff, k, batch=4096):
    import torch
    DEV = _device()
    with torch.no_grad():
        Qa = torch.tensor(np.asarray(Q, np.float32), device=DEV)
        Ra = torch.tensor(np.asarray(R, np.float32), device=DEV)
        ca = torch.tensor(np.asarray(counts, np.int64), device=DEV)
        res = np.empty(len(Q))
        for qs in range(0, len(Q), batch):
            Qc = Qa[qs:qs + batch]
            q2 = (Qc ** 2).sum(1)
            bd = bc = None
            for rs in range(0, len(Ra), EUC_REF_CHUNK):
                Rc = Ra[rs:rs + EUC_REF_CHUNK]
                d = torch.sqrt(torch.clamp(q2[:, None] + (Rc ** 2).sum(1)[None, :] - 2.0 * Qc @ Rc.T, min=0.0))
                bd, bc = _merge(bd, bc, d, ca[rs:rs + EUC_REF_CHUNK][None, :].expand(len(Qc), -1), k)
            res[qs:qs + batch] = _qth_from_candidates(bd, bc, q_eff).double().cpu().numpy()
        return res


def qnn(Q, R, ref_counts, q, metric) -> np.ndarray:
    """q-th NN distance of each query row to the reference rows (class-agnostic), exact, chunked PyTorch."""
    ref_counts = np.asarray(ref_counts, np.int64)
    if len(R) == 0 or len(Q) == 0 or ref_counts.sum() == 0:
        return np.full(len(Q), np.nan)
    q_eff = int(min(q, ref_counts.sum()))
    k = int(min(q_eff, len(R)))
    if metric == "hamming":
        return _hamming_qnn(Q, R, ref_counts, q_eff, k)
    if metric == "euclidean":
        return _euclid_qnn(Q, R, ref_counts, q_eff, k)
    raise ValueError(metric)


def class_conditional_qnn(Q, pred, R, r_cls, r_counts, q, metric) -> np.ndarray:
    """q-th NN distance to the reference rows of the PREDICTED class (NaN if that class has no reference)."""
    pred = np.asarray(pred)
    out = np.full(len(Q), np.nan)
    for c in np.unique(pred):
        qi = np.flatnonzero(pred == c)
        ri = np.flatnonzero(np.asarray(r_cls) == c)
        if len(ri) == 0:
            continue
        out[qi] = qnn(Q[qi], R[ri], np.asarray(r_counts)[ri], q, metric)
    return out
