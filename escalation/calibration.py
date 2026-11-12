"""Score normalisation and budget calibration (paper Sec. "Budget-matched escalation and calibration").

* `ClassECDF`: empirical CDF on D_sel (class-wise or pooled); every signal is mapped to [0, 1] by its D_sel ECDF.
* `crc_lambda` / `sequential_crc`: conformal risk control (Angelopoulos et al.) of the escalation rate,
  lambda_hat = inf{lambda : n/(n+1) R_hat(lambda) + 1/(n+1) <= alpha_k}; sequential over the tiers
  (lambda_1 on all D_crc, lambda_2 on the D_crc samples Tier 1 escalates); empty admissible set -> infeasible.
* `jitter_signal`: randomised tie-breaking inside ties of the raw signal before the ECDF (calibration variant).
* `cp_lambda`: high-probability control with a one-sided Clopper-Pearson bound (delta = 0.1).
* `make_path`, `end_to_end_counts`, `composite_crc`: composite (end-to-end) CRC along a monotone threshold path
  fitted on D_sel; `ltt_fixed_sequence`: Learn-then-Test with binomial p-values and fixed-sequence testing.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binom

CP_DELTA = 0.1
LTT_DELTA = 0.1
PATH_EXPS = [0.25, 0.5, 0.75]          # tier-1 share of the log budget along a path: r1 = rho^e, r2 = rho^(1-e)
MAIN_EXP = 0.5                          # = the main split alpha_1 = alpha_2 = sqrt(alpha)
RHO = np.concatenate([[0.0], np.logspace(-4, 0, 401)])


# ------------------------------------------------------------------------------------------------
# ECDF normalisation
# ------------------------------------------------------------------------------------------------
class ClassECDF:
    """N~(x) = fraction of D_sel samples with the same predicted class whose score is <= N(x).
    Classes without D_sel samples fall back to the pooled ECDF. NaN scores map to 1 (maximally novel)."""

    def __init__(self, scores, pred=None):
        s = np.asarray(scores, float)
        s = np.where(np.isnan(s), np.inf, s)
        self.pooled = np.sort(s)
        self.by = {}
        if pred is not None:
            pred = np.asarray(pred)
            for c in np.unique(pred):
                self.by[int(c)] = np.sort(s[pred == c])

    @staticmethod
    def _ecdf(ref, x):
        if len(ref) == 0:
            return np.ones_like(x)
        return np.searchsorted(ref, x, side="right") / len(ref)

    def transform(self, scores, pred=None):
        x = np.asarray(scores, float)
        nan = np.isnan(x)
        x = np.where(nan, np.inf, x)
        if pred is None or not self.by:
            out = self._ecdf(self.pooled, x)
        else:
            pred = np.asarray(pred)
            out = np.empty(len(x))
            for c in np.unique(pred):
                m = pred == c
                out[m] = self._ecdf(self.by.get(int(c), self.pooled), x[m])
        out[nan] = 1.0
        return out


def jitter_signal(raw, sel_idx, rng):
    """Randomised tie-breaking: dense rank + U[0, 0.999) (order-preserving between distinct values, random within
    ties), pooled D_sel ECDF, then the within-level order (normalised rank) appended for the CRC:
    score = (n_sel * ECDF + u) / (n_sel + 1)."""
    raw = np.asarray(raw, float)
    nan = np.isnan(raw)
    j = np.full(len(raw), np.nan)
    _, dense = np.unique(raw[~nan], return_inverse=True)
    j[~nan] = dense + rng.uniform(0.0, 0.999, (~nan).sum())
    e = ClassECDF(j[sel_idx]).transform(j)
    order = np.argsort(np.where(nan, np.inf, j), kind="stable")
    u = np.empty(len(j))
    u[order] = np.arange(len(j)) / max(1, len(j))
    n_sel = len(sel_idx)
    return (n_sel * e + u) / (n_sel + 1)


# ------------------------------------------------------------------------------------------------
# Sequential conformal risk control
# ------------------------------------------------------------------------------------------------
def escalate(C, s, tau, lam):
    """Escalate if C < tau OR s > lambda."""
    return (np.asarray(C) < tau) | (np.asarray(s) > lam)


def crc_lambda(C, s, tau, alpha, finite_sample=True):
    """lambda_hat = inf{lambda in [0,1] : n/(n+1) R_hat(lambda) + 1/(n+1) <= alpha},
    R_hat(lambda) = mean 1[C < tau or s > lambda].  Returns None if the admissible set is empty.
    finite_sample=False gives the plug-in rule R_hat(lambda) <= alpha (no-CRC ablation)."""
    C = np.asarray(C, float)
    s = np.asarray(s, float)
    n = len(C)
    if n == 0:
        return None
    low = C < tau
    cand = np.unique(np.concatenate([[0.0], np.clip(s, 0, 1), [1.0]]))
    # R_hat is non-increasing in lambda; evaluate on all candidate points (inf is attained at one of them)
    srt = np.sort(s[~low])
    n_low = low.sum()
    exceed = len(srt) - np.searchsorted(srt, cand, side="right")   # #{not low and s > lambda}
    R = (n_low + exceed) / n
    bound = n / (n + 1) * R + 1 / (n + 1) if finite_sample else R
    ok = np.flatnonzero(bound <= alpha + 1e-12)
    if len(ok) == 0:
        return None
    return float(cand[ok[0]])


def sequential_crc(C1, s1, C2, s2, tau1, tau2, alpha1, alpha2, finite_sample=True):
    """lambda_1 on all calibration samples; lambda_2 on the calibration samples that T1 escalates.
    If T1 escalates no calibration sample, T2 cannot be calibrated -> lambda_2 = 1 (confidence-only at T2)."""
    lam1 = crc_lambda(C1, s1, tau1, alpha1, finite_sample)
    if lam1 is None:
        return None, None
    e1 = escalate(C1, s1, tau1, lam1)
    if e1.sum() == 0:
        return lam1, 1.0
    lam2 = crc_lambda(np.asarray(C2)[e1], np.asarray(s2)[e1], tau2, alpha2, finite_sample)
    if lam2 is None:
        return None, None
    return lam1, lam2


# ------------------------------------------------------------------------------------------------
# Clopper-Pearson (high-probability) control
# ------------------------------------------------------------------------------------------------
def cp_lambda(s, alpha_k, delta=CP_DELTA):
    """Smallest lambda in the candidate grid whose one-sided Clopper-Pearson (1 - delta) upper bound of
    mean(s > lambda) is <= alpha_k; None if no lambda qualifies."""
    s = np.asarray(s, float)
    n = len(s)
    if n == 0:
        return None
    cand = np.unique(np.concatenate([[0.0], np.clip(s, 0, 1), [1.0]]))
    srt = np.sort(s)
    x = n - np.searchsorted(srt, cand, side="right")
    ub = np.where(x >= n, 1.0, beta_dist.ppf(1 - delta, x + 1, np.maximum(n - x, 1)))
    ok = np.flatnonzero(ub <= alpha_k + 1e-12)
    return None if len(ok) == 0 else float(cand[ok[0]])


def sequential_cp(v1, v2, a1, a2, delta=CP_DELTA):
    """(lam1, lam2) with Clopper-Pearson control at each tier (tier 2 on the samples tier 1 escalates), or None."""
    l1 = cp_lambda(v1, a1, delta)
    if l1 is None:
        return None
    e1 = v1 > l1
    if e1.sum() == 0:
        return (l1, 1.0)
    l2 = cp_lambda(v2[e1], a2, delta)
    return None if l2 is None else (l1, l2)


# ------------------------------------------------------------------------------------------------
# Composite (end-to-end) CRC and Learn-then-Test along a monotone path
# ------------------------------------------------------------------------------------------------
def rate_lambda(s, rate):
    """Smallest candidate lambda with mean(s > lambda) <= rate (s on the D_sel ECDF scale, in [0, 1])."""
    if len(s) == 0 or rate >= 1:
        return 0.0 if len(s) else 1.0
    srt = np.sort(s)
    cand = np.unique(np.concatenate([[0.0], np.clip(srt, 0, 1), [1.0]]))
    frac = (len(srt) - np.searchsorted(srt, cand, side="right")) / len(srt)
    return float(cand[np.flatnonzero(frac <= rate + 1e-12)[0]])


def make_path(a_sel, b_sel, e):
    """Monotone path rho -> (lam1, lam2), fitted on D_sel only (fixed before calibration): D_sel tier-1 rate rho^e
    and conditional tier-2 rate rho^(1-e); lam2 made non-increasing in rho (running minimum), lam1 is non-increasing
    by construction. rho = 0 -> (1, 1) (nothing escalated). End-to-end escalation is therefore non-decreasing in rho."""
    l1 = np.empty(len(RHO))
    l2 = np.empty(len(RHO))
    for i, r in enumerate(RHO):
        if r == 0:
            l1[i], l2[i] = 1.0, 1.0
            continue
        l1[i] = rate_lambda(a_sel, r ** e)
        e1 = a_sel > l1[i]
        l2[i] = rate_lambda(b_sel[e1], r ** (1 - e)) if e1.any() else 1.0
    l2 = np.minimum.accumulate(l2)
    return l1, l2


def end_to_end_counts(a, b, l1, l2):
    """Number of calibration samples escalated by BOTH tiers at every path point."""
    return np.array([int(((a > x) & (b > y)).sum()) for x, y in zip(l1, l2)])


def composite_crc(k, n, alpha):
    """Index of the largest rho with n/(n+1) R_hat + 1/(n+1) <= alpha (loss monotone along the path), or None."""
    if n == 0:
        return None
    bound = n / (n + 1) * (k / n) + 1 / (n + 1)
    ok = np.flatnonzero(bound <= alpha + 1e-12)
    if len(ok) == 0:
        return None
    # monotone path -> admissible set is a prefix; take the last index of the leading admissible run
    last = ok[0]
    for j in ok[1:]:
        if j != last + 1:
            break
        last = j
    return int(last) if ok[0] == 0 else None


def ltt_fixed_sequence(k, n, alpha, delta):
    """Fixed-sequence testing of H_j: R(rho_j) > alpha in increasing rho; p_j = P(Bin(n, alpha) <= k_j)
    (exact binomial tail). Returns the index of the last certified rho (None if the first test fails)."""
    p = binom.cdf(k, n, alpha)
    last = None
    for j in range(len(p)):
        if p[j] <= delta:
            last = j
        else:
            break
    return last
