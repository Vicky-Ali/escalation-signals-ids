"""Two-tier routing and escalation policies (paper Secs. "Problem formulation" and "Budget-matched escalation").

Routing rule at tier k: escalate x iff C_k(x) < tau_k OR s_k(x) > lambda_k; a sample accepted by Tier 1 is
resolved there, a sample escalated by Tier 1 and accepted by Tier 2 is resolved at Tier 2, otherwise it goes to the
analyst (tier 3). The analyst budget is Esc_open = Esc_1 * Esc_2 <= alpha with tier budgets alpha_1 alpha_2 = alpha
(default alpha_k = sqrt(alpha)).

* Budget-matched signals (main comparison): tau_k = 0, s_k = D_sel ECDF of the signal, lambda_k by sequential CRC
  on D_crc (`calibrate`, `budget_matched`), or by the calibration variants (jitter, Clopper-Pearson, composite CRC,
  Learn-then-Test: `calibrate_composite`).
* SCOD-style weight w chosen by Algorithm 1 (leave-one-family-out pseudo-UAL) on LOFO score bundles
  (`choose_scod_w`).
* Conflict-aware hybrid policy (original proposal): escalate if C_k < tau_k OR kappa_k > lambda_k, tau_k from
  {0.5, ..., 0.9} by Algorithm 1 (`lofo_select`), lambda_k by sequential CRC (`conflict_aware`).
"""
from __future__ import annotations

import itertools

import numpy as np

from . import calibration as CB
from .calibration import escalate
from .metrics import expected_cost, route_metrics
from .signals import W_GRID, signal_keys

TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
C3_HEAD = 300.0          # analyst cost per sample (seconds) used for tie-breaking by expected cost


def cascade_route(C1, s1, C2, s2, tau1, tau2, lam1, lam2):
    """Returns tier (1, 2 or 3 = escalated to analyst) at which each sample is resolved."""
    e1 = escalate(C1, s1, tau1, lam1)
    e2 = escalate(C2, s2, tau2, lam2)
    tier = np.where(~e1, 1, np.where(~e2, 2, 3))
    return tier


# ------------------------------------------------------------------------------------------------
# Budget-matched signals
# ------------------------------------------------------------------------------------------------
def calibrate(sc, s1, s2, a1, a2, method="crc"):
    """(lam1, lam2) or None (infeasible). tau = 0 (no confidence gate) for every budget-matched signal.
    method: 'crc' (sequential CRC; also used for jittered keys) or 'cp' (Clopper-Pearson)."""
    c1, c2 = sc.get("crc", "C1"), sc.get("crc", "C2")
    v1, v2 = sc.get("crc", s1), sc.get("crc", s2)
    if method in ("crc", "jitter"):
        l1, l2 = CB.sequential_crc(c1, v1, c2, v2, 0.0, 0.0, a1, a2, finite_sample=True)
        return None if l1 is None else (l1, l2)
    if method == "cp":
        return CB.sequential_cp(v1, v2, a1, a2)
    raise ValueError(method)


def budget_matched(sc, signal, alpha, w=None, a1=None, a2=None, method="crc", prefix="e_"):
    """Budget-matched routing of one of the nine signals: CRC at budget alpha on D_crc, then all routing metrics.
    Infeasible calibration -> nothing escalated (lambda = 1), flagged feasible=False."""
    a1 = float(np.sqrt(alpha)) if a1 is None else a1
    a2 = float(np.sqrt(alpha)) if a2 is None else a2
    k1, k2 = signal_keys(signal, w)
    s1, s2 = f"{prefix}{k1}", f"{prefix}{k2}"
    lam = calibrate(sc, s1, s2, a1, a2, method)
    feasible = lam is not None
    if not feasible:
        lam = (1.0, 1.0)
    r = route_metrics(sc, s1, s2, (0.0, 0.0), lam, alpha, feasible)
    r["alpha1"], r["alpha2"] = a1, a2
    return r


def choose_scod_w(lofo: dict, a1, a2, t1_fixed=None, prefix="e_"):
    """SCOD w by Algorithm 1 on the LOFO bundles: lowest mean pseudo-UAL (ties: fewer infeasible families first,
    then lower expected D_sel cost, then w closest to 0.5)."""
    ranked = []
    for w in W_GRID:
        s1 = t1_fixed or f"{prefix}scod1_{w}"
        s2 = f"{prefix}scod2_{w}"
        uals, costs, n_inf = [], [], 0
        for m, sc in lofo.items():
            lam = calibrate(sc, s1, s2, a1, a2, "crc")
            if lam is None:
                n_inf += 1
                continue
            uals.append(float((sc.tiers("pseudo", s1, s2, (0.0, 0.0), lam) <= 2).mean()))
            costs.append(expected_cost(sc.tiers("sel", s1, s2, (0.0, 0.0), lam), sc.c1, sc.c2, C3_HEAD))
        mu = round(float(np.mean(uals)), 12) if uals else np.inf
        mc = float(np.mean(costs)) if costs else np.inf
        ranked.append((n_inf, mu, mc, abs(w - 0.5), w))
    ranked.sort()
    b = ranked[0]
    return b[4], dict(scod_w=b[4], scod_lofo_n_infeasible=b[0], scod_lofo_mean_pseudo_UAL=b[1])


# ------------------------------------------------------------------------------------------------
# Composite CRC and Learn-then-Test
# ------------------------------------------------------------------------------------------------
def pseudo_ual(lofo, s1, s2, lam):
    vals = []
    for sc in lofo.values():
        if len(sc.u.get("pseudo", [])) == 0:
            continue
        vals.append(float((sc.tiers("pseudo", s1, s2, (0.0, 0.0), lam) <= 2).mean()))
    return float(np.mean(vals)) if vals else np.nan


def calibrate_composite(sc, s1, s2, alpha, method="composite_crc", lofo=None, delta=CB.LTT_DELTA):
    """End-to-end calibration along monotone threshold paths fitted on D_sel.
    'composite_crc': CRC of the end-to-end escalation rate on the main path (alpha_1 = alpha_2 split).
    'ltt': Learn-then-Test on 3 pre-ordered paths (fixed-sequence testing, Bonferroni delta / 3); among certified
    paths the one with the lowest LOFO pseudo-UAL (then lower D_sel rate, then closest to the main split).
    Returns (lam1, lam2, info) with lam = None if infeasible."""
    a_sel, b_sel = sc.get("sel", s1), sc.get("sel", s2)
    a_crc, b_crc = sc.get("crc", s1), sc.get("crc", s2)
    n = len(a_crc)
    exps = [CB.MAIN_EXP] if method == "composite_crc" else CB.PATH_EXPS
    paths = {e: CB.make_path(a_sel, b_sel, e) for e in exps}
    kc = {e: CB.end_to_end_counts(a_crc, b_crc, *paths[e]) for e in exps}
    if method == "composite_crc":
        l1, l2 = paths[CB.MAIN_EXP]
        j = CB.composite_crc(kc[CB.MAIN_EXP], n, alpha)
        if j is None:
            return None, dict(path_exp=CB.MAIN_EXP, rho=np.nan, crc_endtoend_rate=np.nan)
        return (float(l1[j]), float(l2[j])), dict(path_exp=CB.MAIN_EXP, rho=float(CB.RHO[j]),
                                                  crc_endtoend_rate=float(kc[CB.MAIN_EXP][j] / n))
    if method == "ltt":
        cands = []
        for e in CB.PATH_EXPS:
            jj = CB.ltt_fixed_sequence(kc[e], n, alpha, delta / len(CB.PATH_EXPS))
            if jj is None:
                continue
            lm = (float(paths[e][0][jj]), float(paths[e][1][jj]))
            sel_rate = float(((a_sel > lm[0]) & (b_sel > lm[1])).mean())
            cands.append((pseudo_ual(lofo or {}, s1, s2, lm), sel_rate, abs(e - CB.MAIN_EXP), e, jj, lm))
        if not cands:
            return None, dict(path_exp=np.nan, rho=np.nan, crc_endtoend_rate=np.nan, ltt_pseudo_UAL=np.nan,
                              ltt_n_paths_certified=0)
        cands.sort(key=lambda c: (np.inf if not np.isfinite(c[0]) else c[0], c[1], c[2]))
        pu, sr, _, e, jj, lm = cands[0]
        return lm, dict(path_exp=e, rho=float(CB.RHO[jj]), crc_endtoend_rate=float(kc[e][jj] / n),
                        ltt_pseudo_UAL=pu, ltt_n_paths_certified=len(cands))
    raise ValueError(method)


# ------------------------------------------------------------------------------------------------
# Conflict-aware hybrid policy (LOFO tau + CRC lambda on kappa)
# ------------------------------------------------------------------------------------------------
def calibrate_hybrid(sc, s1, s2, tau, a, mode="crc"):
    """Sequential CRC with confidence gate tau on D_crc ('crc') or the plug-in rule on D_sel ('plugin')."""
    part, fs = ("crc", True) if mode == "crc" else ("sel", False)
    l1, l2 = CB.sequential_crc(sc.get(part, "C1"), sc.get(part, s1), sc.get(part, "C2"),
                               sc.get(part, s2), tau[0], tau[1], a, a, finite_sample=fs)
    return None if l1 is None else (l1, l2)


def lofo_select(lofo: dict, s1, s2, mode, a):
    """Algorithm 1 over the tau grid; returns configs ranked by (n infeasible families, mean pseudo-UAL, mean cost)
    and the per-family detail rows."""
    rows, ranked = [], []
    for tau in itertools.product(TAU_GRID, TAU_GRID):
        uals, costs, n_inf = [], [], 0
        for m, sc in lofo.items():
            lam = calibrate_hybrid(sc, s1, s2, tau, a, mode)
            if lam is None:
                n_inf += 1
                rows.append(dict(family=m, tau1=tau[0], tau2=tau[1], feasible=False))
                continue
            ual = float((sc.tiers("pseudo", s1, s2, tau, lam) <= 2).mean())
            cost = expected_cost(sc.tiers("sel", s1, s2, tau, lam), sc.c1, sc.c2, C3_HEAD)
            uals.append(ual)
            costs.append(cost)
            rows.append(dict(family=m, tau1=tau[0], tau2=tau[1], feasible=True, lam1=lam[0], lam2=lam[1],
                             pseudo_UAL=ual, sel_cost=cost))
        if len(uals) == 0:
            continue
        ranked.append((n_inf, round(float(np.mean(uals)), 12), float(np.mean(costs)), tau))
    ranked.sort(key=lambda r: (r[0], r[1], r[2]))
    return ranked, rows


def conflict_aware(sc, lofo: dict, alpha, s1="k1", s2="k2", mode="crc"):
    """Conflict-aware hybrid policy: tau by Algorithm 1 on the LOFO bundles, lambda by sequential CRC on D_crc of
    the final bundle. The best-ranked feasible tau is used; none feasible -> nothing escalated (flagged)."""
    a = float(np.sqrt(alpha))
    ranked, _ = lofo_select(lofo, s1, s2, mode, a)
    chosen = lam = None
    for _, _, _, tau in ranked:
        lam = calibrate_hybrid(sc, s1, s2, tau, a, mode)
        if lam is not None:
            chosen = tau
            break
    feasible = chosen is not None
    if not feasible:
        chosen, lam = (0.0, 0.0), (1.0, 1.0)
    r = route_metrics(sc, s1, s2, chosen, lam, alpha, feasible)
    r.update(tau1=chosen[0], tau2=chosen[1], alpha_k=a)
    return r
