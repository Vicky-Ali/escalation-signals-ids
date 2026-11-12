"""Evaluation metrics (paper Sec. "Metrics").

* UAL (unseen-attack leakage) = fraction of unseen-attack samples accepted at Tier 1 or Tier 2, split into
  missed (accepted as benign) and alerted (accepted as another, known attack); UAL = missed + alerted.
* HCUL_0.9 = accepted with confidence >= 0.9 at the accepting tier.
* Esc_1, Esc_2, Esc_open = escalation rates on known Test data (Esc_open = fraction reaching the analyst; the budget
  constraint is Esc_open <= alpha).
* FE10k_id / FE10k_shift = false escalations to the analyst per 10,000 benign samples (in-distribution / shift).
* Known-class macro P/R/F1 of the cascade (escalated samples take the analyst's ground-truth label), expected cost.
* `open_vs_known_auroc`: AUROC of a signal for separating unseen attacks from known Test samples.
"""
from __future__ import annotations

import numpy as np

HC = 0.9


def open_set_metrics(tier, pred1, pred2, C1, C2, benign=0, hc=HC):
    """UAL = fraction of unseen-attack samples accepted at T1 or T2; missed = accepted as benign,
    alerted = accepted as another (known) attack. HCUL_hc = accepted with confidence >= hc at the accepting tier."""
    tier = np.asarray(tier)
    n = len(tier)
    if n == 0:
        return dict(n_open=0, UAL=np.nan, UAL_missed=np.nan, UAL_alerted=np.nan, HCUL=np.nan)
    pred = np.where(tier == 1, pred1, pred2)
    conf = np.where(tier == 1, C1, C2)
    acc = tier <= 2
    return dict(n_open=n, UAL=acc.mean(), UAL_missed=(acc & (pred == benign)).mean(),
                UAL_alerted=(acc & (pred != benign)).mean(), HCUL=(acc & (conf >= hc)).mean())


def escalation_metrics(tier):
    tier = np.asarray(tier)
    n = len(tier)
    e1 = tier >= 2
    esc1 = e1.mean() if n else np.nan
    esc2 = (tier[e1] == 3).mean() if e1.any() else 0.0
    return dict(Esc_1=esc1, Esc_2=esc2, Esc_open=(tier == 3).mean() if n else np.nan)


def false_escalations_per_10k(tier):
    """False escalations to the analyst per 10,000 benign samples (`tier` of benign samples only)."""
    tier = np.asarray(tier)
    return 1e4 * float((tier == 3).mean()) if len(tier) else np.nan


def macro_prf(y, yhat, n_classes):
    """Macro precision / recall / F1 over the known classes (classes absent from y are skipped)."""
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    P, R, F = [], [], []
    for c in range(n_classes):
        sup = (y == c).sum()
        if sup == 0:
            continue
        tp = ((y == c) & (yhat == c)).sum()
        pp = (yhat == c).sum()
        p = tp / pp if pp else 0.0
        r = tp / sup
        P.append(p)
        R.append(r)
        F.append(2 * p * r / (p + r) if p + r else 0.0)
    return dict(macro_P=float(np.mean(P)), macro_R=float(np.mean(R)), macro_F1=float(np.mean(F)))


def cascade_known_prediction(tier, pred1, pred2, y_true):
    """Final label on known data: T1/T2 prediction where accepted, ground truth (analyst oracle) if escalated."""
    return np.where(tier == 1, pred1, np.where(tier == 2, pred2, y_true))


def expected_cost(tier, c1, c2, c3):
    tier = np.asarray(tier)
    return float(np.mean(c1 + (tier >= 2) * c2 + (tier == 3) * c3)) if len(tier) else np.nan


def open_vs_known_auroc(score_known, score_open):
    """AUROC of a (higher = more novel) score for unseen attacks (positives) vs known Test samples; NaN ignored."""
    from sklearn.metrics import roc_auc_score
    yb = np.r_[np.zeros(len(score_known)), np.ones(len(score_open))]
    v = np.r_[score_known, score_open].astype(float)
    ok = np.isfinite(v)
    return float(roc_auc_score(yb[ok], v[ok])) if len(score_open) and yb[ok].min() < 1 else np.nan


def route_metrics(sc, s1, s2, tau, lam, alpha, feasible=True, hc=HC):
    """All per-run routing metrics of one calibrated configuration on a `ScoreBundle` with partitions
    test (known Test), open (unseen attacks) and optionally shift (benign shift)."""
    benign = sc.classes.index("benign") if "benign" in sc.classes else 0
    t = {p: sc.tiers(p, s1, s2, tau, lam) for p in ("test", "open", "shift", "crc") if p in sc.u}
    r = dict(alpha=alpha, feasible=feasible, lam1=lam[0], lam2=lam[1])
    r.update(open_set_metrics(t["open"], sc.get("open", "pred1"), sc.get("open", "pred2"),
                              sc.get("open", "C1"), sc.get("open", "C2"), benign=benign, hc=hc))
    r.update(escalation_metrics(t["test"]))
    if "crc" in t:
        r["Esc_open_crc"] = float((t["crc"] == 3).mean())
    r["budget_ok"] = bool(r["Esc_open"] <= alpha)
    bt = sc.y["test"] == benign
    r["FE10k_id"] = false_escalations_per_10k(t["test"][bt]) if bt.any() else np.nan
    r["FE10k_shift"] = false_escalations_per_10k(t["shift"]) if "shift" in t else np.nan
    final = cascade_known_prediction(t["test"], sc.get("test", "pred1"), sc.get("test", "pred2"), sc.y["test"])
    r.update({f"cascade_{k}": v for k, v in macro_prf(sc.y["test"], final, sc.K).items()})
    return r
