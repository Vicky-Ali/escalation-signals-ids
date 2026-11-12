"""Runnable demo on SYNTHETIC data (no dataset needed; about a minute on a CPU).

Walks through the method end to end:
  1. synthetic flow-like data: benign + 3 known attack classes, one UNSEEN attack family (open set) and a shifted
     benign population; known data split 60/10/10/20 into D_tr / D_sel / D_crc / Test;
  2. Tier 1 (XGBoost) and Tier 2 (MLP + supervised contrastive encoder + linear head) trained on D_tr;
  3. raw scores (leaf-path / embedding novelty, Mahalanobis) and the nine escalation signals (ScoreBundle);
  4. leave-one-family-out (LOFO) bundles for Algorithm 1 (SCOD weight selection);
  5. calibration of every signal at alpha = 0.05 with composite (end-to-end) conformal risk control on D_crc;
  6. escalation rate on known Test (Esc_open, budget alpha) and unseen-attack leakage (UAL = missed + alerted).
Small model sizes are set here only to keep the demo fast; the package defaults are the paper's settings.
"""
import time

import numpy as np

from escalation import cascade as CA
from escalation import metrics as MT
from escalation.signals import SIGNALS, ScoreBundle, compute_raw_scores, signal_keys
from escalation.tier1 import T1Flow
from escalation.tier2 import T2Flow

ALPHA = 0.05
T1_PARAMS = dict(max_depth=3, learning_rate=0.3, n_estimators=30, class_weight_gamma=1.0)   # demo size
T2_CFG = dict(steps=200, batch=256)                                                           # demo size


def make_data(seed=0, d=16):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 1.0, (5, d))
    sizes = {0: 3000, 1: 800, 2: 800, 3: 800, 4: 800}         # 0 benign, 1-3 known attacks, 4 unseen attack
    X, y = [], []
    for c, n in sizes.items():
        X.append(centres[c] + rng.normal(0, 1.0, (n, d)))
        y.append(np.full(n, c))
    X, y = np.vstack(X), np.concatenate(y)
    X_shift = centres[0] + 0.5 + rng.normal(0, 1.2, (1000, d))   # benign shift (never used for fitting)
    return X.astype(np.float32), y, X_shift.astype(np.float32)


def split(y, rng, ratios=(0.6, 0.1, 0.1, 0.2)):
    parts = {p: [] for p in ("tr", "sel", "crc", "test")}
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        cut = np.cumsum([int(round(r * len(idx))) for r in ratios[:-1]])
        for p, chunk in zip(parts, np.split(idx, cut)):
            parts[p].append(chunk)
    return {p: np.sort(np.concatenate(v)) for p, v in parts.items()}


def build_bundle(X, y, part_rows, extra_rows, classes, seed=0):
    """Train T1/T2 on part_rows['tr'] and return a ScoreBundle over all partitions (rows are indices into X)."""
    K = len(classes)
    tr, sel = part_rows["tr"], part_rows["sel"]
    t1 = T1Flow(K, T1_PARAMS, seed, device="cpu").fit(X[tr], y[tr])
    t2 = T2Flow(K, seed, T2_CFG).fit(X[tr], y[tr], X[sel], y[sel], log=lambda *a: None)
    lg1, leaves = t1.logits(X), t1.leaves(X)
    lg2, emb = t2.predict(X)
    rows = {**part_rows, **extra_rows}
    query = np.unique(np.concatenate([v for p, v in rows.items() if p != "tr"]))
    raw = compute_raw_scores(leaves, emb, lg1, lg2, tr, y[tr], query, backend="numpy")
    return ScoreBundle(classes, rows, {p: y[v] for p, v in rows.items()}, lg1, lg2, **raw)


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    X, y_all, X_shift = make_data()
    known = y_all < 4
    classes = ["benign", "attack_A", "attack_B", "attack_C"]
    Xk, yk = X[known], y_all[known]
    Xall = np.vstack([Xk, X[~known], X_shift])                       # rows: known | open | shift
    yall = np.concatenate([yk, np.full((~known).sum(), -1), np.zeros(len(X_shift), int)])
    n_k, n_o = len(yk), int((~known).sum())
    parts = split(yk, rng)
    extra = {"open": np.arange(n_k, n_k + n_o), "shift": np.arange(n_k + n_o, len(yall))}
    sc = build_bundle(Xall, yall, parts, extra, classes)
    print(f"final cascade trained ({time.time() - t0:.0f}s): T1 known macro-F1 "
          f"{MT.macro_prf(sc.y['test'], sc.get('test', 'pred1'), sc.K)['macro_F1']:.3f}, "
          f"T2 {MT.macro_prf(sc.y['test'], sc.get('test', 'pred2'), sc.K)['macro_F1']:.3f}")

    # LOFO bundles: retrain without each known attack family m; family m in D_sel is the pseudo-open set
    lofo = {}
    for m in (1, 2, 3):
        keep = {p: v[yk[v] != m] for p, v in parts.items()}
        lofo[classes[m]] = build_bundle(Xall, np.where(yall == m, -1, yall) - (yall > m),
                                        keep, {"pseudo": parts["sel"][yk[parts["sel"]] == m]},
                                        [c for i, c in enumerate(classes) if i != m])
    print(f"LOFO bundles trained ({time.time() - t0:.0f}s)")

    a = float(np.sqrt(ALPHA))
    w, _ = CA.choose_scod_w(lofo, a, a)
    print(f"\nalpha = {ALPHA}: composite CRC on D_crc (n = {len(sc.u['crc'])}); SCOD w = {w}")
    print(f"{'signal':14s} {'feasible':>8s} {'Esc_open':>9s} {'UAL':>7s} {'missed':>7s} {'alerted':>8s} "
          f"{'FE10k_shift':>11s}")
    for name in SIGNALS:
        k1, k2 = signal_keys(name, w)
        s1, s2 = f"e_{k1}", f"e_{k2}"
        lam, _ = CA.calibrate_composite(sc, s1, s2, ALPHA, "composite_crc")
        feas = lam is not None
        r = MT.route_metrics(sc, s1, s2, (0.0, 0.0), lam if feas else (1.0, 1.0), ALPHA, feas)
        print(f"{name:14s} {str(feas):>8s} {r['Esc_open']:9.3f} {r['UAL']:7.3f} {r['UAL_missed']:7.3f} "
              f"{r['UAL_alerted']:8.3f} {r['FE10k_shift']:11.0f}")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
