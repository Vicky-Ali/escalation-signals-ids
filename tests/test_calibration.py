"""Monte Carlo checks of the calibration guarantees (paper Sec. "Budget-matched escalation and calibration").

Signals are Uniform(0, 1) (the D_sel-ECDF scale of a continuous score), so the true escalation rate of any
threshold is known in closed form: P(s > lam) = 1 - lam, and for independent tiers the end-to-end rate is
(1 - lam1)(1 - lam2). Calibration and test data are exchangeable by construction.
"""
import numpy as np

from escalation import calibration as CB

N_REP = 400
N_CRC = 400
ALPHA = 0.05


def _true_rate(l1, l2):
    return (1.0 - l1) * (1.0 - l2)


def test_crc_lambda_mean_spend_le_alpha():
    rng = np.random.default_rng(1)
    risks = []
    for _ in range(N_REP):
        s = rng.uniform(size=N_CRC)
        lam = CB.crc_lambda(np.ones(N_CRC), s, 0.0, ALPHA)
        risks.append(1.0 - lam)
    se = np.std(risks) / np.sqrt(N_REP)
    assert np.mean(risks) <= ALPHA + 3 * se


def test_sequential_crc_mean_spend_le_alpha():
    rng = np.random.default_rng(2)
    a = np.sqrt(ALPHA)
    r1, r = [], []
    for _ in range(N_REP):
        s1, s2 = rng.uniform(size=N_CRC), rng.uniform(size=N_CRC)
        l1, l2 = CB.sequential_crc(np.ones(N_CRC), s1, np.ones(N_CRC), s2, 0.0, 0.0, a, a)
        assert l1 is not None
        r1.append(1.0 - l1)
        r.append(_true_rate(l1, l2))
    assert np.mean(r1) <= a + 3 * np.std(r1) / np.sqrt(N_REP)
    assert np.mean(r) <= ALPHA + 3 * np.std(r) / np.sqrt(N_REP)


def test_composite_crc_mean_spend_le_alpha():
    rng = np.random.default_rng(3)
    risks, n_inf = [], 0
    for _ in range(N_REP // 2):
        a_sel, b_sel = rng.uniform(size=N_CRC), rng.uniform(size=N_CRC)
        a_crc, b_crc = rng.uniform(size=N_CRC), rng.uniform(size=N_CRC)
        l1, l2 = CB.make_path(a_sel, b_sel, CB.MAIN_EXP)
        k = CB.end_to_end_counts(a_crc, b_crc, l1, l2)
        assert np.all(np.diff(k) >= 0)                     # loss monotone along the path
        j = CB.composite_crc(k, N_CRC, ALPHA)
        if j is None:
            n_inf += 1
            risks.append(0.0)                              # infeasible -> nothing escalated
            continue
        risks.append(_true_rate(l1[j], l2[j]))
    se = np.std(risks) / np.sqrt(len(risks))
    assert n_inf == 0
    assert np.mean(risks) <= ALPHA + 3 * se


def test_ltt_exceedance_le_delta():
    rng = np.random.default_rng(4)
    delta = CB.LTT_DELTA
    exceed = []
    for _ in range(N_REP // 2):
        a_sel, b_sel = rng.uniform(size=N_CRC), rng.uniform(size=N_CRC)
        a_crc, b_crc = rng.uniform(size=N_CRC), rng.uniform(size=N_CRC)
        l1, l2 = CB.make_path(a_sel, b_sel, CB.MAIN_EXP)
        k = CB.end_to_end_counts(a_crc, b_crc, l1, l2)
        j = CB.ltt_fixed_sequence(k, N_CRC, ALPHA, delta)
        exceed.append(False if j is None else _true_rate(l1[j], l2[j]) > ALPHA)
    p = np.mean(exceed)
    assert p <= delta + 3 * np.sqrt(delta * (1 - delta) / len(exceed))


def test_clopper_pearson_exceedance_le_delta():
    rng = np.random.default_rng(5)
    delta = CB.CP_DELTA
    exceed = []
    for _ in range(N_REP):
        lam = CB.cp_lambda(rng.uniform(size=N_CRC), ALPHA, delta)
        exceed.append(False if lam is None else (1.0 - lam) > ALPHA)
    p = np.mean(exceed)
    assert p <= delta + 3 * np.sqrt(delta * (1 - delta) / N_REP)


def test_crc_infeasible_when_budget_too_small():
    # n/(n+1) R + 1/(n+1) <= alpha is impossible when 1/(n+1) > alpha
    assert CB.crc_lambda(np.ones(10), np.linspace(0, 1, 10), 0.0, 0.05) is None


def test_class_ecdf_basic():
    e = CB.ClassECDF(np.array([0.1, 0.2, 0.3, 0.4]))
    out = e.transform(np.array([0.0, 0.25, 0.4, 9.0, np.nan]))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0, 1.0, 1.0])
    ec = CB.ClassECDF(np.array([0.1, 0.2, 5.0]), pred=np.array([0, 0, 1]))
    np.testing.assert_allclose(ec.transform(np.array([0.15, 0.15]), np.array([0, 1])), [0.5, 0.0])


def test_jitter_preserves_order_and_range():
    rng = np.random.default_rng(6)
    raw = rng.integers(0, 5, 300).astype(float)
    j = CB.jitter_signal(raw, np.arange(100), rng)
    assert np.all((j >= 0) & (j <= 1))
    assert np.all(j[raw == 4].min() > j[raw == 3].max())
