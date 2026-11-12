"""Each of the nine escalation signals on a toy input (paper Sec. "Escalation signals") and the novelty kernels."""
import numpy as np
import pytest
from scipy.special import log_softmax, logsumexp

from escalation import novelty as NV
from escalation.signals import SIGNALS, MahalanobisOOD, ScoreBundle, compute_raw_scores, signal_keys


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    K, n = 3, 600
    y = rng.integers(0, K, n)
    emb = np.eye(K)[y] * 2 + rng.normal(0, 0.5, (n, K))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    leaves = np.stack([y * 2 + rng.integers(0, 2, n) for _ in range(20)], 1)
    lg1 = 3 * np.eye(K)[y] + rng.normal(0, 1.0, (n, K))
    lg2 = 3 * np.eye(K)[y] + rng.normal(0, 1.0, (n, K))
    idx = rng.permutation(n)
    u = dict(tr=idx[:300], sel=idx[300:380], crc=idx[380:460], test=idx[460:540], open=idx[540:], shift=idx[540:])
    raw = compute_raw_scores(leaves, emb, lg1, lg2, u["tr"], y[u["tr"]], idx[300:])
    return ScoreBundle(["benign", "a", "b"], u, {p: y[v] for p, v in u.items()}, lg1, lg2, **raw), lg1, lg2


def test_nine_signals_present_and_bounded(toy):
    sc, _, _ = toy
    assert list(SIGNALS) == ["confidence", "novelty_cc", "knn_agnostic", "kappa", "mahalanobis", "scod",
                             "energy", "entropy", "disagreement"]
    for name in SIGNALS:
        for key in signal_keys(name, 0.5):
            v = sc.get("crc", f"e_{key}")
            assert np.all((v >= 0) & (v <= 1)), (name, key)
            assert np.all(np.isfinite(sc.get("test", key))), (name, key)


def test_signal_definitions(toy):
    sc, lg1, lg2 = toy
    U = sc.U
    np.testing.assert_allclose(U["k1"], U["C1"] * U["Nc1"])
    np.testing.assert_allclose(U["conf2"], 1 - U["C2"])
    np.testing.assert_allclose(U["scod1_0.25"], 0.25 * U["conf1"] + 0.75 * U["Na1"])
    T = sc.temps[2]
    z = lg2 / T
    np.testing.assert_allclose(U["energy2"], -T * logsumexp(z, axis=1), rtol=1e-10)
    lp = log_softmax(z, axis=1)
    np.testing.assert_allclose(U["entropy2"], -(np.exp(lp) * lp).sum(1), rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(U["dis2"], (U["pred1"] != U["pred2"]) + 0.5 * U["conf2"])
    assert signal_keys("mahalanobis")[0] == "conf1" and signal_keys("disagreement")[0] == "conf1"


def test_novelty_numpy_equals_torch():
    rng = np.random.default_rng(1)
    R = rng.integers(0, 4, (200, 30))
    Qm = rng.integers(0, 4, (50, 30))
    cnt = rng.integers(1, 3, 200)
    a = NV.agnostic_novelty(Qm, R, cnt, 10, "hamming")
    b = NV.qnn(Qm, R, cnt, 10, "hamming")
    np.testing.assert_allclose(a, b, atol=1e-6)
    E, F = rng.normal(size=(200, 8)), rng.normal(size=(50, 8))
    np.testing.assert_allclose(NV.agnostic_novelty(F, E, cnt, 10, "euclidean"),
                               NV.qnn(F, E, cnt, 10, "euclidean"), atol=1e-4)


def test_qth_nn_with_multiplicities_and_missing_class():
    D = np.array([[0.1, 0.2, 0.3]])
    assert NV.qth_nn_distance(D, np.array([1, 1, 1]), 2)[0] == 0.2
    assert NV.qth_nn_distance(D, np.array([2, 1, 1]), 2)[0] == 0.1
    out = NV.class_conditional_novelty(np.zeros((2, 3)), np.array([0, 1]), np.zeros((4, 3)), np.array([0, 0, 0, 0]),
                                       np.ones(4, int), 2, "euclidean")
    assert out[0] == 0.0 and np.isnan(out[1])


def test_mahalanobis_min_class_distance():
    rng = np.random.default_rng(2)
    X = np.r_[rng.normal(0, 1, (200, 2)), rng.normal(6, 1, (200, 2))]
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    m = MahalanobisOOD().fit(X, y)
    s = m.score(np.array([[0.0, 0.0], [6.0, 6.0], [3.0, -10.0]]))
    assert s[2] > s[0] and s[2] > s[1]
