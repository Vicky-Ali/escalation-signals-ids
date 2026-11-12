"""Metric identities (paper Sec. "Metrics"): UAL = missed + alerted, HCUL <= UAL, Esc_open = Esc_1 * Esc_2."""
import numpy as np

from escalation import metrics as MT
from escalation.cascade import cascade_route


def test_ual_equals_missed_plus_alerted():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 500
        tier = rng.integers(1, 4, n)
        p1, p2 = rng.integers(0, 4, n), rng.integers(0, 4, n)
        c1, c2 = rng.uniform(size=n), rng.uniform(size=n)
        m = MT.open_set_metrics(tier, p1, p2, c1, c2, benign=0)
        assert abs(m["UAL"] - (m["UAL_missed"] + m["UAL_alerted"])) < 1e-12
        assert m["HCUL"] <= m["UAL"] + 1e-12
        assert abs(m["UAL"] - np.mean(tier <= 2)) < 1e-12


def test_escalation_rates():
    tier = np.array([1, 1, 2, 3, 3, 2, 1, 3])
    e = MT.escalation_metrics(tier)
    assert e["Esc_1"] == 5 / 8 and e["Esc_2"] == 3 / 5 and e["Esc_open"] == 3 / 8
    assert abs(e["Esc_1"] * e["Esc_2"] - e["Esc_open"]) < 1e-12


def test_routing_rule():
    C1 = np.array([0.95, 0.4, 0.95, 0.95])
    s1 = np.array([0.1, 0.1, 0.9, 0.9])
    C2 = np.array([0.9, 0.9, 0.9, 0.9])
    s2 = np.array([0.0, 0.0, 0.1, 0.9])
    np.testing.assert_array_equal(cascade_route(C1, s1, C2, s2, 0.5, 0.0, 0.5, 0.5), [1, 2, 2, 3])


def test_fe10k_macro_f1_auroc():
    assert MT.false_escalations_per_10k(np.array([1, 3, 2, 1])) == 2500.0
    assert MT.macro_prf(np.array([0, 1, 1]), np.array([0, 1, 1]), 2)["macro_F1"] == 1.0
    assert MT.open_vs_known_auroc(np.array([0.1, 0.2]), np.array([0.8, 0.9])) == 1.0
