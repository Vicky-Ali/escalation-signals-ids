"""escalation: budget-matched escalation signals for a two-tier intrusion-detection cascade.

Implementation of the method of "Which Signal Should Trigger Escalation? A Budget-Matched Evaluation of
Uncertainty-Driven Deferral in Cascaded Intrusion Detection".

Modules
  tier1        XGBoost Tier 1, temperature scaling, leaf-path novelty
  tier2        MLP encoder with supervised contrastive training, linear head, embedding novelty
  novelty      exact q-th nearest-neighbour distances (NumPy and chunked PyTorch)
  signals      the nine escalation signals and the per-run ScoreBundle
  calibration  D_sel ECDF, sequential CRC, jitter, Clopper-Pearson, composite CRC, Learn-then-Test
  cascade      routing, budget-matched calibration, SCOD weight selection, conflict-aware hybrid policy
  metrics      UAL (missed / alerted), HCUL_0.9, escalation rates, FE10k, macro-F1, AUROC
"""
from .signals import SIGNALS, ScoreBundle  # noqa: F401

__version__ = "1.0.0"
