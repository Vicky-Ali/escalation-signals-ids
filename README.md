# Which Signal Should Trigger Escalation?

Code for the method in **"Which Signal Should Trigger Escalation? A Budget-Matched Evaluation of
Uncertainty-Driven Deferral in Cascaded Intrusion Detection"** (IEEE Transactions on Network and Service
Management, under review).

## Method

A two-tier intrusion-detection cascade: Tier 1 is an XGBoost classifier and Tier 2 is a neural encoder (an MLP
trained with supervised contrastive learning) with a linear head. Either tier can accept a sample or pass it on,
and whatever Tier 2 escalates goes to a human analyst. At each tier we compute **nine escalation signals**:
confidence, class-conditional novelty (q-th nearest neighbour in leaf-path or embedding space), class-agnostic
kNN novelty, the confidence-novelty conflict kappa, Mahalanobis distance, a SCOD-style combination, energy,
predictive entropy and T1/T2 disagreement. The signals are compared **at the same analyst budget**. Each signal
is mapped to [0, 1] by its ECDF on a selection split. Per-tier thresholds are then calibrated on a separate
calibration split with **conformal risk control** (sequential or composite CRC; Learn-then-Test and
Clopper-Pearson variants are included too), so that the fraction of known traffic escalated to the analyst stays
within the budget alpha. The main outcome is the unseen-attack leakage (UAL): the fraction of attacks from a
held-out family that the cascade accepts without escalation. UAL is split into missed attacks (labelled benign)
and alerted ones (labelled as another attack).

## Repository layout

```
escalation/
  tier1.py        XGBoost Tier 1, temperature scaling, leaf-path novelty
  tier2.py        MLP encoder + supervised contrastive loss, linear head, embedding novelty
  novelty.py      exact q-th nearest-neighbour distances (NumPy, and a chunked PyTorch version for large data)
  signals.py      the nine escalation signals; ScoreBundle (per-run scores of all data partitions)
  calibration.py  D_sel ECDF, sequential CRC, jitter, Clopper-Pearson, composite CRC, Learn-then-Test
  cascade.py      routing, budget-matched calibration, SCOD weight by LOFO, conflict-aware hybrid policy
  metrics.py      UAL (missed / alerted), HCUL_0.9, Esc_1/Esc_2/Esc_open, FE10k, macro-F1, AUROC
example.py        end-to-end demo on synthetic data
tests/            pytest: CRC / composite CRC spend, LTT / Clopper-Pearson exceedance, metric identities, signals
```

## Installation

Python 3.12:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is optional. XGBoost and PyTorch run on the CPU when no GPU is present.

## Quick start

```bash
python example.py     # about 20 s on a CPU
pytest                # about 15 s
```

`example.py` trains a small cascade on synthetic data with one unseen attack family. It then computes the nine
signals and calibrates each one at alpha = 0.05 with composite CRC. Expected output on a CPU-only machine. With a
GPU, Tier 2 trains on the GPU and the numbers change slightly (e.g. confidence UAL 0.660 instead of 0.588):

```
alpha = 0.05: composite CRC on D_crc (n = 540); SCOD w = 0.25
signal         feasible  Esc_open     UAL  missed  alerted FE10k_shift
confidence         True     0.044   0.588   0.096    0.491        1540
novelty_cc         True     0.035   0.254   0.036    0.217        2550
knn_agnostic       True     0.034   0.328   0.015    0.312        3250
kappa              True     0.042   0.761   0.168    0.594        2390
mahalanobis        True     0.036   0.361   0.013    0.349        3140
scod               True     0.034   0.388   0.021    0.366        2760
energy             True     0.049   0.494   0.125    0.369        1950
entropy            True     0.044   0.591   0.096    0.495        1540
disagreement       True     0.048   0.486   0.084    0.403        1740
```

## Using your own Tier-1 / Tier-2 scores

You need per-sample logits of both tiers and raw novelty scores; `signals.compute_raw_scores` computes these from
leaf indices and embeddings. Each partition is given as an index array into the rows:

```python
import numpy as np
from escalation.signals import ScoreBundle
from escalation import cascade

# rows = unique samples; u[part] = row index of every sample of a partition, y[part] = its label
u = dict(tr=idx_tr, sel=idx_sel, crc=idx_crc, test=idx_test, open=idx_unseen, shift=idx_benign_shift)
y = {p: labels[v] for p, v in u.items()}
sc = ScoreBundle(["benign", "dos", "scan"], u, y, t1_logits, t2_logits,
                 rawNc1, rawNa1, rawNc2, rawNa2, rawM2)

r = cascade.budget_matched(sc, "confidence", alpha=0.05)        # sequential CRC, alpha_1 = alpha_2 = sqrt(alpha)
print(r["Esc_open"], r["UAL"], r["UAL_missed"], r["UAL_alerted"], r["FE10k_shift"])

lam, info = cascade.calibrate_composite(sc, "e_Na1", "e_Na2", alpha=0.05)   # composite CRC (end-to-end)
```

The `scod` signal needs its weight w, chosen with `cascade.choose_scod_w` on leave-one-family-out bundles
(a `pseudo` partition holds the left-out family). `cascade.conflict_aware` runs the conflict-aware hybrid policy.

## Datasets used in the paper

- **5G-NIDD** (University of Oulu): https://doi.org/10.23729/e80ac9df-d9fb-47e7-8d0d-01384a415361
- **CSE-CIC-IDS2018**, corrected release (Liu et al., IEEE CNS 2022): https://intrusion-detection.distrinet-research.be/CNS2022/
- **WARP Falco alerts**: https://github.com/simabagheri1/Falco-Alerts-Dataset-with-APT-attacks

This repository contains the method only. It includes no data, no dataset preparation code and no scripts for
reproducing the paper's experiments, tables or figures.

## Citation



## Licence

MIT, see [LICENSE](LICENSE).

## Contact

Syed Waqas Ali, waqas.ali@uajk.edu.pk
