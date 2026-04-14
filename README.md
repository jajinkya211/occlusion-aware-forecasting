# Uncertainty Guided Interaction Networks (UGIN)

**Occlusion-Aware Trajectory Forecasting with Structurally Isolated Uncertainty Channels**

[![Paper](https://img.shields.io/badge/paper-Autonomous%20Intelligent%20Systems-blue)](https://link.springer.com/journal/43684)

---

## Overview

UGIN is a lightweight trajectory forecasting framework for autonomous driving that decomposes predicted uncertainty into structurally isolated **aleatoric** and **epistemic** channels. The epistemic channel is conditioned exclusively on per-agent visibility, guaranteeing *r*(vis, σ²_epi) < 0 by construction. The aleatoric channel is trained by NLL to track prediction error.

**Key results on nuScenes v1.0-mini:**

| Metric | Value | Target |
|--------|-------|--------|
| minADE | 3.214 m | — |
| minFDE | 3.540 m | — |
| r(vis, σ²_epi) | −0.806 | < −0.20 ✓ |
| r(err, σ²_tot) | 0.581 | > +0.30 ✓ |
| ECE | 0.3115 | — |
| CRPS | 1.837 m | — |
| Miss Rate @95% CI | 9.9% | — |
| Parameters | 629K | — |

No baseline model meets either calibration threshold.

---

## Architecture

```
past [B, T_obs, N, 6]
  -> TemporalEncoder (per-agent transformer)
  -> FiLM_1(vis)          pre-social conditioning
  -> SpatialAttn          multi-head social attention (L=2 layers)
  -> FiLM_2(vis)          post-social conditioning
  -> PredHead
       ├── Trajectory MLP         -> K x T_pred x 2 means
       ├── var_ale MLP (h)        -> log σ²_ale   [trained by NLL + err_pearson]
       └── epi MLP (1-vis)        -> log σ²_epi   [trained by vis_pearson only]
              log σ²_total = log σ²_ale + log σ²_epi
```

**Gradient isolation:** NLL and err_pearson update `var_ale` only. `vis_pearson` updates the epi MLP and scalar `w` only. No gradient crosses between branches.

---

## Installation

```bash
pip install -r requirements.txt
```

For nuScenes, download v1.0-mini from [https://www.nuscenes.org/download](https://www.nuscenes.org/download) and set `NUSCENES_DATAROOT` in `src/config.py`.

---

## Usage

### Train

```bash
# nuScenes v1.0-mini (requires download)
python train.py

# Synthetic dataset (no download needed, for quick testing)
python train.py --synthetic

# Override epoch count
python train.py --epochs 150
```

### Evaluate

```bash
python evaluate.py                              # uses best_model.pth
python evaluate.py --checkpoint path/to/ckpt   # custom checkpoint
python evaluate.py --synthetic
```

Produces:
- `paper_results_final.json`
- `uncertainty_calibration.png`
- `per_bin_evaluation.png`

### Ablation Study

```bash
python ablation.py
python ablation.py --epochs 10   # more epochs for tighter estimates
```

---

## Repository Structure

```
.
├── src/
│   ├── config.py       — all hyperparameters and device setup
│   ├── datasets.py     — SyntheticDataset and NuScenesDataset
│   ├── model.py        — OcclusionAwareForecaster + ablation variants
│   └── losses.py       — TotalLoss, vis_pearson_loss, err_pearson_loss
├── train.py            — training loop
├── evaluate.py         — full evaluation suite (Tables 2 & 3, figures)
├── ablation.py         — ablation study
├── requirements.txt
└── README.md
```

---

## Loss Design

| Loss | Acts on | Updates |
|------|---------|---------|
| NLL + Huber + CE | log σ²_ale (winning mode) | var_ale MLP |
| smooth_loss | log σ²_ale | var_ale MLP |
| vis_pearson_loss | log σ²_epi (all modes) | epi MLP + w |
| err_pearson_loss | log σ²_total (all modes) | var_ale MLP |

`vis_pearson_loss` and `err_pearson_loss` are differentiable batch Pearson objectives, not pairwise hinge losses. They have nonzero gradient until *r* = ±1, providing a continuous training signal directly aligned with the evaluation metrics.

---

## Citation

If you use this code, please cite:

```bibtex
@article{ugin2025,
  title   = {Uncertainty Guided Interaction Networks: A Lightweight Approach
             for Occlusion Aware Trajectory Forecasting on Limited Data},
  journal = {Autonomous Intelligent Systems},
  year    = {2025},
}
```

---

## License

MIT
