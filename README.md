# Uncertainty-Guided Occlusion-Aware Trajectory Forecasting

Official implementation of the paper:

> **Uncertainty-Guided Occlusion-Aware Trajectory Forecasting**  
> Temporal Transformer + Spatial Cross-Attention with dual FiLM visibility conditioning.

---

## Core Idea

Most trajectory forecasters treat all agents identically regardless of how well the sensor observed them. This is wrong. A cyclist half-hidden behind a bus should receive a wider uncertainty cone than a fully visible pedestrian,  not because the model is less capable, but because the information was genuinely absent.

This work wires per-agent visibility scores directly into the model's internal representations using Feature-wise Linear Modulation (FiLM), injected at two points: before and after social attention. Two pairwise calibration losses then enforce the desired correlations during training:

- `r(vis, sigma2)`,  less-visible agents must receive higher predicted variance
- `r(err, sigma2)`,  higher-error agents must receive higher predicted variance

Neither property emerges from standard NLL training alone.

---

## Results on nuScenes v1.0-mini

| Metric | Value | Target | Status |
|---|---|---|---|
| minADE | 3.81 m |,  |,  |
| minFDE | 5.23 m |,  |,  |
| r(vis, sigma2) | −0.211 | < −0.20 | PASS |
| r(err, sigma2) | +0.352 | > +0.30 | PASS |
| ECE | 0.315 |,  |,  |
| CRPS | 2.00 m |,  |,  |
| Miss Rate @95% CI | 28.4% |,  |,  |

---

## Architecture

```
past_trajectory
    #  TemporalEncoder          per-agent, batched
    #  VisibilityFiLM (pre)     conditions embeddings on visibility before social attention
    #  SpatialAttention x 2     models agent interactions
    #  VisibilityFiLM (post)    re-conditions after social aggregation
    #  MultiModalPredictionHead K=6 modes, aleatoric + epistemic variance
```

616K trainable parameters. Trains on a single GPU in ~3–4 hours for 300 epochs.

---

## Repository Structure

```
.
├── config.py        All hyperparameters in one dataclass
├── datasets.py      SyntheticDataset and NuScenesDataset
├── model.py         OcclusionAwareForecaster, FiLM, TemporalEncoder, SpatialAttention
├── losses.py        WTA-NLL, visibility calibration loss, error calibration loss
├── train.py         Training loop, LR schedule, checkpointing
├── evaluate.py      Full metric suite: ECE, CRPS, Brier, Miss Rate
├── ablation.py      Ablation study runner
├── visualise.py     All figures (training curves, trajectories, calibration scatter)
├── main.py          Entry point,  runs everything end to end
└── requirements.txt
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/<jajinkya211>/occlusion-aware-forecasting.git
cd occlusion-aware-forecasting
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download nuScenes mini**

Download `v1.0-mini` from [https://www.nuscenes.org/download](https://www.nuscenes.org/download) and extract it. Then update `nuscenes_dataroot` in `config.py`:

```python
nuscenes_dataroot: str = '/path/to/your/v1.0-mini'
```

---

## Running

**Full pipeline (train + evaluate + ablation + all figures)**

```bash
python main.py
```

**Synthetic data only (no nuScenes required)**

In `config.py`, set:

```python
use_synthetic: bool = True
```

Then run:

```bash
python main.py
```

**Outputs**

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint (by composite score) |
| `paper_results.json` | All metrics and ablation results |
| `training_curves.png` | Loss, minADE, r(err,sigma2), r(vis,sigma2) over epochs |
| `trajectory_predictions.png` | Qualitative predictions with uncertainty cones |
| `uncertainty_calibration.png` | Calibration scatter plots |
| `per_bin_evaluation.png` | Per-visibility-bin breakdown |

---

## Configuration

All hyperparameters live in `config.py` as a single `Config` dataclass. Key fields:

| Parameter | Default | Description |
|---|---|---|
| `obs_len` | 4 | Observation timesteps (2 s at 2 Hz) |
| `pred_len` | 6 | Prediction timesteps (3 s at 2 Hz) |
| `hidden_dim` | 128 | Transformer hidden dimension |
| `num_modes` | 6 | GMM components |
| `lambda_vis_calib` | 0.6 | Weight for visibility calibration loss |
| `lambda_err_calib` | 0.3 | Weight for error calibration loss |
| `lambda_regression` | 1.5 | Weight for Huber regression term |
| `target_r_err_sigma2` | 0.30 | r(err, sigma2) early-stop threshold |
| `target_r_vis_sigma2` | −0.20 | r(vis, sigma2) early-stop threshold |

---

## Ablation

The ablation study trains three reduced variants for 5 epochs each and compares calibration:

| Variant | minADE (m) | r(err, s2) | r(vis, s2) |
|---|---|---|---|
| Full model (ours) | **3.81** | **+0.352** | **−0.211** |
| No FiLM conditioning | ~7.5 | −0.34 | ~0.00 |
| No vis calibration loss | ~4.x | +0.xx | −0.99 |
| No err calibration loss | ~4.x | −0.34 | −0.2x |

Fill in exact numbers from `paper_results.json` after running.

---

## Notation

Throughout this codebase, two calibration metrics are used consistently:

- **`r(vis, sigma2)`**,  Pearson correlation between per-agent visibility score and mean predicted variance. A negative value means the model assigns higher uncertainty to less-visible agents.
- **`r(err, sigma2)`**,  Pearson correlation between per-agent prediction error (minADE) and mean predicted variance. A positive value means the model is more uncertain about agents it predicts poorly.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{occlusion_forecasting_2025,
  title     = {Uncertainty-Guided Occlusion-Aware Trajectory Forecasting},
  author    = {[Authors]},
  booktitle = {[Venue]},
  year      = {2025}
}
```

---

## License

MIT License. See `LICENSE` for details.
