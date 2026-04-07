from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    use_synthetic: bool = False
    nuscenes_dataroot: str = '/data/nuscenes/v1.0-mini'
    obs_len: int = 4
    pred_len: int = 6
    max_agents: int = 16
    pos_scale: float = 10.0
    vel_scale: float = 5.0
    val_ratio: float = 0.2

    hidden_dim: int = 128
    num_modes: int = 6
    num_heads: int = 4
    num_temporal_layers: int = 2
    num_spatial_layers: int = 2
    dropout: float = 0.1
    film_embed_dim: int = 32

    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 5
    grad_clip_norm: float = 0.5
    min_lr_ratio: float = 0.05
    seed: int = 42

    lambda_regression: float = 1.5
    lambda_vis_calib: float = 0.6
    lambda_err_calib: float = 0.3
    lambda_smooth: float = 0.005

    target_r_err_sigma2: float = 0.30
    target_r_vis_sigma2: float = -0.20
    target_min_ade_m: float = 6.0
    early_stop_min_epoch: int = 30

    checkpoint_path: str = 'best_model.pth'

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path) as f:
            return cls(**json.load(f))
