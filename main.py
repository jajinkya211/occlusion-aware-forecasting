import gc
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from config import Config
from datasets import SyntheticDataset, NuScenesDataset, get_nuscenes_scene_splits
from model import OcclusionAwareForecaster
from train import train
from evaluate import evaluate, print_results, save_results
from ablation import run_all_ablations
from visualise import (
    plot_training_curves,
    plot_trajectory_predictions,
    plot_uncertainty_calibration,
    plot_per_bin_evaluation,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: Config, device: torch.device):
    if cfg.use_synthetic:
        print('Dataset: synthetic')
        full_dataset = SyntheticDataset(
            num_scenes=1500,
            obs_len=cfg.obs_len,
            pred_len=cfg.pred_len,
            max_agents=cfg.max_agents,
            seed=cfg.seed,
        )
        num_val   = int(cfg.val_ratio * len(full_dataset))
        num_train = len(full_dataset) - num_val
        train_dataset, val_dataset = random_split(
            full_dataset,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    else:
        print('Dataset: nuScenes v1.0-mini')
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-mini', dataroot=cfg.nuscenes_dataroot, verbose=False)
        train_tokens, val_tokens = get_nuscenes_scene_splits(nusc, cfg.val_ratio, cfg.seed)
        train_dataset = NuScenesDataset(nusc, train_tokens, cfg.obs_len, cfg.pred_len, cfg.max_agents)
        val_dataset   = NuScenesDataset(nusc, val_tokens,   cfg.obs_len, cfg.pred_len, cfg.max_agents)

    pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, drop_last=True, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin,
    )

    print(f'Train: {len(train_dataset)}  Val: {len(val_dataset)}  Batches/epoch: {len(train_loader)}')
    return train_loader, val_loader


def main():
    cfg    = Config()
    device = get_device()
    set_seeds(cfg.seed)

    print(f'Device: {device}')

    train_loader, val_loader = build_dataloaders(cfg, device)

    model = OcclusionAwareForecaster(
        hidden_dim=cfg.hidden_dim,
        pred_len=cfg.pred_len,
        num_modes=cfg.num_modes,
        num_heads=cfg.num_heads,
        num_temporal_layers=cfg.num_temporal_layers,
        num_spatial_layers=cfg.num_spatial_layers,
        film_embed_dim=cfg.film_embed_dim,
        dropout=cfg.dropout,
    ).to(device)

    print(f'Parameters: {model.count_parameters():,}')

    history = train(model, train_loader, val_loader, cfg, device)

    plot_training_curves(history, cfg)

    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    plot_trajectory_predictions(model, val_loader, cfg, device)

    eval_results = evaluate(model, val_loader, cfg, device)
    print_results(eval_results, cfg, checkpoint)
    plot_uncertainty_calibration(eval_results, cfg)
    plot_per_bin_evaluation(eval_results, cfg)

    ablation_results = run_all_ablations(model, train_loader, val_loader, device, cfg)

    save_results(eval_results, ablation_results, checkpoint)


if __name__ == '__main__':
    main()
