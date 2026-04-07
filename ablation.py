import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from losses import TotalLoss
from model import OcclusionAwareForecaster, NoFiLMForecaster
from train import clear_device_cache, pearson_r


def evaluate_calibration(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Tuple[float, float, float]:
    model.eval()
    errors_list:       List[np.ndarray] = []
    variances_list:    List[np.ndarray] = []
    visibilities_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            past   = batch['past_trajectory'].to(device)
            future = batch['future_trajectory'].to(device)
            vis    = batch['visibility_mask'].to(device)
            mask   = batch['agent_mask'].to(device)

            out = model(past, vis, mask)

            traj_cpu   = out['trajectories'].cpu()
            logvar_cpu = out['log_variances'].cpu()
            fut_cpu    = future.cpu()
            vis_cpu    = vis.cpu()
            mask_cpu   = mask.cpu()

            gt_expanded = fut_cpu.permute(0, 2, 1, 3).unsqueeze(2).expand_as(traj_cpu)
            per_agent_error    = (traj_cpu - gt_expanded).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
            per_agent_variance = logvar_cpu.clamp(-4.0, 1.0).exp().mean(dim=[2, 3, 4])

            valid = mask_cpu
            errors_list.append(per_agent_error[valid].numpy())
            variances_list.append(per_agent_variance[valid].numpy())
            visibilities_list.append(vis_cpu.squeeze(-1)[valid].numpy())

    errors_all       = np.concatenate(errors_list)
    variances_all    = np.concatenate(variances_list)
    visibilities_all = np.concatenate(visibilities_list)

    return (
        float(errors_all.mean() * cfg.pos_scale),
        pearson_r(errors_all, variances_all),
        pearson_r(visibilities_all, variances_all),
    )


def run_ablation_variant(
    model_class,
    loss_fn: TotalLoss,
    label: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    num_epochs: int = 5,
) -> Tuple[float, float, float]:
    mdl = model_class(
        hidden_dim=cfg.hidden_dim,
        pred_len=cfg.pred_len,
        num_modes=cfg.num_modes,
        num_heads=cfg.num_heads,
        num_temporal_layers=cfg.num_temporal_layers,
        num_spatial_layers=cfg.num_spatial_layers,
        film_embed_dim=cfg.film_embed_dim,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(mdl.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    t_start = time.time()
    mdl.train()
    for _ in range(num_epochs):
        for batch in train_loader:
            past   = batch['past_trajectory'].to(device)
            future = batch['future_trajectory'].to(device)
            vis    = batch['visibility_mask'].to(device)
            mask   = batch['agent_mask'].to(device)
            opt.zero_grad(set_to_none=True)
            preds = mdl(past, vis, mask)
            losses = loss_fn(
                preds,
                {'future_trajectory': future, 'agent_mask': mask, 'visibility_mask': vis},
            )
            if torch.isfinite(losses['total']):
                losses['total'].backward()
                opt.step()

    result = evaluate_calibration(mdl, val_loader, device, cfg)
    elapsed = time.time() - t_start

    r_err_ok = result[1] > cfg.target_r_err_sigma2
    r_vis_ok = result[2] < cfg.target_r_vis_sigma2
    print(
        f'{label:<26}: '
        f'minADE={result[0]:.2f}m  '
        f'r(err,sigma2)={"OK " if r_err_ok else "   "}{result[1]:+.3f}  '
        f'r(vis,sigma2)={"OK " if r_vis_ok else "   "}{result[2]:+.3f}  '
        f'({elapsed:.0f}s)'
    )

    del mdl, opt
    clear_device_cache(device)
    return result


def run_all_ablations(
    full_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Dict[str, Tuple[float, float, float]]:
    print('Running ablation study (5 epochs per variant)...')
    print()

    ablation_results: Dict[str, Tuple[float, float, float]] = {}

    full_result = evaluate_calibration(full_model, val_loader, device, cfg)
    ablation_results['Full model (ours)'] = full_result
    print(
        f'{"Full model (ours)":<26}: '
        f'minADE={full_result[0]:.2f}m  '
        f'r(err,sigma2)={full_result[1]:+.3f}  '
        f'r(vis,sigma2)={full_result[2]:+.3f}  (from checkpoint)'
    )

    ablation_results['No FiLM conditioning'] = run_ablation_variant(
        NoFiLMForecaster,
        TotalLoss(
            lambda_vis_calib=cfg.lambda_vis_calib,
            lambda_err_calib=cfg.lambda_err_calib,
            lambda_regression=cfg.lambda_regression,
        ),
        'No FiLM conditioning',
        train_loader, val_loader, device, cfg,
    )

    ablation_results['No vis calibration loss'] = run_ablation_variant(
        OcclusionAwareForecaster,
        TotalLoss(
            lambda_vis_calib=0.0,
            lambda_err_calib=cfg.lambda_err_calib,
            lambda_regression=cfg.lambda_regression,
        ),
        'No vis calibration loss',
        train_loader, val_loader, device, cfg,
    )

    ablation_results['No err calibration loss'] = run_ablation_variant(
        OcclusionAwareForecaster,
        TotalLoss(
            lambda_vis_calib=cfg.lambda_vis_calib,
            lambda_err_calib=0.0,
            lambda_regression=cfg.lambda_regression,
        ),
        'No err calibration loss',
        train_loader, val_loader, device, cfg,
    )

    print()
    print('=' * 72)
    print('  ABLATION TABLE  (5-epoch training except Full model)')
    print('=' * 72)
    print(f'  {"Variant":<26}  {"minADE (m)":>10}  {"r(err,s2)":>10}  {"r(vis,s2)":>10}')
    print('  ' + '─' * 60)
    for name, (ade, r_err, r_vis) in ablation_results.items():
        r_err_ok = r_err > cfg.target_r_err_sigma2
        r_vis_ok = r_vis < cfg.target_r_vis_sigma2
        tag = '  <- ours' if 'ours' in name else ''
        print(
            f'  {name:<26}  {ade:>10.2f}  '
            f'{"OK " if r_err_ok else "   "}{r_err:>7.3f}  '
            f'{"OK " if r_vis_ok else "   "}{r_vis:>7.3f}{tag}'
        )
    print('=' * 72)

    return ablation_results
