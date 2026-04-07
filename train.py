import gc
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from losses import TotalLoss


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        try:
            getattr(torch.mps, 'empty_cache', lambda: None)()
        except Exception:
            pass


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


@torch.no_grad()
def compute_min_ade_fde(
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
) -> Tuple[float, float]:
    from losses import masked_mean
    gt = ground_truth.permute(0, 2, 1, 3).unsqueeze(2).expand_as(trajectories)
    l2 = (trajectories - gt).pow(2).sum(-1).sqrt()
    min_ade = masked_mean(l2.mean(-1).min(-1).values, agent_mask).item()
    min_fde = masked_mean(l2[:, :, :, -1].min(-1).values, agent_mask).item()
    return min_ade, min_fde


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: TotalLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: Config,
    is_training: bool,
) -> Dict[str, float]:
    model.train() if is_training else model.eval()

    loss_accumulator = {
        k: 0.0 for k in
        ['total', 'nll', 'regression', 'cls', 'vis_calib', 'err_calib', 'smooth']
    }
    sum_ade = sum_fde = num_valid_batches = num_skipped = 0

    all_errors:       List[np.ndarray] = []
    all_variances:    List[np.ndarray] = []
    all_visibilities: List[np.ndarray] = []

    ctx = torch.enable_grad() if is_training else torch.no_grad()
    with ctx:
        for batch in loader:
            past   = batch['past_trajectory'].to(device)
            future = batch['future_trajectory'].to(device)
            vis    = batch['visibility_mask'].to(device)
            mask   = batch['agent_mask'].to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(past, vis, mask)
            losses = loss_fn(
                predictions,
                {'future_trajectory': future, 'agent_mask': mask, 'visibility_mask': vis},
            )

            if not torch.isfinite(losses['total']):
                num_skipped += 1
                continue

            if is_training:
                losses['total'].backward()
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(
                            param.grad, nan=0.0, posinf=0.0, neginf=0.0
                        )
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            with torch.no_grad():
                traj_cpu   = predictions['trajectories'].cpu()
                logvar_cpu = predictions['log_variances'].cpu()
                fut_cpu    = future.cpu()
                vis_cpu    = vis.cpu()
                mask_cpu   = mask.cpu()

                min_ade, min_fde = compute_min_ade_fde(traj_cpu, fut_cpu, mask_cpu)

                gt_expanded = fut_cpu.permute(0, 2, 1, 3).unsqueeze(2).expand_as(traj_cpu)
                per_agent_error    = (traj_cpu - gt_expanded).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
                per_agent_variance = logvar_cpu.clamp(-4.0, 1.0).exp().mean(dim=[2, 3, 4])
                per_agent_vis      = vis_cpu.squeeze(-1)

                valid = mask_cpu
                all_errors.append(per_agent_error[valid].numpy())
                all_variances.append(per_agent_variance[valid].numpy())
                all_visibilities.append(per_agent_vis[valid].numpy())

            if np.isfinite(min_ade) and np.isfinite(min_fde):
                for key in loss_accumulator:
                    loss_accumulator[key] += losses[key].item()
                sum_ade += min_ade
                sum_fde += min_fde
                num_valid_batches += 1

    if num_valid_batches == 0:
        nan_keys = list(loss_accumulator.keys()) + [
            'min_ade', 'min_fde', 'r_vis_sigma2', 'r_err_sigma2', 'skipped_batches'
        ]
        return {k: float('nan') for k in nan_keys}

    errors_all       = np.concatenate(all_errors)
    variances_all    = np.concatenate(all_variances)
    visibilities_all = np.concatenate(all_visibilities)

    results = {k: v / num_valid_batches for k, v in loss_accumulator.items()}
    results['min_ade']         = sum_ade / num_valid_batches
    results['min_fde']         = sum_fde / num_valid_batches
    results['r_vis_sigma2']    = pearson_r(visibilities_all, variances_all)
    results['r_err_sigma2']    = pearson_r(errors_all, variances_all)
    results['skipped_batches'] = num_skipped
    return results


def build_lr_schedule(cfg: Config):
    def lr_schedule(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
        return max(cfg.min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return lr_schedule


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> List[Dict]:
    loss_fn = TotalLoss(
        lambda_vis_calib=cfg.lambda_vis_calib,
        lambda_err_calib=cfg.lambda_err_calib,
        lambda_regression=cfg.lambda_regression,
        lambda_smooth=cfg.lambda_smooth,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_schedule(cfg))

    history: List[Dict] = []
    best_composite_score = float('inf')
    consecutive_nan_epochs = 0

    header = (
        f"{'Ep':>4}  {'Loss':>8}  {'minADE(m)':>9}  "
        f"{'r(err,s2)':>9}  {'r(vis,s2)':>9}  {'LR':>8}"
    )
    print(header)
    print('─' * len(header))

    for epoch in range(cfg.epochs):
        t_start = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, cfg, True)
        val_metrics   = run_epoch(model, val_loader,   loss_fn, None,      device, cfg, False)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        record = {'epoch': epoch + 1}
        record.update({f'train_{k}': v for k, v in train_metrics.items()})
        record.update({f'val_{k}':   v for k, v in val_metrics.items()})
        history.append(record)

        if not np.isfinite(val_metrics.get('total', float('nan'))):
            consecutive_nan_epochs += 1
            print(f'{epoch + 1:4d}  [NaN]')
            if consecutive_nan_epochs >= 5:
                print('5 consecutive NaN epochs. Stopping.')
                break
            continue
        consecutive_nan_epochs = 0

        r_err_sigma2 = val_metrics['r_err_sigma2']
        r_vis_sigma2 = val_metrics['r_vis_sigma2']
        r_err_ok     = r_err_sigma2 > cfg.target_r_err_sigma2
        r_vis_ok     = r_vis_sigma2 < cfg.target_r_vis_sigma2

        print(
            f"{epoch + 1:4d}  "
            f"{val_metrics['total']:8.4f}  "
            f"{val_metrics['min_ade'] * cfg.pos_scale:9.2f}  "
            f"{'OK' if r_err_ok else '  '}{r_err_sigma2:7.3f}  "
            f"{'OK' if r_vis_ok else '  '}{r_vis_sigma2:7.3f}  "
            f"{current_lr:.2e}  "
            f"({time.time() - t_start:.1f}s)"
        )

        calibration_penalty = (0.0 if r_err_ok else 5.0) + (0.0 if r_vis_ok else 5.0)
        composite_score = val_metrics['min_ade'] + calibration_penalty

        if np.isfinite(composite_score) and composite_score < best_composite_score:
            best_composite_score = composite_score
            torch.save(
                {
                    'epoch':        epoch + 1,
                    'state_dict':   model.state_dict(),
                    'min_ade':      val_metrics['min_ade'],
                    'min_fde':      val_metrics['min_fde'],
                    'r_err_sigma2': r_err_sigma2,
                    'r_vis_sigma2': r_vis_sigma2,
                    'config':       asdict(cfg),
                },
                cfg.checkpoint_path,
            )
            print(
                f"       Checkpoint saved — "
                f"minADE={val_metrics['min_ade'] * cfg.pos_scale:.2f}m  "
                f"r(err,sigma2)={r_err_sigma2:.3f}  "
                f"r(vis,sigma2)={r_vis_sigma2:.3f}"
            )

        if (
            r_err_ok
            and r_vis_ok
            and epoch > cfg.early_stop_min_epoch
            and val_metrics['min_ade'] * cfg.pos_scale < cfg.target_min_ade_m
        ):
            print(f'Both calibration targets met at epoch {epoch + 1}. Stopping.')
            break

    print(f'\nBest composite score: {best_composite_score:.4f}')
    return history
