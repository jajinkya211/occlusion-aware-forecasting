import math
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from train import pearson_r


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> Dict:
    model.eval()

    all_errors:       List[np.ndarray] = []
    all_variances:    List[np.ndarray] = []
    all_visibilities: List[np.ndarray] = []
    all_fde:          List[np.ndarray] = []
    all_best_mu:      List[np.ndarray] = []
    all_ground_truth: List[np.ndarray] = []
    all_best_logvar:  List[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            past   = batch['past_trajectory'].to(device)
            future = batch['future_trajectory'].to(device)
            vis    = batch['visibility_mask'].to(device)
            mask   = batch['agent_mask'].to(device)

            predictions = model(past, vis, mask)

            traj_cpu   = predictions['trajectories'].cpu()
            logvar_cpu = predictions['log_variances'].cpu()
            fut_cpu    = future.cpu()
            vis_cpu    = vis.cpu()
            mask_cpu   = mask.cpu()

            gt = fut_cpu.permute(0, 2, 1, 3)
            gt_expanded = gt.unsqueeze(2).expand_as(traj_cpu)

            per_agent_error    = (traj_cpu - gt_expanded).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
            per_agent_variance = logvar_cpu.clamp(-4.0, 1.0).exp().mean(dim=[2, 3, 4])
            per_agent_fde      = (traj_cpu - gt_expanded).pow(2).sum(-1).sqrt()[:, :, :, -1].min(-1).values

            ade_per_mode = (traj_cpu - gt_expanded).pow(2).sum(-1).sqrt().mean(-1)
            winner = ade_per_mode.argmin(dim=-1)
            B, N = winner.shape
            idx = winner.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(B, N, 1, cfg.pred_len, 2)
            best_mu     = traj_cpu.gather(2, idx).squeeze(2)
            best_logvar = logvar_cpu.gather(2, idx).squeeze(2)

            valid = mask_cpu
            all_errors.append(per_agent_error[valid].numpy())
            all_variances.append(per_agent_variance[valid].numpy())
            all_visibilities.append(vis_cpu.squeeze(-1)[valid].numpy())
            all_fde.append(per_agent_fde[valid].numpy())
            all_best_mu.append(best_mu[valid].numpy())
            all_ground_truth.append(gt[valid].numpy())
            all_best_logvar.append(best_logvar[valid].numpy())

    errors_all       = np.concatenate(all_errors)
    variances_all    = np.concatenate(all_variances)
    visibilities_all = np.concatenate(all_visibilities)
    fde_all          = np.concatenate(all_fde)
    mu_all           = np.concatenate(all_best_mu)
    gt_all           = np.concatenate(all_ground_truth)
    logvar_all       = np.concatenate(all_best_logvar)

    min_ade_m = float(errors_all.mean() * cfg.pos_scale)
    min_fde_m = float(fde_all.mean()    * cfg.pos_scale)

    bin_edges = np.percentile(variances_all, np.linspace(0, 100, 11))
    ece_bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        bin_mask = (variances_all >= lo) & (variances_all < hi)
        if bin_mask.sum() < 3:
            continue
        mean_pred_var  = variances_all[bin_mask].mean()
        mean_actual_se = ((errors_all[bin_mask] * cfg.pos_scale) ** 2) / (cfg.pos_scale ** 2)
        ece_bins.append(abs(mean_pred_var - mean_actual_se.mean()))
    ece = float(np.mean(ece_bins)) if ece_bins else float('nan')

    hard_labels      = (errors_all * cfg.pos_scale > 2.0).astype(float)
    normalised_probs = variances_all / (variances_all.max() + 1e-8)
    brier_score      = float(np.mean((normalised_probs - hard_labels) ** 2))

    sigma_all = np.exp(0.5 * logvar_all)
    z_scores  = (gt_all - mu_all) / (sigma_all + 1e-8)
    phi_z     = np.exp(-0.5 * z_scores ** 2) / np.sqrt(2.0 * np.pi)
    Phi_z     = 0.5 * (1.0 + np.vectorize(lambda z: math.erf(z / math.sqrt(2.0)))(z_scores))
    crps_vals = sigma_all * (z_scores * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - 1.0 / np.sqrt(np.pi))
    crps      = float(crps_vals.mean()) * cfg.pos_scale

    sigma_final  = np.exp(0.5 * logvar_all[:, -1, :])
    gt_final     = gt_all[:, -1, :]
    mu_final     = mu_all[:, -1, :]
    within_95ci  = (np.abs(gt_final - mu_final) / (sigma_final + 1e-8) < 1.96).all(axis=-1)
    miss_rate_95 = float(1.0 - within_95ci.mean())

    r_vis_sigma2 = pearson_r(visibilities_all, variances_all)
    r_err_sigma2 = pearson_r(errors_all, variances_all)

    results = {
        'min_ade_m':     min_ade_m,
        'min_fde_m':     min_fde_m,
        'r_vis_sigma2':  r_vis_sigma2,
        'r_err_sigma2':  r_err_sigma2,
        'ECE':           ece,
        'brier_score':   brier_score,
        'CRPS_m':        crps,
        'miss_rate_95':  miss_rate_95,
        '_arrays': {
            'errors_all':       errors_all,
            'variances_all':    variances_all,
            'visibilities_all': visibilities_all,
            'mu_all':           mu_all,
            'gt_all':           gt_all,
            'logvar_all':       logvar_all,
        },
    }
    return results


def print_results(results: Dict, cfg: Config, checkpoint: Dict) -> None:
    print('=' * 62)
    print('  PAPER RESULTS')
    print('  Uncertainty-Guided Occlusion-Aware Trajectory Forecasting')
    print('=' * 62)
    print(f'  Best epoch     : {checkpoint["epoch"]}')
    print(f'  minADE         : {results["min_ade_m"]:.3f} m')
    print(f'  minFDE         : {results["min_fde_m"]:.3f} m')
    print(
        f'  r(vis, sigma2) : {results["r_vis_sigma2"]:+.3f}  '
        f'(target < {cfg.target_r_vis_sigma2:+.2f})  '
        f'{"PASS" if results["r_vis_sigma2"] < cfg.target_r_vis_sigma2 else "FAIL"}'
    )
    print(
        f'  r(err, sigma2) : {results["r_err_sigma2"]:+.3f}  '
        f'(target > {cfg.target_r_err_sigma2:+.2f})  '
        f'{"PASS" if results["r_err_sigma2"] > cfg.target_r_err_sigma2 else "FAIL"}'
    )
    print(f'  ECE            : {results["ECE"]:.4f}')
    print(f'  Brier score    : {results["brier_score"]:.4f}')
    print(f'  CRPS           : {results["CRPS_m"]:.4f} m')
    print(f'  Miss rate @95% : {results["miss_rate_95"]:.3f}  ({results["miss_rate_95"] * 100:.1f}%)')
    print('=' * 62)


def save_results(results: Dict, ablation_results: Dict, checkpoint: Dict, path: str = 'paper_results.json') -> None:
    serialisable = {k: v for k, v in results.items() if k != '_arrays'}
    serialisable['best_epoch'] = int(checkpoint['epoch'])
    serialisable['ablations'] = {
        name: {
            'min_ade_m':    float(ade),
            'r_err_sigma2': float(r_err),
            'r_vis_sigma2': float(r_vis),
        }
        for name, (ade, r_err, r_vis) in ablation_results.items()
    }
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f'Saved: {path}')
