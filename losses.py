from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.float()).sum() / (mask.float().sum() + 1e-8)


def winner_takes_all_nll(
    trajectories: torch.Tensor,
    log_variances: torch.Tensor,
    mode_logits: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    regression_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_agents, num_modes, pred_len, _ = trajectories.shape
    gt = ground_truth.permute(0, 2, 1, 3)

    ade_per_mode = (
        (trajectories - gt.unsqueeze(2).expand_as(trajectories))
        .pow(2).sum(-1).sqrt().mean(-1)
    )
    winner_indices = ade_per_mode.argmin(dim=-1)

    idx = winner_indices.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(
        batch_size, num_agents, 1, pred_len, 2
    )
    winner_traj   = trajectories.gather(2, idx).squeeze(2)
    winner_logvar = log_variances.gather(2, idx).squeeze(2).clamp(-4.0, 1.0)
    winner_var    = winner_logvar.exp() + 1e-4

    nll_per_agent = (
        0.5 * (winner_logvar + (gt - winner_traj).pow(2) / winner_var)
        .sum(-1).mean(-1)
    )
    nll_loss = masked_mean(torch.nan_to_num(nll_per_agent), agent_mask)

    regression_loss = masked_mean(
        F.huber_loss(winner_traj, gt, reduction='none', delta=1.0).sum(-1).mean(-1),
        agent_mask,
    )

    classification_loss = masked_mean(
        F.cross_entropy(
            mode_logits.view(batch_size * num_agents, num_modes),
            winner_indices.view(batch_size * num_agents),
            reduction='none',
        ).view(batch_size, num_agents),
        agent_mask,
    )

    total = nll_loss + regression_weight * regression_loss + 0.5 * classification_loss
    return total, nll_loss, regression_loss, classification_loss


def visibility_calibration_loss(
    log_variances: torch.Tensor,
    visibility: torch.Tensor,
    agent_mask: torch.Tensor,
    margin: float = 0.1,
    visibility_gap: float = 0.05,
) -> torch.Tensor:
    mean_var = log_variances.clamp(-4.0, 1.0).exp().mean(dim=[2, 3, 4])
    vis = visibility.squeeze(-1)
    mask = agent_mask.float()

    var_i = mean_var.unsqueeze(2)
    var_j = mean_var.unsqueeze(1)
    vis_i = vis.unsqueeze(2)
    vis_j = vis.unsqueeze(1)

    valid_pairs  = mask.unsqueeze(2) * mask.unsqueeze(1)
    less_visible = (vis_i < vis_j - visibility_gap).float()

    num_pairs  = (valid_pairs * less_visible).sum().clamp(min=1.0)
    violations = F.relu(var_j - var_i + margin)
    return (violations * valid_pairs * less_visible).sum() / num_pairs


def error_calibration_loss(
    log_variances: torch.Tensor,
    trajectories: torch.Tensor,
    ground_truth: torch.Tensor,
    agent_mask: torch.Tensor,
    margin: float = 0.05,
    error_gap: float = 0.02,
) -> torch.Tensor:
    gt = ground_truth.permute(0, 2, 1, 3).unsqueeze(2).expand_as(trajectories)
    min_ade = (
        (trajectories - gt).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
    )
    mean_var = log_variances.clamp(-4.0, 1.0).exp().mean(dim=[2, 3, 4])
    mask = agent_mask.float()

    err_i = min_ade.unsqueeze(2)
    err_j = min_ade.unsqueeze(1)
    var_i = mean_var.unsqueeze(2)
    var_j = mean_var.unsqueeze(1)

    valid_pairs  = mask.unsqueeze(2) * mask.unsqueeze(1)
    higher_error = (err_i > err_j + error_gap).float()

    num_pairs  = (valid_pairs * higher_error).sum().clamp(min=1.0)
    violations = F.relu(var_j - var_i + margin)
    return (violations * valid_pairs * higher_error).sum() / num_pairs


def variance_smoothness_loss(
    log_variances: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    temporal_diff = (log_variances[:, :, :, 1:, :] - log_variances[:, :, :, :-1, :]).pow(2)
    return masked_mean(temporal_diff.mean(dim=[2, 3, 4]), agent_mask)


class TotalLoss(nn.Module):

    def __init__(
        self,
        lambda_vis_calib: float = 0.6,
        lambda_err_calib: float = 0.3,
        lambda_regression: float = 1.5,
        lambda_smooth: float = 0.005,
    ) -> None:
        super().__init__()
        self.lambda_vis_calib  = lambda_vis_calib
        self.lambda_err_calib  = lambda_err_calib
        self.lambda_regression = lambda_regression
        self.lambda_smooth     = lambda_smooth

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        traj    = predictions['trajectories']
        log_var = predictions['log_variances']
        logits  = predictions['mode_logits']
        gt      = targets['future_trajectory']
        mask    = targets['agent_mask']
        vis     = targets['visibility_mask']

        wta_total, nll, reg, cls = winner_takes_all_nll(
            traj, log_var, logits, gt, mask, self.lambda_regression
        )
        vis_calib = visibility_calibration_loss(log_var, vis, mask)
        err_calib = error_calibration_loss(log_var, traj, gt, mask)
        smooth    = variance_smoothness_loss(log_var, mask)

        total = (
            wta_total
            + self.lambda_vis_calib * vis_calib
            + self.lambda_err_calib * err_calib
            + self.lambda_smooth    * smooth
        )

        return {
            'total':      total,
            'nll':        nll,
            'regression': reg,
            'cls':        cls,
            'vis_calib':  vis_calib,
            'err_calib':  err_calib,
            'smooth':     smooth,
        }
