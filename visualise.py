from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from config import Config


def plot_training_curves(history: List[Dict], cfg: Config, save_path: str = 'training_curves.png') -> None:
    epochs_axis = [r['epoch'] for r in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Training History', fontsize=13, fontweight='bold')

    plot_specs = [
        ('Total Loss',           'train_total',      'val_total',      None),
        ('NLL Loss',             'train_nll',        'val_nll',        None),
        ('Huber Regression',     'train_regression', 'val_regression', None),
        ('minADE (m)',            None,               None,             'ade'),
        (r'r(err, $\sigma^2$)',  None,               None,             'r_err'),
        (r'r(vis, $\sigma^2$)',  None,               None,             'r_vis'),
    ]

    for ax, (title, train_key, val_key, special) in zip(axes.flat, plot_specs):
        if train_key is not None:
            ax.plot(epochs_axis, [r[train_key] for r in history], label='train')
            ax.plot(epochs_axis, [r[val_key]   for r in history], label='val')
        elif special == 'ade':
            ax.plot(epochs_axis, [r['val_min_ade'] * cfg.pos_scale for r in history], label='val')
            ax.axhline(1.5, ls='--', color='green', lw=1.5, label='goal: 1.5 m')
        elif special == 'r_err':
            ax.plot(epochs_axis, [r['val_r_err_sigma2'] for r in history], label='val')
            ax.axhline(cfg.target_r_err_sigma2, ls='--', color='green', lw=1.5,
                       label=f'target: > {cfg.target_r_err_sigma2}')
        elif special == 'r_vis':
            ax.plot(epochs_axis, [r['val_r_vis_sigma2'] for r in history], label='val')
            ax.axhline(cfg.target_r_vis_sigma2, ls='--', color='red', lw=1.5,
                       label=f'target: < {cfg.target_r_vis_sigma2}')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def plot_trajectory_predictions(
    model,
    val_loader,
    cfg: Config,
    device,
    save_path: str = 'trajectory_predictions.png',
) -> None:
    import torch

    model.eval()
    vis_batch = next(iter(val_loader))
    with torch.no_grad():
        predictions = model(
            vis_batch['past_trajectory'].to(device),
            vis_batch['visibility_mask'].to(device),
            vis_batch['agent_mask'].to(device),
        )

    past_np        = vis_batch['past_trajectory'].numpy()
    future_np      = vis_batch['future_trajectory'].permute(0, 2, 1, 3).numpy()
    pred_traj_np   = predictions['trajectories'].cpu().numpy()
    pred_logvar_np = predictions['log_variances'].cpu().numpy()
    mode_probs_np  = torch.softmax(predictions['mode_logits'], dim=-1).cpu().numpy()
    visibility_np  = vis_batch['visibility_mask'].squeeze(-1).numpy()
    agent_mask_np  = vis_batch['agent_mask'].numpy()

    scene_idx = 0
    valid_agent_indices = np.where(agent_mask_np[scene_idx])[0][:6]

    try:
        colormap = plt.colormaps['RdYlGn']
    except AttributeError:
        colormap = plt.cm.get_cmap('RdYlGn')

    fig, axes = plt.subplots(1, len(valid_agent_indices), figsize=(4 * len(valid_agent_indices), 5))
    if len(valid_agent_indices) == 1:
        axes = [axes]

    for ax, agent_idx in zip(axes, valid_agent_indices):
        vis_score   = float(np.nan_to_num(visibility_np[scene_idx, agent_idx], nan=0.5))
        agent_color = colormap(vis_score)

        obs_x = past_np[scene_idx, :, agent_idx, 0] * cfg.pos_scale
        obs_y = past_np[scene_idx, :, agent_idx, 1] * cfg.pos_scale
        ax.plot(obs_x, obs_y, 'k-o', ms=4, lw=1.5, label='observed')

        gt_x = future_np[scene_idx, agent_idx, :, 0] * cfg.pos_scale + obs_x[-1]
        gt_y = future_np[scene_idx, agent_idx, :, 1] * cfg.pos_scale + obs_y[-1]
        ax.plot(np.r_[obs_x[-1], gt_x], np.r_[obs_y[-1], gt_y], 'g-', lw=2, label='ground truth')

        probs = np.nan_to_num(mode_probs_np[scene_idx, agent_idx], nan=1.0 / cfg.num_modes)
        probs = probs / (probs.sum() + 1e-8)
        top_mode_indices = np.argsort(probs)[::-1][:3]

        for rank, mode_k in enumerate(top_mode_indices):
            prob_k  = float(np.clip(probs[mode_k], 0, 1))
            pred_x  = np.r_[obs_x[-1], pred_traj_np[scene_idx, agent_idx, mode_k, :, 0] * cfg.pos_scale + obs_x[-1]]
            pred_y  = np.r_[obs_y[-1], pred_traj_np[scene_idx, agent_idx, mode_k, :, 1] * cfg.pos_scale + obs_y[-1]]
            sigma_y = np.r_[0, np.exp(0.5 * np.clip(pred_logvar_np[scene_idx, agent_idx, mode_k, :, 1], -6, 4)) * cfg.pos_scale]

            ax.plot(
                pred_x, pred_y, '-',
                color=agent_color,
                alpha=float(np.clip(0.2 + prob_k * 0.7, 0.05, 0.95)),
                lw=max(0.5, 2.5 - rank * 0.7),
                label=f'mode {mode_k} (p={prob_k:.2f})' if rank < 2 else '_',
            )
            ax.fill_between(
                pred_x, pred_y - sigma_y, pred_y + sigma_y,
                alpha=float(np.clip(0.04 + 0.06 * prob_k, 0.01, 0.25)),
                color=agent_color,
            )

        ax.plot(obs_x[-1], obs_y[-1], 's', ms=8, color=agent_color,
                zorder=10, markeredgecolor='k', markeredgewidth=0.8)
        ax.set_title(f'Agent {agent_idx}  vis={vis_score:.2f}', color=agent_color, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        if agent_idx == valid_agent_indices[0]:
            ax.legend(fontsize=8)

    fig.colorbar(
        plt.cm.ScalarMappable(cmap=colormap, norm=mcolors.Normalize(0, 1)),
        ax=axes,
        label='Visibility score  (0 = fully occluded, 1 = fully visible)',
        shrink=0.7,
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def plot_uncertainty_calibration(
    eval_results: Dict,
    cfg: Config,
    save_path: str = 'uncertainty_calibration.png',
) -> None:
    arrays       = eval_results['_arrays']
    errors_all   = arrays['errors_all']
    variances_all = arrays['variances_all']
    visibilities_all = arrays['visibilities_all']
    r_vis_sigma2 = eval_results['r_vis_sigma2']
    r_err_sigma2 = eval_results['r_err_sigma2']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Uncertainty Calibration', fontsize=12, fontweight='bold')

    plot_data = [
        (
            visibilities_all, variances_all,
            'Visibility score', r'$\sigma^2$ (predicted variance)',
            errors_all * cfg.pos_scale,
            r'r(vis, $\sigma^2$): less visible $\rightarrow$ higher variance',
            r_vis_sigma2,
        ),
        (
            errors_all * cfg.pos_scale, variances_all,
            'minADE (m)', r'$\sigma^2$ (predicted variance)',
            visibilities_all,
            r'r(err, $\sigma^2$): higher error $\rightarrow$ higher variance',
            r_err_sigma2,
        ),
    ]

    for ax, (x, y, xlabel, ylabel, color_vals, title, r_val) in zip(axes, plot_data):
        scatter = ax.scatter(
            x, y, c=color_vals, cmap='RdYlGn_r',
            alpha=0.4, s=15,
            vmin=0, vmax=np.percentile(color_vals, 95),
        )
        plt.colorbar(scatter, ax=ax)
        trend_coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(trend_coeffs, x_line), 'k--', lw=2, label=f'r = {r_val:+.3f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')


def plot_per_bin_evaluation(
    eval_results: Dict,
    cfg: Config,
    save_path: str = 'per_bin_evaluation.png',
) -> None:
    from train import pearson_r

    arrays           = eval_results['_arrays']
    errors_all       = arrays['errors_all']
    variances_all    = arrays['variances_all']
    visibilities_all = arrays['visibilities_all']

    VISIBILITY_BINS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    BIN_LABELS      = ['0–0.25\n(occluded)', '0.25–0.5\n(partial)', '0.5–0.75\n(mostly vis)', '0.75–1.0\n(visible)']
    BIN_COLORS      = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']

    bin_min_ade:      List[float] = []
    bin_mean_var:     List[float] = []
    bin_r_err_sigma2: List[float] = []
    bin_counts:       List[int]   = []

    for lo, hi in VISIBILITY_BINS:
        bin_mask = (visibilities_all >= lo) & (visibilities_all < hi)
        if bin_mask.sum() < 3:
            bin_min_ade.append(0.0)
            bin_mean_var.append(0.0)
            bin_r_err_sigma2.append(0.0)
            bin_counts.append(0)
            continue
        bin_min_ade.append(float((errors_all[bin_mask] * cfg.pos_scale).mean()))
        bin_mean_var.append(float(variances_all[bin_mask].mean()))
        r_within = pearson_r(errors_all[bin_mask], variances_all[bin_mask]) if bin_mask.sum() > 5 else 0.0
        bin_r_err_sigma2.append(r_within)
        bin_counts.append(int(bin_mask.sum()))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Per-Visibility-Bin Evaluation', fontsize=13, fontweight='bold')
    x_pos = np.arange(len(VISIBILITY_BINS))

    bar_specs = [
        (bin_min_ade,      'minADE (m)',             'minADE by Visibility Bin'),
        (bin_mean_var,     r'Mean $\sigma^2$',        r'Predicted $\sigma^2$ by Visibility Bin'),
        (bin_r_err_sigma2, r'r(err, $\sigma^2$)',     r'r(err, $\sigma^2$) by Visibility Bin'),
    ]

    for ax, (values, ylabel, title) in zip(axes, bar_specs):
        ax.bar(x_pos, values, color=BIN_COLORS, edgecolor='k', linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(BIN_LABELS, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        if 'r(' in ylabel:
            ax.axhline(cfg.target_r_err_sigma2, ls='--', color='green', lw=1.5,
                       label=f'target: {cfg.target_r_err_sigma2}')
            ax.legend()

    for bar, count in zip(axes[0].patches, bin_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'n={count}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')
