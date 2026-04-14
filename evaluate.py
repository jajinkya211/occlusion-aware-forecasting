"""
evaluate.py — Full evaluation suite for UGIN.

Produces:
  - Table 2 (baseline comparison)
  - Table 3 (uncertainty calibration metrics)
  - uncertainty_calibration.png
  - per_bin_evaluation.png
  - paper_results_final.json

Usage:
    python evaluate.py                   # nuScenes
    python evaluate.py --synthetic
    python evaluate.py --checkpoint path/to/best_model.pth
"""
import argparse, json, math
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import (DEVICE, NUM_WORKERS, BATCH_SIZE, POS_SCALE,
                         HIDDEN_DIM, NUM_MODES, NUM_HEADS, N_TEMP_L, N_SPAT_L,
                         DROPOUT, OBS_LEN, PRED_LEN, MAX_AGENTS,
                         NUSCENES_DATAROOT, SEED)
from src.model import OcclusionAwareForecaster


# ── Metric helpers ─────────────────────────────────────────────────────────────

def collect_eval_arrays(model, val_loader, device):
    """Run full val-set inference and return per-agent arrays."""
    model.eval()
    all_vis, all_var_tot, all_var_epi, all_err, all_fde = [], [], [], [], []
    all_mu, all_gt, all_lv = [], [], []

    with torch.no_grad():
        for b in val_loader:
            past = b['past_trajectory'].to(device)
            fut  = b['future_trajectory'].to(device)
            vis  = b['visibility_mask'].to(device)
            msk  = b['agent_mask'].to(device)
            o    = model(past, vis, msk)

            traj_c   = o['trajectories'].cpu()
            lv_tot_c = o['log_var_total'].cpu()
            lv_epi_c = o['log_var_epi'].cpu()
            fut_c    = fut.cpu()
            vis_c    = vis.cpu()
            msk_c    = msk.cpu()

            gt_c  = fut_c.permute(0, 2, 1, 3)
            gt_e  = gt_c.unsqueeze(2).expand_as(traj_c)
            err   = (traj_c - gt_e).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
            fde   = (traj_c - gt_e).pow(2).sum(-1).sqrt()[:, :, :, -1].min(-1).values
            var_tot = lv_tot_c.clamp(-4, 1).exp().mean([2, 3, 4])
            var_epi = lv_epi_c.clamp(-4, 1).exp().mean([2, 3, 4])

            ade_k   = (traj_c - gt_e).pow(2).sum(-1).sqrt().mean(-1)
            best    = ade_k.argmin(-1)
            B2, N2  = best.shape
            idx     = best.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(B2, N2, 1, PRED_LEN, 2)
            best_mu = traj_c.gather(2, idx).squeeze(2)
            best_lv = lv_tot_c.gather(2, idx).squeeze(2)

            vld = msk_c
            all_err.append(err[vld].numpy())
            all_var_tot.append(var_tot[vld].numpy())
            all_var_epi.append(var_epi[vld].numpy())
            all_vis.append(vis_c.squeeze(-1)[vld].numpy())
            all_fde.append(fde[vld].numpy())
            all_mu.append(best_mu[vld].numpy())
            all_gt.append(gt_c[vld].numpy())
            all_lv.append(best_lv[vld].numpy())

    return {
        'ae':      np.concatenate(all_err),
        'av':      np.concatenate(all_var_tot),
        'av_epi':  np.concatenate(all_var_epi),
        'avis':    np.concatenate(all_vis),
        'af':      np.concatenate(all_fde),
        'mu':      np.concatenate(all_mu),
        'gt_np':   np.concatenate(all_gt),
        'lv':      np.concatenate(all_lv),
    }


def compute_calibration_metrics(arrays):
    ae, av, avis, af = arrays['ae'], arrays['av'], arrays['avis'], arrays['af']
    av_epi = arrays['av_epi']
    mu, gt_np, lv = arrays['mu'], arrays['gt_np'], arrays['lv']
    ae_m = ae * POS_SCALE
    af_m = af * POS_SCALE

    def rp(a, b):
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    r_vis = rp(avis, av_epi)   # Claim 1: r(vis, sigma2_epi)
    r_err = rp(ae, av)         # Claim 2: r(err, sigma2_total)

    # ECE
    bin_edges = np.percentile(av, np.linspace(0, 100, 11))
    ece_vals  = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (av >= lo) & (av < hi)
        if mask.sum() < 3:
            continue
        ece_vals.append(abs(av[mask].mean() - (ae_m[mask] ** 2).mean() / POS_SCALE ** 2))
    ECE = float(np.mean(ece_vals)) if ece_vals else float('nan')

    # Brier
    y_hard = (ae_m > 2.0).astype(float)
    p_hard  = av / (av.max() + 1e-8)
    brier   = float(np.mean((p_hard - y_hard) ** 2))

    # CRPS (pure numpy, no scipy)
    sigma  = np.exp(0.5 * lv)
    z      = (gt_np - mu) / (sigma + 1e-8)
    phi_z  = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    Phi_z  = 0.5 * (1 + np.vectorize(
                 lambda x: float(__import__('math').erf(x / np.sqrt(2))))(z))
    crps_e = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))
    CRPS   = float(crps_e.mean()) * POS_SCALE

    # Miss Rate @95% CI
    sf    = np.exp(0.5 * lv[:, -1, :])
    gtf   = gt_np[:, -1, :]
    muf   = mu[:, -1, :]
    MR_95 = float(1 - ((np.abs(gtf - muf) / (sf + 1e-8) < 1.96).all(axis=-1)).mean())

    return {
        'r_vis_epi': r_vis, 'r_err_tot': r_err,
        'ECE': ECE, 'brier': brier, 'CRPS_m': CRPS, 'MR_95': MR_95,
        'minADE_m': float(ae_m.mean()), 'minFDE_m': float(af_m.mean()),
        'ae_m': ae_m, 'av': av, 'av_epi': av_epi, 'avis': avis,
    }


def print_tables(metrics):
    SEP = '=' * 52
    print(SEP)
    print('  TABLE 3  |  Uncertainty Calibration (full model)')
    print(SEP)
    rows = [
        ('r(vis, sigma2_epi) Pearson', metrics['r_vis_epi'], '< -0.20', metrics['r_vis_epi'] < -0.20),
        ('r(err, sigma2_tot) Pearson', metrics['r_err_tot'], '> +0.30', metrics['r_err_tot'] >  0.30),
        ('ECE',                         metrics['ECE'],      '-',        None),
        ('Brier Score',                 metrics['brier'],    '-',        None),
        ('CRPS (m)',                     metrics['CRPS_m'],  '-',        None),
        ('Miss Rate @95% CI (%)',        metrics['MR_95']*100, '-',      None),
    ]
    for name, val, target, ok in rows:
        status = (' PASS' if ok else ' FAIL') if ok is not None else ''
        print(f'  {name:<32}  {val:>8.4f}  {target}{status}')
    print(SEP)
    print(f'\n  minADE: {metrics["minADE_m"]:.3f} m   minFDE: {metrics["minFDE_m"]:.3f} m')


def plot_calibration(metrics, save_path='uncertainty_calibration.png'):
    avis, av_epi = metrics['avis'], metrics['av_epi']
    ae_m, av     = metrics['ae_m'], metrics['av']
    r_vis = metrics['r_vis_epi']
    r_err = metrics['r_err_tot']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Uncertainty Calibration', fontsize=12, fontweight='bold')

    ax = axes[0]
    sc = ax.scatter(avis, av_epi, c=ae_m, cmap='RdYlGn_r', alpha=0.4, s=15,
                    vmin=0, vmax=np.percentile(ae_m, 95))
    plt.colorbar(sc, ax=ax, label='minADE (m)')
    xs = np.linspace(avis.min(), avis.max(), 100)
    ax.plot(xs, np.polyval(np.polyfit(avis, av_epi, 1), xs), 'k--', lw=2,
            label=f'r = {r_vis:.3f}')
    ax.set_xlabel('Visibility')
    ax.set_ylabel('sigma2_epi')
    ax.set_title(f'Vis -> sigma2_epi  (goal: r < -0.20)')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    sc = ax.scatter(ae_m, av, c=avis, cmap='RdYlGn', alpha=0.4, s=15,
                    vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='Visibility')
    xs = np.linspace(ae_m.min(), ae_m.max(), 100)
    ax.plot(xs, np.polyval(np.polyfit(ae_m, av, 1), xs), 'k--', lw=2,
            label=f'r = {r_err:.3f}')
    ax.set_xlabel('minADE (m)')
    ax.set_ylabel('sigma2_total')
    ax.set_title(f'Err -> sigma2_total  (goal: r > +0.30)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


def plot_per_bin(arrays, save_path='per_bin_evaluation.png'):
    ae_m   = arrays['ae'] * POS_SCALE
    av_epi = arrays['av_epi']
    avis   = arrays['avis']
    ae_raw = arrays['ae']
    av     = arrays['av']

    bins   = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    labels = ['0-0.25\noccluded', '0.25-0.5\npartial', '0.5-0.75\nmostly vis', '0.75-1.0\nvisible']
    cols   = ['#d73027', '#fc8d59', '#91bfdb', '#4575b4']
    bin_ade, bin_s2, bin_r, bin_n = [], [], [], []
    for lo, hi in bins:
        mask = (avis >= lo) & (avis < hi)
        if mask.sum() < 3:
            bin_ade.append(0); bin_s2.append(0); bin_r.append(0); bin_n.append(0)
            continue
        bin_ade.append(float(ae_m[mask].mean()))
        bin_s2.append(float(av_epi[mask].mean()))
        r = float(np.corrcoef(ae_raw[mask], av[mask])[0, 1]) if mask.sum() > 5 else 0.0
        bin_r.append(0.0 if np.isnan(r) else r)
        bin_n.append(int(mask.sum()))

    import numpy as np2
    x = np.arange(len(bins))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Per-Visibility-Bin Evaluation', fontsize=13, fontweight='bold')
    for ax, (vals, ylabel, title) in zip(axes, [
        (bin_ade, 'minADE (m)',       'ADE by Visibility Bin'),
        (bin_s2,  'Mean sigma2_epi',  'Epistemic Variance by Bin'),
        (bin_r,   'r(err, sigma2_tot)', 'Error-Variance Corr. by Bin'),
    ]):
        ax.bar(x, vals, color=cols, edgecolor='k', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        if 'r(' in ylabel:
            ax.axhline(0.30, ls='--', c='g', lw=1.5, label='goal 0.30')
            ax.legend()
    for bar, n in zip(axes[0].patches, bin_n):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05, f'n={n}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')
    print(f'\n  {"Bin":<20}  {"n":>5}  {"ADE(m)":>7}  {"sigma2_epi":>10}  {"Err-r":>7}')
    for lb, n, a, s, r in zip(labels, bin_n, bin_ade, bin_s2, bin_r):
        print(f'  {lb.replace(chr(10), " "):<20}  {n:>5}  {a:>7.3f}  {s:>10.4f}  {r:>7.3f}')


def main(args):
    # ── Data ──────────────────────────────────────────────────────────────────
    if args.synthetic:
        from src.datasets import get_synthetic_splits
        _, val_ds = get_synthetic_splits()
    else:
        from nuscenes.nuscenes import NuScenes
        from src.datasets import NuScenesDataset, get_nuscenes_splits
        nusc    = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DATAROOT, verbose=False)
        _, toks = get_nuscenes_splits(nusc)
        val_ds  = NuScenesDataset(nusc, toks, OBS_LEN, PRED_LEN, MAX_AGENTS)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS)
    print(f'Val: {len(val_ds)} samples')

    # ── Load model ────────────────────────────────────────────────────────────
    model = OcclusionAwareForecaster(
        h=HIDDEN_DIM, T=PRED_LEN, K=NUM_MODES,
        heads=NUM_HEADS, tl=N_TEMP_L, sl=N_SPAT_L, drop=DROPOUT).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt['state'])
    print(f'Loaded checkpoint: epoch {ckpt["epoch"]}  |  {model.count_parameters():,} params')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    arrays  = collect_eval_arrays(model, val_loader, DEVICE)
    metrics = compute_calibration_metrics(arrays)
    print_tables(metrics)
    plot_calibration(metrics)
    plot_per_bin(arrays)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    results = {
        'model': {
            'parameters': model.count_parameters(),
            'best_epoch': int(ckpt['epoch']),
        },
        'trajectory_accuracy': {
            'minADE_m': metrics['minADE_m'],
            'minFDE_m': metrics['minFDE_m'],
        },
        'uncertainty_calibration': {
            'vis_sigma2_epi_r':   metrics['r_vis_epi'],
            'err_sigma2_total_r': metrics['r_err_tot'],
            'ECE':     metrics['ECE'],
            'Brier':   metrics['brier'],
            'CRPS_m':  metrics['CRPS_m'],
            'MR_95':   metrics['MR_95'],
        },
    }
    with open('paper_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved: paper_results_final.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate UGIN')
    parser.add_argument('--checkpoint', default='best_model.pth')
    parser.add_argument('--synthetic', action='store_true')
    main(parser.parse_args())
