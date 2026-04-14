"""
ablation.py — Ablation study for UGIN.

Trains four variants for 5 epochs each and reports ADE, r(err, sigma2_tot),
r(vis, sigma2_epi) for each.

Usage:
    python ablation.py
    python ablation.py --synthetic
    python ablation.py --epochs 10
"""
import argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (DEVICE, NUM_WORKERS, BATCH_SIZE, LR, WEIGHT_DECAY,
                         POS_SCALE, HIDDEN_DIM, NUM_MODES, NUM_HEADS,
                         N_TEMP_L, N_SPAT_L, DROPOUT, OBS_LEN, PRED_LEN,
                         MAX_AGENTS, LAMBDA_CALIB, LAMBDA_ERR_CALIB,
                         LAMBDA_REG, NUSCENES_DATAROOT, clear_device_cache)
from src.model import OcclusionAwareForecaster, NoFiLMForecaster, TemporalOnlyForecaster
from src.losses import TotalLoss


def run_ablation(MdlClass, loss_fn, label, train_loader, val_loader, epochs=5):
    mdl = MdlClass(h=HIDDEN_DIM, T=PRED_LEN, K=NUM_MODES,
                   heads=NUM_HEADS, tl=N_TEMP_L, sl=N_SPAT_L, drop=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    t0  = time.time()

    # Train
    mdl.train()
    for _ in range(epochs):
        for b in train_loader:
            past = b['past_trajectory'].to(DEVICE)
            fut  = b['future_trajectory'].to(DEVICE)
            vis  = b['visibility_mask'].to(DEVICE)
            msk  = b['agent_mask'].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            p = mdl(past, vis, msk)
            L = loss_fn(p, {'future_trajectory': fut, 'agent_mask': msk, 'visibility_mask': vis})
            if torch.isfinite(L['total']):
                L['total'].backward()
                for param in mdl.parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 0.5)
                opt.step()

    # Evaluate
    mdl.eval()
    all_e, all_v_tot, all_v_epi, all_vis_ = [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            past = b['past_trajectory'].to(DEVICE)
            fut  = b['future_trajectory'].to(DEVICE)
            vis  = b['visibility_mask'].to(DEVICE)
            msk  = b['agent_mask'].to(DEVICE)
            o    = mdl(past, vis, msk)
            tc   = o['trajectories'].cpu()
            lv_t = o['log_var_total'].cpu()
            lv_e = o['log_var_epi'].cpu()
            fc   = fut.cpu(); vc = vis.cpu(); mc = msk.cpu()
            gt_e = fc.permute(0, 2, 1, 3).unsqueeze(2).expand_as(tc)
            err  = (tc - gt_e).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
            var_tot = lv_t.clamp(-4, 1).exp().mean([2, 3, 4])
            var_epi = lv_e.clamp(-4, 1).exp().mean([2, 3, 4])
            vld = mc
            all_e.append(err[vld].numpy())
            all_v_tot.append(var_tot[vld].numpy())
            all_v_epi.append(var_epi[vld].numpy())
            all_vis_.append(vc.squeeze(-1)[vld].numpy())

    ae     = np.concatenate(all_e)
    av_tot = np.concatenate(all_v_tot)
    av_epi = np.concatenate(all_v_epi)
    avis   = np.concatenate(all_vis_)

    def rp(a, b):
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    result = (float(ae.mean() * POS_SCALE), rp(ae, av_tot), rp(avis, av_epi))
    print(f'{label:<26}: ADE={result[0]:.2f}m  '
          f'r(err,tot)={result[1]:.3f}  r(vis,epi)={result[2]:.3f}  '
          f'({time.time()-t0:.0f}s)')
    del mdl, opt
    clear_device_cache()
    return result


def main(args):
    # ── Data ──────────────────────────────────────────────────────────────────
    if args.synthetic:
        from src.datasets import get_synthetic_splits
        train_ds, val_ds = get_synthetic_splits()
    else:
        from nuscenes.nuscenes import NuScenes
        from src.datasets import NuScenesDataset, get_nuscenes_splits
        nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DATAROOT, verbose=False)
        tr_t, va_t = get_nuscenes_splits(nusc)
        train_ds = NuScenesDataset(nusc, tr_t, OBS_LEN, PRED_LEN, MAX_AGENTS)
        val_ds   = NuScenesDataset(nusc, va_t, OBS_LEN, PRED_LEN, MAX_AGENTS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)

    epochs = args.epochs
    print(f'Running ablations ({epochs} epochs each)...\n')

    results = {}

    # Full model from checkpoint
    full_mdl = OcclusionAwareForecaster(
        h=HIDDEN_DIM, T=PRED_LEN, K=NUM_MODES,
        heads=NUM_HEADS, tl=N_TEMP_L, sl=N_SPAT_L, drop=DROPOUT).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    full_mdl.load_state_dict(ckpt['state'])
    full_mdl.eval()
    all_e, all_v_tot, all_v_epi, all_vis_ = [], [], [], []
    with torch.no_grad():
        for b in val_loader:
            past = b['past_trajectory'].to(DEVICE); fut = b['future_trajectory'].to(DEVICE)
            vis  = b['visibility_mask'].to(DEVICE);  msk = b['agent_mask'].to(DEVICE)
            o = full_mdl(past, vis, msk)
            tc = o['trajectories'].cpu(); lv_t = o['log_var_total'].cpu()
            lv_e = o['log_var_epi'].cpu(); vc = vis.cpu(); mc = msk.cpu()
            gt_e = fut.cpu().permute(0, 2, 1, 3).unsqueeze(2).expand_as(tc)
            err     = (tc - gt_e).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
            var_tot = lv_t.clamp(-4, 1).exp().mean([2, 3, 4])
            var_epi = lv_e.clamp(-4, 1).exp().mean([2, 3, 4])
            vld = mc
            all_e.append(err[vld].numpy()); all_v_tot.append(var_tot[vld].numpy())
            all_v_epi.append(var_epi[vld].numpy()); all_vis_.append(vc.squeeze(-1)[vld].numpy())
    ae = np.concatenate(all_e); av_tot = np.concatenate(all_v_tot)
    av_epi = np.concatenate(all_v_epi); avis = np.concatenate(all_vis_)
    def rp(a, b):
        c = np.corrcoef(a, b)[0, 1]; return float(c) if not np.isnan(c) else 0.0
    results['Full Model (ours)'] = (float(ae.mean()*POS_SCALE), rp(ae,av_tot), rp(avis,av_epi))
    print(f'Full Model (trained)     : ADE={results["Full Model (ours)"][0]:.2f}m  '
          f'r(err,tot)={results["Full Model (ours)"][1]:.3f}  '
          f'r(vis,epi)={results["Full Model (ours)"][2]:.3f}')
    del full_mdl; clear_device_cache()

    # Ablation variants
    results['No VisibilityFiLM'] = run_ablation(
        NoFiLMForecaster,
        TotalLoss(lc=LAMBDA_CALIB, lec=LAMBDA_ERR_CALIB, lreg=LAMBDA_REG),
        'No VisibilityFiLM', train_loader, val_loader, epochs)

    results['No vis_calib (lc=0)'] = run_ablation(
        OcclusionAwareForecaster,
        TotalLoss(lc=0.0, lec=LAMBDA_ERR_CALIB, lreg=LAMBDA_REG),
        'No vis_calib (lc=0)', train_loader, val_loader, epochs)

    results['No err_calib (lec=0)'] = run_ablation(
        OcclusionAwareForecaster,
        TotalLoss(lc=LAMBDA_CALIB, lec=0.0, lreg=LAMBDA_REG),
        'No err_calib (lec=0)', train_loader, val_loader, epochs)

    # Print table
    print()
    print('=' * 68)
    print(f'  ABLATION TABLE  ({epochs}-epoch training)')
    print('=' * 68)
    print(f'  {"Variant":<26}  {"ADE (m)":>8}  {"r(err,tot)":>10}  {"r(vis,epi)":>10}')
    print('  ' + '-' * 60)
    for name, (ade, ev, sv) in results.items():
        tag  = ' <- ours' if 'ours' in name else ''
        ev_s = 'PASS' if ev > 0.3  else '    '
        sv_s = 'PASS' if sv < -0.2 else '    '
        print(f'  {name:<26}  {ade:>8.2f}  {ev_s} {ev:>5.3f}  {sv_s} {sv:>5.3f}{tag}')
    print('=' * 68)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UGIN ablation study')
    parser.add_argument('--checkpoint', default='best_model.pth')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    main(parser.parse_args())
