"""
train.py — Training loop for UGIN.

Usage:
    python train.py                      # nuScenes (default)
    python train.py --synthetic          # synthetic dataset (no download needed)
    python train.py --epochs 150         # override epoch count
"""
import argparse, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (DEVICE, NUM_WORKERS, BATCH_SIZE, EPOCHS, LR,
                         WEIGHT_DECAY, WARMUP_EPOCHS, LAMBDA_CALIB,
                         LAMBDA_ERR_CALIB, LAMBDA_REG, POS_SCALE,
                         HIDDEN_DIM, NUM_MODES, NUM_HEADS, N_TEMP_L, N_SPAT_L,
                         DROPOUT, OBS_LEN, PRED_LEN, MAX_AGENTS,
                         NUSCENES_DATAROOT, USE_SYNTHETIC, clear_device_cache)
from src.model import OcclusionAwareForecaster
from src.losses import TotalLoss


def masked_mean(x, m):
    return (x * m.float()).sum() / (m.float().sum() + 1e-8)


@torch.no_grad()
def ade_fde(traj, gt, mask):
    gt_ = gt.permute(0, 2, 1, 3).unsqueeze(2).expand_as(traj)
    l2  = (traj - gt_).pow(2).sum(-1).sqrt()
    return (masked_mean(l2.mean(-1).min(-1).values, mask).item(),
            masked_mean(l2[:, :, :, -1].min(-1).values, mask).item())


def run_epoch(model, loader, loss_fn, opt, device, train):
    model.train() if train else model.eval()
    tots = {k: 0.0 for k in ['total', 'nll', 'reg', 'cls', 'vis_calib', 'err_calib', 'smooth']}
    as_ = fs_ = n = skipped = 0
    all_err, all_var, all_vis = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in loader:
            past = b['past_trajectory'].to(device)
            fut  = b['future_trajectory'].to(device)
            vis  = b['visibility_mask'].to(device)
            msk  = b['agent_mask'].to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            p = model(past, vis, msk)
            L = loss_fn(p, {'future_trajectory': fut, 'agent_mask': msk, 'visibility_mask': vis})
            if not torch.isfinite(L['total']):
                skipped += 1
                continue
            if train:
                L['total'].backward()
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            with torch.no_grad():
                a, f = ade_fde(p['trajectories'], fut, msk)
                gt_e = fut.permute(0, 2, 1, 3).unsqueeze(2).expand_as(p['trajectories'])
                err  = (p['trajectories'] - gt_e).pow(2).sum(-1).sqrt().mean(-1).min(-1).values
                var  = p['log_var_total'].clamp(-4, 1).exp().mean([2, 3, 4])
                vld  = msk.cpu()
                all_err.append(err.cpu()[vld].numpy())
                all_var.append(var.cpu()[vld].numpy())
                all_vis.append(vis.squeeze(-1).cpu()[vld].numpy())
            if np.isfinite(a) and np.isfinite(f):
                for k in tots:
                    tots[k] += L[k].item()
                as_ += a
                fs_ += f
                n   += 1
    if n == 0:
        return {k: float('nan') for k in list(tots.keys()) + ['ade', 'fde', 'ev', 'sv', 'skipped']}
    ae   = np.concatenate(all_err)
    av   = np.concatenate(all_var)
    avis = np.concatenate(all_vis)
    def rp(a, b):
        c = np.corrcoef(a, b)[0, 1]
        return float(c) if not np.isnan(c) else 0.0
    r = {k: v / n for k, v in tots.items()}
    r.update({'ade': as_ / n, 'fde': fs_ / n,
              'ev': rp(ae, av), 'sv': rp(avis, av), 'skipped': skipped})
    return r


def lr_lambda(ep):
    if ep < WARMUP_EPOCHS:
        return (ep + 1) / WARMUP_EPOCHS
    prog = (ep - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return max(0.05, 0.5 * (1 + np.cos(np.pi * prog)))


def main(args):
    # ── Data ──────────────────────────────────────────────────────────────────
    if args.synthetic:
        from src.datasets import get_synthetic_splits
        train_ds, val_ds = get_synthetic_splits(n=1500)
    else:
        from nuscenes.nuscenes import NuScenes
        from src.datasets import NuScenesDataset, get_nuscenes_splits
        nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DATAROOT, verbose=False)
        train_toks, val_toks = get_nuscenes_splits(nusc)
        train_ds = NuScenesDataset(nusc, train_toks, OBS_LEN, PRED_LEN, MAX_AGENTS)
        val_ds   = NuScenesDataset(nusc, val_toks,   OBS_LEN, PRED_LEN, MAX_AGENTS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS)
    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  |  {len(train_loader)} batches/ep')

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = OcclusionAwareForecaster(
        h=HIDDEN_DIM, T=PRED_LEN, K=NUM_MODES,
        heads=NUM_HEADS, tl=N_TEMP_L, sl=N_SPAT_L, drop=DROPOUT).to(DEVICE)
    print(f'Parameters: {model.count_parameters():,}  |  device={DEVICE}')

    loss_fn   = TotalLoss(lc=LAMBDA_CALIB, lec=LAMBDA_ERR_CALIB, lreg=LAMBDA_REG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = []
    best_score = float('inf')
    consecutive_nan = 0
    epochs = args.epochs or EPOCHS

    hdr = f"{'Ep':>4}  {'Loss':>8}  {'ADE_m':>6}  {'Err-r':>7}  {'Vis-r':>7}  {'LR':>8}"
    print(hdr)
    print('-' * len(hdr))

    for ep in range(epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, optimizer, DEVICE, True)
        va = run_epoch(model, val_loader,   loss_fn, None,      DEVICE, False)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        row = {'ep': ep + 1}
        row.update({f'tr_{k}': v for k, v in tr.items()})
        row.update({f'va_{k}': v for k, v in va.items()})
        history.append(row)

        if not np.isfinite(va.get('total', float('nan'))):
            consecutive_nan += 1
            print(f'{ep + 1:4d}  [NaN]')
            if consecutive_nan >= 5:
                break
            continue
        consecutive_nan = 0

        ev_ok = va['ev'] > 0.30
        sv_ok = va['sv'] < -0.20
        print(f"{ep+1:4d}  {va['total']:8.4f}  {va['ade']*POS_SCALE:6.2f}m  "
              f"{'ok' if ev_ok else '  '}{va['ev']:6.3f}  "
              f"{'ok' if sv_ok else '  '}{va['sv']:6.3f}  "
              f"{lr_now:.2e}  {time.time()-t0:.1f}s")

        score = va['ade'] + (0 if ev_ok else 5.0) + (0 if sv_ok else 5.0)
        if np.isfinite(score) and score < best_score:
            best_score = score
            torch.save({'epoch': ep + 1, 'state': model.state_dict(),
                        'ade': va['ade'], 'ev': va['ev'], 'sv': va['sv']},
                       'best_model.pth')
            print(f"       saved  ADE={va['ade']*POS_SCALE:.2f}m  "
                  f"Err-r={va['ev']:.3f}  Vis-r={va['sv']:.3f}")

    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'\nTraining complete. Best score: {best_score*POS_SCALE:.2f}m')
    print('Saved: best_model.pth  training_history.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UGIN')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic dataset (no nuScenes download required)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epoch count from config.py')
    main(parser.parse_args())
