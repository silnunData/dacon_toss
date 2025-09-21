#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ae_light.py — tuned version
- Stronger defaults (capacity↑, epochs↑)
- Early stopping on validation metric
- LR scheduler (ReduceLROnPlateau)
- Gradient clipping + AMP
- Keeps output schema: per-part parquet with row_idx/ID, z[*], recon_error

Note: This script assumes the same input conventions used in your pipeline:
- dataloader_streaming_full.py provides Iterable DataLoaders that yield dicts:
  { 'row_idx': Tensor, 'ID': Optional[Tensor], 'seq_ids': LongTensor[B, L],
    'seq_len': LongTensor[B], 'tab_x': FloatTensor[B, D_tab], 'y': LongTensor[B] }
- For TARGET_CLASS=c, the loader already streams that class with optional
  negative subsample for c==0.
Adjust the `get_loaders` shim below if your function name differs.
"""

from __future__ import annotations
import os, sys, json, math, time, argparse, random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

# Optional: if pyarrow not available, install before running.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    pa = None
    pq = None

# ----------------------------
# Repro
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Model
# ----------------------------

class SeqEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, h_gru: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.bigru = nn.GRU(d_model, h_gru, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = 2 * h_gru

    def forward(self, seq_ids: torch.Tensor, seq_len: torch.Tensor):
        # seq_ids: [B, L]
        x = self.emb(seq_ids)
        x = self.drop(x)
        # pack for speed
        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.bigru(packed)
        # h: [num_layers*2, B, h_gru] -> concat last layer forward/backward
        h_fw = h[-2]
        h_bw = h[-1]
        rep = torch.cat([h_fw, h_bw], dim=1)
        return rep  # [B, 2*h_gru]

class TabEncoder(nn.Module):
    def __init__(self, in_dim: int, d_tab: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_tab),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = d_tab

    def forward(self, x):
        return self.net(x)

class AE_Light(nn.Module):
    def __init__(self, vocab_size: int, tab_dim: int,
                 d_model: int = 128, h_gru: int = 128,
                 d_tab: int = 128, d_lat: int = 48,
                 dropout: float = 0.15):
        super().__init__()
        self.seq_enc = SeqEncoder(vocab_size, d_model, h_gru, dropout)
        self.tab_enc = TabEncoder(tab_dim, d_tab, dropout)
        fused_dim = self.seq_enc.out_dim + self.tab_enc.out_dim
        self.to_lat = nn.Sequential(
            nn.Linear(fused_dim, d_lat),
            nn.ReLU(),
        )
        # decoder reconstructs tab only (as in original)
        self.dec = nn.Sequential(
            nn.Linear(d_lat, d_tab),
            nn.ReLU(),
            nn.Linear(d_tab, tab_dim),
        )

    def forward(self, seq_ids, seq_len, tab_x):
        s = self.seq_enc(seq_ids, seq_len)   # [B, 2*h_gru]
        t = self.tab_enc(tab_x)              # [B, d_tab]
        h = torch.cat([s, t], dim=1)
        z = self.to_lat(h)                   # [B, d_lat]
        recon = self.dec(z)                  # [B, tab_dim]
        return z, recon

# ----------------------------
# Data loader shim
# ----------------------------

"""You may need to adapt this import depending on your file.
Expected function:
    get_streaming_loaders(split: str, target_class: int, neg_subsample_p: float,
                          batch_max_len: int, batch_size: int, num_workers: int)
Returns (loader, meta) where meta contains {'tab_dim': int, 'vocab_size': int}.
"""
try:
    from dataloader_streaming_full import get_streaming_loaders  # type: ignore
except Exception:
    get_streaming_loaders = None  # will assert later

# ----------------------------
# Training utils
# ----------------------------

@dataclass
class TrainCfg:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model: int = 128
    h_gru: int = 128
    d_tab: int = 128
    d_lat: int = 48
    dropout: float = 0.15
    lr: float = 5e-4
    epochs: int = 20
    clip_norm: float = 1.0
    amp: bool = True
    batch_size: int = 1024
    batch_max_len: int = 512
    num_workers: int = 4
    target_class: int = 0
    neg_subsample_p: float = 1.0  # for c0 only; can set 0.1~0.2 for speed
    valid_batches: int = 100      # take first N batches as a small validation window
    patience: int = 3             # early stop patience on valid metric
    out_dir: str = './ae_out'
    part_rows: int = 200_000      # rows per parquet part
    seed: int = 42


def loss_fn(recon, tab_x):
    return F.mse_loss(recon, tab_x, reduction='none').mean(dim=1)  # per-sample


def save_part(buffer, out_path, part_idx, has_id: bool):
    if pq is None:
        print('[WARN] pyarrow not available. Skipping parquet dump.')
        return
    table = pa.table(buffer)
    fn = os.path.join(out_path, f'part_{part_idx:04d}.parquet')
    pq.write_table(table, fn)
    print(f'[dump] wrote {fn} rows={len(buffer["recon_error"])}')


@torch.no_grad()
def run_epoch_extract(model, loader, device, out_path, part_rows, target_class):
    model.eval()
    buf = {}
    part_idx = 0
    rows_acc = 0
    keys = ['row_idx', 'ID', 'recon_error']
    # z columns will be added dynamically

    for step, batch in enumerate(loader):
        row_idx = batch.get('row_idx')
        ID = batch.get('ID')
        seq_ids = batch['seq_ids'].to(device)
        seq_len = batch['seq_len'].to(device)
        tab_x   = batch['tab_x'].to(device)
        z, recon = model(seq_ids, seq_len, tab_x)
        err = loss_fn(recon, tab_x)  # [B]

        z_np = z.detach().cpu().numpy()
        err_np = err.detach().cpu().numpy()

        # Initialize buffer dict lazily
        if not buf:
            buf['recon_error'] = []
            if row_idx is not None:
                buf['row_idx'] = []
            if ID is not None:
                buf['ID'] = []
            for i in range(z_np.shape[1]):
                buf[f'z_{i}'] = []

        # Append
        if row_idx is not None:
            buf['row_idx'].extend(batch['row_idx'].tolist())
        if ID is not None:
            buf['ID'].extend(batch['ID'].tolist())
        buf['recon_error'].extend(err_np.tolist())
        for i in range(z_np.shape[1]):
            buf[f'z_{i}'].extend(z_np[:, i].tolist())

        rows_acc += len(err_np)

        # dump
        if rows_acc >= part_rows:
            save_part(buf, out_path, part_idx, has_id=(ID is not None))
            part_idx += 1
            buf = {}
            rows_acc = 0

    # tail
    if buf:
        save_part(buf, out_path, part_idx, has_id=('ID' in buf))


def split_for_valid(loader, take_batches: int):
    """Split the iterable loader into two iterables: small valid, then train.
    Implementation wraps the original iterator. If your loader can be re-created
    cheaply per epoch, prefer that route. Here we take the first N batches for valid.
    """
    it = iter(loader)
    valid_batches = []
    for _ in range(take_batches):
        try:
            valid_batches.append(next(it))
        except StopIteration:
            break

    def valid_iter():
        for b in valid_batches:
            yield b

    def train_iter():
        for b in it:
            yield b

    return valid_iter, train_iter


def evaluate(model, loader, device) -> Tuple[float, float]:
    """Return (mean_recon_err, mean_gap_metric) on validation slice.
    gap metric approximates class-specific fitness by combining error mean and z norm.
    """
    model.eval()
    errs = []
    gaps = []
    with torch.no_grad():
        for batch in loader():
            seq_ids = batch['seq_ids'].to(device)
            seq_len = batch['seq_len'].to(device)
            tab_x   = batch['tab_x'].to(device)
            z, recon = model(seq_ids, seq_len, tab_x)
            err = loss_fn(recon, tab_x)  # [B]
            errs.append(err.mean().item())
            gaps.append((err.mean() + 0.01 * z.pow(2).sum(dim=1).mean()).item())
    if not errs:
        return float('inf'), float('inf')
    return float(np.mean(errs)), float(np.mean(gaps))


def train_one(model, train_iter, valid_iter, cfg: TrainCfg, tag: str):
    device = cfg.device
    model.to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = GradScaler(enabled=cfg.amp)

    best_metric = float('inf')
    best_state = None
    patience_left = cfg.patience

    print(f'[train] start tag={tag} epochs={cfg.epochs} device={device}')

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        n_batches = 0
        n_samples = 0
        epoch_loss = 0.0

        for batch in train_iter():
            seq_ids = batch['seq_ids'].to(device)
            seq_len = batch['seq_len'].to(device)
            tab_x   = batch['tab_x'].to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):
                z, recon = model(seq_ids, seq_len, tab_x)
                err = loss_fn(recon, tab_x)  # [B]
                loss = err.mean()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            scaler.step(opt)
            scaler.update()

            bs = tab_x.size(0)
            n_batches += 1
            n_samples += bs
            epoch_loss += loss.item() * bs
            if n_batches % 50 == 0:
                print(f'[ep {ep}] step {n_batches} loss {loss.item():.6f} lr {opt.param_groups[0]["lr"]:.2e}')

        # validation
        val_err, val_gap = evaluate(model, valid_iter, device)
        sched.step(val_gap)

        dur = time.time() - t0
        mean_train = epoch_loss / max(1, n_samples)
        print(f'[ep {ep}] train_loss={mean_train:.6f} val_err={val_err:.6f} val_gap={val_gap:.6f} secs={dur:.1f}')

        if val_gap + 1e-7 < best_metric:
            best_metric = val_gap
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
            print(f'[ep {ep}] best updated. metric={best_metric:.6f}')
        else:
            patience_left -= 1
            print(f'[ep {ep}] no improve. patience_left={patience_left}')
            if patience_left <= 0:
                print('[early-stop] patience exhausted')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_class', type=int, default=0, choices=[0,1])
    ap.add_argument('--neg_subsample_p', type=float, default=1.0)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--batch_max_len', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--h_gru', type=int, default=128)
    ap.add_argument('--d_tab', type=int, default=128)
    ap.add_argument('--d_lat', type=int, default=48)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--clip_norm', type=float, default=1.0)
    ap.add_argument('--valid_batches', type=int, default=100)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--out_dir', type=str, default='./ae_out')
    ap.add_argument('--part_rows', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    cfg = TrainCfg(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        d_model=args.d_model,
        h_gru=args.h_gru,
        d_tab=args.d_tab,
        d_lat=args.d_lat,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        clip_norm=args.clip_norm,
        amp=True,
        batch_size=args.batch_size,
        batch_max_len=args.batch_max_len,
        num_workers=args.num_workers,
        target_class=args.target_class,
        neg_subsample_p=args.neg_subsample_p,
        valid_batches=args.valid_batches,
        out_dir=args.out_dir,
        part_rows=args.part_rows,
        seed=args.seed,
    )

    set_seed(cfg.seed)

    assert get_streaming_loaders is not None, 'Import get_streaming_loaders from dataloader_streaming_full failed. Please adapt the shim.'

    os.makedirs(cfg.out_dir, exist_ok=True)

    # Build loaders
    print('[loaders] building streaming loaders...')
    full_loader, meta = get_streaming_loaders(
        split='train',
        target_class=cfg.target_class,
        neg_subsample_p=(cfg.neg_subsample_p if cfg.target_class == 0 else 1.0),
        batch_max_len=cfg.batch_max_len,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    vocab_size = int(meta['vocab_size'])
    tab_dim = int(meta['tab_dim'])
    print(f'[meta] vocab_size={vocab_size} tab_dim={tab_dim}')

    # Split a small validation window
    valid_iter, train_iter = split_for_valid(full_loader, cfg.valid_batches)

    # Model
    model = AE_Light(
        vocab_size=vocab_size,
        tab_dim=tab_dim,
        d_model=cfg.d_model,
        h_gru=cfg.h_gru,
        d_tab=cfg.d_tab,
        d_lat=cfg.d_lat,
        dropout=cfg.dropout,
    )

    # Train
    tag = f'c{cfg.target_class}'
    model = train_one(model, train_iter, valid_iter, cfg, tag)

    # Extract representations on full stream (train split) and dump parts
    out_train = os.path.join(cfg.out_dir, f'train_ae_{tag}')
    os.makedirs(out_train, exist_ok=True)
    print(f'[extract/train] dumping to {out_train}')
    # Rebuild a fresh full loader for exhaustive pass
    full_loader2, _ = get_streaming_loaders(
        split='train',
        target_class=cfg.target_class,
        neg_subsample_p=1.0,  # dump must cover all rows of the class
        batch_max_len=cfg.batch_max_len,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    run_epoch_extract(model, full_loader2, cfg.device, out_train, cfg.part_rows, cfg.target_class)

    # Test stream
    test_loader, _ = get_streaming_loaders(
        split='test',
        target_class=cfg.target_class,
        neg_subsample_p=1.0,
        batch_max_len=cfg.batch_max_len,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    out_test = os.path.join(cfg.out_dir, f'test_ae_{tag}')
    os.makedirs(out_test, exist_ok=True)
    print(f'[extract/test] dumping to {out_test}')
    run_epoch_extract(model, test_loader, cfg.device, out_test, cfg.part_rows, cfg.target_class)

    print('[done]')


if __name__ == '__main__':
    main()
