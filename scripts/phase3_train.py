"""Phase 3: train the gyro denoiser on EuRoC sequences.

Train on one set of sequences, validate on a held-out one (cross-sequence, the
honest test). Loss = multi-horizon relative-orientation error between the
orientation integrated from the CORRECTED gyro and GT orientation.

Usage:
    python scripts/phase3_train.py                       # defaults below
    python scripts/phase3_train.py --train V1_01_easy --val V1_02_medium
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import RESULTS_DIR                       # noqa: E402
from ainav.dataset import prepare_sequence, ImuWindowDataset  # noqa: E402
from ainav.denoiser import GyroDenoiser                   # noqa: E402
from ainav.train_loss import integration_loss             # noqa: E402


def build_loaders(train_names, val_names, window, stride, batch):
    train_seqs = [prepare_sequence(n) for n in train_names]
    val_seqs = [prepare_sequence(n) for n in val_names]
    train_ds = ImuWindowDataset(train_seqs, window=window, stride=stride)
    # val uses the TRAIN normalization stats (no peeking at val distribution)
    val_ds = ImuWindowDataset(val_seqs, window=window, stride=window,
                              normalize=False)
    val_ds.mean, val_ds.std = train_ds.mean, train_ds.std
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)
    return train_ds, train_dl, val_dl


def run_epoch(model, dl, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for batch in dl:
        imu_norm = batch["imu_norm"].to(device)     # (B,6,W)
        imu_raw = batch["imu_raw"].to(device)        # (B,W,6)
        quat = batch["quat"].to(device)             # (B,W,4)
        dt = batch["dt"].to(device)                 # (B,W,1)
        gyro_raw = imu_raw[:, :, :3]                 # (B,W,3)
        corr = model(imu_norm).transpose(1, 2)      # (B,W,3)
        gyro_corr = gyro_raw + corr
        loss = integration_loss(gyro_corr, quat, dt)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(loss) * imu_norm.size(0)
        n += imu_norm.size(0)
    return tot / max(n, 1)


def raw_baseline_loss(dl, device):
    """Loss with ZERO correction -- how good is raw gyro already? Gives the
    number the network must beat on the val set."""
    tot, n = 0.0, 0
    for batch in dl:
        gyro_raw = batch["imu_raw"][:, :, :3].to(device)
        quat = batch["quat"].to(device)
        dt = batch["dt"].to(device)
        loss = integration_loss(gyro_raw, quat, dt)
        tot += float(loss) * gyro_raw.size(0)
        n += gyro_raw.size(0)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["V1_01_easy"])
    ap.add_argument("--val", nargs="+", default=["V1_02_medium"])
    ap.add_argument("--window", type=int, default=400)   # 2 s @200Hz
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="checkpoint path (default results/phase3_denoiser.pt)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[device] {device}")
    print(f"[data] train={args.train}  val={args.val}")

    train_ds, train_dl, val_dl = build_loaders(
        args.train, args.val, args.window, args.stride, args.batch)
    print(f"[data] {len(train_ds)} train windows, "
          f"{len(val_dl.dataset)} val windows")

    model = GyroDenoiser(alpha=args.alpha).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] GyroDenoiser, {nparam} params, alpha={args.alpha}")

    base = raw_baseline_loss(val_dl, device)
    print(f"[base] raw-gyro val loss (deg): {np.degrees(base):.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = float("inf")
    ckpt = Path(args.out) if args.out else RESULTS_DIR / "phase3_denoiser.pt"
    hist = []
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(model, train_dl, device, opt)
        with torch.no_grad():
            va = run_epoch(model, val_dl, device, None)
        sched.step()
        hist.append({"epoch": ep, "train_deg": np.degrees(tr),
                     "val_deg": np.degrees(va)})
        tag = ""
        if va < best:
            best = va
            torch.save({"model": model.state_dict(),
                        "mean": train_ds.mean, "std": train_ds.std,
                        "alpha": args.alpha, "window": args.window},
                       ckpt)
            tag = "  *"
        print(f"[ep {ep:3d}] train {np.degrees(tr):.4f}  "
              f"val {np.degrees(va):.4f} deg{tag}")

    improve = 100.0 * (base - best) / base
    print(f"\n[result] raw {np.degrees(base):.4f} deg  ->  "
          f"denoised {np.degrees(best):.4f} deg  "
          f"({improve:+.1f}% relative-orientation error)")
    print(f"[ckpt] {ckpt}")
    (ckpt.with_suffix(".history.json")).write_text(json.dumps(
        {"base_deg": np.degrees(base), "best_deg": np.degrees(best),
         "history": hist}, indent=2))


if __name__ == "__main__":
    main()
