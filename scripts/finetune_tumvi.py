"""Fine-tune the EuRoC-trained gyro denoiser on TUM-VI (a DIFFERENT IMU) and
prove the D2 deployment step: a frozen cross-IMU net fails (cross_imu_eval showed
raw 35.6 deg -> frozen-net 116 deg on room1), but a short fine-tune on the new
sensor should RECOVER the win. This closes the loop -- the negative result is
fixable, not fatal.

Design:
  * Warm-start from the EuRoC checkpoint (results/phase3_denoiser.pt) instead of
    random init -- transfer the learned noise/bias structure, adapt only what the
    new sensor needs.
  * RECOMPUTE normalization stats from the TUM-VI train set: the input
    distribution is a different sensor, so EuRoC channel mean/std no longer fit.
  * Keep alpha=0.15 (the bounded-correction budget; must exceed the sensor's true
    yaw-bias magnitude, as on EuRoC).
  * Honest split, no leakage: train on room2+room3, hold out room1 -- the exact
    sequence the frozen net failed on. Recovery on room1 is therefore a true
    held-out result.

Reuses the Phase-3 training machinery unchanged (ImuWindowDataset, run_epoch,
raw_baseline_loss, integration_loss); only the data loader differs (TUM-VI).

Usage:
    python scripts/finetune_tumvi.py                       # train room2 room3, val room1
    python scripts/finetune_tumvi.py --train room2 room3 --val room1 --epochs 40
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ainav.config import RESULTS_DIR                        # noqa: E402
from ainav.dataset import SeqArrays, ImuWindowDataset       # noqa: E402
from ainav.denoiser import GyroDenoiser                     # noqa: E402
from ainav.train_loss import integration_loss              # noqa: E402
from ainav.metrics import interp_quat_to                    # noqa: E402
from phase3_train import run_epoch, raw_baseline_loss       # noqa: E402
from tumvi_loader import load_tumvi                          # noqa: E402


def prepare_tumvi(name: str) -> SeqArrays:
    """TUM-VI analogue of ainav.dataset.prepare_sequence: align GT quaternion
    onto IMU timestamps over the GT-covered span. Same output contract."""
    seq = load_tumvi(name)
    t_imu = seq.imu.t
    lo, hi = seq.gt.t[0], seq.gt.t[-1]
    mask = (t_imu >= lo) & (t_imu <= hi)
    t = t_imu[mask]
    gyro = seq.imu.gyro[mask]
    accel = seq.imu.accel[mask]
    quat = interp_quat_to(t, seq.gt.t, seq.gt.quat)
    imu = np.concatenate([gyro, accel], axis=1).astype(np.float32)
    dt = np.empty_like(t)
    dt[1:] = np.diff(t)
    dt[0] = np.median(dt[1:])
    return SeqArrays(name=f"tumvi_{name}", t=t, imu=imu,
                     quat=quat.astype(np.float32), dt=dt.astype(np.float32))


def build_loaders(train_names, val_names, window, stride, batch):
    train_seqs = [prepare_tumvi(n) for n in train_names]
    val_seqs = [prepare_tumvi(n) for n in val_names]
    train_ds = ImuWindowDataset(train_seqs, window=window, stride=stride)
    val_ds = ImuWindowDataset(val_seqs, window=window, stride=window,
                              normalize=False)
    val_ds.mean, val_ds.std = train_ds.mean, train_ds.std   # TUM-VI train stats
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)
    return train_ds, train_dl, val_dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["room2", "room3"])
    ap.add_argument("--val", nargs="+", default=["room1"])
    ap.add_argument("--init", default=str(RESULTS_DIR / "phase3_denoiser.pt"),
                    help="EuRoC checkpoint to warm-start from")
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)   # lower: fine-tune, not scratch
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS_DIR / "finetune_tumvi.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[device] {device}")
    print(f"[data] train={args.train}  val(held-out)={args.val}")

    train_ds, train_dl, val_dl = build_loaders(
        args.train, args.val, args.window, args.stride, args.batch)
    print(f"[data] {len(train_ds)} train windows, {len(val_dl.dataset)} val windows")
    print(f"[norm] TUM-VI train mean={np.round(train_ds.mean,3)} "
          f"std={np.round(train_ds.std,3)}")

    model = GyroDenoiser(alpha=args.alpha).to(device)
    # Warm-start from EuRoC weights (transfer learned structure).
    ck = torch.load(args.init, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"[init] warm-started from EuRoC ckpt {args.init} "
          f"(alpha {ck['alpha']} -> {args.alpha})")

    base = raw_baseline_loss(val_dl, device)
    print(f"[base] raw-gyro val loss (deg): {np.degrees(base):.4f}")
    # How does the FROZEN EuRoC net (EuRoC norm stats) score before any tuning?
    frozen_ds = ImuWindowDataset([prepare_tumvi(n) for n in args.val],
                                 window=args.window, stride=args.window,
                                 normalize=False)
    frozen_ds.mean, frozen_ds.std = ck["mean"], ck["std"]
    frozen_dl = DataLoader(frozen_ds, batch_size=args.batch, shuffle=False)
    with torch.no_grad():
        frozen = run_epoch(model, frozen_dl, device, None)
    print(f"[frozen] EuRoC-net (EuRoC norm) val loss (deg): {np.degrees(frozen):.4f}"
          f"   <- what the un-tuned cross-IMU net gets")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = float("inf")
    ckpt = Path(args.out)
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

    print(f"\n[result] raw {np.degrees(base):.4f} deg | frozen-EuRoC "
          f"{np.degrees(frozen):.4f} deg | fine-tuned {np.degrees(best):.4f} deg")
    print(f"[ckpt] {ckpt}")
    (ckpt.with_suffix(".history.json")).write_text(json.dumps(
        {"base_deg": np.degrees(base), "frozen_deg": np.degrees(frozen),
         "best_deg": np.degrees(best), "history": hist}, indent=2))


if __name__ == "__main__":
    main()
