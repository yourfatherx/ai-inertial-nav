"""Harden Approach B (1): multi-seed error bars.

Train the gyro denoiser from N different random seeds, then evaluate every
resulting checkpoint through the SAME Phase-4 ESKF fusion on the held-out
sequences. Report mean +/- std of the denoised-filter metrics so we can say the
improvement is stable across initialisations, not a lucky seed.

The raw baseline (bg0=0) and the oracle (bg0=GT) do not depend on the network,
so they are computed once per sequence and shown for reference.

Usage:
    python scripts/harden_multiseed.py                       # seeds 0..4
    python scripts/harden_multiseed.py --seeds 0 1 2 --epochs 40
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ainav.config import RESULTS_DIR                          # noqa: E402
from ainav.euroc import load_sequence                         # noqa: E402
from ainav.eskf import run_eskf                               # noqa: E402
from ainav.denoiser import GyroDenoiser                       # noqa: E402
from ainav.metrics import (interp_to, interp_quat_to,         # noqa: E402
                           position_ate, orientation_error)

from phase3_train import build_loaders, run_epoch, raw_baseline_loss  # noqa: E402
from phase4_fusion import denoise_full, yaw_error_deg         # noqa: E402

TRAIN = ["MH_01_easy", "MH_03_medium", "V1_01_easy",
         "V1_03_difficult", "V2_01_easy", "V2_02_medium"]
HELDOUT = ["V1_02_medium", "MH_02_easy", "V2_03_difficult"]
TUNE = dict(accel_gate=0.05, tilt_meas_std=0.5)
ALPHA = 0.15


def train_seed(seed, epochs, window, stride, batch, lr, device,
               train_seqs, val_seq, tag="multiseed"):
    """Train one denoiser from a given seed. Returns (ckpt_path, best_val_deg)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds, train_dl, val_dl = build_loaders(
        train_seqs, [val_seq], window, stride, batch)  # val only steers ckpt pick
    model = GyroDenoiser(alpha=ALPHA).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        run_epoch(model, train_dl, device, opt)
        with torch.no_grad():
            va = run_epoch(model, val_dl, device, None)
        sched.step()
        if va < best:
            best = va
            best_state = {"model": {k: v.clone() for k, v in
                                    model.state_dict().items()},
                          "mean": train_ds.mean, "std": train_ds.std,
                          "alpha": ALPHA, "window": window}
    out = RESULTS_DIR / "seeds" / f"{tag}_seed_{seed}.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save(best_state, out)
    print(f"[seed {seed}] best val {np.degrees(best):.4f} deg -> {out.name}")
    return out, float(np.degrees(best))


def eval_seq(ckpt_path, seq_name, device, want_baseline):
    """Run the three-config ESKF fusion for one ckpt on one held-out seq.
    Returns denoised metrics (+ baseline/oracle when want_baseline)."""
    seq = load_sequence(seq_name)
    imu, gt = seq.imu, seq.gt
    mask = (imu.t >= gt.t[0]) & (imu.t <= gt.t[-1])
    t = imu.t[mask]
    gyro_raw = imu.gyro[mask]
    accel = imu.accel[mask]
    imu6 = np.concatenate([gyro_raw, accel], axis=1).astype(np.float32)

    q0 = interp_quat_to(t[:1], gt.t, gt.quat)[0]
    v0 = interp_to(t[:1], gt.t, gt.vel)[0]
    p0 = interp_to(t[:1], gt.t, gt.pos)[0]
    bg0_gt = interp_to(t[:1], gt.t, gt.bias_gyro)[0]
    ba0_gt = interp_to(t[:1], gt.t, gt.bias_accel)[0]
    p_gt = interp_to(t, gt.t, gt.pos)
    q_gt = interp_quat_to(t, gt.t, gt.quat)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GyroDenoiser(alpha=ck["alpha"]).to(device)
    model.load_state_dict(ck["model"])
    gyro_dn = denoise_full(model, imu6, ck["mean"], ck["std"], device)

    def metrics(est):
        oe = orientation_error(est["q"], q_gt)
        _, ye = yaw_error_deg(est["q"], q_gt)
        ate = position_ate(est["p"], p_gt)
        return {"orient_rmse": oe["rmse"], "orient_final": oe["final"],
                "yaw_rmse": ye["rmse"], "yaw_final": ye["final"],
                "ate_rmse": ate["rmse"], "ate_max": ate["max"]}

    dn = run_eskf(t, gyro_dn, accel, q0, v0, p0,
                  bg0=np.zeros(3), ba0=ba0_gt, **TUNE)
    out = {"denoised": metrics(dn)}
    if want_baseline:
        raw = run_eskf(t, gyro_raw, accel, q0, v0, p0,
                       bg0=np.zeros(3), ba0=ba0_gt, **TUNE)
        orc = run_eskf(t, gyro_raw, accel, q0, v0, p0,
                       bg0=bg0_gt, ba0=ba0_gt, **TUNE)
        out["raw_baseline"] = metrics(raw)
        out["oracle"] = metrics(orc)
    return out


def agg(vals):
    a = np.asarray(vals, float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
            "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train", nargs="+", default=TRAIN)
    ap.add_argument("--heldout", nargs="+", default=HELDOUT)
    ap.add_argument("--tag", default="multiseed",
                    help="output name: results/harden_<tag>.json + seeds/<tag>_*")
    args = ap.parse_args()

    train_seqs, heldout = args.train, args.heldout
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    print(f"[seeds] {args.seeds}  epochs={args.epochs}  tag={args.tag}")
    print(f"[train] {train_seqs}")
    print(f"[heldout] {heldout}\n")

    # --- train every seed ---
    ckpts, best_vals = [], []
    for s in args.seeds:
        ck, bv = train_seed(s, args.epochs, args.window, args.stride,
                            args.batch, args.lr, device,
                            train_seqs, heldout[0], tag=args.tag)
        ckpts.append(ck)
        best_vals.append(bv)

    # --- evaluate every seed on every held-out seq ---
    # per seq: list of denoised-metric dicts (one per seed) + baseline/oracle once
    per_seq = {sq: {"denoised": [], "raw_baseline": None, "oracle": None}
               for sq in heldout}
    for i, ck in enumerate(ckpts):
        for sq in heldout:
            r = eval_seq(ck, sq, device, want_baseline=(i == 0))
            per_seq[sq]["denoised"].append(r["denoised"])
            if i == 0:
                per_seq[sq]["raw_baseline"] = r["raw_baseline"]
                per_seq[sq]["oracle"] = r["oracle"]
        print(f"[eval] seed ckpt {ck.name} done "
              f"({i+1}/{len(ckpts)})")

    # --- aggregate ---
    KEYS = ["orient_rmse", "yaw_rmse", "ate_rmse"]
    summary = {"seeds": args.seeds, "best_val_deg": agg(best_vals),
               "sequences": {}}
    print("\n=== MULTI-SEED SUMMARY (denoised ESKF, bg0=0) ===")
    print(f"best val (deg): {agg(best_vals)['mean']:.4f} "
          f"+/- {agg(best_vals)['std']:.4f}\n")
    for sq in heldout:
        d = per_seq[sq]["denoised"]
        aggd = {k: agg([m[k] for m in d]) for k in KEYS}
        base = per_seq[sq]["raw_baseline"]
        orc = per_seq[sq]["oracle"]
        summary["sequences"][sq] = {"denoised": aggd, "raw_baseline": base,
                                    "oracle": orc, "n_seeds": len(d)}
        print(f"[{sq}]")
        for k in KEYS:
            unit = "m" if "ate" in k else "deg"
            print(f"  {k:12s} denoised {aggd[k]['mean']:8.3f} "
                  f"+/- {aggd[k]['std']:6.3f} {unit}   "
                  f"| raw {base[k]:8.3f}  oracle {orc[k]:8.3f}")
        print()

    out = RESULTS_DIR / f"harden_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[json] {out}")


if __name__ == "__main__":
    main()
