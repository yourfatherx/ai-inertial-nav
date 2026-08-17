"""Cross-IMU test: run the EuRoC-trained gyro denoiser on a DIFFERENT physical
IMU (TUM-VI room sequences, Bosch BMI160-class MEMS) through the same ESKF.

This is the sensor-shift question the deployment track (PDF section D2) raises:
the denoiser learned the systematic error of EuRoC's ADIS16448 -- most visibly
its ~+4.34 deg/s constant gyro yaw bias, which it subtracts with a bounded,
zero-init correction. A different IMU has a DIFFERENT bias. So the honest, stated
expectation is:

    the fixed learned correction may not match the new sensor's bias, so the
    denoised result may only partially beat -- or fail to beat -- the raw ESKF.

That negative/partial outcome is itself the finding: it quantifies how
sensor-specific the learned correction is, and motivates per-IMU fine-tuning.
We report it straight, whichever way it lands.

No oracle here: TUM-VI ships no ground-truth IMU bias, so the only honest
comparison is raw-ESKF(bg0=0) vs denoised-ESKF(bg0=0) -- both estimating bias
online from zero, the realistic GPS-denied case. Same filter tuning as Phase 4.

Usage:
    python scripts/cross_imu_eval.py                 # room1
    python scripts/cross_imu_eval.py --seq room2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ainav.config import RESULTS_DIR                          # noqa: E402
from ainav.eskf import run_eskf                               # noqa: E402
from ainav.denoiser import GyroDenoiser                       # noqa: E402
from ainav.metrics import (interp_to, interp_quat_to,         # noqa: E402
                           position_ate, orientation_error)
from ainav.rotations import geodesic_angle, quat_to_euler     # noqa: E402
# Reuse the exact denoise + yaw-error routines Phase 4 uses.
from phase4_fusion import denoise_full, yaw_error_deg         # noqa: E402
from tumvi_loader import load_tumvi                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="room1", help="TUM-VI room sequence")
    ap.add_argument("--ckpt", default=str(RESULTS_DIR / "phase3_denoiser.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data + GT-aligned initial state (same masking as Phase 4) ---
    seq = load_tumvi(args.seq)
    imu, gt = seq.imu, seq.gt
    mask = (imu.t >= gt.t[0]) & (imu.t <= gt.t[-1])
    t = imu.t[mask]
    gyro_raw = imu.gyro[mask]
    accel = imu.accel[mask]
    imu6 = np.concatenate([gyro_raw, accel], axis=1).astype(np.float32)

    q0 = interp_quat_to(t[:1], gt.t, gt.quat)[0]
    v0 = interp_to(t[:1], gt.t, gt.vel)[0]
    p0 = interp_to(t[:1], gt.t, gt.pos)[0]
    # No GT bias on TUM-VI -> both filters start from zero bias (honest).
    ba0 = np.zeros(3)

    p_gt = interp_to(t, gt.t, gt.pos)
    q_gt = interp_quat_to(t, gt.t, gt.quat)

    print(f"[{seq.name}] {len(t)} samples over {t[-1]-t[0]:.1f}s  "
          f"(cross-IMU: denoiser trained on EuRoC ADIS16448, tested on this IMU)")
    print(f"[imu] |accel| mean {np.linalg.norm(accel,axis=1).mean():.3f} m/s^2, "
          f"gyro |mean| {np.degrees(np.abs(gyro_raw.mean(0)))} deg/s")

    # --- denoise the full gyro track with the EuRoC-trained net ---
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GyroDenoiser(alpha=ck["alpha"]).to(device)
    model.load_state_dict(ck["model"])
    gyro_dn = denoise_full(model, imu6, ck["mean"], ck["std"], device)
    dbias = np.degrees(np.mean(gyro_dn - gyro_raw, axis=0))
    print(f"[denoiser] mean gyro correction (deg/s): "
          f"[{dbias[0]:+.3f} {dbias[1]:+.3f} {dbias[2]:+.3f}]  "
          f"(learned EuRoC z-bias was ~-4.5; does it fit THIS sensor?)")

    # --- two ESKF runs, both bg0=0 (no oracle possible) ---
    TUNE = dict(accel_gate=0.05, tilt_meas_std=0.5)
    runs = {
        "raw ESKF (bg0=0)":      run_eskf(t, gyro_raw, accel, q0, v0, p0,
                                          bg0=np.zeros(3), ba0=ba0, **TUNE),
        "denoised ESKF (bg0=0)": run_eskf(t, gyro_dn, accel, q0, v0, p0,
                                          bg0=np.zeros(3), ba0=ba0, **TUNE),
    }

    # --- metrics table ---
    print(f"\n  {'config':24s} {'orient rmse':>11s} {'orient fin':>10s} "
          f"{'yaw rmse':>9s} {'yaw fin':>8s} {'ATE rmse':>9s} {'ATE max':>9s}")
    stats = {}
    for label, est in runs.items():
        oe = orientation_error(est["q"], q_gt)
        _, ye = yaw_error_deg(est["q"], q_gt)
        ate = position_ate(est["p"], p_gt)
        stats[label] = (oe, ye, ate)
        print(f"  {label:24s} {oe['rmse']:10.3f}° {oe['final']:9.3f}° "
              f"{ye['rmse']:8.3f}° {ye['final']:7.3f}° "
              f"{ate['rmse']:8.2f}m {ate['max']:8.2f}m")

    base = stats["raw ESKF (bg0=0)"]
    dn = stats["denoised ESKF (bg0=0)"]
    def imp(a, b):
        return 100.0 * (a - b) / a if a else 0.0
    print(f"\n  denoised vs raw baseline (both bg0=0): "
          f"orient rmse {imp(base[0]['rmse'], dn[0]['rmse']):+.1f}%, "
          f"yaw rmse {imp(base[1]['rmse'], dn[1]['rmse']):+.1f}%, "
          f"ATE rmse {imp(base[2]['rmse'], dn[2]['rmse']):+.1f}%")
    verdict = ("denoiser HELPS on this IMU" if dn[0]['rmse'] < base[0]['rmse']
               else "denoiser does NOT beat raw on this IMU (expected: learned "
                    "EuRoC bias doesn't fit this sensor)")
    print(f"  VERDICT: {verdict}")

    # --- plot (2x2), raw vs denoised only ---
    tt = t - t[0]
    raw = runs["raw ESKF (bg0=0)"]
    dnr = runs["denoised ESKF (bg0=0)"]
    yaw_gt_d = np.degrees(np.unwrap(quat_to_euler(q_gt)[:, 2]))
    yaw_raw_d = np.degrees(np.unwrap(quat_to_euler(raw["q"])[:, 2]))
    yaw_dn_d = np.degrees(np.unwrap(quat_to_euler(dnr["q"])[:, 2]))

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    ax[0, 0].plot(p_gt[:, 0], p_gt[:, 1], "k-", lw=1.4, label="ground truth")
    ax[0, 0].plot(raw["p"][:, 0], raw["p"][:, 1], color="tab:red", lw=1.0,
                  label="raw ESKF")
    ax[0, 0].plot(dnr["p"][:, 0], dnr["p"][:, 1], color="tab:green", lw=1.0,
                  label="denoised ESKF")
    ax[0, 0].scatter(*p_gt[0, :2], c="b", s=40, zorder=5, label="start")
    ax[0, 0].set_title(f"Trajectory top-down — {seq.name}")
    ax[0, 0].set_xlabel("x [m]"); ax[0, 0].set_ylabel("y [m]")
    ax[0, 0].axis("equal"); ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    ax[0, 1].plot(tt, np.degrees(geodesic_angle(raw["q"], q_gt)),
                  color="tab:red", lw=0.9, label="raw ESKF")
    ax[0, 1].plot(tt, np.degrees(geodesic_angle(dnr["q"], q_gt)),
                  color="tab:green", lw=0.9, label="denoised ESKF")
    ax[0, 1].set_title("Orientation error vs GT")
    ax[0, 1].set_xlabel("t [s]"); ax[0, 1].set_ylabel("geodesic error [°]")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=0.3)

    ax[1, 0].plot(tt, yaw_gt_d, "k-", lw=1.2, label="GT")
    ax[1, 0].plot(tt, yaw_raw_d, color="tab:red", lw=0.9, label="raw ESKF")
    ax[1, 0].plot(tt, yaw_dn_d, color="tab:green", lw=0.9, label="denoised ESKF")
    ax[1, 0].set_title("Yaw / heading (the unobservable axis)")
    ax[1, 0].set_xlabel("t [s]"); ax[1, 0].set_ylabel("yaw [°]")
    ax[1, 0].legend(); ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(tt, np.linalg.norm(raw["p"] - p_gt, axis=1),
                  color="tab:red", lw=0.9, label="raw ESKF")
    ax[1, 1].plot(tt, np.linalg.norm(dnr["p"] - p_gt, axis=1),
                  color="tab:green", lw=0.9, label="denoised ESKF")
    ax[1, 1].set_title("Position error over time")
    ax[1, 1].set_xlabel("t [s]"); ax[1, 1].set_ylabel("position error [m]")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Cross-IMU — EuRoC-trained denoiser + ESKF on {seq.name}",
                 fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / f"cross_imu_{args.seq}.png"
    fig.savefig(out, dpi=120)
    print(f"\n[plot] {out}")


if __name__ == "__main__":
    main()
