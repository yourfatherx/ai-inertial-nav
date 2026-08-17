"""Phase 3 eval: full-sequence open-loop attitude drift, raw vs denoised gyro.

The training loss looked at short windows. The real question is: over an ENTIRE
held-out sequence, integrated open-loop from the GT initial orientation, how far
does attitude drift with raw gyro vs the denoised gyro? Lower = the network
removed real bias/noise, not just fit windows.

Usage:
    python scripts/phase3_eval.py                 # val = V1_02_medium
    python scripts/phase3_eval.py --seq V1_01_easy
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import RESULTS_DIR                       # noqa: E402
from ainav.dataset import prepare_sequence                # noqa: E402
from ainav.denoiser import GyroDenoiser                   # noqa: E402
from ainav.integrate import integrate_orientation         # noqa: E402
from ainav.metrics import orientation_error               # noqa: E402
from ainav.rotations import geodesic_angle, quat_to_euler  # noqa: E402


def denoise_full(model, seq, mean, std, device, chunk=4000, pad=256):
    """Run the denoiser over a full sequence in overlapping chunks (the net is
    convolutional, so we pad each chunk and trim the padded edges to avoid
    boundary artefacts). Returns corrected gyro (N,3)."""
    imu = seq.imu                                   # (N,6) raw
    N = len(imu)
    imu_norm = (imu - mean) / std
    corr = np.zeros((N, 3), np.float32)
    model.eval()
    with torch.no_grad():
        start = 0
        while start < N:
            lo = max(0, start - pad)
            hi = min(N, start + chunk + pad)
            x = torch.from_numpy(imu_norm[lo:hi].T[None]).to(device)  # (1,6,L)
            c = model(x)[0].cpu().numpy().T                  # (L,3)
            # place only the valid (unpadded) interior
            a = start - lo
            b = a + min(chunk, N - start)
            corr[start:start + (b - a)] = c[a:b]
            start += chunk
    return imu[:, :3] + corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="V1_02_medium")
    ap.add_argument("--ckpt", default=str(RESULTS_DIR / "phase3_denoiser.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GyroDenoiser(alpha=ck["alpha"]).to(device)
    model.load_state_dict(ck["model"])
    mean, std = ck["mean"], ck["std"]

    seq = prepare_sequence(args.seq)
    t, gyro_raw, quat_gt = seq.t, seq.imu[:, :3], seq.quat
    q0 = quat_gt[0]

    gyro_dn = denoise_full(model, seq, mean, std, device)

    q_raw = integrate_orientation(t, gyro_raw, q0)
    q_dn = integrate_orientation(t, gyro_dn, q0)

    e_raw = orientation_error(q_raw, quat_gt)
    e_dn = orientation_error(q_dn, quat_gt)

    print(f"[seq] {args.seq}  {t[-1]-t[0]:.1f}s open-loop attitude drift")
    print(f"  {'metric':8s} {'raw gyro':>12s} {'denoised':>12s} {'improve':>9s}")
    for k in ("rmse", "mean", "max", "final"):
        imp = 100 * (e_raw[k] - e_dn[k]) / e_raw[k]
        print(f"  {k:8s} {e_raw[k]:11.3f}° {e_dn[k]:11.3f}° {imp:8.1f}%")

    # mean corrected bias removed (rough): mean of (denoised - raw)
    bias_removed = np.degrees(np.mean(gyro_dn - gyro_raw, axis=0))
    print(f"  mean correction (deg/s): "
          f"[{bias_removed[0]:+.3f} {bias_removed[1]:+.3f} {bias_removed[2]:+.3f}]")

    # ---- plot ----
    ang_raw = np.degrees(geodesic_angle(q_raw, quat_gt))
    ang_dn = np.degrees(geodesic_angle(q_dn, quat_gt))
    eul_gt = np.degrees(quat_to_euler(quat_gt))
    eul_raw = np.degrees(quat_to_euler(q_raw))
    eul_dn = np.degrees(quat_to_euler(q_dn))
    names = ["roll", "pitch", "yaw"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(t, ang_raw, label="raw gyro", color="tab:red")
    ax[0, 0].plot(t, ang_dn, label="denoised", color="tab:green")
    ax[0, 0].set_title(f"Open-loop attitude error vs GT — {args.seq}")
    ax[0, 0].set_xlabel("time (s)"); ax[0, 0].set_ylabel("geodesic error (deg)")
    ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

    for i, (a, nm) in enumerate(zip([ax[0, 1], ax[1, 0], ax[1, 1]], names)):
        a.plot(t, eul_gt[:, i], "k-", lw=1.2, label="GT")
        a.plot(t, eul_raw[:, i], color="tab:red", alpha=0.7, label="raw")
        a.plot(t, eul_dn[:, i], color="tab:green", alpha=0.8, label="denoised")
        a.set_title(f"{nm} (deg)"); a.set_xlabel("time (s)")
        a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / f"phase3_eval_{args.seq}.png"
    fig.savefig(out, dpi=110)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
