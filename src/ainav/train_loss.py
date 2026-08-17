"""Orientation-integration loss for the gyro denoiser.

We have no clean per-sample gyro ground truth. What we DO trust is GT
orientation. So the supervision is:

    for a set of horizons h (in samples), take the relative rotation the
    corrected gyro produces over h steps, and compare it to the relative
    rotation GT underwent over the same h steps. Penalize the geodesic angle.

Using several horizons (e.g. 1, 8, 32, 128 samples) makes the net fix both
fast noise (short h) and slow bias (long h) -- a single long horizon alone
lets high-frequency noise average out and go uncorrected.
"""
from __future__ import annotations
import torch

from .torch_rot import quat_mul, quat_conj, geodesic_angle, integrate_gyro


def relative_rotations(quat: torch.Tensor, horizon: int) -> torch.Tensor:
    """Relative rotation q_t^{-1} ⊗ q_{t+h} for a quaternion series.

    quat: (B, T, 4) -> (B, T-h, 4)
    """
    q0 = quat[:, :-horizon, :]
    qh = quat[:, horizon:, :]
    return quat_mul(quat_conj(q0), qh)


def integration_loss(gyro_corr: torch.Tensor, quat_gt: torch.Tensor,
                     dt: torch.Tensor,
                     horizons=(1, 8, 32, 128)) -> torch.Tensor:
    """Multi-horizon relative-orientation loss.

    gyro_corr: (B, T, 3) corrected body rates (rad/s)
    quat_gt:   (B, T, 4) GT orientation at each sample
    dt:        (B, T, 1) per-step dt
    returns scalar mean geodesic error (rad) averaged over horizons.
    """
    B, T, _ = gyro_corr.shape
    # integrated orientation from corrected gyro, starting at GT q at t=0
    q0 = quat_gt[:, 0, :]
    q_int = integrate_gyro(gyro_corr, dt, q0)          # (B, T+1, 4)
    q_int = q_int[:, :T, :]                             # align to samples 0..T-1

    total = gyro_corr.new_zeros(())
    for h in horizons:
        if h >= T:
            continue
        rel_int = relative_rotations(q_int, h)         # (B,T-h,4)
        rel_gt = relative_rotations(quat_gt, h)        # (B,T-h,4)
        # geodesic angle between the two relative rotations
        err = geodesic_angle(rel_int, rel_gt)          # (B,T-h)
        total = total + err.mean()
    return total / len(horizons)
