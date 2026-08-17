"""Differentiable quaternion helpers for training the gyro denoiser.

Mirrors the numpy conventions in ainav.rotations: quaternions are [w,x,y,z],
Hamilton product, body->world. Everything here is torch + autograd friendly so
we can backprop an orientation-integration loss into the network.
"""
from __future__ import annotations
import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a ⊗ b, broadcasting over leading dims. (...,4)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def rotvec_to_quat(rv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rotation vector (axis*angle) (...,3) -> quaternion (...,4)."""
    angle = rv.norm(dim=-1, keepdim=True)                    # (...,1)
    small = angle < eps
    safe = torch.where(small, torch.ones_like(angle), angle)
    axis = rv / safe
    half = angle / 2.0
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    # for very small angles sin(half)/... -> axis*half is fine via the branch
    xyz = torch.where(small, rv * 0.5, xyz)
    return torch.cat([w, xyz], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def geodesic_angle(q1: torch.Tensor, q2: torch.Tensor,
                   eps: float = 1e-7) -> torch.Tensor:
    """Smallest angle (rad) between orientations. (...,4),(...,4) -> (...)."""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = (q1 * q2).sum(-1).abs().clamp(max=1.0 - eps)
    return 2.0 * torch.arccos(dot)


def cumulative_quat(dq: torch.Tensor) -> torch.Tensor:
    """Inclusive cumulative Hamilton product along dim=1.

    dq: (B, T, 4) per-step rotation increments.
    returns P: (B, T, 4) with P[:, k] = dq_0 ⊗ dq_1 ⊗ ... ⊗ dq_k.

    Quaternion product is associative, so the prefix product is computed with a
    Hillis-Steele parallel scan in ceil(log2 T) vectorized steps instead of T
    sequential ones. The op is non-commutative, so the earlier index stays on
    the LEFT: new[t] = old[t-shift] ⊗ old[t].
    """
    T = dq.shape[1]
    P = quat_normalize(dq)
    shift = 1
    while shift < T:
        left = P[:, :-shift, :]                        # old[t-shift], t>=shift
        combined = quat_mul(left, P[:, shift:, :])      # old[t-shift] ⊗ old[t]
        P = torch.cat([P[:, :shift, :], combined], dim=1)
        P = quat_normalize(P)                           # curb fp norm drift
        shift *= 2
    return P


def integrate_gyro(gyro: torch.Tensor, dt: torch.Tensor,
                   q0: torch.Tensor) -> torch.Tensor:
    """Integrate a body-rate gyro sequence into orientation quaternions.

    gyro: (B, T, 3) rad/s
    dt:   scalar or (B,T,1) seconds
    q0:   (B, 4) initial orientation [w,x,y,z]
    returns (B, T+1, 4): q0 followed by q after each step.

    Vectorized: build all per-step increments dq at once, take their cumulative
    product via a parallel scan, then left-apply q0. q0 factors out of the prefix
    product (q_after[k] = q0 ⊗ dq_0 ⊗ ... ⊗ dq_k), so no sequential loop is
    needed.
    """
    B, T, _ = gyro.shape
    if not torch.is_tensor(dt):
        dt = gyro.new_tensor(dt)
    if dt.dim() == 0:
        dt = dt.view(1, 1, 1).expand(B, T, 1)
    q0 = quat_normalize(q0)
    dq = rotvec_to_quat(gyro * dt)                       # (B,T,4)
    P = cumulative_quat(dq)                              # (B,T,4)
    q_after = quat_normalize(quat_mul(q0.unsqueeze(1), P))   # (B,T,4)
    return torch.cat([q0.unsqueeze(1), q_after], dim=1)     # (B,T+1,4)
