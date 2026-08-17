"""Open-loop strapdown integration of raw IMU (no filtering).

This is the Phase 1 baseline: it shows how fast raw-IMU dead reckoning drifts,
motivating both the EKF (Phase 2) and the learned corrector (Phase 3+).

Orientation: integrate gyro as successive small rotations.
Velocity/Position: rotate accel to world, subtract gravity, double-integrate.
"""
from __future__ import annotations
import numpy as np

from .rotations import quat_mul, rotvec_to_quat, quat_normalize, quat_to_rot


def integrate_orientation(t: np.ndarray, gyro: np.ndarray,
                          q0: np.ndarray) -> np.ndarray:
    """Integrate body-rate gyro into orientation quaternions.

    t:    (N,) seconds
    gyro: (N,3) rad/s (body frame)
    q0:   (4,) initial orientation [w,x,y,z] (body->world)
    returns (N,4) quaternions.
    """
    N = len(t)
    q = np.empty((N, 4))
    q[0] = quat_normalize(q0)
    for k in range(1, N):
        dt = t[k] - t[k - 1]
        # rotation increment from body rate over dt
        dq = rotvec_to_quat(gyro[k - 1] * dt)
        q[k] = quat_normalize(quat_mul(q[k - 1], dq))  # body-frame compose
    return q


def integrate_full(t: np.ndarray, gyro: np.ndarray, accel: np.ndarray,
                   q0: np.ndarray, v0: np.ndarray, p0: np.ndarray,
                   gravity: float = 9.81) -> dict:
    """Full open-loop INS mechanization (orientation + velocity + position).

    Gravity vector assumed [0,0,-g] in world frame (EuRoC world z-up).
    returns dict with keys q (N,4), v (N,3), p (N,3).
    """
    N = len(t)
    q = np.empty((N, 4)); v = np.empty((N, 3)); p = np.empty((N, 3))
    q[0] = quat_normalize(q0); v[0] = v0; p[0] = p0
    g_world = np.array([0.0, 0.0, -gravity])

    for k in range(1, N):
        dt = t[k] - t[k - 1]
        # orientation update
        dq = rotvec_to_quat(gyro[k - 1] * dt)
        q[k] = quat_normalize(quat_mul(q[k - 1], dq))
        # specific force -> world, remove gravity
        R = quat_to_rot(q[k - 1])
        a_world = R @ accel[k - 1] + g_world
        # integrate
        v[k] = v[k - 1] + a_world * dt
        p[k] = p[k - 1] + v[k - 1] * dt + 0.5 * a_world * dt * dt
    return {"q": q, "v": v, "p": p}
