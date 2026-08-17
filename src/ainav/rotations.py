"""Minimal quaternion / rotation helpers. Quaternions are [w,x,y,z]."""
from __future__ import annotations
import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b. Supports broadcasting over leading dims."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(...,4) -> (...,3,3) rotation matrix (body->world)."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - z*w)
    R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] = 2*(x*y + z*w)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] = 2*(x*z - y*w)
    R[..., 2, 1] = 2*(y*z + x*w)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R


def rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    """Rotation vector (axis*angle), (...,3) -> quaternion (...,4)."""
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    small = angle < 1e-8
    axis = np.where(small, 0.0, rv / np.where(small, 1.0, angle))
    half = angle / 2.0
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """[w,x,y,z] -> roll,pitch,yaw (rad), (...,3)."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.stack([roll, pitch, yaw], axis=-1)


def geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Smallest angle (rad) between two orientations. (...,4),(...,4)->(...)."""
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v]_x so that [v]_x @ u == cross(v, u). (3,)->(3,3)."""
    x, y, z = v
    return np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])
