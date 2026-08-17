"""Trajectory / orientation error metrics + time alignment helpers."""
from __future__ import annotations
import numpy as np

from .rotations import geodesic_angle, quat_normalize


def interp_to(t_ref: np.ndarray, t_src: np.ndarray,
              x_src: np.ndarray) -> np.ndarray:
    """Linearly interpolate vector series x_src(t_src) onto t_ref. (M,D)."""
    out = np.empty((len(t_ref), x_src.shape[1]))
    for d in range(x_src.shape[1]):
        out[:, d] = np.interp(t_ref, t_src, x_src[:, d])
    return out


def interp_quat_to(t_ref: np.ndarray, t_src: np.ndarray,
                   q_src: np.ndarray) -> np.ndarray:
    """Nearest-neighbour quaternion resampling onto t_ref (good enough at 200Hz)."""
    idx = np.searchsorted(t_src, t_ref)
    idx = np.clip(idx, 0, len(t_src) - 1)
    return quat_normalize(q_src[idx])


def position_ate(p_est: np.ndarray, p_gt: np.ndarray) -> dict:
    """Absolute Trajectory Error (position). Assumes same timestamps.

    Returns rmse, mean, max (metres). No SE3 alignment — we start from GT
    initial state, so raw error is meaningful.
    """
    e = np.linalg.norm(p_est - p_gt, axis=1)
    return {"rmse": float(np.sqrt(np.mean(e**2))),
            "mean": float(np.mean(e)),
            "max": float(np.max(e))}


def orientation_error(q_est: np.ndarray, q_gt: np.ndarray) -> dict:
    """Geodesic orientation error in degrees. Assumes same timestamps."""
    ang = np.degrees(geodesic_angle(q_est, q_gt))
    return {"rmse": float(np.sqrt(np.mean(ang**2))),
            "mean": float(np.mean(ang)),
            "max": float(np.max(ang)),
            "final": float(ang[-1])}
