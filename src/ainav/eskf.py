"""Error-State Kalman Filter (ESKF) for IMU dead-reckoning.

Phase 2 baseline: a *real* filter (no AI yet) that keeps drift bounded far
better than the naive integration of Phase 1.

Design (canonical Sola convention, "local"/body error angle):

  Nominal state (integrated normally, lives on the manifold):
      q  (4,)  orientation, body->world, [w,x,y,z]
      v  (3,)  velocity in world
      p  (3,)  position in world
      bg (3,)  gyro bias
      ba (3,)  accel bias

  Error state (15,) -- the KF actually estimates THIS small deviation, which
  lives in a flat tangent space where ordinary linear-Gaussian algebra is valid:
      [ dtheta(0:3), dv(3:6), dp(6:9), dbg(9:12), dba(12:15) ]

  Two operations per step:
    predict(gyro, accel, dt)  -- mechanize nominal forward + propagate cov P.
    update_gravity(accel)     -- when near-non-accelerating, the measured
                                 specific force points along -gravity, giving an
                                 absolute roll/pitch (tilt) reference. This is
                                 the observation that stops orientation (and
                                 hence gravity-leak) from running away.

Yaw about gravity stays UNOBSERVABLE with IMU only -- expected. The gyro
denoiser (Phase 3+) shrinks the yaw/attitude drift rate in the predict step.
"""
from __future__ import annotations
import numpy as np

from .rotations import (quat_mul, quat_normalize, quat_to_rot,
                        rotvec_to_quat, skew)
from . import config

_G_WORLD = np.array([0.0, 0.0, -config.GRAVITY])   # gravity vector, world frame

# error-state index slices
I_TH = slice(0, 3)
I_V = slice(3, 6)
I_P = slice(6, 9)
I_BG = slice(9, 12)
I_BA = slice(12, 15)


class ESKF:
    def __init__(self, q0, v0, p0, bg0=None, ba0=None,
                 gyro_noise=config.GYRO_NOISE_DENSITY,
                 accel_noise=config.ACCEL_NOISE_DENSITY,
                 gyro_walk=config.GYRO_RANDOM_WALK,
                 accel_walk=config.ACCEL_RANDOM_WALK,
                 tilt_meas_std=0.5,          # m/s^2, accel-as-gravity noise
                 accel_gate=0.5,             # |a|-g gate for applying tilt update
                 p0_std=(1e-3, 1e-3, 1e-2, 1e-3, 1e-2)):
        # nominal state
        self.q = quat_normalize(np.asarray(q0, float))
        self.v = np.asarray(v0, float).copy()
        self.p = np.asarray(p0, float).copy()
        self.bg = np.zeros(3) if bg0 is None else np.asarray(bg0, float).copy()
        self.ba = np.zeros(3) if ba0 is None else np.asarray(ba0, float).copy()

        # noise densities
        self.ng2 = gyro_noise ** 2
        self.na2 = accel_noise ** 2
        self.nbg2 = gyro_walk ** 2
        self.nba2 = accel_walk ** 2
        self.tilt_var = tilt_meas_std ** 2
        self.accel_gate = accel_gate

        # error-state covariance P (15x15). Start small: we trust GT init.
        s_th, s_v, s_p, s_bg, s_ba = p0_std
        d = np.concatenate([np.full(3, s_th**2), np.full(3, s_v**2),
                            np.full(3, s_p**2), np.full(3, s_bg**2),
                            np.full(3, s_ba**2)])
        self.P = np.diag(d)

    # ---- prediction -------------------------------------------------------
    def predict(self, gyro, accel, dt):
        """Advance nominal state by one IMU sample and propagate covariance."""
        w = np.asarray(gyro, float) - self.bg     # bias-corrected body rate
        a = np.asarray(accel, float) - self.ba    # bias-corrected specific force
        R = quat_to_rot(self.q)                    # body->world
        a_world = R @ a + _G_WORLD

        # --- nominal integration (same mechanization as Phase 1) ---
        self.p = self.p + self.v * dt + 0.5 * a_world * dt * dt
        self.v = self.v + a_world * dt
        self.q = quat_normalize(quat_mul(self.q, rotvec_to_quat(w * dt)))

        # --- error-state transition F (continuous), then discretize ---
        F = np.zeros((15, 15))
        F[I_TH, I_TH] = -skew(w)          # dtheta_dot = -[w]x dtheta - dbg
        F[I_TH, I_BG] = -np.eye(3)
        F[I_V, I_TH] = -R @ skew(a)       # dv_dot = -R[a]x dtheta - R dba
        F[I_V, I_BA] = -R
        F[I_P, I_V] = np.eye(3)           # dp_dot = dv
        Phi = np.eye(15) + F * dt         # 1st-order discretization (fine @200Hz)

        # --- discrete process noise Q (variance accrued over dt) ---
        Q = np.zeros((15, 15))
        Q[I_TH, I_TH] = np.eye(3) * self.ng2 * dt
        Q[I_V, I_V] = np.eye(3) * self.na2 * dt
        Q[I_BG, I_BG] = np.eye(3) * self.nbg2 * dt
        Q[I_BA, I_BA] = np.eye(3) * self.nba2 * dt

        self.P = Phi @ self.P @ Phi.T + Q

    # ---- gravity / tilt measurement update --------------------------------
    def update_gravity(self, accel):
        """Use accel as a gravity-direction reference (roll/pitch + accel bias).

        Only applied when the platform is close to non-accelerating, i.e.
        |accel| ~ g; otherwise the "accel == -gravity" assumption is false.
        Returns True if the update was applied.
        """
        accel = np.asarray(accel, float)
        if abs(np.linalg.norm(accel) - config.GRAVITY) > self.accel_gate:
            return False

        R = quat_to_rot(self.q)
        # predicted specific force at rest: h = R^T(-g_world) + ba
        h = R.T @ (-_G_WORLD) + self.ba
        y = accel - h                             # innovation (3,)

        # measurement Jacobian H (3x15) wrt error state
        H = np.zeros((3, 15))
        H[:, I_TH] = skew(R.T @ (-_G_WORLD))      # d h / d dtheta (local error)
        H[:, I_BA] = np.eye(3)                     # d h / d dba
        Rm = np.eye(3) * self.tilt_var

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)        # Kalman gain (15x3)
        dx = K @ y                                 # error-state correction

        self._inject(dx)
        # covariance update (Joseph form for numerical symmetry)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        return True

    # ---- inject error into nominal, then reset ----------------------------
    def _inject(self, dx):
        dth = dx[I_TH]
        self.q = quat_normalize(quat_mul(self.q, rotvec_to_quat(dth)))  # local
        self.v += dx[I_V]
        self.p += dx[I_P]
        self.bg += dx[I_BG]
        self.ba += dx[I_BA]
        # error state is now folded in -> reset to zero (implicit; P kept)

    # convenience snapshot
    def state(self):
        return {"q": self.q.copy(), "v": self.v.copy(), "p": self.p.copy(),
                "bg": self.bg.copy(), "ba": self.ba.copy()}


def run_eskf(t, gyro, accel, q0, v0, p0, bg0=None, ba0=None,
             tilt_every=1, **kw):
    """Run the ESKF over a full IMU sequence. Returns dict of (N,·) arrays.

    tilt_every: apply the gravity update every k-th sample (1 = every sample).
    """
    N = len(t)
    f = ESKF(q0, v0, p0, bg0, ba0, **kw)
    q = np.empty((N, 4)); v = np.empty((N, 3)); p = np.empty((N, 3))
    bg = np.empty((N, 3)); ba = np.empty((N, 3))
    q[0] = f.q; v[0] = f.v; p[0] = f.p; bg[0] = f.bg; ba[0] = f.ba
    n_upd = 0
    for k in range(1, N):
        dt = t[k] - t[k - 1]
        f.predict(gyro[k], accel[k], dt)
        if tilt_every and (k % tilt_every == 0):
            n_upd += f.update_gravity(accel[k])
        s = f.state()
        q[k] = s["q"]; v[k] = s["v"]; p[k] = s["p"]
        bg[k] = s["bg"]; ba[k] = s["ba"]
    return {"q": q, "v": v, "p": p, "bg": bg, "ba": ba, "n_updates": n_upd}
