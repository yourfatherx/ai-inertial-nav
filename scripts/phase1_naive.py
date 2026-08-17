"""Phase 1: naive open-loop integration of raw IMU -> show the drift.

Starts from the ground-truth initial state (orientation, velocity, position)
so the drift shown is purely from IMU error accumulation, not init error.

Outputs:
  results/phase1_<seq>.png   trajectory + orientation-error-over-time
  prints ATE and orientation error.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.euroc import load_sequence            # noqa: E402
from ainav.integrate import integrate_full        # noqa: E402
from ainav.metrics import (interp_to, interp_quat_to,   # noqa: E402
                           position_ate, orientation_error)
from ainav.config import RESULTS_DIR              # noqa: E402


def main(name: str = "V1_01_easy") -> None:
    seq = load_sequence(name)
    imu, gt = seq.imu, seq.gt

    # restrict IMU to the ground-truth time span (GT often starts a bit later)
    mask = (imu.t >= gt.t[0]) & (imu.t <= gt.t[-1])
    t = imu.t[mask]; gyro = imu.gyro[mask]; accel = imu.accel[mask]

    # initial state from GT (interpolated to first IMU time)
    q0 = interp_quat_to(t[:1], gt.t, gt.quat)[0]
    v0 = interp_to(t[:1], gt.t, gt.vel)[0]
    p0 = interp_to(t[:1], gt.t, gt.pos)[0]

    print(f"[{name}] integrating {len(t)} samples over {t[-1]-t[0]:.1f}s ...")
    est = integrate_full(t, gyro, accel, q0, v0, p0)

    # ground truth resampled to IMU times
    p_gt = interp_to(t, gt.t, gt.pos)
    q_gt = interp_quat_to(t, gt.t, gt.quat)

    ate = position_ate(est["p"], p_gt)
    oe = orientation_error(est["q"], q_gt)
    print(f"  position ATE  rmse={ate['rmse']:.2f} m  max={ate['max']:.2f} m")
    print(f"  orient error  rmse={oe['rmse']:.2f}°  final={oe['final']:.2f}°")

    # plots
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(p_gt[:, 0], p_gt[:, 1], "k-", lw=1.2, label="ground truth")
    ax1.plot(est["p"][:, 0], est["p"][:, 1], "r-", lw=1.0, label="naive INS")
    ax1.scatter(*p_gt[0, :2], c="g", s=40, zorder=5, label="start")
    ax1.set_title("Trajectory (top-down x-y)")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.axis("equal"); ax1.legend()

    from ainav.rotations import geodesic_angle
    ang = np.degrees(geodesic_angle(est["q"], q_gt))
    perr = np.linalg.norm(est["p"] - p_gt, axis=1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t - t[0], ang, "b-", lw=0.8, label="orientation err [°]")
    ax2b = ax2.twinx()
    ax2b.plot(t - t[0], perr, "r-", lw=0.8, label="position err [m]")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("orientation error [°]", color="b")
    ax2b.set_ylabel("position error [m]", color="r")
    ax2.set_title("Drift grows without bound (open loop)")

    fig.suptitle(f"Phase 1 — naive raw-IMU dead reckoning ({name})", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / f"phase1_{name}.png"
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "V1_01_easy")
