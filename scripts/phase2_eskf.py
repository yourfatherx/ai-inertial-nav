"""Phase 2: ESKF baseline (no AI) vs. the naive integration of Phase 1.

Runs both filters from the same GT initial state and reports position ATE +
orientation error, then plots trajectories and drift-over-time side by side.

Outputs:
  results/phase2_<seq>.png
  prints a comparison table.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.euroc import load_sequence                    # noqa: E402
from ainav.integrate import integrate_full                # noqa: E402
from ainav.eskf import run_eskf                            # noqa: E402
from ainav.rotations import geodesic_angle                 # noqa: E402
from ainav.metrics import (interp_to, interp_quat_to,      # noqa: E402
                           position_ate, orientation_error)
from ainav.config import RESULTS_DIR                        # noqa: E402


def main(name: str = "V1_01_easy") -> None:
    seq = load_sequence(name)
    imu, gt = seq.imu, seq.gt

    mask = (imu.t >= gt.t[0]) & (imu.t <= gt.t[-1])
    t = imu.t[mask]; gyro = imu.gyro[mask]; accel = imu.accel[mask]

    q0 = interp_quat_to(t[:1], gt.t, gt.quat)[0]
    v0 = interp_to(t[:1], gt.t, gt.vel)[0]
    p0 = interp_to(t[:1], gt.t, gt.pos)[0]
    bg0 = interp_to(t[:1], gt.t, gt.bias_gyro)[0]
    ba0 = interp_to(t[:1], gt.t, gt.bias_accel)[0]

    p_gt = interp_to(t, gt.t, gt.pos)
    q_gt = interp_quat_to(t, gt.t, gt.quat)

    print(f"[{name}] {len(t)} samples over {t[-1]-t[0]:.1f}s")

    # --- Phase 1 naive (for reference) ---
    naive = integrate_full(t, gyro, accel, q0, v0, p0)

    # --- Phase 2 ESKF (start bias from GT too, fair to naive which had none) ---
    eskf = run_eskf(t, gyro, accel, q0, v0, p0, bg0=bg0, ba0=ba0)

    rows = []
    for label, est in [("naive (Phase 1)", naive), ("ESKF (Phase 2)", eskf)]:
        ate = position_ate(est["p"], p_gt)
        oe = orientation_error(est["q"], q_gt)
        rows.append((label, ate, oe))
        print(f"  {label:18s}  ATE rmse={ate['rmse']:10.2f} m  max={ate['max']:10.2f} m"
              f"  |  orient rmse={oe['rmse']:6.2f}°  final={oe['final']:6.2f}°")
    print(f"  ESKF tilt updates applied: {eskf['n_updates']} / {len(t)}")

    # --- plots ---
    tt = t - t[0]
    fig = plt.figure(figsize=(15, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(p_gt[:, 0], p_gt[:, 1], "k-", lw=1.4, label="ground truth")
    ax1.plot(eskf["p"][:, 0], eskf["p"][:, 1], "b-", lw=1.0, label="ESKF")
    ax1.scatter(*p_gt[0, :2], c="g", s=40, zorder=5, label="start")
    ax1.set_title("Trajectory top-down (ESKF vs GT)")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.axis("equal"); ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(tt, np.degrees(geodesic_angle(naive["q"], q_gt)), "r-", lw=0.7,
             label="naive")
    ax2.plot(tt, np.degrees(geodesic_angle(eskf["q"], q_gt)), "b-", lw=0.9,
             label="ESKF")
    ax2.set_title("Orientation error over time")
    ax2.set_xlabel("t [s]"); ax2.set_ylabel("geodesic error [°]")
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(tt, np.linalg.norm(eskf["p"] - p_gt, axis=1), "b-", lw=0.9,
             label="ESKF")
    ax3.set_title("ESKF position error over time")
    ax3.set_xlabel("t [s]"); ax3.set_ylabel("position error [m]")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(tt, eskf["bg"], lw=0.8)
    ax4.plot(tt, gt.bias_gyro[np.searchsorted(gt.t, t)], "k--", lw=0.6, alpha=0.5)
    ax4.set_title("Estimated gyro bias (solid) vs GT (dashed)")
    ax4.set_xlabel("t [s]"); ax4.set_ylabel("bias [rad/s]")

    fig.suptitle(f"Phase 2 — ESKF baseline ({name})", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / f"phase2_{name}.png"
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "V1_01_easy")
