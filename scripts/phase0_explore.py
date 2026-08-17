"""Phase 0 deliverable: load a sequence and plot raw IMU vs. ground truth.

Produces results/phase0_<seq>.png with:
  - raw gyro (3 axes) over time
  - raw accel (3 axes) over time
  - ground-truth 3D trajectory
  - ground-truth orientation (euler) over time

Confirms the data pipeline end-to-end before we build anything on it.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.euroc import load_sequence          # noqa: E402
from ainav.rotations import quat_to_euler       # noqa: E402
from ainav.config import RESULTS_DIR            # noqa: E402


def main(name: str = "MH_01_easy") -> None:
    seq = load_sequence(name)
    imu, gt = seq.imu, seq.gt

    print(f"Loaded {seq.name}: {len(imu.t)} IMU samples over {imu.t[-1]:.1f}s")

    fig = plt.figure(figsize=(14, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    for i, c in zip(range(3), "xyz"):
        ax1.plot(imu.t, imu.gyro[:, i], lw=0.4, label=f"ω_{c}")
    ax1.set_title("Raw gyroscope"); ax1.set_ylabel("rad/s")
    ax1.set_xlabel("t [s]"); ax1.legend(loc="upper right", fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    for i, c in zip(range(3), "xyz"):
        ax2.plot(imu.t, imu.accel[:, i], lw=0.4, label=f"a_{c}")
    ax2.set_title("Raw accelerometer"); ax2.set_ylabel("m/s²")
    ax2.set_xlabel("t [s]"); ax2.legend(loc="upper right", fontsize=8)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3.plot(gt.pos[:, 0], gt.pos[:, 1], gt.pos[:, 2], lw=0.8)
    ax3.scatter(*gt.pos[0], c="g", s=30, label="start")
    ax3.scatter(*gt.pos[-1], c="r", s=30, label="end")
    ax3.set_title("Ground-truth trajectory")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)
    eul = np.degrees(quat_to_euler(gt.quat))
    for i, c in enumerate(["roll", "pitch", "yaw"]):
        ax4.plot(gt.t, eul[:, i], lw=0.6, label=c)
    ax4.set_title("Ground-truth orientation"); ax4.set_ylabel("deg")
    ax4.set_xlabel("t [s]"); ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"EuRoC {seq.name} — Phase 0 data check", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / f"phase0_{seq.name}.png"
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")

    # quick stats to sanity-check accel ~ gravity at rest-ish start
    a0 = np.linalg.norm(imu.accel[:200].mean(axis=0))
    print(f"|accel| over first second (should be ~9.81 if near-static): {a0:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MH_01_easy")
