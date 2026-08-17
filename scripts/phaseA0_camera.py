"""Phase A0 deliverable: load EuRoC camera stream, verify IMU sync + undistortion.

Produces results/phaseA0_<seq>.png with:
  - a raw frame and its undistorted version (calibration sanity)
  - camera timestamps overlaid on the IMU clock (sync check)
  - inter-frame dt histogram (confirms ~20 Hz, no dropped frames)

This is the classical data check before any VIO backend goes in.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.euroc import load_sequence                       # noqa: E402
from ainav.camera import load_camera, read_gray, undistort  # noqa: E402
from ainav.config import RESULTS_DIR, DATA_DIR                # noqa: E402


def main(name: str = "V1_01_easy") -> None:
    seq = load_sequence(name)
    imu = seq.imu

    # Share the IMU's zero reference so camera & IMU are on one clock.
    imu_csv = DATA_DIR / name / "mav0" / "imu0" / "data.csv"
    t0_ns = int(np.loadtxt(imu_csv, delimiter=",", skiprows=1,
                           usecols=0, dtype=np.int64, max_rows=1))
    cam = load_camera(name, "cam0", t0_ns=t0_ns)

    dt = np.diff(cam.t)
    print(f"Loaded {name}: {len(cam.files)} cam0 frames over {cam.t[-1]:.1f}s")
    print(f"  camera rate     : ~{1/np.mean(dt):.1f} Hz "
          f"(nominal {cam.calib.rate_hz:.0f})")
    print(f"  IMU  rate       : ~{1/np.mean(np.diff(imu.t)):.0f} Hz")
    print(f"  cam t range     : [{cam.t[0]:.3f}, {cam.t[-1]:.3f}] s")
    print(f"  IMU t range     : [{imu.t[0]:.3f}, {imu.t[-1]:.3f}] s")
    print(f"  frame dt        : min {dt.min()*1e3:.1f} ms, "
          f"max {dt.max()*1e3:.1f} ms, std {dt.std()*1e3:.2f} ms")
    print(f"  intrinsics fu,fv,cu,cv: "
          f"{cam.calib.K[0,0]:.1f}, {cam.calib.K[1,1]:.1f}, "
          f"{cam.calib.K[0,2]:.1f}, {cam.calib.K[1,2]:.1f}")
    print(f"  distortion k1k2p1p2   : {cam.calib.dist}")

    # every camera frame should sit inside the IMU time span
    assert cam.t[0] >= imu.t[0] - 1e-6, "camera starts before IMU"
    assert cam.t[-1] <= imu.t[-1] + 1e-6, "camera ends after IMU"
    print("  [ok] camera timeline nested inside IMU timeline")

    mid = len(cam.files) // 2
    raw = read_gray(cam.files[mid])
    und = undistort(raw, cam.calib)

    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(raw, cmap="gray"); ax1.set_title(f"Raw frame #{mid}")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(und, cmap="gray"); ax2.set_title("Undistorted (radtan removed)")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(imu.t, np.zeros_like(imu.t), "|", ms=4, alpha=0.15,
             label="IMU 200Hz")
    ax3.plot(cam.t, np.ones_like(cam.t), "|", ms=8, color="C1",
             label="cam0 20Hz")
    ax3.set_ylim(-0.5, 1.5); ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["IMU", "cam"]); ax3.set_xlabel("t [s]")
    ax3.set_title("Shared clock — camera nested in IMU")
    ax3.legend(loc="upper right", fontsize=8)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(dt * 1e3, bins=40, color="C1")
    ax4.axvline(1e3 / cam.calib.rate_hz, color="k", ls="--",
                label=f"{cam.calib.rate_hz:.0f}Hz nominal")
    ax4.set_xlabel("inter-frame dt [ms]"); ax4.set_ylabel("count")
    ax4.set_title("Frame timing"); ax4.legend(fontsize=8)

    fig.suptitle(f"EuRoC {name} — Phase A0 camera/IMU data check", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / f"phaseA0_{name}.png"
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "V1_01_easy")
