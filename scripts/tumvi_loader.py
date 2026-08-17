"""TUM-VI room-sequence loader, returning the same Sequence contract as
ainav.euroc.load_sequence so the existing ESKF / denoiser eval code is reused
verbatim.

Why a separate loader (TUM-VI differs from EuRoC in three ways):
  1. Ground truth lives in mav0/mocap0/data.csv, not
     state_groundtruth_estimate0/, and is POSE-ONLY:
         timestamp[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z
     -> no velocity, no gyro/accel bias columns (EuRoC has all three).
  2. Mocap runs at 120 Hz (EuRoC 200 Hz); IMU is 200 Hz for both.
  3. Ground-truth pose is already in the IMU frame (EuRoC GT is body->world with
     a body-IMU extrinsic) -- which actually makes the orientation comparison
     cleaner here, no extrinsic to apply.

Consequences we handle:
  * vel: synthesized by finite-difference of mocap position (only v0 is used to
    seed the ESKF, so a coarse estimate is fine).
  * bias_gyro / bias_accel: filled with zeros. TUM-VI ships no ground-truth
    bias, so there is NO oracle baseline on this dataset -- the honest test is
    raw-ESKF(bg0=0) vs denoised-ESKF(bg0=0), which is exactly the cross-IMU
    question we want to answer.

The IMU here is a Bosch BMI160-class MEMS unit, NOT EuRoC's ADIS16448 -- that
sensor change is the whole point of the cross-IMU test.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reuse the exact dataclasses the rest of the pipeline expects.
from ainav.euroc import ImuData, GroundTruth, Sequence      # noqa: E402

TUMVI_DIR = ROOT / "data" / "tumvi"


def load_tumvi(name: str = "room1") -> Sequence:
    seq_dir = TUMVI_DIR / name / "mav0"
    imu_csv = seq_dir / "imu0" / "data.csv"
    gt_csv = seq_dir / "mocap0" / "data.csv"
    for p in (imu_csv, gt_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run scripts/fetch_tumvi_csv.py {name} first.")

    imu_raw = np.loadtxt(imu_csv, delimiter=",", skiprows=1)
    gt_raw = np.loadtxt(gt_csv, delimiter=",", skiprows=1)

    # TUM-VI IMU csv: ts[ns], w_x, w_y, w_z, a_x, a_y, a_z  (same as EuRoC)
    if imu_raw.shape[1] < 7:
        raise ValueError(f"IMU csv has {imu_raw.shape[1]} cols, expected >=7")
    # TUM-VI mocap csv: ts[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z  (pose only)
    if gt_raw.shape[1] < 8:
        raise ValueError(f"mocap csv has {gt_raw.shape[1]} cols, expected 8 "
                         f"(ts,pos,quat) -- got a different layout")

    # Common zero reference = first IMU timestamp. ns -> s.
    t0 = imu_raw[0, 0]
    imu = ImuData(
        t=(imu_raw[:, 0] - t0) * 1e-9,
        gyro=imu_raw[:, 1:4].copy(),
        accel=imu_raw[:, 4:7].copy(),
    )

    gt_t = (gt_raw[:, 0] - t0) * 1e-9
    gt_pos = gt_raw[:, 1:4].copy()
    gt_quat = gt_raw[:, 4:8].copy()          # [w,x,y,z]

    # Synthesize velocity from position (central difference; endpoints one-sided).
    # Only v0 seeds the filter, so precision here is non-critical.
    gt_vel = np.gradient(gt_pos, gt_t, axis=0)

    gt = GroundTruth(
        t=gt_t,
        pos=gt_pos,
        quat=gt_quat,
        vel=gt_vel,
        bias_gyro=np.zeros_like(gt_pos),     # no GT bias in TUM-VI
        bias_accel=np.zeros_like(gt_pos),
    )
    return Sequence(name=f"tumvi_{name}", imu=imu, gt=gt)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "room1"
    seq = load_tumvi(name)
    print(f"Loaded {seq.name}")
    print(f"  IMU:  {len(seq.imu.t):>7d} samples, {seq.imu.t[-1]:.1f}s, "
          f"~{1/np.mean(np.diff(seq.imu.t)):.0f} Hz")
    print(f"  GT :  {len(seq.gt.t):>7d} samples, {seq.gt.t[-1]:.1f}s, "
          f"~{1/np.mean(np.diff(seq.gt.t)):.0f} Hz")
    print(f"  gyro range rad/s: [{seq.imu.gyro.min():.2f}, {seq.imu.gyro.max():.2f}]")
    print(f"  |accel| mean (near-static?): {np.linalg.norm(seq.imu.accel,axis=1).mean():.2f} m/s^2")
    print(f"  pos span (m): {np.ptp(seq.gt.pos,axis=0)}")
