"""EuRoC MAV dataset loader (ASL format).

Directory layout of one sequence (e.g. MH_01_easy):
    <seq>/mav0/imu0/data.csv
    <seq>/mav0/state_groundtruth_estimate0/data.csv

IMU csv columns:
    timestamp[ns], w_x, w_y, w_z (rad/s), a_x, a_y, a_z (m/s^2)
Ground-truth csv columns:
    timestamp[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z,
    b_w_x, b_w_y, b_w_z, b_a_x, b_a_y, b_a_z
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import DATA_DIR


@dataclass
class ImuData:
    t: np.ndarray        # (N,)  seconds, zero-based
    gyro: np.ndarray     # (N,3) rad/s
    accel: np.ndarray    # (N,3) m/s^2


@dataclass
class GroundTruth:
    t: np.ndarray        # (M,)  seconds, same zero as ImuData
    pos: np.ndarray      # (M,3) m, world frame
    quat: np.ndarray     # (M,4) [w,x,y,z], body->world
    vel: np.ndarray      # (M,3) m/s, world frame
    bias_gyro: np.ndarray  # (M,3)
    bias_accel: np.ndarray # (M,3)


@dataclass
class Sequence:
    name: str
    imu: ImuData
    gt: GroundTruth


def _seq_dir(name: str) -> Path:
    d = DATA_DIR / name
    if (d / "mav0").exists():
        return d / "mav0"
    # some extractions nest one deeper: <name>/<name>/mav0
    alt = d / name / "mav0"
    if alt.exists():
        return alt
    raise FileNotFoundError(
        f"Could not find mav0/ under {d}. Did you download & extract {name}?"
    )


def load_sequence(name: str = "MH_01_easy") -> Sequence:
    mav0 = _seq_dir(name)

    imu_csv = mav0 / "imu0" / "data.csv"
    gt_csv = mav0 / "state_groundtruth_estimate0" / "data.csv"
    for p in (imu_csv, gt_csv):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    imu_raw = np.loadtxt(imu_csv, delimiter=",", skiprows=1)
    gt_raw = np.loadtxt(gt_csv, delimiter=",", skiprows=1)

    # Common zero reference = first IMU timestamp. ns -> s.
    t0 = imu_raw[0, 0]
    imu = ImuData(
        t=(imu_raw[:, 0] - t0) * 1e-9,
        gyro=imu_raw[:, 1:4].copy(),
        accel=imu_raw[:, 4:7].copy(),
    )
    gt = GroundTruth(
        t=(gt_raw[:, 0] - t0) * 1e-9,
        pos=gt_raw[:, 1:4].copy(),
        quat=gt_raw[:, 4:8].copy(),      # already [w,x,y,z] in EuRoC
        vel=gt_raw[:, 8:11].copy(),
        bias_gyro=gt_raw[:, 11:14].copy(),
        bias_accel=gt_raw[:, 14:17].copy(),
    )
    return Sequence(name=name, imu=imu, gt=gt)


if __name__ == "__main__":
    seq = load_sequence()
    print(f"Loaded {seq.name}")
    print(f"  IMU:  {len(seq.imu.t):>7d} samples, "
          f"{seq.imu.t[-1]:.1f}s, ~{1/np.mean(np.diff(seq.imu.t)):.0f} Hz")
    print(f"  GT :  {len(seq.gt.t):>7d} samples, "
          f"{seq.gt.t[-1]:.1f}s, ~{1/np.mean(np.diff(seq.gt.t)):.0f} Hz")
    print(f"  gyro range rad/s: [{seq.imu.gyro.min():.2f}, {seq.imu.gyro.max():.2f}]")
    print(f"  accel range m/s2: [{seq.imu.accel.min():.2f}, {seq.imu.accel.max():.2f}]")
