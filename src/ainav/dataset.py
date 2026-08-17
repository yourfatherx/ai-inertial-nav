"""Windowed IMU dataset for training the gyro denoiser.

Each sample is a fixed-length window of raw IMU (gyro+accel) plus the GT
orientation quaternion resampled onto the IMU timestamps for that window.
The training loss integrates the corrected gyro across the window and compares
the resulting relative rotations against GT, so we hand the trainer both the
raw IMU (network input) and the GT quaternions (supervision).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

from .euroc import load_sequence
from .metrics import interp_quat_to


@dataclass
class SeqArrays:
    name: str
    t: np.ndarray          # (N,) IMU time, seconds
    imu: np.ndarray        # (N,6) [gyro(3), accel(3)]
    quat: np.ndarray       # (N,4) GT orientation resampled to IMU time
    dt: np.ndarray         # (N,) per-step dt (dt[k] = t[k]-t[k-1], dt[0]=median)


def prepare_sequence(name: str) -> SeqArrays:
    """Load a EuRoC sequence and align GT orientation onto IMU timestamps,
    restricting to the GT-covered time span (GT starts/ends inside the IMU)."""
    seq = load_sequence(name)
    t_imu = seq.imu.t
    # keep only IMU samples covered by GT so quaternion resampling is valid
    lo, hi = seq.gt.t[0], seq.gt.t[-1]
    mask = (t_imu >= lo) & (t_imu <= hi)
    t = t_imu[mask]
    gyro = seq.imu.gyro[mask]
    accel = seq.imu.accel[mask]
    quat = interp_quat_to(t, seq.gt.t, seq.gt.quat)
    imu = np.concatenate([gyro, accel], axis=1).astype(np.float32)
    dt = np.empty_like(t)
    dt[1:] = np.diff(t)
    dt[0] = np.median(dt[1:])
    return SeqArrays(name=name, t=t, imu=imu, quat=quat.astype(np.float32),
                     dt=dt.astype(np.float32))


class ImuWindowDataset(Dataset):
    """Sliding windows over one or more prepared sequences.

    Returns per item:
        imu   (6, W)  raw IMU, channels-first for Conv1d
        quat  (W, 4)  GT quaternion at each sample in the window
        dt    (W, 1)  per-step dt
    """

    def __init__(self, seqs: list[SeqArrays], window: int = 400,
                 stride: int = 200, normalize: bool = True):
        self.window = window
        self.seqs = seqs
        self.index: list[tuple[int, int]] = []  # (seq_idx, start)
        for si, s in enumerate(seqs):
            n = len(s.t)
            for start in range(0, n - window, stride):
                self.index.append((si, start))
        # channel-wise normalization stats from all IMU data (train set)
        if normalize:
            allimu = np.concatenate([s.imu for s in seqs], axis=0)
            self.mean = allimu.mean(0).astype(np.float32)
            self.std = (allimu.std(0) + 1e-6).astype(np.float32)
        else:
            self.mean = np.zeros(6, np.float32)
            self.std = np.ones(6, np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, start = self.index[i]
        s = self.seqs[si]
        sl = slice(start, start + self.window)
        imu_raw = s.imu[sl]                      # (W,6) unnormalized (rad/s, m/s2)
        imu_norm = (imu_raw - self.mean) / self.std
        quat = s.quat[sl]                        # (W,4)
        dt = s.dt[sl][:, None]                   # (W,1)
        return {
            "imu_norm": torch.from_numpy(imu_norm.T.copy()),   # (6,W) net input
            "imu_raw": torch.from_numpy(imu_raw.copy()),       # (W,6) for gyro
            "quat": torch.from_numpy(quat.copy()),             # (W,4)
            "dt": torch.from_numpy(dt.copy()),                 # (W,1)
        }
