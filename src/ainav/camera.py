"""EuRoC camera loader + calibration (Approach A, visual-inertial).

One camera in a sequence (e.g. cam0):
    <seq>/mav0/cam0/sensor.yaml   pinhole intrinsics + radtan distortion + T_BS
    <seq>/mav0/cam0/data.csv      timestamp[ns], filename
    <seq>/mav0/cam0/data/*.png    752x480 mono8 frames @ ~20 Hz

Intrinsics in sensor.yaml: [fu, fv, cu, cv]
Distortion (radial-tangential): [k1, k2, p1, p2]
T_BS: 4x4 sensor->body (camera frame expressed in body/IMU frame).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
import cv2

from .config import DATA_DIR


@dataclass
class CameraCalib:
    K: np.ndarray            # (3,3) pinhole intrinsics
    dist: np.ndarray         # (4,) radtan [k1,k2,p1,p2]
    resolution: tuple        # (w, h)
    T_BS: np.ndarray         # (4,4) camera->body extrinsics
    rate_hz: float


@dataclass
class CameraData:
    t: np.ndarray            # (F,) seconds, zero-based on the same t0 as ImuData
    files: list              # (F,) absolute Paths to the .png frames
    calib: CameraCalib


def _cam_dir(name: str, cam: str) -> Path:
    d = DATA_DIR / name
    mav0 = d / "mav0" if (d / "mav0").exists() else d / name / "mav0"
    cd = mav0 / cam
    if not cd.exists():
        raise FileNotFoundError(f"Missing {cd}. Extracted {name} with images?")
    return cd


def load_calib(name: str, cam: str = "cam0") -> CameraCalib:
    with open(_cam_dir(name, cam) / "sensor.yaml") as f:
        y = yaml.safe_load(f)
    fu, fv, cu, cv_ = y["intrinsics"]
    K = np.array([[fu, 0.0, cu], [0.0, fv, cv_], [0.0, 0.0, 1.0]])
    dist = np.array(y["distortion_coefficients"], dtype=float)
    T_BS = np.array(y["T_BS"]["data"], dtype=float).reshape(4, 4)
    w, h = y["resolution"]
    return CameraCalib(K=K, dist=dist, resolution=(w, h),
                       T_BS=T_BS, rate_hz=float(y["rate_hz"]))


def load_camera(name: str = "V1_01_easy", cam: str = "cam0",
                t0_ns: float | None = None) -> CameraData:
    """Load a camera's frame index. Frames are read lazily from `files`.

    t0_ns: shared zero reference (pass the IMU's first timestamp so camera and
    IMU share a clock). If None, uses the first camera timestamp.
    """
    cd = _cam_dir(name, cam)
    raw = np.loadtxt(cd / "data.csv", delimiter=",", skiprows=1,
                     usecols=0, dtype=np.int64)
    if t0_ns is None:
        t0_ns = raw[0]
    files = [cd / "data" / f"{ts}.png" for ts in raw]
    return CameraData(t=(raw - t0_ns) * 1e-9, files=files,
                      calib=load_calib(name, cam))


def read_gray(path: Path) -> np.ndarray:
    """Read one EuRoC frame as (H,W) uint8 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read {path}")
    return img


def undistort(img: np.ndarray, calib: CameraCalib) -> np.ndarray:
    """Undistort a frame with the radtan model, keeping the same K."""
    return cv2.undistort(img, calib.K, calib.dist)
