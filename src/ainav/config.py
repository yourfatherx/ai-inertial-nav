"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # D:/AI-inertial-nav
DATA_DIR = ROOT / "data" / "euroc"
RESULTS_DIR = ROOT / "results"

# EuRoC IMU runs at 200 Hz.
IMU_RATE_HZ = 200.0
IMU_DT = 1.0 / IMU_RATE_HZ

# Gravity magnitude (EuRoC world frame, ENU-ish; z up). m/s^2.
GRAVITY = 9.81

# EuRoC ADIS16448 IMU noise densities (from mav0/imu0/sensor.yaml).
# Continuous-time densities; the ESKF turns these into discrete process noise.
GYRO_NOISE_DENSITY = 1.6968e-4    # rad/s/sqrt(Hz)   white noise on gyro
ACCEL_NOISE_DENSITY = 2.0000e-3   # m/s^2/sqrt(Hz)   white noise on accel
GYRO_RANDOM_WALK = 1.9393e-5      # rad/s^2/sqrt(Hz) gyro-bias drift
ACCEL_RANDOM_WALK = 3.0000e-3     # m/s^3/sqrt(Hz)   accel-bias drift

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
