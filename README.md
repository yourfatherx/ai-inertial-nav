# AI-Based Inertial Navigation for GPS-Denied Environments

Research prototype. **Approach B first:** learning-based IMU denoising feeding an
Error-State Kalman Filter (ESKF). Later, **Approach A:** AI Visual-Inertial Odometry.

## Goal (Approach B)
Show that a small neural network that *corrects the raw IMU signal* reduces
dead-reckoning drift versus a classical filter alone, on the EuRoC MAV dataset.

Pipeline:
```
raw IMU (gyro, accel) --> [NN corrector] --> corrected IMU --> [ESKF] --> pose
```

## Roadmap
- **Phase 0** — env + EuRoC loader + raw-vs-truth plot   ← current
- **Phase 1** — naive integration baseline (show drift)
- **Phase 2** — Error-State EKF, no AI (real baseline)
- **Phase 3** — gyro-denoising CNN (open-loop eval)
- **Phase 4** — fuse denoised signal into ESKF (money plot)
- **Phase 5** — demo polish (3D trajectory, metrics table)

## Environment
- Isolated Python 3.12 venv in `.venv/` (managed by `uv`).
- PyTorch + CUDA 12.4 (RTX 4050 Laptop, 6 GB).
- Activate:  `.venv/Scripts/python.exe`  (or `source .venv/Scripts/activate`)

## Layout
```
src/ainav/     core library (data, filters, models, metrics)
scripts/       runnable entry points (download, train, eval, plot)
notebooks/     exploration
data/euroc/    EuRoC MAV sequences (gitignored)
results/       plots, checkpoints, metrics (gitignored)
tests/         sanity checks
```

## Data
EuRoC MAV: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
Start with `MH_01_easy` (ASL format).
