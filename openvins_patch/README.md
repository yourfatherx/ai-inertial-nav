# OpenVINS patch — external-track front-end (Phase A2)

Our A2 contribution to OpenVINS: a headless driver that feeds **pre-computed
feature tracks** (exported from our Python front-ends) into the MSCKF back-end,
so the same filter runs on either the classical KLT or the learned
SuperPoint+LightGlue tracks and the trajectory ATE can be compared. Only the
front-end changes; the filter is untouched.

Mechanism: OpenVINS already exposes `VioManager::feed_measurement_simulation(t,
camids, feats)` (used by its own simulator) which accepts `(feature_id,
raw_pixel_uv)` pairs and undistorts internally via `TrackSIM`. We reuse that as
the injection point — no filter code is modified.

## Files
- `run_tracksfile.cpp` — the driver. Reads `imu.txt`, `tracks.txt`, `gt_init.txt`
  (produced by `scripts/phaseA2_export_tracks.py`), feeds IMU + tracks by
  timestamp, seeds init from ground truth (`R_GtoI` matrix → JPL quat via
  OpenVINS' own `rot_2_quat`), and writes a TUM trajectory.
- `cmake_add_run_tracksfile.txt` — the CMake lines to register the target.

## Apply (inside the WSL OpenVINS clone `~/open_vins`)
```bash
cp /mnt/d/AI-inertial-nav/openvins_patch/run_tracksfile.cpp \
   ~/open_vins/ov_msckf/src/run_tracksfile.cpp
# append the CMake block after the run_simulation target:
cat /mnt/d/AI-inertial-nav/openvins_patch/cmake_add_run_tracksfile.txt \
   >> ~/open_vins/ov_msckf/cmake/ROS1.cmake
```
Then rebuild in the container (`catkin build ov_msckf`). See
`docs/A1_openvins_wsl.md` → "Stage A2" for the full run recipe.

Derived from OpenVINS `run_simulation.cpp` (GPLv3); this file is GPLv3.
