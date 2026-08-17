# config/euroc_mono — mono cam0 OpenVINS config for A2

The A2 back-end test runs OpenVINS **mono** (cam0 only), because both our Python
front-ends track cam0 only. Rather than hand-author a config (parser-mismatch
risk), derive it from OpenVINS' shipped `euroc_mav` config with two edits — its
cam0 intrinsics/distortion/extrinsics are already the EuRoC VI-sensor values that
match `V1_03_difficult/mav0/cam0/sensor.yaml` exactly (verified: the Python
pre-flight reprojects clean KLT tracks to ~1.1 px median with these same numbers).

## Build it inside the container (OpenVINS clone at `/catkin_ws/src/open_vins`)
```bash
cd /catkin_ws/src/open_vins/config
cp -r euroc_mav euroc_mono
cd euroc_mono
# mono: one camera, no stereo epipolar tracking
sed -i 's/^\(\s*max_cameras:\).*/\1 1/'   estimator_config.yaml
sed -i 's/^\(\s*use_stereo:\).*/\1 false/' estimator_config.yaml
```
`max_cameras: 1` makes OpenVINS read only `cam0` from the copied
`kalibr_imucam_chain.yaml` (the `cam1` block is ignored, harmless). IMU noise
(ADIS16448) comes from the copied `kalibr_imu_chain.yaml`; `gravity_mag: 9.81`
and all estimator/filter settings stay identical to A1's euroc_mav run, so the
ONLY variable across our A2 runs is the front-end tracks we inject.

Config path passed to `run_tracksfile`:
`/catkin_ws/src/open_vins/config/euroc_mono/estimator_config.yaml`
