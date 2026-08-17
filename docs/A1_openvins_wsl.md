# A1 — OpenVINS classical VIO baseline (WSL2 + Docker) runbook

Repeatable checklist to stand up OpenVINS on EuRoC as the Approach-A classical
baseline. Runs in WSL2 Ubuntu on Windows; the host cannot build C++/ROS VIO, so
everything real happens inside a Linux container. MSCKF is CPU-only — headless run,
no GPU/RViz needed for ATE.

Backend rationale: OpenVINS has a clean feature-tracker front-end / MSCKF back-end
split, so the A2 learned front-end becomes a module swap, not a rewrite.

---

## Stage 0 — WSL2 install on D: (space-constrained, C: nearly full)

Admin PowerShell on Windows:

```
wsl --install --no-distribution          # WSL platform only, no distro yet
# reboot
wsl --update
wsl --install Ubuntu-22.04 --location D:\WSL\ubuntu-a1 --name ubuntu-a1
```

Whole distro VHDX + Docker images + EuRoC bags live inside the D:-hosted ext4 VHDX
(a block device, native I/O — not the slow 9p `/mnt/d` mount). Only the small WSL
platform feature stays on C:. Budget ~10–12 GB on D:.

### Fix: `Wsl/CallMsi/Install/REGDB_E_CLASSNOTREG` ("Class not registered")

Seen on `wsl --install` / `wsl --update`. The System32 `wsl.exe` is a stub that
forwards to the WSL store package; if that package/COM class isn't registered the
stub throws this and cannot self-heal. Fix, in **Admin** PowerShell:

```
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
**Reboot** (mandatory — the class registers on restart), then:
```
wsl --update
wsl --set-default-version 2
wsl --status          # must print version info, not the class error
```
If `wsl --update` still throws it: install the WSL MSI directly from
https://github.com/microsoft/WSL/releases (latest `wsl.x64.msi`), reboot, retry.

---

## Stage 1 — Docker engine (inside Ubuntu)

apt `docker.io` (not Docker Desktop). WSL2 has no systemd by default, so the daemon
is started with the SysV `service` command, not `systemctl`.

```
cd ~
sudo apt update && sudo apt -y upgrade
sudo apt -y install docker.io
sudo usermod -aG docker $USER
```
Refresh group: `wsl --shutdown` on Windows, reopen Ubuntu. Then start daemon + test:
```
sudo service docker start
docker run --rm hello-world      # "Hello from Docker!" with no sudo
```
`permission denied ... docker.sock` almost always = group not refreshed OR daemon
not started. Re-check with `groups` (must list `docker`) and `sudo service docker start`.

---

## Stage 2 — Build the OpenVINS image (~30–60 min)

No prebuilt image exists; build locally from the ROS1 Noetic Dockerfile (EuRoC bags
are ROS1 bags). ROS + OpenCV + Ceres compile inside the container. Work in native
`~`, never `/mnt/d`.

```
cd ~
git clone https://github.com/rpng/open_vins.git
cd open_vins
export VERSION=ros1_20_04
docker build -t ov_$VERSION -f Dockerfile_$VERSION .
```
Verify: `docker images | grep ov_ros1_20_04`. The Dockerfile does NOT build the repo
itself — OpenVINS is built inside the running container in Stage 4.

---

## Stage 3 — EuRoC data (native ext4)

Start with V1_01_easy (matches A0). Download the ROS1 `.bag`, not the ASL mav0 folders:

```
mkdir -p ~/euroc && cd ~/euroc
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.bag
```
~1.4 GB. Keep on ext4 (`~/euroc`), not `/mnt/d`. Add MH_01_easy later the same way.

---

## Stage 4 — Run OpenVINS, dump trajectory (headless)

Launch container, bind-mount the repo (build target) + data:

```
docker run -it --net=host \
  --mount type=bind,source=$HOME/open_vins,target=/catkin_ws/src/open_vins \
  --mount type=bind,source=$HOME/euroc,target=/datasets \
  ov_ros1_20_04 bash
```
Inside:
```
cd /catkin_ws && catkin build          # first-time OpenVINS build (~10 min)
source devel/setup.bash
```
Run the EuRoC estimator (`config/euroc_mav/`) with trajectory recording — the
`run_subscribe_msckf` node against a `rosbag play` of the bag. Set the estimate
output to TUM format (`dosave`/`path_est` param), e.g. `/datasets/est_V1_01.txt`.
Confirm the file fills with `timestamp tx ty tz qx qy qz qw`.

---

## Stage 5 — Evaluate ATE

Either tool:
- **evo** (pip): `evo_ape euroc <mav0 GT csv> est_V1_01.txt -va` (euroc GT reader,
  SE(3) alignment).
- **ov_eval** (built into OpenVINS): RPG-style ATE, no extra install.

**Deliverable:** OpenVINS ATE on V1_01 (bounded, cm–dm) vs Approach-B dead-reckoning
ATE (unbounded, hundreds of m). The camera bounds absolute position — this is the
reference number the A2 learned front-end must beat.

---

## Non-goals (A1)

- No GPU / RViz (headless CPU is enough for ATE).
- No learned front-end (that's A2).
- No multi-sequence sweep until V1_01 works end-to-end; then MH_01.

---

# Stage A2 — Learned vs classical front-end through the SAME back-end

The decisive A2 number: feed each front-end's tracks into the identical OpenVINS
MSCKF and compare trajectory ATE. Only the tracker changes, so the ATE gap IS the
front-end's contribution. We do NOT rewrite the filter — OpenVINS' own
`feed_measurement_simulation(t, camids, feats)` accepts external `(id, raw_uv)`
tracks and undistorts internally. Mono cam0; the apples-to-apples baseline is KLT
through this same path (not A1's stereo 5.4 cm).

### A2.1 — Export tracks (Windows, already scripted)
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/phaseA2_export_tracks.py V1_03_difficult 0
```
Writes `results/a2_backend/V1_03_difficult/`: `imu.txt`, `gt_init.txt`,
`tracks_{KLT,Learned}_{clean,lowlight}.txt`, `cam0.yaml`. (Pre-flight
`scripts/phaseA2_preflight.py` first confirms tracks reproject to ~1 px vs GT —
run it before touching the container.)

### A2.2 — Patch OpenVINS + build the mono config (in WSL Ubuntu, native shell)
```
cp /mnt/d/AI-inertial-nav/openvins_patch/run_tracksfile.cpp \
   ~/open_vins/ov_msckf/src/run_tracksfile.cpp
cat /mnt/d/AI-inertial-nav/openvins_patch/cmake_add_run_tracksfile.txt \
   >> ~/open_vins/ov_msckf/cmake/ROS1.cmake
# stage the exported bundle on native ext4 (also serves as the container mount):
cp -r /mnt/d/AI-inertial-nav/results/a2_backend/V1_03_difficult ~/euroc/a2_V1_03
```

### A2.3 — Build + run (inside the container)
```
docker run -it --net=host \
  --mount type=bind,source=$HOME/open_vins,target=/catkin_ws/src/open_vins \
  --mount type=bind,source=$HOME/euroc,target=/datasets \
  ov_ros1_20_04 bash
```
Inside:
```
cd /catkin_ws && catkin build ov_msckf && source devel/setup.bash
# make the mono config (see config/euroc_mono/README.md):
cd /catkin_ws/src/open_vins/config && cp -r euroc_mav euroc_mono && cd euroc_mono
sed -i 's/^\(\s*max_cameras:\).*/\1 1/'   estimator_config.yaml
sed -i 's/^\(\s*use_stereo:\).*/\1 false/' estimator_config.yaml
CFG=/catkin_ws/src/open_vins/config/euroc_mono/estimator_config.yaml
BIN=/catkin_ws/devel/lib/ov_msckf/run_tracksfile
D=/datasets/a2_V1_03
for cond in clean lowlight; do for tr in KLT Learned; do
  $BIN $CFG $D/imu.txt $D/tracks_${tr}_${cond}.txt $D/gt_init.txt \
       $D/est_V1_03_${tr}_${cond}.txt
done; done
```
Each run prints `wrote N poses`. `run_tracksfile` is a plain executable (no
roscore/ROS spin needed).

### A2.4 — Pull trajectories back + compute ATE (Windows)
From the WSL shell (NOT the container — the est files are in the bind-mount):
```
cp ~/euroc/a2_V1_03/est_V1_03_*.txt /mnt/d/AI-inertial-nav/results/
```
Then on Windows:
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/compareA2_backend.py V1_03_difficult
```
→ `results/compareA2_backend.{png,json}` + the ATE table (KLT vs Learned ×
clean/lowlight). Optional in-container cross-check: `rosrun ov_eval
error_singlerun se3 <gt_tum> est_V1_03_KLT_clean.txt`.

### A2 verdict (how we read it, then stop)
- **Learned ATE ≤ KLT under low-light while KLT drifts** → cleaner tracks convert
  to pose accuracy → real win; wire the learned front-end in permanently.
- **Learned ATE ≥ KLT everywhere** → honest negative; its short/fragmented tracks
  (median ~2–4 frames, the known lower bound) starve the MSCKF despite being
  geometrically cleaner → "naive learned front-end needs a track-manager first."
  Bank B + A1 as the prototype.

Build gotcha: if the compiler rejects an IMU getter, the accessors are
`state->_imu->quat()` (JPL q_GtoI) and `state->_imu->pos()` — used by OpenVINS'
own visualizer.

---

## A2-REDO — fair head-to-head with RANSAC gate (2026-07-14)

**Why:** the banked A2 fed raw tracks through OpenVINS' SIM path, which bypasses the
native epipolar outlier rejection — so KLT's long high-leverage tracks carried ~15%
outliers into the filter and diverged 3/4 runs. This redo adds that gate back,
**identically for both front-ends**, as a pure post-process on the exact banked track
files. Only variable vs the banked run = the gate. Timeboxed: one gate, one re-run,
one number. Win → A3. Lose again → bank for good, no more iterating.

**Pre-flight already passed (Windows, no container):**
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/phaseA2_ransac_gate.py V1_03_difficult
```
Wrote `tracks_{KLT,Learned}_{clean,lowlight}_gated.txt`. Inlier ratios lifted
(KLT_lowlight 0.568→0.864, all others →~0.85–0.95) while keeping 73–88% of obs — gate
culls outliers without starving the filter. Green-lit the runs below.

### Steps (your end — WSL/Docker, same container as A2)
1. **Copy the 4 gated track files into the existing bundle** (WSL native shell):
   ```
   cp /mnt/d/AI-inertial-nav/results/a2_backend/V1_03_difficult/tracks_*_gated.txt ~/euroc/a2_V1_03/
   ```
   (imu.txt / gt_init.txt / config are unchanged — reuse them.)

2. **Run the container** (identical to A2.3; the binary + mono config already exist —
   no rebuild needed unless the container is fresh):
   ```
   docker run -it --net=host \
     --mount type=bind,source=$HOME/open_vins,target=/catkin_ws/src/open_vins \
     --mount type=bind,source=$HOME/euroc,target=/datasets \
     ov_ros1_20_04 bash
   ```
   Inside (rebuild line only if this is a fresh container):
   ```
   source /catkin_ws/devel/setup.bash    # or: cd /catkin_ws && catkin build ov_msckf -j2 && source devel/setup.bash
   CFG=/catkin_ws/src/open_vins/config/euroc_mono/estimator_config.yaml
   BIN=/catkin_ws/devel/lib/ov_msckf/run_tracksfile
   D=/datasets/a2_V1_03
   for cond in clean lowlight; do for tr in KLT Learned; do
     $BIN $CFG $D/imu.txt $D/tracks_${tr}_${cond}_gated.txt $D/gt_init.txt \
          $D/est_V1_03_${tr}_${cond}_gated.txt
   done; done
   ```
   (Note `-j2` if rebuilding — the OOM gotcha from the banked run.)

3. **Pull the 4 gated est files back** (WSL shell, not container):
   ```
   cp ~/euroc/a2_V1_03/est_V1_03_*_gated.txt /mnt/d/AI-inertial-nav/results/
   ```

4. **Tell me "gated est copied"** — I run the compare and read the verdict.
   Sanity check per run: `p_IinG` should stay within a ~10–30 m room (banked KLT
   printed thousands of metres = divergence). If all 4 now print bounded positions,
   we likely have the fair bake-off the banked run never got.
