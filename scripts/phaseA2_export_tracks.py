"""A2 back-end integration: export front-end tracks for OpenVINS.

The A2 question is decided by ONE end-to-end number: feed each front-end's
tracks into the *same* OpenVINS MSCKF back-end and compare trajectory ATE.
OpenVINS exposes `VioManager::feed_measurement_simulation(t, camids, feats)` --
its external-feature API -- which takes `(feature_id, raw_pixel_uv)` pairs and
runs the filter untouched. So we never rewrite the filter; we just hand it our
tracks. This script produces everything the C++ side (`run_tracksfile`) reads:

  results/a2_backend/<seq>/
    imu.txt                      t wx wy wz ax ay az           (>= t_init)
    gt_init.txt                  t px py pz  R_GtoI(9 row-major)  vx vy vz  bg(3) ba(3)
    cam0.yaml                    intrinsics dump (reference)
    tracks_KLT_<cond>.txt        t  feat_id  u  v               (raw distorted px)
    tracks_Learned_<cond>.txt    t  feat_id  u  v

KEY DETAILS
  * Tracks run on RAW (distorted) frames -- OpenVINS/TrackSIM stores the raw uv
    and undistorts internally with the cam0 model, so we must NOT undistort here.
  * All timestamps are seconds on a shared clock zeroed at the first IMU sample.
  * Init is seeded from the first ground-truth sample (GT starts ~1.8 s into the
    run for V1_03). R_GtoI = R_WB^T is exported as a matrix so the C++ converts
    it to a JPL quaternion with OpenVINS' own rot_2_quat -- no convention guessing.
  * Both front-ends are mono (cam0). The apples-to-apples baseline is KLT fed
    through this same path (NOT A1's stereo 5.4 cm); the only variable is the tracker.
"""
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.camera import load_camera, read_gray            # noqa: E402
from ainav.frontend import KLTTracker, LearnedTracker, degrade  # noqa: E402
from ainav.config import RESULTS_DIR                        # noqa: E402
from ainav.euroc import _seq_dir                            # noqa: E402  (mav0 locator)

CONDITIONS = ["clean", "lowlight"]


def _mav0(seq: str) -> Path:
    """Locate <seq>/mav0 (handles the nested <seq>/<seq>/mav0 layout)."""
    return _seq_dir(seq)          # already returns .../mav0


def _hamilton_to_R_WB(qw, qx, qy, qz) -> np.ndarray:
    """Hamilton quaternion (body->world) -> rotation matrix R_WB (v_W = R_WB v_B)."""
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])


def main(seq: str, n_frames: int) -> None:
    mav0 = _mav0(seq)
    out_dir = RESULTS_DIR / "a2_backend" / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load IMU + GT (raw ns) and fix the shared clock at the first IMU sample ---
    imu = np.loadtxt(mav0 / "imu0" / "data.csv", delimiter=",", skiprows=1)
    gt = np.loadtxt(mav0 / "state_groundtruth_estimate0" / "data.csv",
                    delimiter=",", skiprows=1)
    t0_ns = np.int64(imu[0, 0])
    imu_t = (imu[:, 0] - t0_ns) * 1e-9
    gt_t = (gt[:, 0] - t0_ns) * 1e-9
    t_init = float(gt_t[0])                       # GT starts ~1.8 s in
    print(f"[export] {seq}: t0={t0_ns}, GT init at t={t_init:.3f}s "
          f"({len(imu)} imu, {len(gt)} gt rows)")

    # --- imu.txt (feed a couple samples before t_init to bracket propagation) ---
    keep = imu_t >= (t_init - 0.02)
    with open(out_dir / "imu.txt", "w") as f:
        for row, t in zip(imu[keep], imu_t[keep]):
            f.write(f"{t:.9f} {row[1]:.10f} {row[2]:.10f} {row[3]:.10f} "
                    f"{row[4]:.10f} {row[5]:.10f} {row[6]:.10f}\n")
    print(f"[export] imu.txt: {int(keep.sum())} samples (t>={t_init-0.02:.3f})")

    # --- gt_init.txt: init state from the first GT row ---
    g = gt[0]
    p = g[1:4]
    R_WB = _hamilton_to_R_WB(g[4], g[5], g[6], g[7])
    R_GtoI = R_WB.T
    v = g[8:11]
    bg = g[11:14]
    ba = g[14:17]
    with open(out_dir / "gt_init.txt", "w") as f:
        f.write(f"{t_init:.9f}\n")
        f.write(" ".join(f"{x:.10f}" for x in p) + "\n")
        f.write(" ".join(f"{x:.10f}" for x in R_GtoI.reshape(-1)) + "\n")
        f.write(" ".join(f"{x:.10f}" for x in v) + "\n")
        f.write(" ".join(f"{x:.10f}" for x in bg) + "\n")
        f.write(" ".join(f"{x:.10f}" for x in ba) + "\n")
    print(f"[export] gt_init.txt: p={p}, |v|={np.linalg.norm(v):.3f}")

    # --- camera frames (RAW, absolute clock on the IMU t0) ---
    cam = load_camera(seq, "cam0", t0_ns=float(t0_ns))
    n = len(cam.files) if n_frames <= 0 else min(n_frames, len(cam.files))
    cam_t = cam.t[:n]
    raw = [read_gray(fp) for fp in cam.files[:n]]
    print(f"[export] {n} raw cam0 frames, {raw[0].shape}; "
          f"first frame t={cam_t[0]:.3f}s")

    # dump intrinsics for reference / the mono config
    c = cam.calib
    with open(out_dir / "cam0.yaml", "w") as f:
        f.write(f"# {seq} cam0 (reference for config/euroc_mono)\n")
        f.write(f"resolution: [{c.resolution[0]}, {c.resolution[1]}]\n")
        f.write(f"intrinsics: [{c.K[0,0]}, {c.K[1,1]}, {c.K[0,2]}, {c.K[1,2]}]\n")
        f.write(f"distortion_coefficients: {list(c.dist)}\n")
        f.write(f"T_BS_rowmajor: {list(c.T_BS.reshape(-1))}\n")

    trackers = {"KLT": KLTTracker(max_features=400),
                "Learned": LearnedTracker(max_features=400)}
    print(f"[export] learned device: {trackers['Learned'].device}")

    for cond in CONDITIONS:
        frames = [degrade(im, cond, np.random.default_rng(i))
                  for i, im in enumerate(raw)]
        for tname, T in trackers.items():
            T._next_id = 0
            res = T.run(iter(frames))
            path = out_dir / f"tracks_{tname}_{cond}.txt"
            nobs = 0
            with open(path, "w") as f:
                for fi, fo in enumerate(res.frames):
                    t = float(cam_t[fi])
                    if t < t_init:                 # only feed within GT span
                        continue
                    for i, (u, v_) in zip(fo.ids, fo.pts):
                        f.write(f"{t:.9f} {int(i)} {float(u):.4f} {float(v_):.4f}\n")
                        nobs += 1
            print(f"[export] {path.name}: {nobs} obs "
                  f"(meanLen={res.mean_track_len:.1f}, inlier={res.inlier_ratio:.3f})")

    print(f"[export] done -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="V1_03_difficult")
    ap.add_argument("n_frames", nargs="?", type=int, default=0,
                    help="0 = all frames")
    a = ap.parse_args()
    main(a.seq, a.n_frames)
