"""A2 pre-flight: prove exported tracks are geometrically consistent with GT.

Before feeding tracks to the OpenVINS container (a slow build/run loop), validate
the DATA in a fast Python loop. For each long track we:
  1. take its (t, u, v) observations on RAW pixels,
  2. undistort to normalized camera coords with the cam0 model,
  3. build the camera pose in world from GT: T_WC(t) = T_WB(t) . T_BS,
  4. triangulate the 3D world point from the two widest-baseline views,
  5. reproject into every observation and measure the pixel residual.

A correct pipeline (right intrinsics, distortion, extrinsics, timestamps, and
quaternion convention) makes real tracks reproject to a few pixels. If residuals
are huge, there is a coordinate/timing bug to fix here -- NOT in the container.

Bonus: this residual IS the direct drift metric. Clean tracks reproject tight;
drifting KLT low-light tracks reproject loose. We report per-tracks-file medians,
so the number that OpenVINS will later confirm is already visible here.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.camera import load_calib                          # noqa: E402
from ainav.config import RESULTS_DIR                         # noqa: E402
from ainav.euroc import _seq_dir                             # noqa: E402


def _load_gt(mav0):
    gt = np.loadtxt(mav0 / "state_groundtruth_estimate0" / "data.csv",
                    delimiter=",", skiprows=1)
    t0 = np.int64(np.loadtxt(mav0 / "imu0" / "data.csv", delimiter=",",
                             skiprows=1, usecols=0, dtype=np.int64,
                             max_rows=1))
    t = (gt[:, 0] - t0) * 1e-9
    return t, gt


def _R_WB(qw, qx, qy, qz):
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ])


def _cam_pose_world(t_query, gt_t, gt, T_BS):
    """R_WC, p_WC at time t_query: nearest-GT rotation, linear-interp position."""
    j = int(np.searchsorted(gt_t, t_query))
    j = max(1, min(j, len(gt_t) - 1))
    # linear interp position between j-1 and j
    t0_, t1_ = gt_t[j - 1], gt_t[j]
    a = 0.0 if t1_ == t0_ else (t_query - t0_) / (t1_ - t0_)
    a = float(np.clip(a, 0.0, 1.0))
    p_WB = (1 - a) * gt[j - 1, 1:4] + a * gt[j, 1:4]
    k = j if a >= 0.5 else j - 1                    # nearest for rotation
    R_WB = _R_WB(gt[k, 4], gt[k, 5], gt[k, 6], gt[k, 7])
    R_BS = T_BS[:3, :3]
    p_BS = T_BS[:3, 3]
    R_WC = R_WB @ R_BS
    p_WC = p_WB + R_WB @ p_BS
    return R_WC, p_WC


def _read_tracks(path):
    """path -> dict[id] = list of (t, u, v), and sorted unique frame times."""
    arr = np.loadtxt(path)
    tracks = {}
    for t, i, u, v in arr:
        tracks.setdefault(int(i), []).append((t, u, v))
    return tracks


def _eval_file(path, gt_t, gt, K, dist, T_BS, min_len=5, max_tracks=400):
    tracks = _read_tracks(path)
    long_ids = [i for i, obs in tracks.items() if len(obs) >= min_len]
    rng = np.random.default_rng(0)
    if len(long_ids) > max_tracks:
        long_ids = list(rng.choice(long_ids, max_tracks, replace=False))

    residuals = []
    for i in long_ids:
        obs = sorted(tracks[i])
        ts = np.array([o[0] for o in obs])
        uv = np.array([[o[1], o[2]] for o in obs], np.float32)
        # undistort raw pixels -> normalized camera coords (K=I after this)
        norm = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        poses = [_cam_pose_world(t, gt_t, gt, T_BS) for t in ts]
        Ps = []
        for R_WC, p_WC in poses:
            R_CW = R_WC.T
            t_CW = -R_CW @ p_WC
            Ps.append(np.hstack([R_CW, t_CW.reshape(3, 1)]))
        # triangulate from widest-baseline pair (first & last)
        X = cv2.triangulatePoints(Ps[0], Ps[-1],
                                  norm[0].reshape(2, 1),
                                  norm[-1].reshape(2, 1)).reshape(4)
        if abs(X[3]) < 1e-9:
            continue
        Xw = X[:3] / X[3]
        # reproject into every view -> normalized residual -> pixels (x focal)
        f = 0.5 * (K[0, 0] + K[1, 1])
        for P, n_obs in zip(Ps, norm):
            xc = P @ np.array([Xw[0], Xw[1], Xw[2], 1.0])
            if xc[2] <= 1e-6:
                continue
            proj = xc[:2] / xc[2]
            residuals.append(np.linalg.norm(proj - n_obs) * f)
    residuals = np.array(residuals)
    return residuals, len(long_ids)


def main(seq: str) -> None:
    mav0 = _seq_dir(seq)
    out_dir = RESULTS_DIR / "a2_backend" / seq
    calib = load_calib(seq, "cam0")
    K, dist, T_BS = calib.K, calib.dist, calib.T_BS
    gt_t, gt = _load_gt(mav0)
    print(f"[preflight] {seq}: GT {len(gt)} rows, "
          f"f={0.5*(K[0,0]+K[1,1]):.1f}px")

    files = sorted(out_dir.glob("tracks_*.txt"))
    if not files:
        print(f"[preflight] no tracks in {out_dir} -- run phaseA2_export_tracks.py")
        return
    print(f"{'tracks file':32s} {'#tracks':>8s} {'med px':>8s} "
          f"{'p90 px':>8s} {'<2px %':>7s}")
    for p in files:
        res, ntr = _eval_file(p, gt_t, gt, K, dist, T_BS)
        if len(res) == 0:
            print(f"{p.name:32s} {ntr:8d}   (no valid reprojections)")
            continue
        med = np.median(res)
        p90 = np.percentile(res, 90)
        frac = 100.0 * np.mean(res < 2.0)
        print(f"{p.name:32s} {ntr:8d} {med:8.2f} {p90:8.2f} {frac:7.1f}")

    print("\n[preflight] interpret: clean tracks should reproject to a few px "
          "(median < ~2-3). If ALL files show huge residuals, a convention/timing "
          "bug exists -- fix before the container. If clean is tight but low-light "
          "KLT is loose, that IS the drift the ATE test will confirm.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="V1_03_difficult")
    main(ap.parse_args().seq)
