"""Approach A1 (OpenVINS stereo-VIO) vs Approach B (ESKF dead-reckoning) on V1_01_easy.

The headline of Approach A: a camera bounds *absolute* position. This script puts
both estimators on the SAME sequence and shows it directly:

  A1  = OpenVINS MSCKF trajectory (results/est_V1_01.txt, produced in WSL, TUM format).
        In OpenVINS' own world frame, so we SE(3)-align it to GT with Umeyama here —
        the same alignment `ov_eval se3` used, so the recomputed ATE should reproduce
        the ~5.4 cm we already measured (a sanity check on the file handoff).
  B    = our ESKF (no AI, IMU-only), rerun on V1_01 from the GT initial state. Already
        in the GT frame, so raw error is meaningful (position_ate, no alignment).

Outputs:
  results/compareA1_vs_B.png   (position error over time + trajectory panels)
  results/compareA1_vs_B.json  (the ATE numbers)
  prints the comparison table.
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.euroc import load_sequence, _seq_dir              # noqa: E402
from ainav.eskf import run_eskf                              # noqa: E402
from ainav.integrate import integrate_full                  # noqa: E402
from ainav.metrics import interp_to, interp_quat_to, position_ate  # noqa: E402
from ainav.config import RESULTS_DIR                          # noqa: E402


def load_tum(path: Path):
    """Load a TUM / OpenVINS trajectory: `t tx ty tz qx qy qz qw` (+ extra cols).

    Returns (t [s], pos (N,3), quat (N,4) xyzw). Skips the `#` header line.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = line.split()
            rows.append([float(x) for x in v[:8]])
    a = np.asarray(rows)
    return a[:, 0], a[:, 1:4], a[:, 4:8]


def umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """Rigid SE(3) alignment (rotation + translation, NO scale) so dst ~= R@src + t.

    Standard Umeyama without scaling — the metric-VIO case. src, dst: (N,3).
    """
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    S = src - mu_s
    D = dst - mu_d
    H = (D.T @ S) / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    t = mu_d - R @ mu_s
    return R, t


def main(name: str = "V1_01_easy") -> None:
    seq = load_sequence(name)
    imu, gt = seq.imu, seq.gt

    # ---------- A1: OpenVINS VIO (SE3-aligned to GT) ----------
    a1_path = RESULTS_DIR / "est_V1_01.txt"
    if not a1_path.exists():
        sys.exit(f"[missing] {a1_path} — copy it out of WSL first:\n"
                 f"    cp ~/euroc/est_V1_01.txt /mnt/d/AI-inertial-nav/results/est_V1_01.txt")
    t_a1, p_a1, _ = load_tum(a1_path)
    # OpenVINS emits absolute EuRoC epoch seconds; our loader zeros the clock at the
    # first IMU timestamp. Shift A1 onto the same zero so timestamps line up.
    imu_csv = _seq_dir(name) / "imu0" / "data.csv"
    t0_abs = float(np.loadtxt(imu_csv, delimiter=",", skiprows=1, max_rows=1)[0]) * 1e-9
    t_a1 = t_a1 - t0_abs
    m = (t_a1 >= gt.t[0]) & (t_a1 <= gt.t[-1])
    t_a1, p_a1 = t_a1[m], p_a1[m]
    p_gt_a1 = interp_to(t_a1, gt.t, gt.pos)
    R, tvec = umeyama_se3(p_a1, p_gt_a1)
    p_a1_aln = (R @ p_a1.T).T + tvec
    err_a1 = np.linalg.norm(p_a1_aln - p_gt_a1, axis=1)
    ate_a1 = {"rmse": float(np.sqrt(np.mean(err_a1**2))),
              "mean": float(np.mean(err_a1)),
              "max": float(np.max(err_a1))}

    # ---------- B: ESKF dead-reckoning (no AI) on the same sequence ----------
    mask = (imu.t >= gt.t[0]) & (imu.t <= gt.t[-1])
    t = imu.t[mask]; gyro = imu.gyro[mask]; accel = imu.accel[mask]
    q0 = interp_quat_to(t[:1], gt.t, gt.quat)[0]
    v0 = interp_to(t[:1], gt.t, gt.vel)[0]
    p0 = interp_to(t[:1], gt.t, gt.pos)[0]
    bg0 = interp_to(t[:1], gt.t, gt.bias_gyro)[0]
    ba0 = interp_to(t[:1], gt.t, gt.bias_accel)[0]
    p_gt_b = interp_to(t, gt.t, gt.pos)

    eskf = run_eskf(t, gyro, accel, q0, v0, p0, bg0=bg0, ba0=ba0)
    err_b = np.linalg.norm(eskf["p"] - p_gt_b, axis=1)
    ate_b = position_ate(eskf["p"], p_gt_b)

    naive = integrate_full(t, gyro, accel, q0, v0, p0)
    err_n = np.linalg.norm(naive["p"] - p_gt_b, axis=1)
    ate_n = position_ate(naive["p"], p_gt_b)

    # ---------- report ----------
    print(f"[{name}] position ATE on the SAME sequence")
    print(f"  A1  OpenVINS VIO (stereo+IMU)   rmse={ate_a1['rmse']*100:8.2f} cm"
          f"   max={ate_a1['max']*100:8.2f} cm")
    print(f"  B   ESKF (IMU-only, no AI)       rmse={ate_b['rmse']:8.2f} m "
          f"   max={ate_b['max']:8.2f} m")
    print(f"  B   naive integration           rmse={ate_n['rmse']:8.2f} m "
          f"   max={ate_n['max']:8.2f} m")
    ratio = ate_b["rmse"] / ate_a1["rmse"]
    print(f"  --> camera bounds position: A1 is {ratio:,.0f}x tighter than B's ESKF")

    out_json = {
        "sequence": name,
        "A1_openvins_vio": {**ate_a1, "unit": "m"},
        "B_eskf_no_ai": {**ate_b, "unit": "m"},
        "B_naive": {**ate_n, "unit": "m"},
        "ate_ratio_B_over_A1": ratio,
    }
    (RESULTS_DIR / "compareA1_vs_B.json").write_text(json.dumps(out_json, indent=2))

    # ---------- plots ----------
    tt_a1 = t_a1 - t_a1[0]
    tt_b = t - t[0]
    fig = plt.figure(figsize=(15, 9))

    # (1) the money plot: position error over time, log scale
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.semilogy(tt_b, np.maximum(err_n, 1e-3), "r-", lw=0.7, alpha=0.6,
                 label=f"B naive (ATE {ate_n['rmse']:.0f} m)")
    ax1.semilogy(tt_b, np.maximum(err_b, 1e-3), "b-", lw=1.0,
                 label=f"B ESKF, no AI (ATE {ate_b['rmse']:.0f} m)")
    ax1.semilogy(tt_a1, np.maximum(err_a1, 1e-3), "g-", lw=1.4,
                 label=f"A1 OpenVINS VIO (ATE {ate_a1['rmse']*100:.1f} cm)")
    ax1.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.5)
    ax1.set_title("Absolute position error over time — camera bounds drift")
    ax1.set_xlabel("t [s]"); ax1.set_ylabel("position error [m] (log)")
    ax1.legend(loc="center right")

    # (2) A1 vs GT — tight
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(p_gt_a1[:, 0], p_gt_a1[:, 1], "k-", lw=1.4, label="ground truth")
    ax2.plot(p_a1_aln[:, 0], p_a1_aln[:, 1], "g-", lw=1.0, label="A1 OpenVINS (aligned)")
    ax2.set_title(f"A1 tracks GT (ATE {ate_a1['rmse']*100:.1f} cm)")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.axis("equal"); ax2.legend()

    # (3) B ESKF vs GT — diverges
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(p_gt_b[:, 0], p_gt_b[:, 1], "k-", lw=1.4, label="ground truth")
    ax3.plot(eskf["p"][:, 0], eskf["p"][:, 1], "b-", lw=1.0, label="B ESKF (no AI)")
    ax3.scatter(*p_gt_b[0, :2], c="g", s=30, zorder=5, label="start")
    ax3.set_title(f"B drifts unbounded (ATE {ate_b['rmse']:.0f} m)")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]")
    ax3.axis("equal"); ax3.legend()

    fig.suptitle(f"Approach A1 (camera-aided VIO) vs Approach B (IMU-only) — {name}",
                 fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / "compareA1_vs_B.png"
    fig.savefig(out, dpi=120)
    print(f"[saved] {out}")
    print(f"[saved] {RESULTS_DIR / 'compareA1_vs_B.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "V1_01_easy")
