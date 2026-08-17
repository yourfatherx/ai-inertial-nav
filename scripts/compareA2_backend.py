"""A2 decisive number: learned vs classical front-end through the SAME back-end.

Both front-ends' tracks were fed into the identical OpenVINS MSCKF (mono cam0)
via run_tracksfile; this reads the resulting TUM trajectories and reports ATE.
The only variable between a KLT run and a Learned run is the tracker, so the ATE
difference IS the front-end's end-to-end contribution to VIO accuracy.

Baseline note: the apples-to-apples reference is KLT fed through this same path
(mono), NOT A1's stereo 5.4 cm. Both front-ends are mono here.

Expects, copied out of WSL into results/ (see docs/A1_openvins_wsl.md Stage A2):
  est_V1_03_KLT_clean.txt      est_V1_03_Learned_clean.txt
  est_V1_03_KLT_lowlight.txt   est_V1_03_Learned_lowlight.txt

Timestamps are already on our zeroed IMU clock (the exporter fed that clock), so
they line up with the loader's gt.t directly -- no epoch shift (unlike A1).

Outputs: results/compareA2_backend.png, results/compareA2_backend.json + table.
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.euroc import load_sequence                          # noqa: E402
from ainav.metrics import interp_to                            # noqa: E402
from ainav.config import RESULTS_DIR                            # noqa: E402

TRACKERS = ["KLT", "Learned"]
CONDS = ["clean", "lowlight"]


def load_tum(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = line.split()
            rows.append([float(x) for x in v[:8]])
    a = np.asarray(rows)
    return a[:, 0], a[:, 1:4]


def umeyama_se3(src, dst):
    """Rigid SE(3) alignment (no scale): dst ~= R@src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = ((dst - mu_d).T @ (src - mu_s)) / len(src)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    return R, mu_d - R @ mu_s


def eval_est(path, gt):
    t, p = load_tum(path)
    m = (t >= gt.t[0]) & (t <= gt.t[-1])
    t, p = t[m], p[m]
    if len(t) < 10:
        return None
    p_gt = interp_to(t, gt.t, gt.pos)
    R, tv = umeyama_se3(p, p_gt)
    p_aln = (R @ p.T).T + tv
    err = np.linalg.norm(p_aln - p_gt, axis=1)
    return {"t": t, "p_aln": p_aln, "p_gt": p_gt,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mean": float(np.mean(err)), "max": float(np.max(err)),
            "n": int(len(t)), "dur": float(t[-1] - t[0])}


def main(name: str = "V1_03_difficult", suffix: str = "") -> None:
    seq = load_sequence(name)
    tag = name.split("_")[0] + "_" + name.split("_")[1]   # V1_03
    res = {}
    for tr in TRACKERS:
        for cond in CONDS:
            p = RESULTS_DIR / f"est_{tag}_{tr}_{cond}{suffix}.txt"
            res.setdefault(cond, {})[tr] = eval_est(p, seq.gt) if p.exists() else None

    # ---------- table ----------
    print(f"\n[A2 back-end] {name}: position ATE (mono, same MSCKF, "
          f"front-end swapped)\n")
    print(f"  {'condition':10s} {'KLT ATE':>12s} {'Learned ATE':>14s} "
          f"{'winner':>10s}")
    verdict = {}
    for cond in CONDS:
        k = res[cond].get("KLT")
        l = res[cond].get("Learned")
        ks = f"{k['rmse']*100:.2f} cm" if k else "  (missing)"
        ls = f"{l['rmse']*100:.2f} cm" if l else "  (missing)"
        win = "-"
        if k and l:
            win = "Learned" if l["rmse"] < k["rmse"] else "KLT"
            verdict[cond] = {"KLT_cm": k["rmse"]*100, "Learned_cm": l["rmse"]*100,
                             "winner": win,
                             "ratio_KLT_over_Learned": k["rmse"]/l["rmse"]}
        print(f"  {cond:10s} {ks:>12s} {ls:>14s} {win:>10s}")

    out = {"sequence": name, "verdict": verdict,
           "detail": {c: {t: ({kk: rr[kk] for kk in ("rmse", "mean", "max", "n", "dur")}
                             if rr else None)
                          for t, rr in res[c].items()} for c in CONDS}}
    stem = "compareA2_backend" + (suffix or "")
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(out, indent=2))

    # ---------- plots: error-over-time + trajectories per condition ----------
    colors = {"KLT": "#c44", "Learned": "#3a7"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for j, cond in enumerate(CONDS):
        # error over time
        ax = axes[0, j]
        for tr in TRACKERS:
            r = res[cond].get(tr)
            if not r:
                continue
            err = np.linalg.norm(r["p_aln"] - r["p_gt"], axis=1)
            ax.plot(r["t"] - r["t"][0], err, color=colors[tr], lw=1.0,
                    label=f"{tr} (ATE {r['rmse']*100:.1f} cm)")
        ax.set_title(f"({'AB'[j]}) position error — {cond}")
        ax.set_xlabel("t [s]"); ax.set_ylabel("error [m]")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        # trajectory
        ax2 = axes[1, j]
        anyr = res[cond].get("KLT") or res[cond].get("Learned")
        if anyr:
            ax2.plot(anyr["p_gt"][:, 0], anyr["p_gt"][:, 1], "k-", lw=1.4,
                     label="ground truth")
        for tr in TRACKERS:
            r = res[cond].get(tr)
            if not r:
                continue
            ax2.plot(r["p_aln"][:, 0], r["p_aln"][:, 1], color=colors[tr],
                     lw=1.0, label=tr)
        ax2.set_title(f"({'CD'[j]}) trajectory — {cond}")
        ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
        ax2.axis("equal"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle(f"A2 end-to-end: learned vs classical front-end through OpenVINS "
                 f"MSCKF — {name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(RESULTS_DIR / f"{stem}.png", dpi=120)
    print(f"\n[saved] {RESULTS_DIR / f'{stem}.png'}")
    print(f"[saved] {RESULTS_DIR / f'{stem}.json'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="V1_03_difficult")
    ap.add_argument("--suffix", default="",
                    help="est-file suffix, e.g. _gated for the RANSAC-gated redo")
    a = ap.parse_args()
    main(a.seq, a.suffix)
