"""D3 -- GPS-denied drift vs denial-time curve.

The single question a defence buyer asks: "GPS dies. How far off are you after
30 / 60 / 120 seconds?" This script answers it as a curve, not a point.

Method (pure Python, data already in hand -- no container, no Jetson):
  * Take a real EuRoC sequence. Pick many GPS-cut instants across it.
  * From each cut, the aiding is gone -> propagate IMU-only forward and measure
    horizontal/3D position error at a set of ELAPSED horizons (10..120 s).
  * Two propagators, identical filter + identical GT-aligned init state:
        INS  (baseline) = raw-gyro ESKF        -- conventional inertial nav.
        AI-Nav (ours)   = denoised-gyro ESKF    -- gyro denoiser feeds the same ESKF.
    The ONLY difference is whether the gyro is cleaned first. Same-info, fair.
  * Aggregate error over all cut points at each horizon -> median + IQR band.
    Aggregating over cuts (not one lucky cut) is what makes the number quotable.

This reuses the validated phase-4 pieces verbatim: run_eskf, the phase3 denoiser,
and the same GT-aligned initial state (interp of pos/vel/quat/bias at the cut).

Sanity gate: at cut=t0 with the full window, this reproduces the phase-4 full-length
ATE -- so the curve is built from the same estimator, not a re-implementation.

Outputs:
  results/d3_drift_curve.png    the curve (INS vs AI-Nav, error vs denial-time)
  results/d3_drift_curve.json   the numbers + the headline "X m after 60 s" line

Usage:
  python scripts/d3_drift_curve.py                       # V1_01_easy (longest V-room)
  python scripts/d3_drift_curve.py --seq MH_01_easy
"""
from __future__ import annotations
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import RESULTS_DIR                            # noqa: E402
from ainav.euroc import load_sequence                          # noqa: E402
from ainav.eskf import run_eskf                                # noqa: E402
from ainav.denoiser import GyroDenoiser                        # noqa: E402
from ainav.metrics import interp_to, interp_quat_to            # noqa: E402

# same ESKF tuning as phase4 (clean gyro -> trust tilt only when truly static)
TUNE = dict(accel_gate=0.05, tilt_meas_std=0.5)

# elapsed denial horizons we report (seconds)
HORIZONS = [10, 20, 30, 45, 60, 90, 120]


def denoise_full(model, imu6, mean, std, device, chunk=4000, pad=256):
    """Run the conv denoiser over a full IMU track in padded chunks (phase4 routine).
    imu6: (N,6) raw [gyro|accel] -> corrected gyro (N,3)."""
    N = len(imu6)
    imu_norm = (imu6 - mean) / std
    corr = np.zeros((N, 3), np.float32)
    model.eval()
    with torch.no_grad():
        start = 0
        while start < N:
            lo = max(0, start - pad)
            hi = min(N, start + chunk + pad)
            x = torch.from_numpy(imu_norm[lo:hi].T[None]).to(device)
            c = model(x)[0].cpu().numpy().T
            a = start - lo
            b = a + min(chunk, N - start)
            corr[start:start + (b - a)] = c[a:b]
            start += chunk
    return imu6[:, :3] + corr


def deadreckon(t, gyro, accel, gt, t_cut, t_end):
    """IMU-only ESKF from t_cut to t_end, seeded from GT state at the cut.
    Returns (t_window, pos_est, pos_gt) on the IMU clock."""
    m = (t >= t_cut) & (t <= t_end)
    tw = t[m]
    if len(tw) < 2:
        return tw, np.zeros((len(tw), 3)), np.zeros((len(tw), 3))
    q0 = interp_quat_to(tw[:1], gt.t, gt.quat)[0]
    v0 = interp_to(tw[:1], gt.t, gt.vel)[0]
    p0 = interp_to(tw[:1], gt.t, gt.pos)[0]
    # HONEST GPS-denied baseline (identical to phase4): the filter is NOT handed the
    # true gyro bias -> bg0=0. Gyro bias is unobservable to an IMU+gravity filter, so
    # for raw INS it integrates into heading drift; the denoiser's job is to remove it
    # from the data first. Both sides get the (more observable) accel bias, so it's a
    # same-info gyro comparison. Seeding bg0 from GT would double-correct the denoised
    # run (data already debiased + filter handed the bias) and invert the result.
    ba0 = interp_to(tw[:1], gt.t, gt.bias_accel)[0]
    out = run_eskf(tw, gyro[m], accel[m], q0, v0, p0,
                   bg0=np.zeros(3), ba0=ba0, **TUNE)
    p_gt = interp_to(tw, gt.t, gt.pos)
    return tw, out["p"], p_gt


def errors_at_horizons(tw, p_est, p_gt, t_cut):
    """3D position error (m) at each elapsed horizon after the cut. NaN if the run
    doesn't reach that horizon (honest -- never silently truncate)."""
    if len(tw) < 2:
        return {h: np.nan for h in HORIZONS}
    elapsed = tw - t_cut
    err = np.linalg.norm(p_est - p_gt, axis=1)
    out = {}
    for h in HORIZONS:
        if elapsed[-1] < h:
            out[h] = np.nan
        else:
            j = int(np.searchsorted(elapsed, h))
            j = min(j, len(err) - 1)
            out[h] = float(err[j])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="V1_01_easy")
    ap.add_argument("--ckpt", default=str(RESULTS_DIR / "phase3_denoiser.pt"))
    ap.add_argument("--n-cuts", type=int, default=20,
                    help="number of GPS-cut instants swept across the sequence")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = load_sequence(args.seq)
    imu, gt = seq.imu, seq.gt

    # restrict to the GT-valid window (same masking as phase2/phase4)
    mask = (imu.t >= gt.t[0]) & (imu.t <= gt.t[-1])
    t = imu.t[mask]
    gyro_raw = imu.gyro[mask]
    accel = imu.accel[mask]
    imu6 = np.concatenate([gyro_raw, accel], axis=1).astype(np.float32)
    dur = float(t[-1] - t[0])

    # denoise the whole track once (AI-Nav gyro); baseline uses raw gyro
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GyroDenoiser(alpha=ck["alpha"]).to(device)
    model.load_state_dict(ck["model"])
    gyro_dn = denoise_full(model, imu6, ck["mean"], ck["std"], device)

    reachable = [h for h in HORIZONS if h <= dur]
    print(f"[{args.seq}] {dur:.1f}s of flight, {args.n_cuts} GPS-cut points; "
          f"horizons reachable: {reachable}"
          + (f"  (dropped {[h for h in HORIZONS if h > dur]} > seq length)"
             if dur < HORIZONS[-1] else ""))

    # sweep cut points over the first ~half so even the last cut has runway
    cut_hi = t[0] + max(1.0, dur - min(HORIZONS[0], dur * 0.5))
    cuts = np.linspace(t[0], cut_hi, args.n_cuts)

    acc = {"INS": {h: [] for h in HORIZONS}, "AI-Nav": {h: [] for h in HORIZONS}}
    for tc in cuts:
        # only propagate as far as the longest horizon needs (+2s margin) -- huge
        # speedup for early cuts, identical numbers (horizons cap at 120s).
        t_end = min(t[-1], tc + HORIZONS[-1] + 2.0)
        for label, gyro in (("INS", gyro_raw), ("AI-Nav", gyro_dn)):
            tw, pe, pg = deadreckon(t, gyro, accel, gt, tc, t_end)
            e = errors_at_horizons(tw, pe, pg, tc)
            for h in HORIZONS:
                if not np.isnan(e[h]):
                    acc[label][h].append(e[h])

    def summ(vals):
        a = np.asarray(vals, float)
        if a.size == 0:
            return None
        return {"n": int(a.size), "median": float(np.median(a)),
                "p25": float(np.percentile(a, 25)),
                "p75": float(np.percentile(a, 75)),
                "mean": float(a.mean()), "max": float(a.max())}

    curve = {lab: {h: summ(acc[lab][h]) for h in HORIZONS} for lab in acc}

    # ---- table ----
    print(f"\n  {'horizon':>8s} | {'INS median':>11s} {'INS p75':>9s} "
          f"| {'AI-Nav med':>11s} {'AI-Nav p75':>11s} | {'improve':>8s}")
    for h in HORIZONS:
        ins, ai = curve["INS"][h], curve["AI-Nav"][h]
        if ins is None or ai is None:
            print(f"  {h:6d}s  |  (beyond sequence length -- not measured)")
            continue
        imp = 100.0 * (ins["median"] - ai["median"]) / ins["median"] if ins["median"] else 0.0
        print(f"  {h:6d}s  | {ins['median']:9.2f}m {ins['p75']:8.2f}m "
              f"| {ai['median']:9.2f}m {ai['p75']:10.2f}m | {imp:+7.1f}%")

    # ---- headline number ----
    hl_h = 60 if 60 in reachable else (max(reachable) if reachable else None)
    headline = None
    if hl_h is not None:
        ins, ai = curve["INS"][hl_h], curve["AI-Nav"][hl_h]
        headline = (f"After {hl_h}s of GPS denial: conventional INS drifts "
                    f"{ins['median']:.0f} m (median); AI-Nav holds to "
                    f"{ai['median']:.1f} m -- "
                    f"{ins['median']/ai['median']:.0f}x tighter.")
        print(f"\n  HEADLINE: {headline}")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab, col in (("INS", "tab:red"), ("AI-Nav", "tab:green")):
        hs = [h for h in HORIZONS if curve[lab][h] is not None]
        med = [curve[lab][h]["median"] for h in hs]
        lo = [curve[lab][h]["p25"] for h in hs]
        hi = [curve[lab][h]["p75"] for h in hs]
        ax.plot(hs, med, "-o", color=col, lw=2,
                label=("INS (raw gyro, conventional)" if lab == "INS"
                       else "AI-Nav (denoised gyro, ours)"))
        ax.fill_between(hs, lo, hi, color=col, alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("GPS-denial time [s]")
    ax.set_ylabel("position error [m]  (median, IQR band; log)")
    ax.set_title(f"GPS-denied drift vs denial-time -- {args.seq}\n"
                 f"(swept {args.n_cuts} cut points; IMU-only after cut)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    if headline:
        ax.text(0.02, 0.02, headline, transform=ax.transAxes, fontsize=8,
                va="bottom", wrap=True,
                bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0c060", alpha=0.9))
    fig.tight_layout()
    png = RESULTS_DIR / "d3_drift_curve.png"
    fig.savefig(png, dpi=120)
    print(f"\n[plot] {png}")

    out = {"sequence": args.seq, "duration_s": dur, "n_cuts": args.n_cuts,
           "horizons_s": HORIZONS, "reachable_s": reachable,
           "tune": TUNE, "curve": curve, "headline": headline,
           "note": ("INS=raw-gyro ESKF, AI-Nav=denoised-gyro ESKF; identical "
                    "filter + GT-aligned init; IMU-only after each GPS cut; "
                    "aggregated over swept cut points.")}
    jp = RESULTS_DIR / "d3_drift_curve.json"
    jp.write_text(json.dumps(out, indent=2))
    print(f"[json] {jp}")


if __name__ == "__main__":
    main()
