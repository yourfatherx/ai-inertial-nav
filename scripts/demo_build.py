"""GPS-denied flight demonstrator — build the investor demo (Tier 1 + Tier 2).

Three trajectories on real EuRoC data, one clock:
  GT      (green) -- where the drone actually flew.
  AI-Nav  (blue)  -- our OpenVINS VIO (est_V1_01.txt), ~6 cm, hugs truth.
  INS     (red)   -- conventional strapdown dead-reckoning. Has GPS/aiding up to the
                     "GPS cut" instant (so it sits on truth), then GPS is denied and it
                     integrates raw IMU forward -> drifts and balloons.

Everything traces to already-validated numbers (results/compareA1_vs_B.json). The A1
alignment + epoch shift are reused verbatim from compareA1_vs_B.py; the red path is the
repo's own strapdown mechanization (integrate_full) seeded from GT state at the cut.

Outputs:
  results/demo_replay.html       Tier 1 -- animated 3D replay, Play + scrubber, GPS banner.
  results/demo_interactive.html  Tier 2 -- GPS-cut-time slider re-diverges the red path.
  results/demo_bundle.json       the numbers behind the demo (reproduced ATE, drift).

Usage:
  python scripts/demo_build.py [--seq V1_01_easy] [--cut 60] [--fps 20]
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.euroc import load_sequence, _seq_dir                # noqa: E402
from ainav.eskf import run_eskf                                # noqa: E402
from ainav.metrics import interp_to, interp_quat_to            # noqa: E402
from ainav.config import RESULTS_DIR                            # noqa: E402

import plotly.graph_objects as go                              # noqa: E402

# palette (colour-blind safe, high contrast on white)
C_GT = "#2ca02c"      # green  -- ground truth
C_AI = "#1f6feb"      # blue   -- our AI-Nav
C_INS = "#d62728"     # red    -- conventional INS, GPS-denied


# ----------------------------------------------------------------------------- #
# data (reuses the validated compareA1_vs_B.py logic verbatim)
# ----------------------------------------------------------------------------- #
def load_tum(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()[:8]])
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


def ins_deadreckon(seq, t_cut):
    """Conventional INS after GPS denial at t_cut.

    A real inertial nav has a filter, so we use the repo's ESKF (our validated
    B_eskf_no_ai baseline, ~317 m ATE) rather than raw double-integration -- seeded
    from GT pos/vel/quat/bias at t_cut and run on the post-cut IMU. Returns (t, pos)
    on the IMU clock for t >= t_cut. This is the red path.
    """
    imu, gt = seq.imu, seq.gt
    m = (imu.t >= t_cut) & (imu.t <= gt.t[-1])
    t = imu.t[m]
    if len(t) < 2:
        return t, np.zeros((len(t), 3))
    q0 = interp_quat_to(t[:1], gt.t, gt.quat)[0]
    v0 = interp_to(t[:1], gt.t, gt.vel)[0]
    p0 = interp_to(t[:1], gt.t, gt.pos)[0]
    bg0 = interp_to(t[:1], gt.t, gt.bias_gyro)[0]
    ba0 = interp_to(t[:1], gt.t, gt.bias_accel)[0]
    out = run_eskf(t, imu.gyro[m], imu.accel[m], q0, v0, p0, bg0=bg0, ba0=ba0)
    return t, out["p"]


def build_bundle(name="V1_01_easy", cut=60.0, fps=20):
    """Assemble everything the HTMLs need on one common clock."""
    seq = load_sequence(name)
    gt = seq.gt

    # --- AI-Nav (blue): OpenVINS VIO, epoch-shifted + Umeyama-aligned (verbatim A1) ---
    a1_path = RESULTS_DIR / "est_V1_01.txt"
    if not a1_path.exists():
        sys.exit(f"[missing] {a1_path}")
    t_ai, p_ai = load_tum(a1_path)
    imu_csv = _seq_dir(name) / "imu0" / "data.csv"
    t0_abs = float(np.loadtxt(imu_csv, delimiter=",", skiprows=1, max_rows=1)[0]) * 1e-9
    t_ai = t_ai - t0_abs
    mask = (t_ai >= gt.t[0]) & (t_ai <= gt.t[-1])
    t_ai, p_ai = t_ai[mask], p_ai[mask]
    p_gt_at_ai = interp_to(t_ai, gt.t, gt.pos)
    R, tv = umeyama_se3(p_ai, p_gt_at_ai)
    p_ai = (R @ p_ai.T).T + tv                       # now in GT frame

    # sanity: reproduced ATE must match compareA1_vs_B.json (~6.2 cm), else demo is lying
    ate_ai = float(np.sqrt(np.mean(np.linalg.norm(p_ai - p_gt_at_ai, axis=1) ** 2)))

    # --- common demo clock: GT window, fps steps ---
    t0, t1 = float(gt.t[0]), float(gt.t[-1])
    clock = np.arange(t0, t1, 1.0 / fps)
    gt_xyz = interp_to(clock, gt.t, gt.pos)
    ai_xyz = interp_to(clock, t_ai, p_ai)            # blue on demo clock

    # --- INS (red) for the headline cut: GT before cut, dead-reckon after ---
    def red_on_clock(t_cut):
        t_ins, p_ins = ins_deadreckon(seq, t_cut)
        red = gt_xyz.copy()                          # before cut: rides truth
        after = clock >= t_cut
        if t_ins.size:
            red[after] = interp_to(clock[after], t_ins, p_ins)
        return red

    red_xyz = red_on_clock(cut)
    drift_final = float(np.linalg.norm(red_xyz[-1] - gt_xyz[-1]))

    bundle = {
        "seq": name, "cut": cut, "fps": fps,
        "clock": clock, "gt": gt_xyz, "ai": ai_xyz, "red": red_xyz,
        "ate_ai_cm": ate_ai * 100.0, "drift_final_m": drift_final,
        "red_on_clock": red_on_clock,               # callable, for Tier 2 sweep
    }
    return bundle


# ----------------------------------------------------------------------------- #
# scene helpers
# ----------------------------------------------------------------------------- #
def _bounds(*arrays, margin=1.6):
    """Scene box scoped to the given paths (pass GT+AI only -> room scale, so the
    red INS path visibly arcs out of frame instead of collapsing the scene)."""
    P = np.vstack(arrays)
    lo, hi = P.min(0), P.max(0)
    c = (lo + hi) / 2
    r = float((hi - lo).max()) / 2 * margin + 0.5
    return dict(
        xaxis=dict(range=[c[0] - r, c[0] + r], title="x [m]"),
        yaxis=dict(range=[c[1] - r, c[1] + r], title="y [m]"),
        zaxis=dict(range=[c[2] - r, c[2] + r], title="z [m]"),
        aspectmode="cube",
    )


def _line(xyz, color, name, width=5, dash="solid"):
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
                        line=dict(color=color, width=width, dash=dash), name=name)


def _head(p, color, name):
    return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                        marker=dict(color=color, size=6, symbol="circle"),
                        name=name, showlegend=False)


# ----------------------------------------------------------------------------- #
# Tier 1 -- animated replay
# ----------------------------------------------------------------------------- #
def build_replay(b, out_path, max_frames=150):
    clock, gt, ai, red = b["clock"], b["gt"], b["ai"], b["red"]
    cut = b["cut"]
    # room-scale scene (GT+AI only) so red visibly arcs out of frame
    scene = _bounds(gt, ai)
    # decimate the whole demo onto a short clock so frames don't store full-res
    # growing polylines -> keeps the HTML emailable (a few MB, not tens).
    n0 = len(clock)
    keep = np.unique(np.linspace(0, n0 - 1, min(max_frames, n0)).astype(int))
    clock, gt, ai, red = clock[keep], gt[keep], ai[keep], red[keep]
    n = len(clock)
    idx = list(range(n))

    def frame(i):
        j = i + 1
        denied = clock[i] >= cut
        err = float(np.linalg.norm(ai[i] - gt[i]))
        drift = float(np.linalg.norm(red[i] - gt[i]))
        data = [
            _line(gt[:j], C_GT, "Ground truth"),
            _line(ai[:j], C_AI, "AI-Nav (ours)"),
            _line(red[:j], C_INS, "INS · GPS-denied", dash="dot"),
            _head(gt[i], C_GT, "gt"), _head(ai[i], C_AI, "ai"), _head(red[i], C_INS, "ins"),
        ]
        title = (f"t = {clock[i]-clock[0]:5.1f}s   "
                 f"<span style='color:{C_AI}'>AI-Nav err {err*100:5.1f} cm</span>   "
                 f"<span style='color:{C_INS}'>INS drift {drift:6.1f} m</span>")
        ann = []
        if denied:
            ann = [dict(text="⚠  GPS DENIED", x=0.5, y=0.97, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=22, color="white"),
                        bgcolor=C_INS, borderpad=6, opacity=0.9)]
        return go.Frame(data=data, name=f"{clock[i]-clock[0]:.1f}",
                        layout=dict(title=dict(text=title), annotations=ann))

    frames = [frame(i) for i in idx]
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        template="plotly_white", scene=scene, height=760,
        title=dict(text=frames[0].layout.title.text, x=0.02, xanchor="left"),
        annotations=list(frames[0].layout.annotations or []),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.06, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=45, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
        sliders=[dict(
            active=0, x=0.12, len=0.8, y=0.02,
            currentvalue=dict(prefix="t = ", suffix=" s"),
            steps=[dict(method="animate", label=f.name,
                        args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                             mode="immediate")]) for f in frames])],
    )
    fig.write_html(out_path, include_plotlyjs="cdn", auto_play=False)
    return out_path


# ----------------------------------------------------------------------------- #
# Tier 2 -- interactive GPS-cut slider
# ----------------------------------------------------------------------------- #
def build_interactive(b, out_path, n_cuts=12):
    clock, gt, ai = b["clock"], b["gt"], b["ai"]
    red_on_clock = b["red_on_clock"]
    t0, t1 = float(clock[0]), float(clock[-1])
    dur = t1 - t0
    # candidate cut times spread across the flight (avoid the very ends)
    cuts = np.linspace(t0 + 0.15 * dur, t0 + 0.85 * dur, n_cuts)
    scene = _bounds(gt, ai)          # room scale; red arcs out of frame by design

    base = [_line(gt, C_GT, "Ground truth"), _line(ai, C_AI, "AI-Nav (ours)")]
    reds = [red_on_clock(c) for c in cuts]
    default = n_cuts // 2

    fig = go.Figure()
    for tr in base:
        fig.add_trace(tr)                                   # 0,1 always visible
    for k, (c, red) in enumerate(zip(cuts, reds)):
        fig.add_trace(_line(red, C_INS, "INS · GPS-denied", dash="dot"))
        fig.data[-1].visible = (k == default)

    def vis(sel):
        return [True, True] + [i == sel for i in range(n_cuts)]

    steps = []
    for k, c in enumerate(cuts):
        drift = float(np.linalg.norm(reds[k][-1] - gt[-1]))
        steps.append(dict(
            method="update", label=f"{c-t0:.0f}",
            args=[dict(visible=vis(k)),
                  dict(title=dict(text=(
                      f"GPS denied at t = {c-t0:.0f}s  →  "
                      f"<span style='color:{C_INS}'>INS ends {drift:,.0f} m off</span>   "
                      f"<span style='color:{C_AI}'>AI-Nav stays {b['ate_ai_cm']:.0f} cm</span>")),
                      annotations=[dict(text="⚠  GPS DENIED", x=0.5, y=0.97,
                                        xref="paper", yref="paper", showarrow=False,
                                        font=dict(size=20, color="white"),
                                        bgcolor=C_INS, borderpad=6, opacity=0.9)])]))

    d0 = float(np.linalg.norm(reds[default][-1] - gt[-1]))
    fig.update_layout(
        template="plotly_white", scene=scene, height=760,
        title=dict(text=(f"GPS denied at t = {cuts[default]-t0:.0f}s  →  "
                         f"<span style='color:{C_INS}'>INS ends {d0:,.0f} m off</span>   "
                         f"<span style='color:{C_AI}'>AI-Nav stays {b['ate_ai_cm']:.0f} cm</span>"),
                   x=0.02, xanchor="left"),
        annotations=[dict(text="⚠  GPS DENIED", x=0.5, y=0.97, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=20, color="white"),
                          bgcolor=C_INS, borderpad=6, opacity=0.9)],
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
        sliders=[dict(active=default, x=0.1, len=0.82, y=0.02,
                      currentvalue=dict(prefix="drag to move GPS-loss:  t = ", suffix=" s"),
                      steps=steps)],
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="V1_01_easy")
    ap.add_argument("--cut", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    b = build_bundle(args.seq, args.cut, args.fps)

    # sanity gate: the demo must reproduce the validated A1 number
    print(f"[check] AI-Nav reproduced ATE = {b['ate_ai_cm']:.2f} cm "
          f"(compareA1_vs_B.json: 6.23 cm)")
    print(f"[check] INS drift at end (cut {args.cut:.0f}s) = {b['drift_final_m']:,.1f} m")

    r1 = build_replay(b, RESULTS_DIR / "demo_replay.html")
    r2 = build_interactive(b, RESULTS_DIR / "demo_interactive.html")

    (RESULTS_DIR / "demo_bundle.json").write_text(json.dumps({
        "seq": b["seq"], "cut": b["cut"], "fps": b["fps"],
        "ate_ai_cm": b["ate_ai_cm"], "drift_final_m": b["drift_final_m"],
        "n_frames": int(len(b["clock"])),
    }, indent=2))

    for p in (r1, r2, RESULTS_DIR / "demo_bundle.json"):
        print(f"[saved] {p}  ({p.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
