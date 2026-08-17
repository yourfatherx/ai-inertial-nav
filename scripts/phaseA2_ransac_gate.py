"""A2 REDO (fair head-to-head): RANSAC-gate the banked track files.

The banked A2 run fed raw external tracks into OpenVINS' SIM path, which SKIPS the
native front-end outlier rejection -- so KLT's long high-leverage tracks carried
~15% outliers straight into the filter and diverged 3/4 runs. This script adds that
missing gate back, IDENTICALLY for both front-ends, as a pure post-process on the
exact banked track files. No re-tracking, no GPU -> the ONLY variable vs the banked
run is the epipolar gate. That is the fair test A2's verdict was missing.

In : results/a2_backend/<seq>/tracks_{KLT,Learned}_{clean,lowlight}.txt   (banked)
Out: results/a2_backend/<seq>/tracks_{...}_gated.txt                      (gated)

Prints a pre-flight table (obs before/after, inlier ratio before/after, mean track
len) so we can judge whether the gate culls outliers WITHOUT starving the filter --
before spending any container runs.

Usage: python scripts/phaseA2_ransac_gate.py [V1_03_difficult] [--thresh 1.0]
"""
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.frontend import FrameObs, ransac_gate, _epipolar_inlier_ratio, _track_lengths  # noqa: E402
from ainav.config import RESULTS_DIR                                      # noqa: E402

TRACKERS = ["KLT", "Learned"]
CONDS = ["clean", "lowlight"]


def load_tracks(path: Path):
    """`t id u v` lines -> list[FrameObs], grouped by timestamp (frame)."""
    a = np.loadtxt(path)
    # group consecutive rows by timestamp (file is written frame-major, in order)
    ts = a[:, 0]
    frames = []
    # unique timestamps preserving order
    _, first_idx = np.unique(ts, return_index=True)
    bounds = np.sort(first_idx)
    for s, e in zip(bounds, list(bounds[1:]) + [len(a)]):
        blk = a[s:e]
        frames.append(FrameObs(ids=blk[:, 1].astype(np.int64),
                               pts=blk[:, 2:4].astype(np.float32)))
    return frames, ts


def write_tracks(path: Path, frames, ts_per_frame):
    n = 0
    with open(path, "w") as f:
        for fo, t in zip(frames, ts_per_frame):
            for i, (u, v) in zip(fo.ids, fo.pts):
                f.write(f"{t:.9f} {int(i)} {float(u):.4f} {float(v):.4f}\n")
                n += 1
    return n


def stats(frames):
    obs = sum(len(fo.ids) for fo in frames)
    lens = _track_lengths(frames)
    mlen = float(np.mean(list(lens.values()))) if lens else 0.0
    inl = _epipolar_inlier_ratio(frames)
    return obs, mlen, inl


def main(seq, thresh, conf):
    d = RESULTS_DIR / "a2_backend" / seq
    print(f"[gate] {seq}  thresh={thresh}px conf={conf}\n")
    print(f"  {'file':22s} {'obs in':>9s} {'obs out':>9s} {'kept%':>6s} "
          f"{'inl in':>7s} {'inl out':>8s} {'mlen in':>8s} {'mlen out':>9s}")
    for tr in TRACKERS:
        for cond in CONDS:
            src = d / f"tracks_{tr}_{cond}.txt"
            if not src.exists():
                print(f"  {src.name:22s}  (missing)")
                continue
            frames, _ = load_tracks(src)
            # timestamp per frame = first row's ts in each block
            tsf = [None] * len(frames)
            # recover per-frame ts by re-reading unique order
            a = np.loadtxt(src)
            uniq = []
            seen = set()
            for t in a[:, 0]:
                if t not in seen:
                    seen.add(t); uniq.append(t)
            tsf = uniq
            obs0, ml0, inl0 = stats(frames)
            gated = ransac_gate(frames, thresh_px=thresh, conf=conf)
            obs1, ml1, inl1 = stats(gated)
            out = d / f"tracks_{tr}_{cond}_gated.txt"
            write_tracks(out, gated, tsf)
            kept = 100.0 * obs1 / obs0 if obs0 else 0.0
            print(f"  {src.name:22s} {obs0:9d} {obs1:9d} {kept:5.1f}% "
                  f"{inl0:7.3f} {inl1:8.3f} {ml0:8.1f} {ml1:9.1f}")
    print(f"\n[gate] wrote *_gated.txt -> {d}")
    print("[gate] verdict guide: inl_out should approach ~1.0 (outliers culled) "
          "while kept% stays high enough to constrain the filter (want >~70%).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="V1_03_difficult")
    ap.add_argument("--thresh", type=float, default=1.0)
    ap.add_argument("--conf", type=float, default=0.99)
    a = ap.parse_args()
    main(a.seq, a.thresh, a.conf)
