"""Phase A2 deliverable: learned vs classical VIO front-end, clean and degraded.

Runs two feature front-ends -- classical KLT (what OpenVINS/A1 uses) and a
learned SuperPoint+LightGlue tracker -- over the same EuRoC frames under four
conditions: clean, motion-blur, low-light, and low-texture. The question A2
asks is not "who wins on clean data" (KLT is excellent there) but "who survives
when the camera degrades" -- the GPS-denied UAV flying at dusk, through haze,
or maneuvering hard enough to smear the image.

Outputs:
  results/phaseA2_frontend_<seq>.png   4-panel: track length, inlier ratio,
                                       tracks-over-time under low-light, and the
                                       degradation thumbnails.
  results/phaseA2_frontend_<seq>.json  every metric, both trackers, all conditions.

Front-end quality (mean track length, tracks/frame, epipolar inlier ratio)
directly caps VIO accuracy: the MSCKF back-end can only be as good as the tracks
it is fed. So a front-end that holds longer, cleaner tracks under degradation is
a strictly better foundation for the A2 learned-VIO pipeline.
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.camera import load_camera, read_gray, undistort   # noqa: E402
from ainav.frontend import KLTTracker, LearnedTracker, degrade  # noqa: E402
from ainav.config import RESULTS_DIR                          # noqa: E402

CONDITIONS = ["clean", "blur", "lowlight", "lowtexture"]
COND_LABEL = {"clean": "clean", "blur": "motion blur",
              "lowlight": "low light", "lowtexture": "low texture"}


def main(name: str = "V1_01_easy", n_frames: int = 200) -> None:
    cam = load_camera(name, "cam0")
    n = min(n_frames, len(cam.files))
    base = [undistort(read_gray(f), cam.calib) for f in cam.files[:n]]
    print(f"[A2] {name}: {n} undistorted frames, {base[0].shape}")

    rng = np.random.default_rng(0)
    trackers = {"KLT": KLTTracker(max_features=400),
                "SuperPoint+LightGlue": LearnedTracker(max_features=400)}
    print(f"[A2] learned tracker device: {trackers['SuperPoint+LightGlue'].device}")

    results = {}   # cond -> tracker -> metrics dict
    curves = {}    # cond -> tracker -> per_frame_count
    for cond in CONDITIONS:
        frames = [degrade(im, cond, np.random.default_rng(i))
                  for i, im in enumerate(base)]
        results[cond] = {}
        curves[cond] = {}
        for tname, T in trackers.items():
            T._next_id = 0                       # fresh id space per run
            r = T.run(iter(frames))
            results[cond][tname] = dict(
                mean_track_len=r.mean_track_len,
                median_track_len=r.median_track_len,
                mean_tracks_per_frame=r.mean_tracks_per_frame,
                inlier_ratio=r.inlier_ratio,
                n_features_total=r.n_features_total)
            curves[cond][tname] = r.per_frame_count
            print(f"  {cond:11s} {tname:22s} "
                  f"meanLen={r.mean_track_len:6.2f} "
                  f"inlier={r.inlier_ratio:.3f} "
                  f"perFrame={r.mean_tracks_per_frame:6.1f}")

    _plot(name, base, results, curves)
    _report(name, results)

    out_json = RESULTS_DIR / f"phaseA2_frontend_{name}.json"
    with open(out_json, "w") as f:
        json.dump({"seq": name, "n_frames": n, "results": results}, f, indent=2)
    print(f"[A2] wrote {out_json}")


def _retention(results, tname, metric):
    """degraded/clean ratio, averaged over the 3 degradations (higher=robust)."""
    base = results["clean"][tname][metric]
    if base == 0:
        return 0.0
    vals = [results[c][tname][metric] / base
            for c in CONDITIONS if c != "clean"]
    return float(np.mean(vals))


def _report(name, results):
    print(f"\n[A2] robustness retention (mean over 3 degradations, higher=better)")
    for tname in results["clean"]:
        rl = _retention(results, tname, "mean_track_len")
        ri = _retention(results, tname, "inlier_ratio")
        print(f"   {tname:22s} track-len {rl*100:5.1f}%   inlier {ri*100:5.1f}%")


def _plot(name, base, results, curves):
    tnames = list(results["clean"].keys())
    colors = {"KLT": "#c44", "SuperPoint+LightGlue": "#3a7"}
    x = np.arange(len(CONDITIONS))
    w = 0.38

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (A) mean track length
    for k, tname in enumerate(tnames):
        vals = [results[c][tname]["mean_track_len"] for c in CONDITIONS]
        ax[0, 0].bar(x + (k - 0.5) * w, vals, w, label=tname,
                     color=colors[tname])
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax[0, 0].set_ylabel("mean track length (frames)")
    ax[0, 0].set_title("(A) How long features survive")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(axis="y", alpha=0.3)

    # (B) epipolar inlier ratio
    for k, tname in enumerate(tnames):
        vals = [results[c][tname]["inlier_ratio"] for c in CONDITIONS]
        ax[0, 1].bar(x + (k - 0.5) * w, vals, w, label=tname,
                     color=colors[tname])
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels([COND_LABEL[c] for c in CONDITIONS])
    ax[0, 1].set_ylabel("epipolar inlier ratio")
    ax[0, 1].set_ylim(0, 1.05)
    ax[0, 1].set_title("(B) Track geometric consistency")
    ax[0, 1].grid(axis="y", alpha=0.3)

    # (C) tracks-per-frame over time under low light
    cond = "lowlight"
    for tname in tnames:
        ax[1, 0].plot(curves[cond][tname], color=colors[tname], label=tname)
    ax[1, 0].set_xlabel("frame index")
    ax[1, 0].set_ylabel("active tracks")
    ax[1, 0].set_title(f"(C) Tracks sustained under {COND_LABEL[cond]}")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    # (D) degradation thumbnails of one sample frame
    fi = min(60, len(base) - 1)
    montage = []
    for c in CONDITIONS:
        d = degrade(base[fi], c, np.random.default_rng(fi))
        montage.append(d)
    strip = np.hstack(montage)
    ax[1, 1].imshow(strip, cmap="gray", vmin=0, vmax=255)
    ax[1, 1].set_title("(D) Frame under each condition "
                       "(clean | blur | low light | low texture)")
    ax[1, 1].axis("off")

    fig.suptitle(f"A2 front-end robustness -- {name}: learned vs classical (KLT)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS_DIR / f"phaseA2_frontend_{name}.png"
    fig.savefig(out, dpi=130)
    print(f"[A2] wrote {out}")


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else "V1_01_easy"
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    main(seq, nf)
