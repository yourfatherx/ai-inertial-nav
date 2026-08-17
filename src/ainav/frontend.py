"""Feature front-ends for visual-inertial odometry (Approach A, Phase A2).

A VIO front-end turns a stream of frames into *tracks*: the same physical
point followed across consecutive frames. The back-end (MSCKF/EKF) only ever
sees these tracks, so front-end quality caps VIO accuracy. Two are provided
behind one interface so they can be benchmarked head-to-head:

  KLTTracker      - Shi-Tomasi corners + pyramidal Lucas-Kanade optical flow.
                    This is what OpenVINS' TrackKLT does (our A1 baseline).
  LearnedTracker  - SuperPoint keypoints + LightGlue matching, frame-to-frame.
                    A learned front-end; the thing A2 is testing.

Both consume grayscale uint8 frames and emit the same TrackResult, so the
evaluation script is agnostic to which one produced the tracks.

Track identity across a whole sequence is maintained the same way for both:
each frame yields a set of (id, x, y) observations; a feature keeps its id as
long as it is re-observed, and dies when it is not. Mean track length (how many
frames a feature survives) and tracks-per-frame are the headline front-end
metrics -- longer, denser tracks give the back-end more constraints.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2


@dataclass
class FrameObs:
    """Feature observations in a single frame."""
    ids: np.ndarray          # (N,) int64 persistent track ids
    pts: np.ndarray          # (N,2) float32 pixel coords (x, y)


@dataclass
class TrackResult:
    """Per-frame observations plus the summary metrics we compare on."""
    frames: list             # list[FrameObs], one per processed frame
    name: str                # tracker name, for plots
    # summary metrics (filled by summarize())
    mean_track_len: float = 0.0
    median_track_len: float = 0.0
    mean_tracks_per_frame: float = 0.0
    inlier_ratio: float = 0.0
    n_features_total: int = 0
    per_frame_count: np.ndarray = field(default_factory=lambda: np.empty(0))


# --------------------------------------------------------------------------- #
#  metrics
# --------------------------------------------------------------------------- #
def _track_lengths(frames: list) -> dict:
    """id -> number of frames that id was observed in."""
    lengths: dict = {}
    for fo in frames:
        for i in fo.ids:
            lengths[int(i)] = lengths.get(int(i), 0) + 1
    return lengths


def _epipolar_inlier_ratio(frames: list) -> float:
    """Fraction of consecutive-frame matches consistent with a fundamental
    matrix (RANSAC). A clean front-end tracks rigid scene points, so most
    matches obey epipolar geometry; a sloppy one drifts and produces outliers.
    """
    inl, tot = 0, 0
    for a, b in zip(frames[:-1], frames[1:]):
        # shared ids between the two frames = putative matches
        ida = {int(i): p for i, p in zip(a.ids, a.pts)}
        idb = {int(i): p for i, p in zip(b.ids, b.pts)}
        common = list(ida.keys() & idb.keys())
        if len(common) < 8:
            continue
        pa = np.array([ida[i] for i in common], dtype=np.float32)
        pb = np.array([idb[i] for i in common], dtype=np.float32)
        _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is None:
            continue
        inl += int(mask.sum())
        tot += len(common)
    return inl / tot if tot else 0.0


def ransac_gate(frames: list, thresh_px: float = 1.0, conf: float = 0.99) -> list:
    """Reject epipolar-outlier observations, mirroring OpenVINS' native front-end
    rejection that the external-track SIM feed bypasses.

    For each consecutive frame pair, run F-matrix RANSAC on the shared-id matches
    (same test as `_epipolar_inlier_ratio`). An observation is DROPPED from frame b
    if its match to frame a is a RANSAC outlier. This is tracker-agnostic -- it sees
    only (id, u, v), so KLT and Learned tracks are gated by the identical rule, which
    is what makes the head-to-head fair.

    Returns a new list[FrameObs] with outlier observations removed. Frame 0 is kept
    as-is (no previous frame to test against).
    """
    if not frames:
        return frames
    gated = [FrameObs(ids=frames[0].ids.copy(), pts=frames[0].pts.copy())]
    prev = frames[0]
    for b in frames[1:]:
        ida = {int(i): p for i, p in zip(prev.ids, prev.pts)}
        idb = {int(i): p for i, p in zip(b.ids, b.pts)}
        common = list(ida.keys() & idb.keys())
        keep_ids = set(int(i) for i in b.ids)      # default: keep all (incl. new)
        if len(common) >= 8:
            pa = np.array([ida[i] for i in common], dtype=np.float32)
            pb = np.array([idb[i] for i in common], dtype=np.float32)
            _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, thresh_px, conf)
            if mask is not None:
                m = mask.reshape(-1).astype(bool)
                outliers = {common[k] for k in range(len(common)) if not m[k]}
                keep_ids -= outliers               # drop only tested outliers
        sel = np.array([int(i) in keep_ids for i in b.ids], dtype=bool)
        kept = FrameObs(ids=b.ids[sel], pts=b.pts[sel])
        gated.append(kept)
        prev = kept                                # chain on kept points
    return gated


def summarize(res: TrackResult) -> TrackResult:
    lengths = _track_lengths(res.frames)
    lv = np.array(list(lengths.values())) if lengths else np.zeros(1)
    counts = np.array([len(f.ids) for f in res.frames])
    res.mean_track_len = float(lv.mean())
    res.median_track_len = float(np.median(lv))
    res.mean_tracks_per_frame = float(counts.mean())
    res.n_features_total = len(lengths)
    res.per_frame_count = counts
    res.inlier_ratio = _epipolar_inlier_ratio(res.frames)
    return res


# --------------------------------------------------------------------------- #
#  classical KLT front-end (mirrors OpenVINS TrackKLT)
# --------------------------------------------------------------------------- #
class KLTTracker:
    name = "KLT"

    def __init__(self, max_features: int = 400, quality: float = 0.01,
                 min_dist: int = 12, redetect_below: float = 0.7):
        self.max_features = max_features
        self.quality = quality
        self.min_dist = min_dist
        self.redetect_below = redetect_below  # top up when tracks thin out
        self._next_id = 0

    def _detect(self, img, mask=None):
        pts = cv2.goodFeaturesToTrack(
            img, self.max_features, self.quality, self.min_dist, mask=mask)
        if pts is None:
            return np.empty((0, 2), np.float32)
        return pts.reshape(-1, 2).astype(np.float32)

    def run(self, frames_iter) -> TrackResult:
        out = []
        prev_img = None
        prev_pts = np.empty((0, 2), np.float32)
        prev_ids = np.empty((0,), np.int64)
        for img in frames_iter:
            if prev_img is None:
                pts = self._detect(img)
                ids = np.arange(self._next_id, self._next_id + len(pts))
                self._next_id += len(pts)
            else:
                if len(prev_pts):
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                        prev_img, img, prev_pts, None,
                        winSize=(21, 21), maxLevel=3)
                    st = st.reshape(-1).astype(bool)
                    pts = nxt[st]
                    ids = prev_ids[st]
                    # drop points that flowed out of the image
                    h, w = img.shape
                    inb = (pts[:, 0] >= 0) & (pts[:, 0] < w) & \
                          (pts[:, 1] >= 0) & (pts[:, 1] < h)
                    pts, ids = pts[inb], ids[inb]
                else:
                    pts = np.empty((0, 2), np.float32)
                    ids = np.empty((0,), np.int64)
                # redetect to keep the map populated (OpenVINS does this too)
                if len(pts) < self.redetect_below * self.max_features:
                    mask = np.full(img.shape, 255, np.uint8)
                    for x, y in pts.astype(int):
                        cv2.circle(mask, (x, y), self.min_dist, 0, -1)
                    new = self._detect(img, mask)
                    room = self.max_features - len(pts)
                    new = new[:max(0, room)]
                    new_ids = np.arange(self._next_id, self._next_id + len(new))
                    self._next_id += len(new)
                    pts = np.vstack([pts, new]) if len(new) else pts
                    ids = np.concatenate([ids, new_ids]) if len(new) else ids
            out.append(FrameObs(ids=ids.astype(np.int64),
                                pts=pts.astype(np.float32)))
            prev_img, prev_pts, prev_ids = img, pts, ids
        return summarize(TrackResult(frames=out, name=self.name))


# --------------------------------------------------------------------------- #
#  learned front-end (SuperPoint + LightGlue)
# --------------------------------------------------------------------------- #
class LearnedTracker:
    name = "SuperPoint+LightGlue"

    def __init__(self, max_features: int = 400, device: str | None = None,
                 match_thresh_px: float = 3.0):
        import torch
        from lightglue import LightGlue, SuperPoint
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sp = SuperPoint(max_num_keypoints=max_features).eval().to(self.device)
        self.lg = LightGlue(features="superpoint").eval().to(self.device)
        self.match_thresh_px = match_thresh_px
        self._next_id = 0

    def _extract(self, img):
        t = self.torch.from_numpy(img.astype(np.float32) / 255.0)
        t = t[None, None].to(self.device)
        with self.torch.no_grad():
            return self.sp.extract(t)

    def run(self, frames_iter) -> TrackResult:
        from lightglue.utils import rbd
        out = []
        prev_feat = None
        prev_ids = None       # (K,) id for each keypoint in prev frame
        for img in frames_iter:
            feat = self._extract(img)
            kpts = rbd(feat)["keypoints"].cpu().numpy()
            if prev_feat is None:
                ids = np.arange(self._next_id, self._next_id + len(kpts))
                self._next_id += len(kpts)
            else:
                with self.torch.no_grad():
                    m = self.lg({"image0": prev_feat, "image1": feat})
                matches = rbd(m)["matches"].cpu().numpy()  # (M,2) idx pairs
                ids = np.full(len(kpts), -1, np.int64)
                for i0, i1 in matches:
                    ids[i1] = prev_ids[i0]          # inherit id from prev frame
                fresh = ids < 0
                nnew = int(fresh.sum())
                ids[fresh] = np.arange(self._next_id, self._next_id + nnew)
                self._next_id += nnew
            out.append(FrameObs(ids=ids.astype(np.int64),
                                pts=kpts.astype(np.float32)))
            prev_feat, prev_ids = feat, ids
        return summarize(TrackResult(frames=out, name=self.name))


# --------------------------------------------------------------------------- #
#  frame degradations (defence scenario: blur / low-light / low-texture)
# --------------------------------------------------------------------------- #
def degrade(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Apply a controlled corruption to a grayscale uint8 frame."""
    if kind == "clean":
        return img
    if kind == "blur":                         # motion blur, ~15px kernel
        k = 15
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kern)
    if kind == "lowlight":                      # crush exposure + read noise
        dark = (img.astype(np.float32) * 0.18)
        noise = rng.normal(0, 6.0, img.shape)
        return np.clip(dark + noise, 0, 255).astype(np.uint8)
    if kind == "lowtexture":                    # heavy smoothing kills detail
        return cv2.GaussianBlur(img, (0, 0), 6.0)
    raise ValueError(f"unknown degradation {kind!r}")
