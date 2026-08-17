"""Build a ROS1 .bag from an EuRoC ASL mav0/ folder (offline, no ROS install).

Why: the ETH .bag mirror is unreachable, but the full ASL folders (cam0, cam1,
imu0) are already on disk from A0. OpenVINS (ROS1 Noetic container) consumes a
ROS1 bag, so convert the folders -> bag here with the pure-python `rosbags` writer
and cv2 (both already in the venv).

Topics match OpenVINS's EuRoC defaults:
    /imu0             sensor_msgs/Imu   (200 Hz)
    /cam0/image_raw   sensor_msgs/Image (mono8, 20 Hz)
    /cam1/image_raw   sensor_msgs/Image (mono8, 20 Hz)

Messages are written in global timestamp order; images are loaded lazily (one at a
time) so peak RAM stays small.

Usage:
    python scripts/asl_to_rosbag.py V1_01_easy
    python scripts/asl_to_rosbag.py V1_01_easy --out D:/path/V1_01_easy.bag
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import DATA_DIR  # noqa: E402

TS = get_typestore(Stores.ROS1_NOETIC)
Imu = TS.types["sensor_msgs/msg/Imu"]
Image = TS.types["sensor_msgs/msg/Image"]
Header = TS.types["std_msgs/msg/Header"]
Time = TS.types["builtin_interfaces/msg/Time"]
Vec3 = TS.types["geometry_msgs/msg/Vector3"]
Quat = TS.types["geometry_msgs/msg/Quaternion"]

_Z9 = np.zeros(9, dtype=np.float64)


def _stamp(ns: int) -> Time:
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def _header(ns: int, seq: int, frame: str) -> Header:
    return Header(seq=seq, stamp=_stamp(ns), frame_id=frame)


def _imu_msg(ns: int, seq: int, gyro, accel) -> Imu:
    # EuRoC IMU has no orientation -> identity + covariance[0]=-1 (unknown, REP-145).
    oc = _Z9.copy(); oc[0] = -1.0
    return Imu(
        header=_header(ns, seq, "imu0"),
        orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        orientation_covariance=oc,
        angular_velocity=Vec3(x=gyro[0], y=gyro[1], z=gyro[2]),
        angular_velocity_covariance=_Z9.copy(),
        linear_acceleration=Vec3(x=accel[0], y=accel[1], z=accel[2]),
        linear_acceleration_covariance=_Z9.copy(),
    )


def _img_msg(ns: int, seq: int, frame: str, img: np.ndarray) -> Image:
    h, w = img.shape
    return Image(
        header=_header(ns, seq, frame),
        height=h, width=w, encoding="mono8", is_bigendian=0, step=w,
        data=np.ascontiguousarray(img).reshape(-1),
    )


def _read_cam_index(mav0: Path, cam: str):
    """Return list of (ns, png_path) from cam*/data.csv, files that exist."""
    csv = mav0 / cam / "data.csv"
    rows = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
    return [(int(ts), mav0 / cam / "data" / f"{ts}.png") for ts in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seq", nargs="?", default="V1_01_easy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mav0 = DATA_DIR / args.seq / "mav0"
    if not mav0.exists():
        raise SystemExit(f"Missing {mav0}")
    out = Path(args.out) if args.out else DATA_DIR / f"{args.seq}.bag"
    if out.exists():
        raise SystemExit(f"{out} already exists — delete it first.")

    # IMU: timestamp[ns], wx,wy,wz, ax,ay,az
    imu = np.loadtxt(mav0 / "imu0" / "data.csv", delimiter=",", skiprows=1)
    imu_ts = imu[:, 0].astype(np.int64)
    gyro, accel = imu[:, 1:4], imu[:, 4:7]

    cams = [c for c in ("cam0", "cam1") if (mav0 / c).exists()]
    cam_topic = {"cam0": "/cam0/image_raw", "cam1": "/cam1/image_raw"}
    cam_idx = {c: _read_cam_index(mav0, c) for c in cams}

    # Merge all events, sort by timestamp. Images carry a path (lazy load).
    events = [(int(t), "imu", i) for i, t in enumerate(imu_ts)]
    for c in cams:
        events += [(t, c, p) for (t, p) in cam_idx[c]]
    events.sort(key=lambda e: e[0])

    print(f"{args.seq}: {len(imu_ts)} imu + "
          + " + ".join(f"{len(cam_idx[c])} {c}" for c in cams)
          + f"  ->  {out}")

    seq_ctr = {"imu": 0, "cam0": 0, "cam1": 0}
    with Writer(out) as w:
        conn = {"/imu0": w.add_connection("/imu0", Imu.__msgtype__, typestore=TS)}
        for c in cams:
            conn[cam_topic[c]] = w.add_connection(
                cam_topic[c], Image.__msgtype__, typestore=TS)

        n = 0
        for ns, kind, ref in events:
            s = seq_ctr[kind]; seq_ctr[kind] = s + 1
            if kind == "imu":
                msg = _imu_msg(ns, s, gyro[ref], accel[ref])
                topic, mt = "/imu0", Imu.__msgtype__
            else:
                img = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise SystemExit(f"cv2 could not read {ref}")
                msg = _img_msg(ns, s, kind, img)
                topic, mt = cam_topic[kind], Image.__msgtype__
            w.write(conn[topic], ns, TS.serialize_ros1(msg, mt))
            n += 1
            if n % 5000 == 0:
                print(f"  wrote {n}/{len(events)} msgs")

    size_gb = out.stat().st_size / 1e9
    print(f"[done] {out}  ({size_gb:.2f} GB, {len(events)} msgs)")


if __name__ == "__main__":
    main()
