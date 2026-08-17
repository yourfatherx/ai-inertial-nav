"""Fetch ONLY the IMU + mocap ground-truth CSVs from a TUM-VI room sequence.

TUM-VI publishes an EuRoC/ASL-format export of every sequence as one .tar:
    https://cvg.cit.tum.de/tumvi/exported/euroc/512_16/dataset-<seq>_512_16.tar

Unlike the EuRoC HuggingFace bundles, the HTTP-range trick does NOT pay off
here: the tar is *uncompressed* but stores all camera images FIRST and the two
small CSVs LAST, and a tar has no central directory -- so reaching the CSVs
means walking ~thousands of 512-byte headers (one ranged GET each) or streaming
the whole 1.7 GB anyway. Cheapest correct path: download the tar once
(resumable), stream-extract only the two CSVs, delete the tar.

We keep only:
    mav0/imu0/data.csv        (200 Hz gyro+accel -- the Bosch MEMS IMU)
    mav0/mocap0/data.csv      (120 Hz pose ground truth, IMU frame, pose-only)

Output layout (mirrors the EuRoC loader's expectation):
    data/tumvi/<seq>/mav0/imu0/data.csv
    data/tumvi/<seq>/mav0/mocap0/data.csv

Usage:
    python scripts/fetch_tumvi_csv.py                 # room1
    python scripts/fetch_tumvi_csv.py room2 room3
    python scripts/fetch_tumvi_csv.py room1 --keep-tar
"""
from __future__ import annotations
import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUMVI_DIR = ROOT / "data" / "tumvi"

URL_TMPL = ("https://cvg.cit.tum.de/tumvi/exported/euroc/512_16/"
            "dataset-{seq}_512_16.tar")

# The only two members we want, matched by suffix (tar prefixes them with the
# top-level "dataset-<seq>_512_16/" directory).
WANTED_SUFFIXES = ("mav0/imu0/data.csv", "mav0/mocap0/data.csv")


def _download(url: str, dest: Path) -> None:
    """Resumable download with a simple progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "ai-inertial-nav/0.1"})
    # Probe total size (ranged 1-byte GET returns Content-Range: bytes 0-0/TOTAL).
    probe = urllib.request.Request(url, headers={"User-Agent": "x", "Range": "bytes=0-0"})
    with urllib.request.urlopen(probe, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        total = int(cr.split("/")[-1]) if "/" in cr else 0

    if have and total and have >= total:
        print(f"[skip] tar already complete ({have} bytes): {dest.name}")
        return

    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"[resume] {dest.name} from {have}/{total} bytes")
    else:
        print(f"[get] {dest.name} 0/{total} bytes")

    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, mode) as f:
        got = have
        while True:
            chunk = r.read(8 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = 100 * got / total
                print(f"\r  {got/1e6:8.1f} / {total/1e6:.1f} MB ({pct:5.1f}%)",
                      end="", flush=True)
        print()


def _extract_csvs(tar_path: Path, out_seq_dir: Path) -> list[Path]:
    """Stream the tar, write only the two wanted CSVs. Returns written paths."""
    written = []
    with tarfile.open(tar_path, "r|") as tf:      # streaming mode, no seeking
        for member in tf:
            if not member.isfile():
                continue
            suffix = next((s for s in WANTED_SUFFIXES
                           if member.name.endswith(s)), None)
            if suffix is None:
                continue
            dst = out_seq_dir / suffix            # e.g. .../mav0/imu0/data.csv
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            with open(dst, "wb") as f:
                f.write(src.read())
            written.append(dst)
            print(f"[extract] {suffix}  ({dst.stat().st_size} bytes)")
            if len(written) == len(WANTED_SUFFIXES):
                break                             # CSVs are last; stop early
    return written


def fetch_seq(seq: str, keep_tar: bool) -> None:
    out_seq_dir = TUMVI_DIR / seq
    imu_csv = out_seq_dir / "mav0" / "imu0" / "data.csv"
    gt_csv = out_seq_dir / "mav0" / "mocap0" / "data.csv"
    if imu_csv.exists() and gt_csv.exists():
        print(f"[skip] {seq}: CSVs already present")
        return

    tar_path = TUMVI_DIR / f"{seq}.tar"
    _download(URL_TMPL.format(seq=seq), tar_path)

    written = _extract_csvs(tar_path, out_seq_dir)
    if len(written) != len(WANTED_SUFFIXES):
        got = {p.name for p in written}
        raise RuntimeError(f"{seq}: expected {WANTED_SUFFIXES}, only wrote {got}")

    if not keep_tar:
        tar_path.unlink()
        print(f"[clean] removed {tar_path.name} ({seq} CSVs kept)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seqs", nargs="*", default=["room1"],
                    help="TUM-VI room sequences (default: room1)")
    ap.add_argument("--keep-tar", action="store_true",
                    help="do not delete the tar after extraction")
    args = ap.parse_args()
    for seq in args.seqs:
        fetch_seq(seq, args.keep_tar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
