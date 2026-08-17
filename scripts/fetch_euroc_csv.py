"""Fetch ONLY the IMU + ground-truth CSVs from a large EuRoC bundle zip,
using HTTP range requests -- no 6 GB image download.

Why this works:
  - The GlowBond mirror serves each vicon_room / machine_hall bundle as one big
    ASL-format zip, but HF's CDN honours `Range: bytes=...` requests.
  - A zip's central directory lives at the END of the file and lists every
    member's offset + compressed size. Python's zipfile only needs to (a) read
    that directory, then (b) read the bytes of the specific members we extract.
  - So by giving zipfile a seekable file-like object that fulfils each read()
    with a ranged HTTP GET, we download a few KB of directory + a few MB of
    CSVs instead of the whole bundle.

Usage:
    python scripts/fetch_euroc_csv.py                 # vicon_room1 -> V1_01/02/03
    python scripts/fetch_euroc_csv.py vicon_room2
    python scripts/fetch_euroc_csv.py machine_hall --list   # just list members
"""
from __future__ import annotations
import argparse
import io
import sys
import time
import zipfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import DATA_DIR  # noqa: E402

BUNDLE_URL = ("https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/"
              "resolve/main/{bundle}.zip")

# We only ever want these two members per sequence.
WANTED_SUFFIXES = (
    "imu0/data.csv",
    "state_groundtruth_estimate0/data.csv",
)


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file backed by HTTP range requests."""

    TIMEOUT = 60
    CHUNK = 8 << 20   # 8 MB read-ahead: turns zipfile's ~4 KB reads into
                      # a few big ranged GETs instead of ~160k tiny ones.

    def __init__(self, url: str, ua: str = "ai-inertial-nav/0.1"):
        self.url = url
        self.ua = ua
        self._pos = 0
        self._buf = b""          # cached bytes
        self._buf_start = 0      # file offset of self._buf[0]
        self._size = self._probe_size()

    def _probe_size(self) -> int:
        # HEAD to HF redirects to the CDN and can stall; a ranged GET of a
        # single byte is reliable and returns the full size in Content-Range:
        #   "bytes 0-0/6042263426"
        headers = {"User-Agent": self.ua, "Range": "bytes=0-0"}
        for i in range(5):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                    cr = r.headers.get("Content-Range")
                    if cr and "/" in cr:
                        return int(cr.rsplit("/", 1)[1])
                    n = r.headers.get("Content-Length")
                    if n is not None:
                        return int(n)
            except (urllib.error.URLError, ConnectionError,
                    TimeoutError, OSError):
                time.sleep(1.5 * (i + 1))
        raise RuntimeError("could not probe bundle size")

    # --- io plumbing ---
    def seekable(self):  # noqa: D401
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(whence)
        return self._pos

    def tell(self):
        return self._pos

    def _range_get(self, start, end):
        """Raw ranged GET of bytes [start, end] inclusive."""
        headers = {"User-Agent": self.ua, "Range": f"bytes={start}-{end}"}
        return self._get_with_retry(headers)

    def read(self, size=-1):
        if size is None or size < 0:
            want_end = self._size
        else:
            want_end = min(self._pos + size, self._size)
        if want_end <= self._pos:
            return b""

        # serve from buffer if fully covered
        need_lo, need_hi = self._pos, want_end
        buf_lo, buf_hi = self._buf_start, self._buf_start + len(self._buf)
        if not (need_lo >= buf_lo and need_hi <= buf_hi):
            # refill buffer starting at need_lo, at least CHUNK (or the request)
            fetch_len = max(self.CHUNK, need_hi - need_lo)
            fetch_hi = min(need_lo + fetch_len, self._size)
            self._buf = self._range_get(need_lo, fetch_hi - 1)
            self._buf_start = need_lo
            buf_lo = need_lo

        off = need_lo - buf_lo
        data = self._buf[off: off + (need_hi - need_lo)]
        self._pos += len(data)
        return data

    def _get_with_retry(self, headers, tries=5):
        last = None
        for i in range(tries):
            try:
                req = urllib.request.Request(self.url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as r:
                    return r.read()
            except (urllib.error.URLError, ConnectionError,
                    TimeoutError, OSError) as e:
                last = e
                time.sleep(1.5 * (i + 1))
        raise RuntimeError(f"range GET failed after {tries} tries: {last}")

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def _seq_from_member(name: str) -> str | None:
    """Map an internal zip path to a sequence dir name, e.g.
    'vicon_room1/V1_01_easy/mav0/imu0/data.csv' -> 'V1_01_easy'."""
    parts = name.split("/")
    for p in parts:
        if p and (p[:2] in ("V1", "V2", "MH")) and "_" in p:
            return p
    return None


def _extract_csvs_from_inner(inner_bytes: bytes, seq: str) -> int:
    """Given the raw bytes of an inner ASL-format sequence zip, pull the two
    CSVs into data/euroc/<seq>/mav0/... and return bytes written."""
    written = 0
    izf = zipfile.ZipFile(io.BytesIO(inner_bytes))
    targets = [m for m in izf.infolist()
               if m.filename.endswith(WANTED_SUFFIXES)]
    for m in targets:
        idx = m.filename.find("mav0/")
        rel = m.filename[idx:] if idx >= 0 else m.filename.split("/", 1)[-1]
        out = DATA_DIR / seq / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        data = izf.read(m)
        out.write_bytes(data)
        written += len(data)
        print(f"[save] {out}  ({len(data)/1e6:.2f} MB)")
    if not targets:
        print(f"[warn] no target CSVs found inside inner zip for {seq}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default="vicon_room1",
                    help="vicon_room1 | vicon_room2 | machine_hall")
    ap.add_argument("--seq", action="append", default=None,
                    help="restrict to sequence(s), e.g. --seq V1_02_medium "
                         "(repeatable). Default: all in the bundle.")
    ap.add_argument("--list", action="store_true",
                    help="only list members, do not download")
    args = ap.parse_args()

    url = BUNDLE_URL.format(bundle=args.bundle)
    print(f"[open] {url}")
    raw = HttpRangeFile(url)
    print(f"[info] bundle size: {raw._size/1e9:.2f} GB")

    zf = zipfile.ZipFile(raw)  # central directory via range requests

    # Preferred path: flat CSV members directly in the outer zip.
    flat = [m for m in zf.infolist() if m.filename.endswith(WANTED_SUFFIXES)]
    # Fallback (GlowBond layout): outer zip nests one deflated ASL .zip per seq.
    inner = [m for m in zf.infolist()
             if m.filename.endswith(".zip") and _seq_from_member(m.filename)]

    print(f"[dir ] {len(zf.infolist())} entries; "
          f"{len(flat)} flat CSVs, {len(inner)} inner sequence zips")

    if args.list:
        for m in inner:
            print(f"       inner: {m.filename}  "
                  f"({m.compress_size/1e6:.0f} MB compressed)")
        for m in flat:
            print(f"       csv  : {m.filename}")
        return

    total = 0
    if flat:
        for m in flat:
            seq = _seq_from_member(m.filename) or "unknown"
            idx = m.filename.find("mav0/")
            rel = m.filename[idx:] if idx >= 0 else m.filename.split("/", 1)[-1]
            out = DATA_DIR / seq / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(m)
            out.write_bytes(data)
            total += len(data)
            print(f"[save] {out}  ({len(data)/1e6:.2f} MB)")
    else:
        want = set(args.seq) if args.seq else None
        for m in inner:
            seq = _seq_from_member(m.filename)
            if want and seq not in want:
                continue
            gt = DATA_DIR / seq / "mav0" / "state_groundtruth_estimate0" / "data.csv"
            imu = DATA_DIR / seq / "mav0" / "imu0" / "data.csv"
            if gt.exists() and imu.exists():
                print(f"[skip] {seq} CSVs already present")
                continue
            print(f"[pull] {m.filename}  "
                  f"({m.compress_size/1e6:.0f} MB compressed) -> inflating "
                  f"inner zip to reach its CSVs")
            inner_bytes = zf.read(m)  # ranged reads + inflate of just this member
            total += _extract_csvs_from_inner(inner_bytes, seq)

    print(f"[done] wrote {total/1e6:.2f} MB of CSVs into {DATA_DIR}")


if __name__ == "__main__":
    main()
