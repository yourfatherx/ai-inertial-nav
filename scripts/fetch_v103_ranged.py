"""Fetch ONLY V1_03_difficult images from the 6GB GlowBond vicon_room1.zip.

The outer bundle (~6GB, ZIP64) nests per-sequence archives; the one we want is
the member `vicon_room1/V1_03_difficult/V1_03_difficult.zip` (~801MB, the
standard ASL zip with cam0/cam1 images + imu + groundtruth). The CDN honors
Range requests (206), so we:
  1. read the ZIP64 End-Of-Central-Directory to locate the central directory,
  2. parse it to find the offset+size of the nested V1_03 zip member,
  3. range-GET just that member (stored, not compressed) to a temp file,
  4. extract the ASL tree from it into data/euroc/V1_03_difficult/,
  5. delete the temp.

Same range-fetch idea as fetch_euroc_csv.py, extended to a ZIP64 archive.
"""
import sys
import struct
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import DATA_DIR  # noqa: E402

RESOLVE = ("https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/"
           "resolve/main/vicon_room1.zip")
MEMBER = "V1_03_difficult/V1_03_difficult.zip"     # the nested ASL zip
WANT = "V1_03_difficult"
nested_zip = DATA_DIR / "_V1_03_nested.zip"


def _resolve_cdn(url):
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "ai-inertial-nav/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl(), int(r.headers["Content-Length"])


def _get(url, start, end):
    req = urllib.request.Request(
        url, headers={"User-Agent": "ai-inertial-nav/0.1",
                      "Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(url=req, timeout=120) as r:
        return r.read()


def _locate_member(url, size):
    tail = _get(url, size - 65536, size - 1)
    locp = tail.rfind(b"PK\x06\x07")               # ZIP64 EOCD locator
    z64_off = struct.unpack("<Q", tail[locp + 8:locp + 16])[0]
    z64 = _get(url, z64_off, z64_off + 56)
    cd_size = struct.unpack("<Q", z64[40:48])[0]
    cd_off = struct.unpack("<Q", z64[48:56])[0]
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    i = 0
    while i < len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        method = struct.unpack("<H", cd[i + 10:i + 12])[0]
        comp = struct.unpack("<I", cd[i + 20:i + 24])[0]
        uncomp = struct.unpack("<I", cd[i + 24:i + 28])[0]
        nlen = struct.unpack("<H", cd[i + 28:i + 30])[0]
        elen = struct.unpack("<H", cd[i + 30:i + 32])[0]
        clen = struct.unpack("<H", cd[i + 32:i + 34])[0]
        lho = struct.unpack("<I", cd[i + 42:i + 46])[0]
        name = cd[i + 46:i + 46 + nlen].decode("utf-8", "replace")
        extra = cd[i + 46 + nlen:i + 46 + nlen + elen]
        if 0xFFFFFFFF in (comp, uncomp, lho):
            j = 0
            while j + 4 <= len(extra):
                hid, dsize = struct.unpack("<HH", extra[j:j + 4])
                if hid == 0x0001:
                    blob = extra[j + 4:j + 4 + dsize]
                    q = 0
                    if uncomp == 0xFFFFFFFF:
                        uncomp = struct.unpack("<Q", blob[q:q + 8])[0]; q += 8
                    if comp == 0xFFFFFFFF:
                        comp = struct.unpack("<Q", blob[q:q + 8])[0]; q += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", blob[q:q + 8])[0]; q += 8
                    break
                j += 4 + dsize
        if name.endswith(MEMBER):
            return name, lho, comp, method
        i += 46 + nlen + elen + clen
    return None


def main():
    url, size = _resolve_cdn(RESOLVE)
    print(f"[cdn] size={size/1e9:.2f} GB")
    m = _locate_member(url, size)
    if not m:
        print(f"[ERROR] member ending {MEMBER!r} not found")
        sys.exit(1)
    name, lho, comp, method = m
    print(f"[member] {name}  comp={comp/1e6:.1f} MB method={method}")

    # local header -> data offset
    lh = _get(url, lho, lho + 29)
    nlen = struct.unpack("<H", lh[26:28])[0]
    elen = struct.unpack("<H", lh[28:30])[0]
    data_off = lho + 30 + nlen + elen

    # stream the nested zip's bytes to a temp file in 8MB range chunks
    print(f"[fetch] pulling {comp/1e6:.0f} MB nested zip ...")
    CHUNK = 8 << 20
    got = 0
    with open(nested_zip, "wb") as f:
        while got < comp:
            a = data_off + got
            b = min(a + CHUNK, data_off + comp) - 1
            buf = _get(url, a, b)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            print(f"\r  {got/1e6:6.0f}/{comp/1e6:.0f} MB", end="", flush=True)
    print()
    if method != 0:
        print(f"[warn] nested member stored with method {method}, expected 0")

    print("[extract] unpacking ASL tree ...")
    out_root = DATA_DIR / WANT
    with zipfile.ZipFile(nested_zip) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        # strip everything up to and including the top "<WANT>/" if present
        for n in names:
            rel = n
            if WANT + "/" in n:
                rel = n[n.index(WANT + "/") + len(WANT) + 1:]
            elif n.startswith("mav0/"):
                rel = n
            if not rel:
                continue
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as out:
                out.write(src.read())
    nested_zip.unlink()
    cam = out_root / "mav0" / "cam0" / "data"
    n = len(list(cam.glob("*.png"))) if cam.exists() else 0
    print(f"[done] {WANT}: {n} cam0 frames at {out_root}")


if __name__ == "__main__":
    main()
