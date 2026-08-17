"""One-off: fetch V1_03_difficult images from the GlowBond vicon_room1 bundle.

The single-sequence EuRoC zips only survive on the ETH host (now unreachable)
and the pepijn223 HF mirror (V1_01 only). GlowBond/EuRoC_MAV_Dataset serves the
whole vicon_room1 bundle (V1_01+V1_02+V1_03, ~6GB). We stream it, extract ONLY
the V1_03_difficult tree into data/euroc/V1_03_difficult/, then delete the zip.
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import DATA_DIR  # noqa: E402

URL = ("https://huggingface.co/datasets/GlowBond/EuRoC_MAV_Dataset/"
       "resolve/main/vicon_room1.zip")
WANT = "V1_03_difficult"
zip_path = DATA_DIR / "vicon_room1.zip"


def main():
    if not zip_path.exists():
        print(f"[download] {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "ai-inertial-nav/0.1"})
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = 100 * done / total if total else 0
                print(f"\r  {done/1e9:6.2f} / {total/1e9:5.2f} GB  {pct:5.1f}%",
                      end="", flush=True)
        print()
    else:
        print(f"[skip dl] {zip_path} exists ({zip_path.stat().st_size/1e9:.2f} GB)")

    print("[scan] locating V1_03_difficult members ...")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        members = [n for n in names if WANT in n and not n.endswith("/")]
        if not members:
            print("[ERROR] no V1_03_difficult members found. Top entries:")
            for n in names[:20]:
                print("   ", n)
            sys.exit(1)
        # figure out the path prefix up to and including "V1_03_difficult/"
        sample = members[0]
        idx = sample.index(WANT)
        prefix = sample[:idx + len(WANT) + 1]   # ".../V1_03_difficult/"
        print(f"[extract] {len(members)} files, prefix={prefix!r}")
        out_root = DATA_DIR / WANT
        for i, n in enumerate(members):
            rel = n[len(prefix):]
            if not rel:
                continue
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, open(dst, "wb") as out:
                out.write(src.read())
            if i % 200 == 0:
                print(f"\r  {i}/{len(members)}", end="", flush=True)
        print(f"\r  {len(members)}/{len(members)} done")

    zip_path.unlink()
    print(f"[cleanup] removed {zip_path.name}")
    cam = DATA_DIR / WANT / "mav0" / "cam0" / "data"
    n = len(list(cam.glob("*.png"))) if cam.exists() else 0
    print(f"[done] {WANT}: {n} cam0 frames")


if __name__ == "__main__":
    main()
