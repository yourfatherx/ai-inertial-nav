"""Download & extract one EuRoC MAV sequence into data/euroc/<name>/.

Usage:
    python scripts/download_euroc.py                # MH_01_easy
    python scripts/download_euroc.py MH_03_medium
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainav.config import DATA_DIR  # noqa: E402

# The official ETH host (robotics.ethz.ch) is frequently unreachable, so we
# use a public HuggingFace mirror that serves the standard ASL-format zips.
# Files available on this mirror (single-sequence zips):
#   V1_01_easy.zip  (see: huggingface.co/datasets/pepijn223/euroc-mirror)
HF_MIRROR = ("https://huggingface.co/datasets/pepijn223/euroc-mirror/"
             "resolve/main/{name}.zip")

BASE = HF_MIRROR


def download(name: str = "MH_01_easy") -> Path:
    out_dir = DATA_DIR / name
    if (out_dir / "mav0").exists():
        print(f"[skip] {name} already extracted at {out_dir}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}.zip"
    url = BASE.format(name=name)

    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ai-inertial-nav/0.1"})
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = 100 * done / total if total else 0
            print(f"\r  {done/1e6:7.1f} MB  {pct:5.1f}%", end="", flush=True)
    print()

    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    zip_path.unlink()
    print(f"[done] {out_dir}")
    return out_dir


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "MH_01_easy"
    download(name)
