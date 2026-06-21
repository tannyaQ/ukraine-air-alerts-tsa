"""Download raw air-raid siren datasets into data/raw/.

Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset (MIT).

The committed snapshot in data/raw/ is the source of truth for the analysis;
this script documents provenance and lets you refresh the snapshot or fetch the
larger (uncommitted) official crosscheck file on demand.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/"
)

# filename -> committed to git? (volunteer is the primary source of truth)
FILES = {
    "volunteer_data_en.csv": True,
    "official_data_en.csv": False,  # ~28 MB crosscheck, fetched on demand
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + name
    dest = RAW_DIR / name
    print(f"-> {url}")
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"   saved {dest.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print(f"   sha256 {sha256(dest)}\n")
    return dest


def main() -> None:
    for name in FILES:
        download(name)


if __name__ == "__main__":
    main()
