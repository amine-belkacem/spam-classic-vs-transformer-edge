"""Download the two UCI datasets and check that they are the expected files.

Usage:
    python scripts/download_data.py --out data/raw

Sources, both under CC BY 4.0 (checked on the UCI pages on 2026-10-26):
  - SMS Spam Collection, Almeida & Hidalgo, https://doi.org/10.24432/C5CC84
  - YouTube Spam Collection, Alberto & Lochter, https://doi.org/10.24432/C58885

The SHA-256 values below are the ones of the extracted files, not of the zips:
UCI could rebuild an archive (new timestamps, new compression) without touching
the data, and that should not break the project. After the hash check, the files
are parsed and the row and spam counts are compared with the UCI tables.

Only the members listed here are extracted, under names chosen by the script.
The YouTube zip also contains macOS metadata (__MACOSX/) that is skipped.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spamedge.data import load_sms, load_youtube, sha256_of  # noqa: E402

DATASETS = {
    "sms": {
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "files": {
            "SMSSpamCollection": "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d",
        },
    },
    "youtube": {
        "url": "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip",
        "files": {
            "Youtube01-Psy.csv": "19797e6c77690e3c8809cfd2853ae7341390636367ba66cf5d4f4083f0b88535",
            "Youtube02-KatyPerry.csv": "902c614f8ef24f987d6f614d7e6111aa5160b89a0646b68e007bd6044a3d123b",
            "Youtube03-LMFAO.csv": "702ef589860a1831956f527760a3d9737ef8a07ab36c7de35b92b8898b8c3928",
            "Youtube04-Eminem.csv": "92f54eb6b22fdf3b7ae85e1f500e5aa7442edd025e504b988a97078756187e76",
            "Youtube05-Shakira.csv": "1d8ab47b71e8037c51183b2fc62f0591a48a4b54f3a4f5d9d3043113b274e98e",
        },
    },
}
TIMEOUT_S = 60


def fetch_and_extract(url: str, members: list[str], dest_dir: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        missing = [m for m in members if m not in names]
        if missing:
            raise FileNotFoundError(f"{url}: members not found in the archive: {missing}")
        for member in members:
            # Write under a name I control instead of zf.extract(), so a crafted
            # path inside the archive can never land outside dest_dir.
            (dest_dir / Path(member).name).write_bytes(zf.read(member))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    failures = 0
    for name, spec in DATASETS.items():
        dest_dir = args.out / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        members = list(spec["files"])
        if all((dest_dir / m).exists() for m in members):
            print(f"{dest_dir} already complete, skipping download")
        else:
            fetch_and_extract(spec["url"], members, dest_dir)

        for member, expected in spec["files"].items():
            digest = sha256_of(dest_dir / member)
            if digest != expected:
                print(f"ECHEC SHA-256 {member} : obtenu {digest}, attendu {expected}", file=sys.stderr)
                failures += 1
    if failures:
        return 1

    # Both loaders raise with a clear message if the counts differ from the UCI tables.
    sms = load_sms(args.out / "sms" / "SMSSpamCollection")
    yt = load_youtube(args.out / "youtube")
    for label, df in (("SMS", sms), ("YouTube", yt)):
        print(f"OK  {label:8s} rows={len(df)}  spam={int(df['label'].sum())}  "
              f"spam rate={df['label'].mean():.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
