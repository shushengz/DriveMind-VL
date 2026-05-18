"""Download the small LingoQA evaluation annotation table.

This helper intentionally downloads metadata only. It does not download
LingoQA videos or training data. The user must explicitly acknowledge the
upstream license/terms before the request is made.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


LINGOQA_EVAL_URL = "https://drive.usercontent.google.com/u/1/uc?id=1I8u6uYysQUstoVYZapyRQkXmOwr-AG3d&export=download"
LINGOQA_REPO_URL = "https://github.com/wayveai/LingoQA"


def download_file(url: str, output: Path, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "DriveMind-VL/metadata-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
    except urllib.error.URLError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"download failed: {exc}") from exc

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink()
        raise RuntimeError("downloaded file is empty")
    tmp_path.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LingoQA evaluation metadata only.")
    parser.add_argument("--output", default="data/external/lingoqa/evaluation.parquet")
    parser.add_argument("--url", default=LINGOQA_EVAL_URL)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--accept_terms",
        action="store_true",
        help="Confirm you have reviewed and accept the upstream LingoQA terms/license.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if not args.accept_terms:
        print("Refusing to download until --accept_terms is provided.")
        print(f"Review upstream repository and license first: {LINGOQA_REPO_URL}")
        print("This helper downloads metadata only, not videos or model weights.")
        raise SystemExit(2)

    if output.exists() and not args.overwrite:
        print(f"exists, skip: {output}")
        print("Use --overwrite to replace it.")
        return

    download_file(args.url, output, timeout=args.timeout)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"downloaded {output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
