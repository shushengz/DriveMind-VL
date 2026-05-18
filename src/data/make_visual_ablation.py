"""Create visual-ablation JSONL variants for grounding checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_blank_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("blank_image ablation requires pillow") from exc
    image = Image.new("RGB", size, color=(128, 128, 128))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "visual ablation blank image", fill=(255, 255, 255))
    image.save(path)


def row_image_paths(row: dict[str, Any]) -> list[str]:
    external = row.get("meta", {}).get("external", {})
    paths = external.get("image_paths", []) if isinstance(external, dict) else []
    if isinstance(paths, list) and paths:
        return [str(path) for path in paths if str(path)]
    image = str(row.get("image", ""))
    return [image] if image else []


def build_variant(rows: list[dict[str, Any]], mode: str, blank_image_path: Path, blank_size: tuple[int, int]) -> list[dict[str, Any]]:
    if not rows:
        return []
    if mode == "wrong_image":
        images = [row_image_paths(row) for row in rows]
    elif mode == "blank_image":
        create_blank_image(blank_image_path, blank_size)
        max_frames = max((len(row_image_paths(row)) for row in rows), default=1)
        images = [[blank_image_path.as_posix()] * max_frames for _ in rows]
    else:
        raise ValueError(f"unsupported mode: {mode}")

    variant: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        new_row = json.loads(json.dumps(row, ensure_ascii=False))
        if mode == "wrong_image":
            replacement_paths = images[(idx + 1) % len(images)]
        else:
            replacement_paths = images[idx]
        new_row["image"] = replacement_paths[0] if replacement_paths else ""
        meta = new_row.setdefault("meta", {})
        external = meta.setdefault("external", {})
        meta["visual_ablation"] = mode
        external["original_image"] = row.get("image", "")
        external["original_image_paths"] = row_image_paths(row)
        external["ablated_image"] = new_row.get("image", "")
        external["image_paths"] = replacement_paths
        external["ablated_image_paths"] = replacement_paths
        variant.append(new_row)
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Create wrong-image or blank-image visual ablation datasets.")
    parser.add_argument("--input", default="data/processed/drivemind_intelli_sample.jsonl")
    parser.add_argument("--output", default="data/processed/drivemind_intelli_sample_wrong_image.jsonl")
    parser.add_argument("--mode", choices=["wrong_image", "blank_image"], required=True)
    parser.add_argument("--blank_image_path", default="outputs/cases/ablation_images/blank.jpg")
    parser.add_argument("--blank_width", type=int, default=640)
    parser.add_argument("--blank_height", type=int, default=360)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    variant = build_variant(
        rows,
        mode=args.mode,
        blank_image_path=Path(args.blank_image_path),
        blank_size=(args.blank_width, args.blank_height),
    )
    write_jsonl(variant, Path(args.output))
    print(f"wrote {len(variant)} {args.mode} samples to {args.output}")


if __name__ == "__main__":
    main()
