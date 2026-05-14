"""Draw perception boxes on an image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.perception.build_perception_json import build_perception
except Exception:
    from build_perception_json import build_perception


def visualize(image_path: Path, output_dir: Path, perception: dict | None = None) -> Path:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("Pillow is required for visualization.") from exc

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    perception = perception or build_perception(str(image_path))
    for obj in perception.get("objects", []):
        bbox = obj.get("bbox", [])
        if len(bbox) == 4:
            draw.rectangle(bbox, outline=(255, 60, 60), width=3)
            draw.text((bbox[0], max(0, bbox[1] - 14)), str(obj.get("class", "object")), fill=(255, 60, 60))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{image_path.stem}_perception.jpg"
    img.save(out, quality=92)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize lightweight perception JSON.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--perception_json", default="")
    parser.add_argument("--output_dir", default="outputs/cases")
    args = parser.parse_args()
    perception = None
    if args.perception_json:
        try:
            perception = json.loads(Path(args.perception_json).read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"failed to read perception json, using default: {exc}")
    out = visualize(Path(args.image), Path(args.output_dir), perception)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
