"""Build CPU-only SFT-v3 data from strict visual-control samples or source JSONL."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.visual_control_formatter import MODE_STRICT, build_variants, read_jsonl, write_jsonl

def _u(hex_text: str) -> str:
    return bytes.fromhex(hex_text).decode("utf-8")


REFUSAL_TEXT = _u("e5bd93e5898de59bbee5838fe4bfa1e681afe4b88de8b6b3efbc8ce697a0e6b395e7a1aee5ae9ae38082")
WRONG_IMAGE_TEXT = _u("e4b88de5ba94e59fbae4ba8ee8afa5e59bbee5838fe7a1aee5ae9ae58e9fe997aee9a298e7ad94e6a188e38082")
TEXT_ONLY_TEXT = _u("e7bcbae5b091e59bbee5838fe4bfa1e681afefbc8ce697a0e6b395e58fafe99da0e588a4e696ade38082")
BLANK_REASON = _u("e8afa5e8be93e585a5e4b8bae7a9bae799bde59bbee5838fefbc8ce4b88de883bde58fafe99da0e59b9ee7ad94e4be9de8b596e8a786e8a789e79a84e997aee9a298e38082")
WRONG_REASON = _u("e8afa5e59bbee5838fe58fafe883bde4b88de5afb9e5ba94e58e9fe997aee9a298efbc8ce5ba94e981bfe5858de68a8ae99499e8afafe59bbee5838fe5bd93e4bd9ce8af81e68daee38082")
TEXT_ONLY_REASON = _u("e6b2a1e69c89e59bbee5838fe8be93e585a5e697b6efbc8ce4b88de5ba94e587ade8afade8a880e58588e9aa8ce78c9ce6b58be8a786e8a789e7ad94e6a188e38082")
SPATIAL_REASON = _u("e7a9bae997b4e585b3e7b3bbe5bf85e9a1bbe794b1e58cb9e9858de59bbee5838fe694afe68c81efbc8ce99499e8afafe68896e7bcbae5a4b1e59bbee5838fe4b88de883bde4bd9ce4b8bae4be9de68daee38082")
SPATIAL_RE = re.compile(r"left|right|front|back|traffic light|lane|vehicle|pedestrian", re.I)

def assistant(answer: str, reason: str) -> dict[str, str]:
    return {"answer": answer, "reason": reason}


def make_sft_row(sample: dict[str, Any], sample_type: str, out: dict[str, str]) -> dict[str, Any]:
    return {
        "id": f"{sample.get('id')}_{sample_type}",
        "dataset": sample.get("dataset", ""),
        "sample_type": sample_type,
        "messages": [
            {"role": "user", "content": sample.get("prompt") or sample.get("question") or ""},
            {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
        ],
        "image_paths": sample.get("image_paths", []),
        "image_labels": sample.get("image_labels", []),
        "prompt": sample.get("prompt") or sample.get("question") or "",
        "assistant": out,
        "metadata": {**(sample.get("metadata") or {}), "source_id": sample.get("id"), "setting": sample.get("setting")},
    }


def variants_from_rows(rows: list[dict[str, Any]], dataset: str, seed: int) -> list[dict[str, dict[str, Any]]]:
    rng = random.Random(seed)
    if rows and {"mode", "setting", "prompt"}.issubset(rows[0].keys()):
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("id")), {})[str(row.get("setting"))] = row
        out = []
        for settings in grouped.values():
            if "normal" in settings:
                normal = settings["normal"]
                out.append({
                    "normal": normal,
                    "text_only": {**normal, "setting": "text_only", "image_paths": [], "image_labels": [], "prompt": normal.get("question", "")},
                    "wrong_image": {**normal, "setting": "wrong_image"},
                    "blank_image": {**normal, "setting": "blank_image", "image_paths": ["outputs/cases/ablation_images/blank.jpg"], "image_labels": ["[Frame 0]"]},
                    **settings,
                })
        return out
    return [build_variants(row, rows, dataset=dataset, mode=MODE_STRICT, rng=rng) for row in rows]


def build_for_dataset(rows: list[dict[str, Any]], dataset: str, seed: int, max_rows: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    groups = variants_from_rows(rows, dataset, seed)
    rng.shuffle(groups)
    out_rows: list[dict[str, Any]] = []
    for group in groups:
        normal = group["normal"]
        gold = str(normal.get("gold") or "")
        out_rows.append(make_sft_row(normal, "normal_visual_qa", assistant(gold, gold)))
        question = str(normal.get("question") or "")
        control_candidates = [
            ("blank_refusal", group.get("blank_image"), assistant(REFUSAL_TEXT, BLANK_REASON)),
            ("wrong_image_caution", group.get("wrong_image"), assistant(WRONG_IMAGE_TEXT, WRONG_REASON)),
            ("text_only_caution", group.get("text_only"), assistant(TEXT_ONLY_TEXT, TEXT_ONLY_REASON)),
        ]
        if SPATIAL_RE.search(question):
            control_candidates.append(("spatial_hard_negative", group.get("wrong_image") or normal, assistant(WRONG_IMAGE_TEXT, SPATIAL_REASON)))
        for sample_type, sample, out in control_candidates:
            if sample:
                out_rows.append(make_sft_row(sample, sample_type, out))
    return out_rows


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = Counter(str(row.get("dataset") or "unknown") for row in rows)
    by_type = Counter(str(row.get("sample_type") or "unknown") for row in rows)
    normal = by_type.get("normal_visual_qa", 0)
    control = len(rows) - normal
    answer_lengths = [len(str(row.get("assistant", {}).get("answer", ""))) for row in rows]
    refusal = sum(1 for row in rows if row.get("sample_type") in {"blank_refusal", "text_only_caution", "wrong_image_caution"})
    spatial = sum(1 for row in rows if row.get("sample_type") == "spatial_hard_negative")
    return {
        "total": len(rows),
        "by_dataset": dict(by_dataset),
        "by_sample_type": dict(by_type),
        "normal_control_ratio": {"normal": normal, "control": control, "normal_fraction": normal / len(rows) if rows else 0.0},
        "average_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        "refusal_sample_ratio": refusal / len(rows) if rows else 0.0,
        "spatial_sample_ratio": spatial / len(rows) if rows else 0.0,
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT-v3 data, CPU-only.")
    parser.add_argument("--input_lingoqa", default="data/processed/visual_control/lingoqa_strict_normal.jsonl")
    parser.add_argument("--input_drivelm", default="data/processed/visual_control/drivelm_strict_normal.jsonl")
    parser.add_argument("--output_dir", default="data/train/sft_v3")
    parser.add_argument("--max_lingoqa", type=int, default=0)
    parser.add_argument("--max_drivelm", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_optional(path: str, limit: int) -> list[dict[str, Any]]:
    p = Path(path)
    return read_jsonl(p, limit) if p.exists() else []


def main() -> None:
    args = parse_args()
    l_limit = args.max_lingoqa or (8 if args.dry_run else 0)
    d_limit = args.max_drivelm or (8 if args.dry_run else 0)
    lingoqa_rows = build_for_dataset(load_optional(args.input_lingoqa, l_limit), "lingoqa", args.seed)
    drivelm_rows = build_for_dataset(load_optional(args.input_drivelm, d_limit), "drivelm", args.seed + 1)
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "lingoqa_sft_v3.jsonl", lingoqa_rows)
    write_jsonl(out_dir / "drivelm_sft_v3.jsonl", drivelm_rows)
    mixed = [*lingoqa_rows, *drivelm_rows]
    random.Random(args.seed).shuffle(mixed)
    write_jsonl(out_dir / "mixed_sft_v3.jsonl", mixed)
    st = stats(mixed)
    write_json(out_dir / "sft_v3_stats.json", st)
    print(json.dumps({"output_dir": out_dir.as_posix(), **st}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
