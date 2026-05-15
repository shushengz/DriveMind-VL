"""Convert small external benchmark exports into DriveMind-Instruct rows.

This is a lightweight adapter scaffold. It intentionally does not download any
benchmark files; users must place legally obtained metadata/images under
data/external/<benchmark>/ first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SOURCES = {
    "intelli_cockpit_bench",
    "nuscenes_qa",
    "drivelm",
    "drivebench",
    "drive_and_act",
    "dmd",
    "generic_vqa",
}

INTELLI_COCKPIT_TASK = "external_vqa"


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
                    if isinstance(item, dict):
                        records.append(item)
        else:
            with path.open("r", encoding="utf-8-sig") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                records = [x for x in obj if isinstance(x, dict)]
            elif isinstance(obj, dict):
                for key in ("data", "records", "samples", "items"):
                    value = obj.get(key)
                    if isinstance(value, list):
                        records = [x for x in value if isinstance(x, dict)]
                        break
                if not records:
                    records = [obj]
    except OSError as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc
    return records


def dump_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise RuntimeError(f"failed to write {path}: {exc}") from exc


def first_present(record: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return default


def normalize_reference(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""


def resolve_image(record: dict[str, Any], image_root: Path | None) -> str:
    image_value = first_present(
        record,
        ["image", "image_path", "img", "img_path", "filename", "file_name", "frame_path"],
        "",
    )
    if isinstance(image_value, list):
        image_value = image_value[0] if image_value else ""
    image_path = str(image_value)
    if image_root and image_path and not Path(image_path).is_absolute():
        return (image_root / image_path).as_posix()
    return Path(image_path).as_posix() if image_path else ""


def infer_task_type(source: str, record: dict[str, Any]) -> str:
    text = json.dumps(record, ensure_ascii=False).lower()
    if source in {"drive_and_act", "dmd"}:
        return "cabin_understanding"
    if "inside" in text or "interior" in text or "driver" in text or "cabin" in text:
        return "cabin_understanding"
    if source == "intelli_cockpit_bench":
        text = json.dumps(
            {
                "question": record.get("question"),
                "reference": record.get("reference"),
                "category": record.get("category"),
                "subcategory": record.get("subcategory"),
                "shooting_angle": record.get("shooting_angle"),
                "weather_conditions": record.get("weather_conditions"),
            },
            ensure_ascii=False,
        ).lower()
        risk_terms = (
            "risk",
            "danger",
            "unsafe",
            "collision",
            "too close",
            "close distance",
            "brake",
            "slow down",
            "caution",
            "safe to",
        )
        if any(term in text for term in risk_terms):
            return "risk_reasoning"
        return INTELLI_COCKPIT_TASK
    if "tool_sequence" in record or "safety" in text or "risk" in text:
        return "risk_reasoning"
    return "risk_reasoning"


def infer_risk_level(record: dict[str, Any], answer_text: str) -> str:
    explicit = str(first_present(record, ["risk_level", "label"], "")).lower()
    if explicit in {"low", "medium", "high"}:
        return explicit
    text = f"{json.dumps(record, ensure_ascii=False)} {answer_text}".lower()
    speed = record.get("speed")
    try:
        speed_value = float(speed)
    except (TypeError, ValueError):
        speed_value = 0.0
    if any(term in text for term in ("collision", "danger", "unsafe", "close", "pedestrian")) and speed_value >= 30:
        return "high"
    if any(term in text for term in ("rain", "fog", "night", "close", "risk")):
        return "medium"
    return "low"


def infer_risk_object(record: dict[str, Any], answer_text: str) -> str:
    explicit = str(first_present(record, ["risk_object", "object"], "")).strip()
    if explicit:
        return explicit
    text = f"{json.dumps(record, ensure_ascii=False)} {answer_text}".lower()
    candidates = (
        ("front_vehicle", ("front vehicle", "front car", "vehicle ahead", "car ahead")),
        ("pedestrian", ("pedestrian", "person crossing", "walker")),
        ("cyclist", ("cyclist", "bike", "bicycle")),
        ("blind_spot_vehicle", ("blind spot", "adjacent lane")),
        ("traffic_light", ("traffic light", "red light")),
        ("lane", ("lane", "lane marking")),
    )
    for label, terms in candidates:
        if any(term in text for term in terms):
            return label
    category = str(first_present(record, ["category"], "unknown")).lower().replace(" ", "_")
    return category or "unknown"


def infer_suggestion(record: dict[str, Any], answer_text: str) -> str:
    explicit = str(first_present(record, ["suggestion", "action"], "")).strip()
    if explicit:
        return explicit
    text = f"{json.dumps(record, ensure_ascii=False)} {answer_text}".lower()
    if any(term in text for term in ("slow", "decelerate", "brake")):
        return "slow_down"
    if any(term in text for term in ("keep distance", "safe distance")):
        return "keep_safe_distance"
    if any(term in text for term in ("stop", "yield")):
        return "brake_and_warn"
    return "keep_attention"


def normalize_answer(source: str, task_type: str, record: dict[str, Any]) -> dict[str, Any]:
    answer_keys = ["reference", "gt_answer", "answer", "response", "label", "activity"]
    answer = first_present(record, answer_keys, "")
    if isinstance(answer, dict):
        return answer
    answer_text = normalize_reference(answer)
    if task_type == INTELLI_COCKPIT_TASK:
        return {
            "task": INTELLI_COCKPIT_TASK,
            "answer": answer_text,
            "category": str(first_present(record, ["category"], "")),
            "subcategory": str(first_present(record, ["subcategory", "sub_category"], "")),
            "reason": answer_text,
        }
    if task_type == "cabin_understanding":
        return {
            "task": "cabin_understanding",
            "driver_state": answer_text if answer else "unknown",
            "passenger_state": "unknown",
            "suggestion": "remind_driver" if "fatigue" in answer_text.lower() else "monitor",
            "source_answer": answer_text,
        }
    return {
        "task": "risk_reasoning",
        "risk_level": infer_risk_level(record, answer_text),
        "risk_object": infer_risk_object(record, answer_text),
        "reason": answer_text,
        "suggestion": infer_suggestion(record, answer_text),
        "source_answer": answer_text,
    }


def convert_record(
    record: dict[str, Any],
    index: int,
    source: str,
    image_root: Path | None,
    split: str,
) -> dict[str, Any]:
    task_type = infer_task_type(source, record)
    question = first_present(record, ["instruction", "question", "query", "prompt"], "")
    scenario = {
        "benchmark_source": source,
        "split": split,
        "original_id": first_present(record, ["id", "sample_id", "token", "question_id"], f"{index:06d}"),
        "category": first_present(record, ["category", "query_type", "task", "activity"], ""),
        "subcategory": first_present(record, ["subcategory", "sub_category"], ""),
        "shooting_angle": first_present(record, ["shooting_angle", "camera", "view"], ""),
        "difficulty": first_present(record, ["difficulty"], "unknown"),
    }
    return {
        "id": f"{source}_{split}_{index:06d}",
        "image": resolve_image(record, image_root),
        "vehicle_state": {
            "speed": first_present(record, ["speed"], 0),
            "weather": first_present(record, ["weather", "scene_condition", "weather_conditions"], "unknown"),
            "distance_to_front_car": first_present(record, ["distance_to_front_car"], None),
            "yaw_rate": first_present(record, ["yaw_rate"], None),
            "gear": first_present(record, ["gear"], "unknown"),
            "time": first_present(record, ["time", "lighting"], "unknown"),
        },
        "perception": {
            "objects": first_present(record, ["objects", "detections"], []),
            "scene": first_present(record, ["scene", "road_type", "location"], "external_benchmark"),
            "risk_hint": first_present(record, ["risk_hint", "category"], ""),
            "roadway": first_present(record, ["roadway"], ""),
            "sub_roadway": first_present(record, ["sub_roadway"], ""),
            "shooting_angle": scenario["shooting_angle"],
        },
        "instruction": str(question),
        "answer": normalize_answer(source, task_type, record),
        "meta": {
            "task_type": task_type,
            "source": "external_benchmark",
            "benchmark_source": source,
            "difficulty": scenario["difficulty"],
            "external": scenario,
        },
    }


def run(args: argparse.Namespace) -> None:
    source = args.source
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported source {source}; choose from {sorted(SUPPORTED_SOURCES)}")
    records = load_records(Path(args.input))
    if args.limit and args.limit > 0:
        records = records[: args.limit]
    image_root = Path(args.image_root) if args.image_root else None
    converted = [
        convert_record(row, index=i, source=source, image_root=image_root, split=args.split)
        for i, row in enumerate(records, start=1)
    ]
    dump_jsonl(converted, Path(args.output))
    print(f"converted {len(converted)} {source} records to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert external benchmark metadata to DriveMind-Instruct JSONL.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL metadata file.")
    parser.add_argument("--output", default="data/processed/drivemind_external.jsonl")
    parser.add_argument("--source", required=True, choices=sorted(SUPPORTED_SOURCES))
    parser.add_argument("--image_root", default="", help="Optional root prepended to relative image paths.")
    parser.add_argument("--split", default="eval", choices=["train", "val", "eval", "test"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
