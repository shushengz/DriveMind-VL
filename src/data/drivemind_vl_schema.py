"""Internal DriveMind-VL schema and leakage-safe data loading helpers."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")
SETTING_SUFFIX = re.compile(r"_(normal|text_only|wrong_image|blank_image)(?:_[A-Za-z0-9_]+)?$")
SAMPLE_SUFFIXES = (
    "normal_replay", "normal_anchor", "spatial_normal_qa", "extra_normal_fill",
    "blank_calibration", "wrong_image_calibration", "text_only_calibration",
    "normal_visual_qa", "spatial_hard_negative", "blank_refusal",
    "wrong_image_caution", "text_only_caution", "normal_anchor_gold_vs_model_wrong",
    "normal_gold_vs_refusal", "control_direct_answer_vs_caution",
    "blank_high_f1_vs_caution", "text_only_gold_overlap_vs_caution",
    "wrong_image_gold_overlap_vs_caution",
)


@dataclass
class DriveMindRecord:
    id: str
    dataset: str
    setting: str
    question: str
    answer: str
    image_paths: list[str] = field(default_factory=list)
    image_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_id(value: Any, row: dict[str, Any] | None = None) -> str:
    if row:
        direct = row.get("source_id") or (row.get("metadata") or {}).get("source_id")
        if direct:
            return normalize_id(direct)
    sample_id = str(value or "")
    for suffix in sorted(SAMPLE_SUFFIXES, key=len, reverse=True):
        marker = "_" + suffix
        if sample_id.endswith(marker):
            sample_id = sample_id[:-len(marker)]
            break
    return SETTING_SUFFIX.sub("", sample_id)


def extract_answer(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("answer", "")).strip()
    text = str(value or "").strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    return str(parsed.get("answer", "")).strip() if isinstance(parsed, dict) else text


def answer_only_json(answer: str) -> str:
    return json.dumps({"answer": str(answer).strip()}, ensure_ascii=False)


def is_answer_only_json(value: str) -> bool:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, dict) and set(parsed) == {"answer"} and isinstance(parsed["answer"], str)


def _setting_from_sft(row: dict[str, Any]) -> str:
    setting = str((row.get("metadata") or {}).get("setting") or row.get("setting") or "")
    if setting in SETTINGS:
        return setting
    sample_type = str(row.get("sample_type", ""))
    if sample_type.startswith("blank"):
        return "blank_image"
    if sample_type.startswith("text_only"):
        return "text_only"
    if sample_type.startswith("wrong_image"):
        return "wrong_image"
    return "normal"


def _question(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("prompt") or ((row.get("messages") or [{}])[0].get("content", "")))


def from_sft_row(row: dict[str, Any]) -> DriveMindRecord:
    messages = row.get("messages") or []
    assistant = next((msg.get("content") for msg in messages if msg.get("role") == "assistant"), row.get("assistant", ""))
    setting = _setting_from_sft(row)
    metadata = {
        "source_id": normalize_id(row.get("id"), row),
        "capability": str((row.get("metadata") or {}).get("capability", "")),
        "split": "train",
        "is_control": setting != "normal",
        "sample_type": row.get("sample_type", ""),
    }
    return DriveMindRecord(str(row.get("id", "")), str(row.get("dataset", "lingoqa")), setting, _question(row), extract_answer(assistant), list(row.get("image_paths") or []), list(row.get("image_labels") or []), metadata)


def from_strict_row(row: dict[str, Any]) -> DriveMindRecord:
    setting = str(row.get("setting", "normal"))
    source_meta = row.get("metadata") or {}
    metadata = {
        "source_id": normalize_id(row.get("id"), row),
        "capability": str(source_meta.get("capability", "")),
        "split": str(source_meta.get("split", "eval")),
        "is_control": setting != "normal",
    }
    return DriveMindRecord(str(row.get("id", "")), str(row.get("dataset", "")), setting, str(row.get("question") or row.get("prompt", "")), extract_answer(row.get("gold", "")), list(row.get("image_paths") or []), list(row.get("image_labels") or []), metadata)


def load_sft_records(path: str | Path) -> list[DriveMindRecord]:
    return [from_sft_row(row) for row in read_jsonl(path)]


def load_strict_records(path: str | Path) -> list[DriveMindRecord]:
    return [from_strict_row(row) for row in read_jsonl(path)]


def load_id_file(path: str | Path) -> set[str]:
    target = Path(path)
    if not target.exists():
        return set()
    payload = json.loads(target.read_text(encoding="utf-8"))
    values = payload.get("ids", []) if isinstance(payload, dict) else payload
    return {normalize_id(value) for value in values}


def excluded_eval_ids() -> set[str]:
    paths = (
        "outputs/final_report/lingoqa_heldout_ids_100.json",
        "outputs/final_report/drivelm_ood_ids_100.json",
        "outputs/final_report/drivelm_ood_ids_300.json",
    )
    result: set[str] = set()
    for path in paths:
        result.update(load_id_file(path))
    return result


def leakage_ids(records: Iterable[DriveMindRecord], excluded: set[str] | None = None) -> list[str]:
    excluded = excluded if excluded is not None else excluded_eval_ids()
    return sorted({record.metadata.get("source_id", normalize_id(record.id)) for record in records if normalize_id(record.metadata.get("source_id", record.id)) in excluded})


def make_user_content(record: DriveMindRecord) -> list[dict[str, str]]:
    content = [{"type": "image", "image": path} for path in record.image_paths] if record.setting != "text_only" else []
    content.append({"type": "text", "text": record.question})
    return content


def make_ms_swift_user_text(record: DriveMindRecord) -> str:
    """ms-swift standard dataset keeps media paths in `images` and tokens in text."""
    prefix = "".join("<image>" for _ in record.image_paths) if record.setting != "text_only" else ""
    return f"{prefix}{record.question}"
