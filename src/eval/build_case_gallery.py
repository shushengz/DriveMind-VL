"""Build a simple CSV/HTML case gallery from offline visual-control artifacts."""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRIORITY = ["wrong_image_confound", "blank_hallucination", "text_prior_bias", "over_refusal", "visual_gain", "spatial_failure", "unresolved"]


def _u(hex_text: str) -> str:
    return bytes.fromhex(hex_text).decode("utf-8")


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def index_samples(paths: list[str]) -> dict[str, dict[str, Any]]:
    idx = {}
    for path in paths:
        for row in read_jsonl(Path(path)):
            idx.setdefault(str(row.get("id")), row)
    return idx


def comment_for(row: dict[str, Any]) -> str:
    return {
        "wrong_image_confound": _u("77726f6e672d696d61676520e5be97e58886e4b88de4bd8ee4ba8e206e6f726d616cefbc8ce79691e4bcbce8a2abe99499e8afafe59bbee5838fe5b9b2e689b0e68896e8afade8a880e58588e9aa8ce8bf87e5bcbae38082"),
        "blank_hallucination": _u("626c616e6b2d696d61676520e4b88be587bae78eb0e8a786e8a789e696ade8a880efbc8ce99c80e8a681e58aa0e5bcbae7a9bae799bde59bbee5838fe6a0a1e58786e38082"),
        "text_prior_bias": _u("746578742d6f6e6c7920e68ea5e8bf91e68896e8b685e8bf87206e6f726d616cefbc8ce8afb4e6988ee997aee9a298e58fafe883bde58fafe8a2abe8afade8a880e58588e9aa8ce78c9ce4b8ade38082"),
        "over_refusal": _u("6e6f726d616c20e59bbee5838fe4b88be68b92e7ad94efbc8ce99c80e981bfe5858de8bf87e5baa6e6a0a1e58786e68d9fe5aeb3e6ada3e5b8b8e59b9ee7ad94e38082"),
        "visual_gain": _u("6e6f726d616c20e6988ee698bee4bc98e4ba8ee68ea7e588b6e9a1b9efbc8ce698afe6ada3e590912076697375616c2067726f756e64696e6720e6a188e4be8be38082"),
    }.get(str(row.get("failure_type", "")), _u("e99c80e8a681e4babae5b7a5e5a48de6a0b8e38082"))

def build_rows(case_scores: list[dict[str, Any]], sample_index: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(case_scores, key=lambda r: PRIORITY.index(r.get("failure_type", "unresolved")) if r.get("failure_type", "unresolved") in PRIORITY else 99)
    if limit:
        ordered = ordered[:limit]
    rows = []
    for case in ordered:
        sample = sample_index.get(str(case.get("id")), {})
        rows.append({"id": case.get("id", ""), "dataset": case.get("dataset") or sample.get("dataset", ""), "image_paths": json.dumps(sample.get("image_paths", []), ensure_ascii=False), "image_labels": json.dumps(sample.get("image_labels", []), ensure_ascii=False), "question": sample.get("question") or case.get("question", ""), "gold": case.get("gold", ""), "normal_prediction": case.get("normal_prediction", ""), "text_only_prediction": case.get("text_only_prediction", ""), "wrong_image_prediction": case.get("wrong_image_prediction", ""), "blank_image_prediction": case.get("blank_image_prediction", ""), "normal_f1": case.get("normal_f1", ""), "text_only_f1": case.get("text_only_f1", ""), "wrong_image_f1": case.get("wrong_image_f1", ""), "blank_image_f1": case.get("blank_image_f1", ""), "case_gap": case.get("case_gap", ""), "failure_type": case.get("failure_type", ""), "chinese_comment": comment_for(case)})
    return rows


def render_image_cell(paths_json: str) -> str:
    try:
        paths = json.loads(paths_json)
    except Exception:
        paths = []
    parts = []
    for path in paths[:3]:
        path = str(path)
        if Path(path).exists():
            parts.append(f'<img src="{html.escape(path)}" style="max-width:160px;max-height:110px">')
        else:
            parts.append(f'<div>{html.escape(path)}</div>')
    return "".join(parts)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    body = []
    for row in rows:
        cells = []
        for key in headers:
            cells.append(f"<td>{render_image_cell(str(row.get(key, '')))}</td>" if key == "image_paths" else f"<td>{html.escape(str(row.get(key, '')))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    table = "<table border='1' cellspacing='0' cellpadding='4'><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>"
    path.write_text("<html><meta charset='utf-8'><body>" + table + "</body></html>\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build case gallery CSV/HTML, CPU-only.")
    parser.add_argument("--case_scores", required=True)
    parser.add_argument("--raw_predictions", default="")
    parser.add_argument("--visual_samples", nargs="*", default=[])
    parser.add_argument("--output_dir", default="outputs/final_report")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = 20 if args.dry_run else args.limit
    rows = build_rows(read_csv(Path(args.case_scores)), index_samples(args.visual_samples), limit)
    out_dir = Path(args.output_dir)
    write_csv(out_dir / "case_gallery.csv", rows)
    write_html(out_dir / "case_gallery.html", rows)
    print(json.dumps({"csv": (out_dir / "case_gallery.csv").as_posix(), "html": (out_dir / "case_gallery.html").as_posix(), "count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
