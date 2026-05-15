"""Create a deterministic train/eval split for DriveMind-Instruct."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"skip line {line_no}: {exc}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_split(rows: list[dict[str, Any]], eval_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("meta", {}).get("task_type", "unknown")].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    total = len(rows)
    for task, task_rows in sorted(grouped.items()):
        rng.shuffle(task_rows)
        task_eval = max(1, round(eval_size * len(task_rows) / total))
        eval_rows.extend(task_rows[:task_eval])
        train_rows.extend(task_rows[task_eval:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows[:eval_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split DriveMind-Instruct JSONL into train/eval sets.")
    parser.add_argument("--input", default="data/processed/drivemind_seed.jsonl")
    parser.add_argument("--train_output", default="data/processed/drivemind_train.jsonl")
    parser.add_argument("--eval_output", default="data/processed/drivemind_eval.jsonl")
    parser.add_argument("--eval_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    train_rows, eval_rows = stratified_split(rows, args.eval_size, args.seed)
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.eval_output), eval_rows)

    print(f"total: {len(rows)}")
    print(f"train: {len(train_rows)} -> {args.train_output}")
    print(f"eval: {len(eval_rows)} -> {args.eval_output}")
    print("train task counts:", dict(Counter(row["meta"]["task_type"] for row in train_rows)))
    print("eval task counts:", dict(Counter(row["meta"]["task_type"] for row in eval_rows)))


if __name__ == "__main__":
    main()

