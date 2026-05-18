"""Evaluate LingoQA predictions with the official Lingo-Judge classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_references(path: Path):
    import pandas as pd

    refs = pd.read_parquet(path)
    refs = refs[["question_id", "segment_id", "question", "answer"]]
    refs = refs.groupby(["question_id", "segment_id", "question"]).agg(list).reset_index()
    refs = refs.rename({"answer": "references"}, axis=1)
    return refs


def load_predictions(path: Path):
    import pandas as pd

    preds = pd.read_csv(path)
    preds = preds.rename({"answer": "prediction"}, axis=1)
    return preds


def load_judge(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval().to(device)
    return tokenizer, model, torch


def score_one(tokenizer: Any, model: Any, torch: Any, device: str, question: str, references: list[str], prediction: str) -> dict[str, Any]:
    texts = [
        f"{tokenizer.cls_token}\nQuestion: {question}\nAnswer: {str(ref).lower().strip()}\nStudent: {str(prediction).lower().strip()}"
        for ref in references
    ]
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.squeeze(-1)
    max_score = float(torch.max(logits).detach().cpu())
    probability = float(torch.sigmoid(torch.tensor(max_score)))
    return {"score": max_score, "probability": probability, "correct": max_score > 0.0}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd
    from tqdm import tqdm

    refs = load_references(Path(args.references))
    preds = load_predictions(Path(args.predictions_csv))
    merged = pd.merge(preds, refs, on=["question_id", "segment_id"])
    if args.max_samples > 0:
        merged = merged.head(args.max_samples)

    tokenizer, model, torch = load_judge(args.model_name, args.device)
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="lingo-judge"):
        result = score_one(
            tokenizer=tokenizer,
            model=model,
            torch=torch,
            device=args.device,
            question=str(row["question"]),
            references=list(row["references"]),
            prediction=str(row["prediction"]),
        )
        rows.append(
            {
                "question_id": row["question_id"],
                "segment_id": row["segment_id"],
                "question": row["question"],
                "prediction": row["prediction"],
                "references": row["references"],
                **result,
            }
        )

    score = sum(1 for row in rows if row["correct"]) / len(rows) if rows else 0.0
    return {
        "predictions_csv": args.predictions_csv,
        "references": args.references,
        "model_name": args.model_name,
        "matched": len(merged),
        "evaluated": len(rows),
        "lingo_judge_score": round(score, 4),
        "avg_probability": round(sum(row["probability"] for row in rows) / len(rows), 4) if rows else 0.0,
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lingo-Judge on exported LingoQA predictions.")
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--references", default="data/external/lingoqa/val.parquet")
    parser.add_argument("--output", default="outputs/eval_results/lingo_judge_metrics.json")
    parser.add_argument("--model_name", default="wayveai/Lingo-Judge")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    report = evaluate(args)
    cases = report.pop("cases")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    cases_output = Path(args.output).with_suffix(".cases.jsonl")
    with cases_output.open("w", encoding="utf-8") as f:
        for row in cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote Lingo-Judge cases to {cases_output}")


if __name__ == "__main__":
    main()
