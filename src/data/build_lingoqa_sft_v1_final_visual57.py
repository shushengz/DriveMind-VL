from __future__ import annotations
import csv, json
from pathlib import Path
from collections import Counter

ROOT = Path('/root/autodl-tmp/DriveMind-VL')
INPUT_JSONL = ROOT / 'data/processed/lingoqa_sft_v1_balanced_67.jsonl'
INPUT_PLAN = ROOT / 'outputs/cases/lingoqa_sft_v1_reviewed_plan_balanced_67.csv'
OUT_JSONL = ROOT / 'data/processed/lingoqa_sft_v1_final_visual57.jsonl'
OUT_PLAN = ROOT / 'outputs/cases/lingoqa_sft_v1_final_visual57_plan.csv'
OUT_SUMMARY = ROOT / 'outputs/eval_results/lingoqa_sft_v1_final_visual57_summary.json'
OUT_REPORT = ROOT / 'docs/lingoqa_sft_v1_final_visual57_report.md'
EXCLUDED_ROLE = 'anti_hallucination_counterfactual'

def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def role(row):
    return row.get('meta', {}).get('sft_v1_role', '')

def source_status(row):
    return row.get('review', {}).get('source_status') or row.get('meta', {}).get('source_status', '')

rows = [row for row in load_jsonl(INPUT_JSONL) if role(row) != EXCLUDED_ROLE]
for row in rows:
    row.setdefault('meta', {})['source'] = 'lingoqa_sft_v1_final_visual57'
    row['meta']['final_train_version'] = 'sft_v1_final_visual57'
    row.setdefault('review', {})['final_train_version'] = 'sft_v1_final_visual57'
    row['review']['excluded_from_plain_sft_roles'] = [EXCLUDED_ROLE]

ids = [row['id'] for row in rows]
dups = [item for item, count in Counter(ids).items() if count > 1]
missing = []
for row in rows:
    for key in ['id', 'image', 'instruction', 'answer', 'meta', 'review']:
        if key not in row:
            missing.append((row.get('id', '?'), 'missing_' + key))
    paths = row.get('meta', {}).get('external', {}).get('image_paths', [])
    if len(paths) < 3:
        missing.append((row.get('id', '?'), 'less_than_3_image_paths', len(paths)))
    for path in paths:
        if not (ROOT / path).exists():
            missing.append((row.get('id', '?'), path))
if dups or missing:
    raise SystemExit({'duplicates': dups, 'missing': missing[:20]})

write_jsonl(OUT_JSONL, rows)

plan_rows = []
if INPUT_PLAN.exists():
    with INPUT_PLAN.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('sft_v1_role') != EXCLUDED_ROLE:
                row['final_train_version'] = 'sft_v1_final_visual57'
                plan_rows.append(row)
    if plan_rows:
        OUT_PLAN.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(plan_rows[0].keys())
        with OUT_PLAN.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(plan_rows)

role_counts = Counter(role(row) for row in rows)
cap_counts = Counter(row.get('meta', {}).get('capability', '') for row in rows)
source_counts = Counter(source_status(row) for row in rows)
summary = {
    'total_samples': len(rows),
    'input_jsonl': str(INPUT_JSONL.relative_to(ROOT)),
    'output_jsonl': str(OUT_JSONL.relative_to(ROOT)),
    'excluded_plain_sft_roles': [EXCLUDED_ROLE],
    'by_sft_v1_role': dict(role_counts),
    'by_capability': dict(cap_counts),
    'by_source_status': dict(source_counts),
    'target_ratio_status': {
        'total_50_80': 50 <= len(rows) <= 80,
        'positive_visual_grounding_at_least_30': role_counts.get('positive_visual_grounding', 0) >= 30,
        'spatial_reasoning_correction_at_least_20': role_counts.get('spatial_reasoning_correction', 0) >= 20,
        'high_quality_fix_rewritten_at_least_7': role_counts.get('high_quality_fix_rewritten', 0) >= 7,
        'anti_hallucination_plain_sft_zero': role_counts.get(EXCLUDED_ROLE, 0) == 0,
    },
    'recommend_smoke_training': True,
    'recommendation_reason': 'Use this visual-only dataset for the next ordinary SFT run; keep anti/counterfactual samples for preference or rejection training, not plain SFT.',
    'training_hparams_recommendation': {
        'epochs': 1,
        'learning_rate': 2e-5,
        'lora_rank': 8,
        'lora_alpha': 16,
        'gradient_accumulation_steps': 4,
        'processor': 'slow use_fast=False default',
    },
    'eval_warning': 'The original 100-control split is contaminated by some source rows. Use a fresh strict visual-control split for final reporting.',
}
OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
OUT_REPORT.write_text('\n'.join([
    '# LingoQA SFT-v1 Final Visual57 Report',
    '',
    'This final ordinary-SFT dataset removes `anti_hallucination_counterfactual` rows from the balanced-67 set because plain SFT on confound rows can strengthen text-only or blank-image priors.',
    '',
    '## Summary',
    '',
    '```json',
    json.dumps(summary, ensure_ascii=False, indent=2),
    '```',
    '',
    '## Decision',
    '',
    'Use `data/processed/lingoqa_sft_v1_final_visual57.jsonl` for the next smoke training. Keep the excluded anti/counterfactual rows for DPO/ORPO/RFT-lite or explicit rejection/uncertainty training.',
    '',
]), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
