from __future__ import annotations
import csv, json
from pathlib import Path
from collections import Counter
from copy import deepcopy

ROOT = Path('/root/autodl-tmp/DriveMind-VL')
NORMAL_JSONL = ROOT / 'data/processed/lingoqa_sft_v1_final_visual57.jsonl'
BALANCED67 = ROOT / 'data/processed/lingoqa_sft_v1_balanced_67.jsonl'
WRONG100 = ROOT / 'data/processed/drivemind_lingoqa_eval_100_wrong_frame.jsonl'
BLANK100 = ROOT / 'data/processed/drivemind_lingoqa_eval_100_blank_frame.jsonl'
OUT_JSONL = ROOT / 'data/processed/lingoqa_sft_v1_control_aware_82.jsonl'
OUT_SUMMARY = ROOT / 'outputs/eval_results/lingoqa_sft_v1_control_aware_82_summary.json'
OUT_PLAN = ROOT / 'outputs/cases/lingoqa_sft_v1_control_aware_82_plan.csv'
OUT_REPORT = ROOT / 'docs/lingoqa_sft_v1_control_aware_82_report.md'

TEXT_ONLY_IDS = [
    'lingoqa_eval_000056_positive_grounding',
    'lingoqa_eval_000005_positive_grounding',
    'lingoqa_eval_000061_positive_grounding',
    'lingoqa_eval_000099_positive_grounding',
    'lingoqa_eval_000004_positive_grounding',
]
BLANK_IDS = [
    'lingoqa_eval_000043_positive_grounding',
    'lingoqa_eval_000063_positive_grounding',
    'lingoqa_eval_000022_positive_grounding',
    'lingoqa_eval_000047_positive_grounding',
    'lingoqa_eval_000031_hard_negative_spatial_reasoning',
]
WRONG_IDS = [
    'lingoqa_eval_000075_wrong_image_confound',
    'lingoqa_eval_000049_wrong_image_confound',
    'lingoqa_eval_000009_wrong_image_confound',
    'lingoqa_eval_000021_wrong_image_confound',
    'lingoqa_eval_000023_wrong_image_confound',
    'lingoqa_eval_000040_wrong_image_confound',
    'lingoqa_eval_000062_wrong_image_confound',
    'lingoqa_eval_000069_wrong_image_confound',
    'lingoqa_eval_000090_wrong_image_confound',
    'lingoqa_eval_000098_wrong_image_confound',
]
REJECTION_ANSWER = 'I cannot determine the answer from the provided visual input.'
REJECTION_REASON = {
    'text_only': 'No image or frame sequence is provided, so the answer would be a language prior rather than visual evidence.',
    'blank_image': 'The provided image is a blank visual-ablation placeholder and does not contain the driving scene needed to answer.',
    'wrong_image': 'The provided frames are not the original scene for this question, so they do not support a reliable answer.',
}

def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def eval_case_id(row_id: str) -> str:
    parts = row_id.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else row_id

def make_rejection(row, mode, ablation_map=None):
    new = deepcopy(row)
    base_id = eval_case_id(row['id'])
    new['id'] = f'{row["id"]}_{mode}_rejection'
    new['answer'] = {
        'task': 'external_vqa',
        'answer': REJECTION_ANSWER,
        'references': [REJECTION_ANSWER],
        'category': 'LingoQA',
        'subcategory': row.get('answer', {}).get('subcategory', row.get('meta', {}).get('capability', '')),
        'reason': REJECTION_REASON[mode],
    }
    meta = new.setdefault('meta', {})
    meta['source'] = 'lingoqa_sft_v1_control_aware_82'
    meta['final_train_version'] = 'sft_v1_control_aware_82'
    meta['sft_v1_role'] = f'control_rejection_{mode}'
    meta['train_input_mode'] = mode
    meta['curation_status'] = 'reviewed_control_rejection_sft'
    if mode == 'text_only':
        new['image'] = ''
        meta.setdefault('external', {})['image_paths'] = []
    else:
        if ablation_map is None or base_id not in ablation_map:
            raise KeyError(f'missing {mode} ablation row for {base_id}')
        ablated = ablation_map[base_id]
        new['image'] = ablated.get('image', '')
        meta['external'] = deepcopy(ablated.get('meta', {}).get('external', {}))
    new['review'] = {
        'final_decision': 'keep_control_rejection_sft',
        'source_status': f'{mode}_control_rejection',
        'preferred_objective': 'control_aware_rejection_sft',
        'original_id': row['id'],
        'original_eval_case_id': base_id,
    }
    return new

normal = load_jsonl(NORMAL_JSONL)
balanced = {row['id']: row for row in load_jsonl(BALANCED67)}
wrong_map = {row['id']: row for row in load_jsonl(WRONG100)}
blank_map = {row['id']: row for row in load_jsonl(BLANK100)}
rows = list(normal)
for rid in TEXT_ONLY_IDS:
    rows.append(make_rejection(balanced[rid], 'text_only'))
for rid in BLANK_IDS:
    rows.append(make_rejection(balanced[rid], 'blank_image', blank_map))
for rid in WRONG_IDS:
    rows.append(make_rejection(balanced[rid], 'wrong_image', wrong_map))

ids = [row['id'] for row in rows]
dups = [key for key, value in Counter(ids).items() if value > 1]
missing = []
for row in rows:
    mode = row.get('meta', {}).get('train_input_mode', 'normal')
    paths = row.get('meta', {}).get('external', {}).get('image_paths', [])
    if mode == 'text_only':
        if paths:
            missing.append((row['id'], 'text_only_has_image_paths'))
    else:
        if not paths:
            missing.append((row['id'], 'no_image_paths'))
        for p in paths:
            if not (ROOT / p).exists():
                missing.append((row['id'], p))
if dups or missing:
    raise SystemExit({'duplicates': dups, 'missing': missing[:20]})

write_jsonl(OUT_JSONL, rows)
plan_rows = []
for row in rows:
    plan_rows.append({
        'id': row['id'],
        'role': row.get('meta', {}).get('sft_v1_role', ''),
        'capability': row.get('meta', {}).get('capability', row.get('answer', {}).get('subcategory', '')),
        'train_input_mode': row.get('meta', {}).get('train_input_mode', 'normal'),
        'source_status': row.get('review', {}).get('source_status', row.get('meta', {}).get('source_status', '')),
        'question': row.get('instruction', ''),
        'answer': row.get('answer', {}).get('answer', ''),
        'reason': row.get('answer', {}).get('reason', ''),
    })
OUT_PLAN.parent.mkdir(parents=True, exist_ok=True)
with OUT_PLAN.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(plan_rows[0].keys()))
    writer.writeheader(); writer.writerows(plan_rows)
summary = {
    'total_samples': len(rows),
    'normal_visual_samples': len(normal),
    'control_rejection_samples': len(rows) - len(normal),
    'by_role': dict(Counter(row.get('meta', {}).get('sft_v1_role', '') for row in rows)),
    'by_train_input_mode': dict(Counter(row.get('meta', {}).get('train_input_mode', 'normal') for row in rows)),
    'by_capability': dict(Counter(row.get('meta', {}).get('capability', row.get('answer', {}).get('subcategory', '')) for row in rows)),
    'purpose': 'Ordinary SFT with explicit rejection targets for text-only, blank-image, and wrong-image controls to approximate strict visual dependency training.',
    'recommend_training': True,
    'recommended_hparams': {'epochs': 1, 'learning_rate': 1e-5, 'lora_rank': 8, 'lora_alpha': 16, 'gradient_accumulation_steps': 4},
    'risk': 'May increase refusal/uncertainty if overtrained; keep LR lower than visual57 and run strict controls immediately after training.',
}
OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
OUT_REPORT.write_text('\n'.join(['# LingoQA SFT-v1 Control-Aware 82 Report', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '']), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
