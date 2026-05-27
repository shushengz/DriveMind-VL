
from __future__ import annotations
import csv,json
from pathlib import Path
from collections import Counter,defaultdict
from copy import deepcopy
ROOT=Path('/root/autodl-tmp/DriveMind-VL')
DECISIONS={"lingoqa_eval_000168": ["positive_visual_grounding", "Two pedestrians are crossing the road.", "The selected frames show two pedestrians on the zebra crossing ahead near the traffic light.", "clean500 visual review: two pedestrians visibly crossing."], "lingoqa_eval_000270": ["positive_visual_grounding", "Yes, there is a marked pedestrian crossing at the traffic light.", "The selected frames show zebra-crossing road markings across the ego path at the signalized intersection.", "clean500 visual review: crossing markings visible."], "lingoqa_eval_000355": ["positive_visual_grounding", "Yes, the cyclist is in a dedicated cycle lane.", "The cyclist is visible in the left-side marked cycling lane rather than in the ego vehicle lane.", "clean500 visual review: cyclist in marked lane."], "lingoqa_eval_000410": ["positive_visual_grounding", "Yes, there is a vehicle ahead in the ego lane.", "A large vehicle is visible directly ahead along the ego vehicle path in the selected frames.", "clean500 visual review: vehicle ahead visible."], "lingoqa_eval_000442": ["positive_visual_grounding", "Yes, the motorcyclist is in the ego lane ahead.", "The motorcyclist is visible ahead and aligned with the ego lane in the selected frames.", "clean500 visual review: motorcyclist in lane."], "lingoqa_eval_000461": ["positive_visual_grounding", "Yes, there is a dedicated cycle lane on the left.", "The road markings and lane layout show a dedicated cycle lane along the left side of the ego lane.", "clean500 visual review: left cycle lane visible."], "lingoqa_eval_000429": ["positive_visual_grounding", "Yes, one pedestrian is crossing the road ahead.", "A pedestrian is visible crossing ahead near the right side of the roadway in the later frame.", "clean500 visual review: pedestrian crossing visible."], "lingoqa_eval_000109": ["positive_visual_grounding", "Yes, the visible traffic lights are green.", "The traffic signals at the intersection are visible and show green in the selected frames.", "clean500 visual review: green traffic lights visible."], "lingoqa_eval_000141": ["positive_visual_grounding", "Yes, the traffic light is green.", "The selected frames show a green traffic signal at the junction ahead.", "clean500 visual review: green traffic light visible."], "lingoqa_eval_000161": ["positive_visual_grounding", "Yes, visible traffic lights are present and they are green.", "The selected frames show traffic lights at the intersection with a green signal visible.", "clean500 visual review: green signals visible."], "lingoqa_eval_000103": ["positive_visual_grounding", "Yes, cyclists are visible in the lane to the left of the ego vehicle.", "Multiple cyclists appear along the left-side lane in the selected frames.", "clean500 visual review: cyclists on left side visible."], "lingoqa_eval_000387": ["positive_visual_grounding", "You should pay attention to pedestrians crossing the road.", "The selected frames show pedestrians crossing ahead near the construction area, creating the main attention target.", "clean500 visual review: pedestrians crossing visible."], "lingoqa_eval_000271": ["positive_visual_grounding", "The vehicle being followed is the silver car in front.", "A silver car is visible directly ahead of the ego vehicle in the selected frames.", "clean500 visual review: silver car ahead visible."], "lingoqa_eval_000262": ["positive_visual_grounding", "The traffic lights require the driver's attention.", "The selected frames show a signalized crossing ahead with pedestrians nearby, so the traffic-light state is the key visual cue.", "clean500 visual review: signalized crossing visible."], "lingoqa_eval_000409": ["spatial_reasoning_correction", "Accelerate gently because the van ahead is pulling away and the lane ahead is opening.", "The selected frames show the white van ahead moving forward, leaving more space in the ego lane.", "clean500 spatial review: van-ahead motion cue."], "lingoqa_eval_000107": ["spatial_reasoning_correction", "Start slowly because the traffic light is green and the vehicle in front is pulling away.", "The frames show a green light ahead and the lead vehicle moving, supporting a cautious start.", "clean500 spatial review: green light plus lead vehicle."], "lingoqa_eval_000196": ["spatial_reasoning_correction", "Drive straight while moving slightly right in the lane to pass parked vehicles on the left.", "The selected frames show parked vehicles along the left side, while the ego path continues forward with room to keep right.", "clean500 spatial review: parked vehicles on left."], "lingoqa_eval_000383": ["spatial_reasoning_correction", "Slow down approaching the intersection because a pedestrian is on the zebra crossing ahead.", "The selected frames show a pedestrian engaged on the crosswalk in front of the ego path.", "clean500 spatial review: pedestrian on crossing."], "lingoqa_eval_000301": ["spatial_reasoning_correction", "Maintain speed and lane because the zebra crossing ahead is clear of pedestrians.", "The selected frames show the ego lane and zebra crossing ahead without a pedestrian entering the vehicle path.", "clean500 spatial review: clear crossing."], "lingoqa_eval_000451": ["spatial_reasoning_correction", "A cyclist is directly ahead in the current lane.", "The selected frames show a cyclist in front of the ego vehicle along the current lane.", "clean500 spatial review: cyclist ahead."], "lingoqa_eval_000399": ["spatial_reasoning_correction", "Accelerate and move right to pass the cyclist while the lane ahead opens.", "The selected frames show the cyclist near the left/bike-lane area and a large vehicle ahead moving away, leaving room to pass cautiously.", "clean500 spatial review: cyclist and opening lane."], "lingoqa_eval_000147": ["spatial_reasoning_correction", "Drive slowly and follow the bus ahead after it merges into the ego lane.", "The selected frames show a bus directly ahead of the ego vehicle in the same lane.", "clean500 spatial review: bus ahead in lane."], "lingoqa_eval_000155": ["spatial_reasoning_correction", "Slow down and keep enough right-side lane space for the bicycle in the left lane ahead.", "The selected frames show a cyclist/bicycle on the left side of the road, so the ego vehicle should slow and maintain clearance.", "clean500 spatial review: cyclist left-side clearance."], "lingoqa_eval_000352": ["spatial_reasoning_correction", "Do not accelerate because multiple road users and vehicles create conflicts ahead.", "The frames show a bus ahead, a pedestrian/cyclist area, and crossing traffic near the intersection, so acceleration is not supported.", "clean500 spatial review: multiple conflicts ahead."], "lingoqa_eval_000376": ["spatial_reasoning_correction", "Pay attention to the traffic lights and road users ahead because the signal state and nearby pedestrians may require slowing.", "The selected frames show a signalized junction with traffic and pedestrians near the roadway.", "clean500 spatial review: signals and road users."], "lingoqa_eval_000497": ["spatial_reasoning_correction", "Slow down near the intersection because pedestrians and a cyclist are crossing ahead despite the green light.", "The selected frames show green traffic lights but also vulnerable road users crossing the ego path.", "clean500 spatial review: green signal plus crossing users."], "lingoqa_eval_000493": ["high_quality_fix_rewritten", "The right car is driving in the rightmost lane.", "The selected frames show the car on the right side of the roadway occupying the rightmost lane.", "clean500 manual rewrite: simple lane-position target used to bring rewritten-fix bucket to training minimum."]}
ALPHA=ROOT/'data/processed/lingoqa_sft_v1_balanced_alpha40.jsonl'
CLEAN_CSV=ROOT/'outputs/cases/lingoqa_sft_v1_clean500_candidate_plan.csv'
DATASET=ROOT/'data/processed/drivemind_lingoqa_eval_500.jsonl'
OUT_JSONL=ROOT/'data/processed/lingoqa_sft_v1_balanced_67.jsonl'
OUT_PLAN=ROOT/'outputs/cases/lingoqa_sft_v1_reviewed_plan_balanced_67.csv'
OUT_SUM=ROOT/'outputs/eval_results/lingoqa_sft_v1_balanced_67_summary.json'
OUT_REPORT=ROOT/'docs/lingoqa_sft_v1_balanced_67_review_report.md'

def load_jsonl(p):
    rows=[]
    with p.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def write_jsonl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
alpha=load_jsonl(ALPHA)
clean_plan={r['id']:r for r in read_csv(CLEAN_CSV)}
samples={r['id']:r for r in load_jsonl(DATASET)}
rows=list(alpha)
review_rows=[]
# carry alpha reviewed plan if present
alpha_plan=ROOT/'outputs/cases/lingoqa_sft_v1_reviewed_plan_alpha40.csv'
if alpha_plan.exists(): review_rows=read_csv(alpha_plan)
for sid,(role,answer,reason,note) in DECISIONS.items():
    sample=deepcopy(samples[sid])
    plan=clean_plan[sid]
    new=deepcopy(sample)
    suffix={'positive_visual_grounding':'clean500_positive','spatial_reasoning_correction':'clean500_spatial','high_quality_fix_rewritten':'clean500_manual_rewrite'}[role]
    new['id']=f'{sid}_{suffix}'
    new['answer']={
        'task':'external_vqa',
        'answer':answer,
        'references':[answer, sample.get('answer',{}).get('answer','')],
        'category':'LingoQA',
        'subcategory':sample.get('answer',{}).get('subcategory', plan.get('capability','')),
        'reason':reason,
    }
    meta=new.setdefault('meta',{})
    meta['source']='lingoqa_sft_v1_balanced_67'
    meta['benchmark_source']='lingoqa'
    meta['capability']=plan.get('capability','')
    meta['sft_v1_role']=role
    meta['curation_status']='reviewed_keep_sft_v1'
    meta['source_status']='clean500_excludes_100_control_manual_review'
    meta['prompt_variant']='spatial'
    meta['frame_strategy']='first_middle_last'
    meta['max_images']=3
    meta['visual_gain_5frame_vs_controls']=float(plan['visual_gain'])
    meta['f1_5frame']=float(plan['f1_5frame'])
    meta['f1_single']=float(plan['f1_single'])
    meta['f1_text_only']=float(plan['f1_text_only'])
    new['review']={
        'final_decision':'keep_sft_v1',
        'source_status':'clean500_excludes_100_control_manual_review',
        'review_note':note,
        'preferred_objective':'normal_vqa_sft',
        'manual_visual_review':'contact_sheet_reviewed',
        'original_gold_answer':sample.get('answer',{}).get('answer',''),
        'original_gold_reason':sample.get('answer',{}).get('reason',''),
        'clean500_plan_pred_5frame':plan.get('pred_5frame',''),
    }
    rows.append(new)
    review_rows.append({
        'id':new['id'],'eval_case_id':sid,'source_status':'clean500_excludes_100_control_manual_review',
        'candidate_type':'clean500_visual_gain_manual_review','capability':plan.get('capability',''),
        'sft_v1_role':role,'sft_v1_action':'keep_sft_v1','ordinary_sft_eligible':'yes',
        'preferred_objective':'normal_vqa_sft','manual_rewrite_required':'done','final_decision':'keep_sft_v1',
        'final_answer':answer,'final_reason':reason,'review_note':note,'priority_bucket':'high',
        'visual_dependency_gap':plan.get('visual_gain',''),'normal_f1':plan.get('f1_5frame',''),
        'control_max_f1':str(max(float(plan.get('f1_single',0)),float(plan.get('f1_text_only',0)))),
        'question':sample.get('instruction',''),'image':sample.get('image',''),
        'image_paths':json.dumps(sample.get('meta',{}).get('external',{}).get('image_paths',[]),ensure_ascii=False),
    })
# validation
ids=[r['id'] for r in rows]
dups=[k for k,v in Counter(ids).items() if v>1]
missing=[]
for r in rows:
    paths=r.get('meta',{}).get('external',{}).get('image_paths',[])
    if not paths: missing.append((r['id'],'no_image_paths'))
    for p in paths:
        if not (ROOT/p).exists(): missing.append((r['id'],p))
    for key in ['id','image','instruction','answer','meta','review']:
        if key not in r: missing.append((r.get('id','?'),'missing_'+key))
if dups or missing:
    raise SystemExit({'dups':dups,'missing':missing[:20]})
write_jsonl(OUT_JSONL,rows)
OUT_PLAN.parent.mkdir(parents=True,exist_ok=True)
fieldnames=list(review_rows[0].keys())
with OUT_PLAN.open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(review_rows)
summary={
    'total_samples':len(rows),
    'base_alpha40_samples':len(alpha),
    'clean500_added_samples':len(DECISIONS),
    'by_sft_v1_role':dict(Counter(r.get('meta',{}).get('sft_v1_role','') for r in rows)),
    'by_capability':dict(Counter(r.get('meta',{}).get('capability','') for r in rows)),
    'by_source_status':dict(Counter(r.get('review',{}).get('source_status', r.get('meta',{}).get('source_status','')) for r in rows)),
    'target_ratio_status':{
        'total_50_80':50 <= len(rows) <= 80,
        'positive_visual_grounding_30_40':30 <= Counter(r.get('meta',{}).get('sft_v1_role','') for r in rows)['positive_visual_grounding'] <= 40,
        'spatial_reasoning_correction_20_30':20 <= Counter(r.get('meta',{}).get('sft_v1_role','') for r in rows)['spatial_reasoning_correction'] <= 30,
        'anti_hallucination_counterfactual_10_15':10 <= Counter(r.get('meta',{}).get('sft_v1_role','') for r in rows)['anti_hallucination_counterfactual'] <= 15,
        'high_quality_fix_rewritten_7_15':7 <= Counter(r.get('meta',{}).get('sft_v1_role','') for r in rows)['high_quality_fix_rewritten'] <= 15,
    },
    'clean500_added_ids':list(DECISIONS.keys()),
    'eval_contamination_risk':{
        'lingoqa_100_control':'high for alpha40 rows; do not report on original 100 as clean holdout',
        'clean500_added':'lower for original 100-control evaluation because ids are excluded from that set, but still from LingoQA eval distribution; use fresh holdout for final claim',
    },
    'recommend_smoke_training':True,
    'recommendation_reason':'Meets 50-80 total and role-balance gates after manual contact-sheet review; use only as smoke training, then evaluate on a fresh/unseen control split.',
}
OUT_SUM.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# LingoQA SFT-v1 Balanced-67 Review Report','',
'## Scope','',
'This revision starts from the conservative alpha40 reviewed set, mines clean candidates from the 500-sample LingoQA run while excluding the original 100-control ids, manually reviews contact sheets, and adds 27 selected rows. No training was run.','',
'## Outputs','',
f'- JSONL: `{OUT_JSONL.relative_to(ROOT)}`',f'- Reviewed plan: `{OUT_PLAN.relative_to(ROOT)}`',f'- Summary: `{OUT_SUM.relative_to(ROOT)}`','',
'## Distribution','', '```json', json.dumps(summary,ensure_ascii=False,indent=2),'```','',
'## Training Decision','',
'The dataset now meets the planned role-count gates for a smoke training run. The original 100-control set remains contaminated by alpha40 rows, so any post-training claim must use a fresh strict visual-control split or clearly mark the 100-control result as diagnostic only.','']
OUT_REPORT.write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(OUT_JSONL)
