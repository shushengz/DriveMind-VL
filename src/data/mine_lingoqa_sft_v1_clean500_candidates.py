
from __future__ import annotations
import csv, json, re
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/root/autodl-tmp/DriveMind-VL')
OUT_CSV=ROOT/'outputs/cases/lingoqa_sft_v1_clean500_candidate_plan.csv'
OUT_SUM=ROOT/'outputs/eval_results/lingoqa_sft_v1_clean500_candidate_summary.json'
SHEET_DIR=ROOT/'outputs/cases/visual_review_sheets_sft_v1_clean500'
SHEET_DIR.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    rows=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def parse_answer(pred):
    text=str(pred or '').strip()
    m=re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
    if m: text=m.group(1).strip()
    try:
        obj=json.loads(text)
        if isinstance(obj,dict):
            return str(obj.get('answer') or obj.get('reason') or '')
    except Exception:
        pass
    return text

def toks(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())

def f1(pred,gold):
    p=toks(pred); g=toks(gold)
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    cp=Counter(p); cg=Counter(g)
    common=sum((cp & cg).values())
    if common==0: return 0.0
    prec=common/len(p); rec=common/len(g)
    return 2*prec*rec/(prec+rec)

def score_row(row):
    gold=row.get('gold') if isinstance(row.get('gold'),dict) else row.get('answer',{})
    refs=[]
    if isinstance(gold,dict):
        refs=[str(gold.get('answer') or gold.get('reason') or '')]+[str(x) for x in gold.get('references',[]) if x]
    pred=parse_answer(row.get('prediction'))
    return max([f1(pred,ref) for ref in refs if ref] or [0.0]), pred

def wrap(txt, width=58, max_lines=4):
    words=str(txt).replace('\n',' ').split()
    lines=[]; cur=''
    for w in words:
        if len(cur)+len(w)+1>width:
            lines.append(cur); cur=w
            if len(lines)>=max_lines: break
        else:
            cur=(cur+' '+w).strip()
    if cur and len(lines)<max_lines: lines.append(cur)
    return '\n'.join(lines[:max_lines])

def selected_frames(paths):
    paths=[p for p in paths if p]
    if len(paths)<=3: return paths
    return [paths[0], paths[len(paths)//2], paths[-1]]

def make_sheet(rows, filename):
    thumb_w, thumb_h = 260, 146
    text_h = 150
    cols=2
    cell_w=thumb_w*3+24
    cell_h=thumb_h+text_h+28
    W=cols*cell_w
    H=((len(rows)+cols-1)//cols)*cell_h
    img=Image.new('RGB',(W,H),'white')
    draw=ImageDraw.Draw(img)
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    except Exception: font=None
    for idx,r in enumerate(rows):
        x=(idx%cols)*cell_w; y=(idx//cols)*cell_h
        draw.rectangle([x,y,x+cell_w-2,y+cell_h-2],outline=(180,180,180))
        title=f"{r['id']} | {r['capability']} | gain {r['visual_gain']} | f5 {r['f1_5frame']}"
        draw.text((x+6,y+5),title,fill=(0,0,0),font=font)
        for j,p in enumerate(selected_frames(json.loads(r['image_paths']))):
            fp=ROOT/p
            tx=x+6+j*(thumb_w+4); ty=y+24
            try:
                im=Image.open(fp).convert('RGB')
                im.thumbnail((thumb_w,thumb_h))
                bg=Image.new('RGB',(thumb_w,thumb_h),(235,235,235))
                bg.paste(im,((thumb_w-im.width)//2,(thumb_h-im.height)//2))
                img.paste(bg,(tx,ty))
            except Exception as e:
                draw.rectangle([tx,ty,tx+thumb_w,ty+thumb_h],fill=(240,220,220),outline=(160,0,0))
                draw.text((tx+4,ty+4),'MISSING '+str(e)[:40],fill=(120,0,0),font=font)
        text=f"Q: {r['question']}\nGold: {r['gold_answer']}\nPred5: {r['pred_5frame']}\nTextF1:{r['f1_text_only']} SingleF1:{r['f1_single']}"
        draw.text((x+6,y+thumb_h+32),wrap(text,105,7),fill=(0,0,0),font=font)
    img.save(SHEET_DIR/filename, quality=92)

normal={r['id']:r for r in load_jsonl(ROOT/'outputs/eval_results/lingoqa_qwen25vl_3b_500_5frame_predictions.jsonl')}
single={r['id']:r for r in load_jsonl(ROOT/'outputs/eval_results/lingoqa_qwen25vl_3b_500_single_frame_predictions.jsonl')}
text={r['id']:r for r in load_jsonl(ROOT/'outputs/eval_results/lingoqa_qwen25vl_3b_500_text_only_predictions.jsonl')}
control100={r['id'] for r in load_jsonl(ROOT/'data/processed/drivemind_lingoqa_eval_100_control.jsonl')}
rows=[]
for sid,row in normal.items():
    if sid in control100: continue
    if sid not in single or sid not in text: continue
    f5,p5=score_row(row); f1s,ps=score_row(single[sid]); f1t,pt=score_row(text[sid])
    gold=row.get('gold') or {}
    cap=gold.get('subcategory') or row.get('meta',{}).get('external',{}).get('capability','')
    image_paths=row.get('meta',{}).get('external',{}).get('image_paths') or row.get('media',{}).get('image_paths') or []
    gain=f5-max(f1s,f1t)
    # Conservative mining labels. Final inclusion still requires visual review.
    if f5 >= 0.45 and gain >= 0.20:
        role='positive_visual_grounding'
    elif cap in {'spatial_localization','reasoning_world_knowledge'} and f5 >= 0.25 and gain >= 0.10:
        role='spatial_reasoning_correction'
    elif f5 >= 0.55 and cap in {'spatial_localization','reasoning_world_knowledge'}:
        role='spatial_reasoning_correction'
    else:
        continue
    rows.append({
        'id':sid,
        'source_status':'clean500_excludes_100_control',
        'candidate_type':'clean500_visual_gain',
        'capability':cap,
        'sft_v1_role':role,
        'recommended_action':'manual_visual_review_then_keep_or_drop',
        'preferred_objective':'normal_vqa_sft',
        'priority_score':round(f5+max(gain,0),4),
        'visual_gain':round(gain,4),
        'f1_5frame':round(f5,4),
        'f1_single':round(f1s,4),
        'f1_text_only':round(f1t,4),
        'question':row.get('meta',{}).get('question','') or '',
        'gold_answer':gold.get('answer',''),
        'gold_reason':gold.get('reason',''),
        'pred_5frame':p5,
        'pred_single':ps,
        'pred_text_only':pt,
        'image':row.get('media',{}).get('image') or (image_paths[0] if image_paths else ''),
        'image_paths':json.dumps(image_paths,ensure_ascii=False),
        'segment_id':row.get('meta',{}).get('external',{}).get('segment_id',''),
    })
# add instruction from dataset by id
samples={r['id']:r for r in load_jsonl(ROOT/'data/processed/drivemind_lingoqa_eval_500.jsonl')}
for r in rows:
    sample=samples.get(r['id'],{})
    r['question']=sample.get('instruction') or r['question']
rows.sort(key=lambda r:(0 if r['sft_v1_role']=='positive_visual_grounding' else 1, -float(r['priority_score'])))
OUT_CSV.parent.mkdir(parents=True,exist_ok=True)
with OUT_CSV.open('w',encoding='utf-8',newline='') as f:
    fieldnames=list(rows[0].keys()) if rows else []
    w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
summary={
    'count':len(rows),
    'excluded_control100_count':len(control100),
    'by_role':dict(Counter(r['sft_v1_role'] for r in rows)),
    'by_capability':dict(Counter(r['capability'] for r in rows)),
    'criteria':'exclude 100 control; positive if 5frame_f1>=0.45 and gain>=0.20; spatial if spatial/reasoning and (f1>=0.25 gain>=0.10 or f1>=0.55); manual visual review required',
}
OUT_SUM.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# sheets: top candidates by role/cap
pos=[r for r in rows if r['sft_v1_role']=='positive_visual_grounding'][:36]
spa=[r for r in rows if r['sft_v1_role']=='spatial_reasoning_correction'][:36]
make_sheet(pos[:18], 'clean500_positive_top18.jpg')
make_sheet(pos[18:36], 'clean500_positive_next18.jpg')
make_sheet(spa[:18], 'clean500_spatial_top18.jpg')
make_sheet(spa[18:36], 'clean500_spatial_next18.jpg')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(OUT_CSV)
print(SHEET_DIR)
