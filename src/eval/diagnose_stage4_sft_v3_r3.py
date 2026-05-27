
"""Stage 4 diagnosis for SFT-v3-r3 answer-only results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BASE="base_qwen25vl_3b"; SFT2="sft_v2"; R2="sft_v3_r2_lingo_smoke"; R3="sft_v3_r3_lingo_smoke"


def read_rows(path:Path) -> list[dict[str,Any]]:
    if not path.exists(): return []
    with path.open("r",encoding="utf-8",newline="") as f: return list(csv.DictReader(f))


def f(row:dict[str,Any]|None,key:str) -> float:
    if not row: return 0.0
    try: return float(row.get(key,0) or 0)
    except Exception: return 0.0


def find(rows:list[dict[str,Any]], model:str, score_mode:str="answer_only") -> dict[str,Any]|None:
    return next((r for r in rows if r.get("model_name")==model and r.get("dataset")=="lingoqa" and r.get("score_mode")==score_mode), None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/final_report/stage4_sft_v3_r3_main_results.csv")
    p.add_argument("--output_json", default="outputs/final_report/stage4_sft_v3_r3_diagnosis.json")
    p.add_argument("--output_md", default="outputs/final_report/stage4_sft_v3_r3_diagnosis.md")
    args=p.parse_args()
    rows=read_rows(Path(args.input))
    out={"input":args.input,"rows":rows,"r3_present":False,"recommend_enter_dpo_v8":False,"recommend_enter_grpo_lite":False}
    yes = "\u662f"
    no = "\u5426"
    lines=["# Stage 4 SFT-v3-r3 Answer-only \u8bca\u65ad", ""]
    if not rows or not find(rows,R3):
        msg="r3 GPU training/eval has not been run yet."
        out["message"]=msg
        lines += [msg, "", "\u5f53\u524d CPU-only \u9636\u6bb5\u53ea\u51c6\u5907\u6570\u636e\u548c\u8bc4\u6d4b\u5de5\u5177\uff0c\u4e0d\u4ea7\u751f r3 \u6a21\u578b\u6027\u80fd\u7ed3\u8bba\u3002"]
    else:
        base=find(rows,BASE); sft2=find(rows,SFT2); r2=find(rows,R2); r3=find(rows,R3)
        out["r3_present"]=True
        checks={
            "normal_f1_ge_base": bool(base and f(r3,"normal_f1") >= f(base,"normal_f1")),
            "normal_f1_close_sft2": bool(sft2 and f(r3,"normal_f1") >= f(sft2,"normal_f1") - 0.03),
            "hallucination_lt_sft2_or_r2": bool((sft2 and f(r3,"control_hallucination_rate") < f(sft2,"control_hallucination_rate")) or (r2 and f(r3,"control_hallucination_rate") < f(r2,"control_hallucination_rate"))),
            "case_gap_ge_base": bool(base and f(r3,"case_gap") >= f(base,"case_gap")),
            "no_over_refusal": bool(base and f(r3,"normal_refusal_rate") <= f(base,"normal_refusal_rate") + 0.05),
            "answer_length_ok": bool(base and f(r3,"avg_answer_length") <= f(base,"avg_answer_length") * 1.5 + 5),
        }
        enter=all(checks[k] for k in ["normal_f1_ge_base","normal_f1_close_sft2","hallucination_lt_sft2_or_r2","case_gap_ge_base","no_over_refusal","answer_length_ok"])
        out.update({"checks":checks,"recommend_enter_dpo_v8":enter})
        lines.append(f"r3 answer-only Normal F1={f(r3,'normal_f1'):.4f}, Case Gap={f(r3,'case_gap'):.4f}, hallucination={f(r3,'control_hallucination_rate'):.4f}, normal refusal={f(r3,'normal_refusal_rate'):.4f}\u3002")
        if base: lines.append(f"Base answer-only Normal F1={f(base,'normal_f1'):.4f}, Case Gap={f(base,'case_gap'):.4f}, hallucination={f(base,'control_hallucination_rate'):.4f}\u3002")
        if sft2: lines.append(f"SFT-v2 answer-only Normal F1={f(sft2,'normal_f1'):.4f}, hallucination={f(sft2,'control_hallucination_rate'):.4f}\u3002")
        if r2: lines.append(f"SFT-v3-r2 answer-only Normal F1={f(r2,'normal_f1'):.4f}, hallucination={f(r2,'control_hallucination_rate'):.4f}\u3002")
        lines += ["", "## \u5224\u65ad"]
        labels={
            "normal_f1_ge_base":"r3 answer-only normal_f1 \u662f\u5426\u9ad8\u4e8e\u6216\u7b49\u4e8e Base",
            "normal_f1_close_sft2":"r3 answer-only normal_f1 \u662f\u5426\u63a5\u8fd1 SFT-v2",
            "hallucination_lt_sft2_or_r2":"r3 control hallucination \u662f\u5426\u4f4e\u4e8e SFT-v2 \u6216 r2",
            "case_gap_ge_base":"r3 case_gap \u662f\u5426\u4f18\u4e8e Base",
            "no_over_refusal":"r3 normal_refusal_rate \u662f\u5426\u672a\u5f02\u5e38\u4e0a\u5347",
            "answer_length_ok":"r3 answer length \u662f\u5426\u65e0\u660e\u663e\u5f02\u5e38",
        }
        for key,val in checks.items(): lines.append(f"- {labels[key]}\uff1a{yes if val else no}")
        lines += ["", "## \u51b3\u7b56", f"\u662f\u5426\u5efa\u8bae\u8fdb\u5165 DPO-v8\uff1a{yes if enter else no}\u3002", "\u662f\u5426\u5efa\u8bae\u8fdb\u5165 GRPO-lite\uff1a\u5426\u3002\u5f53\u524d\u9636\u6bb5\u4e0d\u5141\u8bb8\u76f4\u63a5\u8fdb\u5165 GRPO-lite\u3002"]
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"json":args.output_json,"md":args.output_md,"r3_present":out["r3_present"]},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
