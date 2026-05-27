"""Train a small Qwen2.5-VL-3B LoRA smoke checkpoint on SFT-v3 data.

Default dry-run writes config/stat files and does not load the model.
"""
from __future__ import annotations
import argparse, json, math, random, shlex, sys
from collections import Counter
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def read_jsonl(path: Path, limit: int=0):
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            if limit and len(rows)>=limit: break
    return rows

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def write_config(path: Path, args):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(f"{k}: {v}" for k,v in sorted(vars(args).items()))+"\n",encoding="utf-8")

def dataset_stats(rows):
    return {"count":len(rows),"by_dataset":dict(Counter(str(r.get("dataset","unknown")) for r in rows)),"by_sample_type":dict(Counter(str(r.get("sample_type","unknown")) for r in rows)),"by_setting":dict(Counter(str((r.get("metadata") or {}).get("setting","unknown")) for r in rows))}

def image_paths(row,max_images):
    paths=[]
    for item in row.get("image_paths") or []:
        p=Path(str(item))
        if p.exists(): paths.append(p.as_posix())
        if max_images and len(paths)>=max_images: break
    return paths

def assistant_content(row):
    obj=row.get("assistant") if isinstance(row.get("assistant"),dict) else {}
    out={"answer":str(obj.get("answer",""))}
    if "reason" in obj:
        out["reason"]=str(obj.get("reason",""))
    return json.dumps(out,ensure_ascii=False,separators=(",",":"))

def build_messages(row, include_assistant, args):
    content=[]
    for path in image_paths(row,args.max_images):
        item={"type":"image","image":path}
        if args.max_pixels>0: item["max_pixels"]=args.max_pixels
        content.append(item)
    content.append({"type":"text","text":str(row.get("prompt") or "")})
    messages=[{"role":"user","content":content}]
    if include_assistant: messages.append({"role":"assistant","content":assistant_content(row)})
    return messages

def encode_sample(processor,row,device,args):
    from qwen_vl_utils import process_vision_info
    prompt_messages=build_messages(row,False,args); full_messages=build_messages(row,True,args)
    prompt_text=processor.apply_chat_template(prompt_messages,tokenize=False,add_generation_prompt=True)
    full_text=processor.apply_chat_template(full_messages,tokenize=False,add_generation_prompt=False)
    image_inputs,video_inputs=process_vision_info(full_messages)
    prompt_inputs=processor(text=[prompt_text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
    full_inputs=processor(text=[full_text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt")
    labels=full_inputs["input_ids"].clone(); labels[:,:prompt_inputs["input_ids"].shape[1]]=-100; labels[full_inputs["attention_mask"]==0]=-100; full_inputs["labels"]=labels
    return {k:v.to(device) if hasattr(v,"to") else v for k,v in full_inputs.items()}

def train(args, rows):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    device="cuda" if torch.cuda.is_available() else "cpu"
    dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16 if torch.cuda.is_available() else torch.float32
    qconf=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16) if args.qlora else None
    processor=AutoProcessor.from_pretrained(args.model_path,trust_remote_code=True,use_fast=args.use_fast_processor)
    kwargs={"trust_remote_code":True,"dtype":dtype,"device_map":"auto" if args.qlora else None}
    if qconf is not None: kwargs["quantization_config"]=qconf
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path,**kwargs)
    if not args.qlora: model=model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache=False
    if args.qlora: model=prepare_model_for_kbit_training(model)
    if args.init_adapter:
        adapter_path=Path(args.init_adapter)
        if not (adapter_path/"adapter_config.json").exists():
            raise FileNotFoundError(f"init_adapter adapter_config.json not found: {adapter_path}")
        model=PeftModel.from_pretrained(model, adapter_path.as_posix(), is_trainable=True)
    else:
        model=get_peft_model(model,LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters(); model.train(); opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.learning_rate)
    log_path=Path(args.log_dir)/"loss_log.jsonl"; log_path.parent.mkdir(parents=True,exist_ok=True)
    step=0; opt.zero_grad(set_to_none=True)
    epochs=max(1, math.ceil(args.max_steps*args.gradient_accumulation_steps/max(1,len(rows)))) if args.max_steps else args.epochs
    with log_path.open("w",encoding="utf-8") as log_f:
        for epoch in range(epochs):
            for idx,row in enumerate(rows):
                batch=encode_sample(processor,row,device,args); outputs=model(**batch); raw=float(outputs.loss.detach().cpu())
                if not math.isfinite(raw):
                    raise FloatingPointError(f"NaN/Inf loss at epoch={epoch+1} idx={idx+1}: {raw}")
                (outputs.loss/args.gradient_accumulation_steps).backward()
                if (idx+1)%args.gradient_accumulation_steps==0 or idx+1==len(rows):
                    opt.step(); opt.zero_grad(set_to_none=True); step+=1
                    rec={"epoch":epoch+1,"step":step,"loss":raw}; log_f.write(json.dumps(rec,ensure_ascii=False)+"\n"); log_f.flush(); print(json.dumps(rec),flush=True)
                    if args.save_steps and step%args.save_steps==0:
                        ckpt=Path(args.output_dir)/f"checkpoint-step-{step:06d}"; ckpt.mkdir(parents=True,exist_ok=True); model.save_pretrained(ckpt); processor.save_pretrained(ckpt)
                    if args.max_steps and step>=args.max_steps:
                        model.save_pretrained(args.output_dir); processor.save_pretrained(args.output_dir); return {"success": True, "final_loss": raw, "steps": step}
    model.save_pretrained(args.output_dir); processor.save_pretrained(args.output_dir); return {"success": True, "final_loss": raw if 'raw' in locals() else None, "steps": step}

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=""); p.add_argument("--model_path",default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"); p.add_argument("--train_file",default="data/train/sft_v3/lingoqa_sft_v3.jsonl"); p.add_argument("--init_adapter",default=""); p.add_argument("--output_dir",default="checkpoints/qwen25vl_lora_sft_v3_lingo_smoke"); p.add_argument("--log_dir",default="outputs/train_logs/sft_v3_lingo_smoke"); p.add_argument("--train_samples",type=int,default=1000); p.add_argument("--eval_samples",type=int,default=100); p.add_argument("--epochs",type=int,default=1); p.add_argument("--max_steps",type=int,default=100); p.add_argument("--save_steps",type=int,default=100); p.add_argument("--eval_steps",type=int,default=50); p.add_argument("--learning_rate",type=float,default=1e-5); p.add_argument("--gradient_accumulation_steps",type=int,default=8); p.add_argument("--lora_r",type=int,default=16); p.add_argument("--lora_alpha",type=int,default=32); p.add_argument("--lora_dropout",type=float,default=0.05); p.add_argument("--max_pixels",type=int,default=200704); p.add_argument("--max_images",type=int,default=3); p.add_argument("--bf16",action="store_true"); p.add_argument("--qlora",action="store_true"); p.add_argument("--gradient_checkpointing",action="store_true",default=True); p.add_argument("--use_fast_processor",action="store_true"); p.add_argument("--seed",type=int,default=42); p.add_argument("--dry_run",action="store_true"); args=p.parse_args(); apply_config(args); return args

def parse_scalar(value: str):
    value=value.strip()
    if value.lower() in {"true","yes"}: return True
    if value.lower() in {"false","no"}: return False
    try:
        if any(ch in value for ch in [".","e","E"]): return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")

def load_simple_yaml(path: Path) -> dict[str, Any]:
    data={}
    if not path.exists(): return data
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped=line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped: continue
        key,value=stripped.split(":",1)
        data[key.strip()]=parse_scalar(value)
    return data

def apply_config(args):
    if not args.config: return
    provided=set()
    for item in sys.argv[1:]:
        if item.startswith("--"):
            provided.add(item[2:].replace("-","_"))
    for key,value in load_simple_yaml(Path(args.config)).items():
        if hasattr(args,key) and key not in provided:
            setattr(args,key,value)

def main():
    args=parse_args(); rows=read_jsonl(Path(args.train_file),args.train_samples); random.Random(args.seed).shuffle(rows)
    Path(args.output_dir).mkdir(parents=True,exist_ok=True); Path(args.log_dir).mkdir(parents=True,exist_ok=True)
    write_config(Path(args.log_dir)/"train_config.yaml",args); write_json(Path(args.log_dir)/"dataset_stats.json",dataset_stats(rows)); (Path(args.log_dir)/"command.txt").write_text(" ".join(shlex.quote(x) for x in sys.argv)+"\n",encoding="utf-8")
    if args.dry_run:
        (Path(args.log_dir)/"loss_log.jsonl").write_text("",encoding="utf-8"); summary={"success": True, "dry_run":True,"rows":len(rows),"init_adapter":args.init_adapter,"output_dir":args.output_dir,"log_dir":args.log_dir,"max_steps":args.max_steps,"oom":False,"nan_detected":False}; write_json(Path(args.log_dir)/"train_summary.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2)); return
    summary={"success": False, "dry_run": False, "init_adapter": args.init_adapter, "output_dir": args.output_dir, "max_steps": args.max_steps, "final_loss": None, "oom": False, "nan_detected": False}
    try:
        result=train(args,rows); summary.update(result or {})
    except FloatingPointError as exc:
        summary["nan_detected"]=True; summary["error"]=str(exc); write_json(Path(args.log_dir)/"train_summary.json",summary); raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            summary["oom"]=True; summary["error"]=str(exc); summary["suggestion"]="lower max_pixels, batch size, or max sequence length"
            write_json(Path(args.log_dir)/"train_summary.json",summary); raise
        summary["error"]=str(exc); write_json(Path(args.log_dir)/"train_summary.json",summary); raise
    except Exception as exc:
        summary["error"]=str(exc); write_json(Path(args.log_dir)/"train_summary.json",summary); raise
    write_json(Path(args.log_dir)/"train_summary.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
