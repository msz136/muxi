#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/msz/point
DATA_ROOT=/data/msz/dataset
MODEL=/data/msz/models/Qwen3-VL-8B-Instruct
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$ROOT"/{data,configs,outputs,logs,bad} "$LOG_DIR"
cd "$ROOT"

exec > >(tee -a "$LOG_DIR/onekey.log") 2>&1

echo "[1/6] preflight"
date
hostname
uname -a
command -v mx-smi && mx-smi || true
df -h /data/msz /tmp || true
free -h || true

python - <<'PY'
import os, torch, transformers
print("python ok")
print("torch =", torch.__version__)
print("transformers =", transformers.__version__)
print("cuda/device_count =", torch.cuda.is_available(), torch.cuda.device_count())
try:
    import deepspeed
    print("deepspeed =", deepspeed.__version__)
except Exception as e:
    print("deepspeed import failed:", repr(e))
PY

test -f "$MODEL/config.json" || { echo "missing model: $MODEL/config.json"; exit 1; }
test -d "$DATA_ROOT" || { echo "missing data root: $DATA_ROOT"; exit 1; }

cat >"$ROOT/configs/ds_zero2_point.json" <<'JSON'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "allgather_partitions": true,
    "allgather_bucket_size": 500000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "contiguous_gradients": true
  },
  "bf16": {"enabled": true},
  "fp16": {"enabled": false},
  "zero_allow_untested_optimizer": true,
  "wall_clock_breakdown": false
}
JSON

cat >"$ROOT/point_sft.py" <<'PY'
#!/usr/bin/env python3
import argparse, json, os, random, re, traceback
from pathlib import Path

IMG_EXT = (".jpg",".jpeg",".png",".bmp",".webp")
VID_EXT = (".mp4",".avi",".mov",".mkv",".webm")
SYSTEM = "You are a helpful vision-language assistant. Answer grounding questions with coordinates in the range 0 to 1000 when coordinates are requested."

def jdump(x): return json.dumps(x, ensure_ascii=False)

def iter_records(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line=line.strip()
                if line:
                    try: yield json.loads(line)
                    except Exception: pass
    elif p.suffix == ".json":
        try:
            obj=json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        if isinstance(obj, list):
            for x in obj:
                if isinstance(x, dict): yield x
        elif isinstance(obj, dict):
            for k in ("data","annotations","items","train","samples"):
                if isinstance(obj.get(k), list):
                    for x in obj[k]:
                        if isinstance(x, dict): yield x
                    return
            yield obj

def walk_strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values(): yield from walk_strings(v)
    elif isinstance(x, list):
        for v in x: yield from walk_strings(v)

def resolve_media(s, roots):
    if not isinstance(s, str): return None
    s=s.strip()
    if not s or s.startswith("http"): return None
    ps = [s]
    if s.startswith("file://"): ps.append(s[7:])
    for raw in ps:
        p=Path(raw)
        if p.is_absolute() and p.exists(): return str(p)
        for r in roots:
            for c in (Path(r)/raw, Path(r)/"images"/raw, Path(r)/"image"/raw, Path(r)/"videos"/raw, Path(r)/"video"/raw, Path(r)/p.name):
                if c.exists(): return str(c)
    return None

def fmt_points(v):
    def norm(n):
        try:
            f=float(n)
            if 0 <= f <= 1: f *= 1000
            return int(round(max(0, min(1000, f))))
        except Exception: return None
    pts=[]
    if isinstance(v, dict):
        v = v.get("points") or v.get("point") or v.get("coords") or v.get("coordinate")
    if isinstance(v, (list, tuple)):
        if len(v)>=2 and all(isinstance(a,(int,float,str)) for a in v[:2]):
            pts=[[norm(v[0]), norm(v[1])]]
        else:
            for p in v:
                if isinstance(p, dict):
                    p=[p.get("x"), p.get("y")]
                if isinstance(p,(list,tuple)) and len(p)>=2:
                    x,y=norm(p[0]),norm(p[1])
                    if x is not None and y is not None: pts.append([x,y])
    return "<point>"+jdump(pts)+"</point>" if pts else None

def fmt_box(v):
    def norm(n):
        try:
            f=float(n)
            if 0 <= f <= 1: f *= 1000
            return int(round(max(0, min(1000, f))))
        except Exception: return None
    boxes=[]
    if isinstance(v, dict):
        v = v.get("boxes") or v.get("box") or v.get("bbox") or v.get("bboxes")
    if isinstance(v,(list,tuple)):
        if len(v)>=4 and all(isinstance(a,(int,float,str)) for a in v[:4]):
            b=[norm(a) for a in v[:4]]
            if None not in b: boxes=[b]
        else:
            for b0 in v:
                if isinstance(b0, dict):
                    b0=[b0.get("x1"),b0.get("y1"),b0.get("x2"),b0.get("y2")]
                if isinstance(b0,(list,tuple)) and len(b0)>=4:
                    b=[norm(a) for a in b0[:4]]
                    if None not in b: boxes.append(b)
    return "<box>"+jdump(boxes)+"</box>" if boxes else None

def normalize(rec, roots, dataset):
    conv = rec.get("conversations") or rec.get("messages")
    system, user, answer = SYSTEM, None, None
    if isinstance(conv, list):
        for m in conv:
            role = str(m.get("from", m.get("role",""))).lower()
            val = m.get("value", m.get("content",""))
            if isinstance(val, list):
                val = " ".join(str(x.get("text", x)) for x in val)
            if role in ("system",): system = str(val)
            elif role in ("human","user") and user is None: user = str(val)
            elif role in ("gpt","assistant"): answer = str(val)
    user = user or rec.get("question") or rec.get("instruction") or rec.get("prompt") or rec.get("query") or rec.get("text")
    answer = answer or rec.get("answer") or rec.get("output") or rec.get("response") or rec.get("chosen") or rec.get("caption")
    if not answer:
        answer = fmt_points(rec.get("points") or rec.get("point") or rec.get("coords")) or fmt_box(rec.get("bbox") or rec.get("box") or rec.get("bboxes"))
    if not user or not answer: return None

    imgs, vids = [], []
    for s in walk_strings(rec):
        low=s.lower()
        if low.endswith(IMG_EXT):
            p=resolve_media(s, roots)
            if p and p not in imgs: imgs.append(p)
        elif low.endswith(VID_EXT):
            p=resolve_media(s, roots)
            if p and p not in vids: vids.append(p)
    if imgs and "<image>" not in user:
        user = (" ".join(["<image>"] * len(imgs)) + "\n" + str(user)).strip()
    if vids and "<video>" not in user:
        user = (" ".join(["<video>"] * len(vids)) + "\n" + str(user)).strip()
    return {"dataset": dataset, "image": imgs, "video": vids, "conversations": [
        {"from":"system","value":system},
        {"from":"human","value":str(user).replace("Your answer should","")},
        {"from":"gpt","value":str(answer)}
    ]}

def convert(args):
    specs = ["Struct2D-Set","pixmo-points","ShareRobot","RoboPoint","EmbSpatial","Robo2VLM-1","robovqa","Phys100k","embodied_jsons"]
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    counts={}
    with out.open("w", encoding="utf-8") as w:
        for name in specs:
            bases=[Path(args.data_root)/name]
            if name == "ShareRobot": bases.append(Path(args.data_root)/"embodied_jsons")
            roots=[str(b) for b in bases if b.exists()]
            files=[]
            for b in bases:
                if b.exists():
                    files += [p for p in b.rglob("*") if p.suffix in (".json",".jsonl")]
            files=sorted(set(files), key=lambda p: ("annotation" not in p.name.lower(), "qwen" not in p.name.lower(), str(p)))
            n=0
            for f in files:
                for rec in iter_records(f):
                    ex=normalize(rec, roots or [str(Path(args.data_root))], name)
                    if ex:
                        w.write(jdump(ex)+"\n"); n+=1
                        if args.max_per_dataset and n >= args.max_per_dataset: break
                if args.max_per_dataset and n >= args.max_per_dataset: break
            counts[name]=n
    print("converted:", counts)
    print("output:", out)
    if sum(counts.values()) == 0:
        raise SystemExit("no usable samples converted")

def train(args):
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import AutoConfig, AutoProcessor, Trainer, TrainingArguments, set_seed
    try:
        from transformers import AutoModelForImageTextToText as AutoModel
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq as AutoModel
        except Exception:
            from transformers import AutoModelForCausalLM as AutoModel
    try:
        from qwen_vl_utils import process_vision_info
    except Exception:
        process_vision_info = None

    class D(Dataset):
        def __init__(self, path):
            self.rows=[json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        def __len__(self): return len(self.rows)
        def __getitem__(self, i): return self.rows[i]

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token

    def msg(ex, with_answer):
        cs=ex["conversations"]; sys=cs[0]["value"]; usr=cs[1]["value"]; ans=cs[2]["value"]
        content=[]
        for p in ex.get("image",[]): content.append({"type":"image","image":p})
        for p in ex.get("video",[]): content.append({"type":"video","video":p, "max_pixels":args.max_pixels, "fps":1.0})
        content.append({"type":"text","text":usr.replace("<image>","").replace("<video>","").strip()})
        m=[{"role":"system","content":sys},{"role":"user","content":content}]
        if with_answer: m.append({"role":"assistant","content":ans})
        return m

    class Collator:
        def __init__(self): self.bad=open(args.bad_samples, "a", encoding="utf-8")
        def encode(self, ex):
            messages_full, messages_prompt = msg(ex, True), msg(ex, False)
            text_full = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
            text_prompt = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
            if process_vision_info:
                imgs, vids = process_vision_info(messages_full)
                enc = processor(text=[text_full], images=imgs, videos=vids, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
                encp = processor(text=[text_prompt], images=imgs, videos=vids, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
            else:
                if ex.get("video"): raise ValueError("video sample needs qwen_vl_utils")
                imgs=[Image.open(p).convert("RGB") for p in ex.get("image",[])]
                enc = processor(text=[text_full], images=imgs or None, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
                encp = processor(text=[text_prompt], images=imgs or None, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
            item={k:v.squeeze(0) if hasattr(v, "dim") and v.dim()>0 and v.shape[0]==1 else v for k,v in enc.items()}
            labels=item["input_ids"].clone()
            labels[:encp["input_ids"].shape[-1]]=-100
            item["labels"]=labels
            return item
        def __call__(self, batch):
            items=[]
            for ex in batch:
                try: items.append(self.encode(ex))
                except Exception as e:
                    self.bad.write(jdump({"err":repr(e), "sample":ex})+"\n"); self.bad.flush()
            if not items: raise RuntimeError("empty batch after bad-sample filtering")
            maxlen=max(x["input_ids"].shape[-1] for x in items)
            pad=tok.pad_token_id
            out={}
            for k in ("input_ids","attention_mask","labels"):
                vals=[]
                for x in items:
                    v=x[k]; fill=-100 if k=="labels" else (0 if k=="attention_mask" else pad)
                    vals.append(torch.nn.functional.pad(v, (0,maxlen-v.shape[-1]), value=fill))
                out[k]=torch.stack(vals)
            for k in items[0]:
                if k in out: continue
                vals=[x[k] for x in items if k in x]
                if vals and torch.is_tensor(vals[0]):
                    out[k]=torch.cat(vals, dim=0)
            return out

    set_seed(42)
    cfg=AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(cfg, "use_cache"): cfg.use_cache=False
    model=AutoModel.from_pretrained(args.model_name_or_path, config=cfg, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True, attn_implementation="eager")
    if hasattr(model.config, "use_cache"): model.config.use_cache=False
    if hasattr(model, "gradient_checkpointing_enable"): model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()
    for n,p in model.named_parameters():
        ln=n.lower()
        if ("visual" in ln or "vision" in ln) and "merger" not in ln:
            p.requires_grad=False
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    class SafeTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            except Exception as e:
                if self.is_world_process_zero():
                    with open(args.bad_batches, "a", encoding="utf-8") as f:
                        f.write(repr(e)+"\n"+traceback.format_exc()+"\n")
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                z=next(p for p in model.parameters() if p.requires_grad).sum()*0.0
                self.accelerator.backward(z)
                return z.detach()

    targs=TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_ratio=0.03, weight_decay=0.0, max_grad_norm=1.0, lr_scheduler_type="cosine",
        logging_steps=args.logging_steps, save_strategy=args.save_strategy, save_steps=1000, save_total_limit=1,
        bf16=args.bf16, deepspeed=args.deepspeed, remove_unused_columns=False, dataloader_num_workers=args.dataloader_num_workers,
        report_to=[], gradient_checkpointing=True, eval_strategy="no"
    )
    trainer=SafeTrainer(model=model, args=targs, train_dataset=D(args.data_path), data_collator=Collator(), processing_class=processor)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd", required=True)
    c=sub.add_parser("convert"); c.add_argument("--data-root", required=True); c.add_argument("--out", required=True); c.add_argument("--max-per-dataset", type=int, default=0)
    t=sub.add_parser("train")
    t.add_argument("--model-name-or-path", required=True); t.add_argument("--data-path", required=True); t.add_argument("--output-dir", required=True)
    t.add_argument("--deepspeed", required=True); t.add_argument("--per-device-train-batch-size", type=int, default=8); t.add_argument("--gradient-accumulation-steps", type=int, default=4)
    t.add_argument("--learning-rate", type=float, default=5e-6); t.add_argument("--num-train-epochs", type=float, default=1); t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--model-max-length", type=int, default=16384); t.add_argument("--min-pixels", type=int, default=50176); t.add_argument("--max-pixels", type=int, default=50176)
    t.add_argument("--logging-steps", type=int, default=1); t.add_argument("--save-strategy", default="steps"); t.add_argument("--dataloader-num-workers", type=int, default=4)
    t.add_argument("--bf16", action="store_true"); t.add_argument("--bad-samples", default="/data/msz/point/bad/bad_samples.jsonl"); t.add_argument("--bad-batches", default="/data/msz/point/bad/bad_batches.log")
    t.add_argument("--resume-from-checkpoint", default="")
    a=p.parse_args()
    convert(a) if a.cmd=="convert" else train(a)
if __name__ == "__main__": main()
PY

chmod +x "$ROOT/point_sft.py"

echo "[2/6] convert raw datasets"
python "$ROOT/point_sft.py" convert \
  --data-root "$DATA_ROOT" \
  --out "$ROOT/data/grounding_sft.jsonl"

echo "[3/6] batch probing"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHOSEN_BS=""
for BS in 8 6 4 2 1; do
  echo "probe per_device_train_batch_size=$BS"
  if deepspeed --num_gpus=8 "$ROOT/point_sft.py" train \
    --model-name-or-path "$MODEL" \
    --data-path "$ROOT/data/grounding_sft.jsonl" \
    --output-dir "$ROOT/outputs/probe_bs${BS}_${RUN_ID}" \
    --deepspeed "$ROOT/configs/ds_zero2_point.json" \
    --per-device-train-batch-size "$BS" \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --max-steps 2 \
    --model-max-length 16384 \
    --min-pixels 50176 --max-pixels 50176 \
    --logging-steps 1 \
    --save-strategy no \
    --dataloader-num-workers 4 \
    --bf16; then
      CHOSEN_BS="$BS"; break
  fi
done
test -n "$CHOSEN_BS" || { echo "all batch probes failed"; exit 1; }
echo "$CHOSEN_BS" > "$ROOT/configs/chosen_batch_size.txt"

echo "[4/6] launch full SFT, chosen batch=$CHOSEN_BS"
OUT="$ROOT/outputs/qwen3vl8b_grounding_expert_${RUN_ID}"
for TRY in 1 2 3; do
  echo "train attempt $TRY/3"
  if deepspeed --num_gpus=8 "$ROOT/point_sft.py" train \
    --model-name-or-path "$MODEL" \
    --data-path "$ROOT/data/grounding_sft.jsonl" \
    --output-dir "$OUT" \
    --deepspeed "$ROOT/configs/ds_zero2_point.json" \
    --per-device-train-batch-size "$CHOSEN_BS" \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --num-train-epochs 1 \
    --model-max-length 16384 \
    --min-pixels 50176 --max-pixels 50176 \
    --logging-steps 1 \
    --save-strategy steps \
    --dataloader-num-workers 4 \
    --bf16; then
      echo "[5/6] done: $OUT"
      exit 0
  fi
  sleep $((TRY * 60))
done

echo "[6/6] failed after retries; logs: $LOG_DIR"
exit 1
