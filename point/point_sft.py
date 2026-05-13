#!/usr/bin/env python3
import argparse, inspect, json, math, os, random, re, shutil, sys, time, traceback
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
BAD_STRINGS = {"", "nan", "+nan", "-nan", "none", "null", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
SYSTEM = "You are a helpful vision-language assistant. Answer grounding questions with coordinates in the range 0 to 1000 when coordinates are requested."
DUMMY_EX = {
    "dataset": "__dummy__",
    "image": [],
    "video": [],
    "conversations": [
        {"from": "system", "value": SYSTEM},
        {"from": "human", "value": "This is a synthetic fallback sample used only to keep training alive after all real samples in a batch failed."},
        {"from": "gpt", "value": "ok"},
    ],
}


def _mkdir_for(path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path, obj):
    try:
        _mkdir_for(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(jdump(obj) + "\n")
    except Exception:
        pass


def _append_text(path, text):
    try:
        _mkdir_for(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(str(text).rstrip() + "\n")
    except Exception:
        pass


def _finite_float(x):
    """Return a finite float, or None for NaN/Inf/empty/bool-like bad values."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s.lower() in BAD_STRINGS:
            return None
        x = s
    try:
        f = float(x)
    except Exception:
        return None
    return f if math.isfinite(f) else None


def _clean_text(x, max_chars=200000):
    """Normalize textual fields and reject stringified NaN/None/Inf."""
    if x is None:
        return None
    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if isinstance(x, str):
        s = x.replace("\x00", "").strip()
    else:
        try:
            s = str(x).replace("\x00", "").strip()
        except Exception:
            return None
    if s.lower() in BAD_STRINGS:
        return None
    if max_chars and len(s) > max_chars:
        s = s[:max_chars]
    return s


def _sanitize_json_value(x):
    """Recursively remove non-finite Python floats before conversion/training."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _sanitize_json_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize_json_value(v) for v in x]
    return x


def _safe_json_loads(s):
    # Python's json module accepts NaN/Infinity by default; convert them to None.
    return _sanitize_json_value(json.loads(s, parse_constant=lambda _: None))


def jdump(x):
    return json.dumps(_sanitize_json_value(x), ensure_ascii=False, allow_nan=False)


def iter_records(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _safe_json_loads(line)
                        if isinstance(obj, dict):
                            yield obj
                    except Exception:
                        continue
        except Exception:
            return
    elif p.suffix == ".json":
        try:
            obj = _safe_json_loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return
        if isinstance(obj, list):
            for x in obj:
                if isinstance(x, dict):
                    yield x
        elif isinstance(obj, dict):
            for k in ("data", "annotations", "items", "train", "samples"):
                if isinstance(obj.get(k), list):
                    for x in obj[k]:
                        if isinstance(x, dict):
                            yield x
                    return
            yield obj


def walk_strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from walk_strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk_strings(v)


def resolve_media(s, roots):
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s or s.startswith("http"):
        return None
    ps = [s]
    if s.startswith("file://"):
        ps.append(s[7:])
    for raw in ps:
        p = Path(raw)
        try:
            if p.is_absolute() and p.exists():
                return str(p)
        except Exception:
            continue
        for r in roots:
            for c in (
                Path(r) / raw,
                Path(r) / "images" / raw,
                Path(r) / "image" / raw,
                Path(r) / "videos" / raw,
                Path(r) / "video" / raw,
                Path(r) / p.name,
            ):
                try:
                    if c.exists() and c.is_file():
                        return str(c)
                except Exception:
                    pass
    return None


def fmt_points(v):
    def norm(n):
        f = _finite_float(n)
        if f is None:
            return None
        if 0 <= f <= 1:
            f *= 1000
        return int(round(max(0, min(1000, f))))

    pts = []
    if isinstance(v, dict):
        v = v.get("points") or v.get("point") or v.get("coords") or v.get("coordinate")
    if isinstance(v, (list, tuple)):
        if len(v) >= 2 and all(isinstance(a, (int, float, str)) for a in v[:2]):
            x, y = norm(v[0]), norm(v[1])
            if x is not None and y is not None:
                pts = [[x, y]]
        else:
            for p in v:
                if isinstance(p, dict):
                    p = [p.get("x"), p.get("y")]
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x, y = norm(p[0]), norm(p[1])
                    if x is not None and y is not None:
                        pts.append([x, y])
    return "<point>" + jdump(pts) + "</point>" if pts else None


def fmt_box(v):
    def norm(n):
        f = _finite_float(n)
        if f is None:
            return None
        if 0 <= f <= 1:
            f *= 1000
        return int(round(max(0, min(1000, f))))

    boxes = []
    if isinstance(v, dict):
        v = v.get("boxes") or v.get("box") or v.get("bbox") or v.get("bboxes")
    if isinstance(v, (list, tuple)):
        if len(v) >= 4 and all(isinstance(a, (int, float, str)) for a in v[:4]):
            b = [norm(a) for a in v[:4]]
            if None not in b and b[0] <= b[2] and b[1] <= b[3]:
                boxes = [b]
        else:
            for b0 in v:
                if isinstance(b0, dict):
                    b0 = [b0.get("x1"), b0.get("y1"), b0.get("x2"), b0.get("y2")]
                if isinstance(b0, (list, tuple)) and len(b0) >= 4:
                    b = [norm(a) for a in b0[:4]]
                    if None not in b and b[0] <= b[2] and b[1] <= b[3]:
                        boxes.append(b)
    return "<box>" + jdump(boxes) + "</box>" if boxes else None


def normalize(rec, roots, dataset):
    if not isinstance(rec, dict):
        return None
    rec = _sanitize_json_value(rec)
    conv = rec.get("conversations") or rec.get("messages")
    system, user, answer = SYSTEM, None, None
    if isinstance(conv, list):
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = str(m.get("from", m.get("role", ""))).lower()
            val = m.get("value", m.get("content", ""))
            if isinstance(val, list):
                val = " ".join(_clean_text(x.get("text", x) if isinstance(x, dict) else x) or "" for x in val)
            val = _clean_text(val)
            if val is None:
                continue
            if role in ("system",):
                system = val
            elif role in ("human", "user") and user is None:
                user = val
            elif role in ("gpt", "assistant"):
                answer = val
    user = _clean_text(user or rec.get("question") or rec.get("instruction") or rec.get("prompt") or rec.get("query") or rec.get("text"))
    answer = _clean_text(answer or rec.get("answer") or rec.get("output") or rec.get("response") or rec.get("chosen") or rec.get("caption"))
    if not answer:
        answer = fmt_points(rec.get("points") or rec.get("point") or rec.get("coords")) or fmt_box(rec.get("bbox") or rec.get("box") or rec.get("bboxes"))
    answer = _clean_text(answer)
    if not user or not answer:
        return None

    imgs, vids = [], []
    for s in walk_strings(rec):
        low = s.lower().split("?", 1)[0]
        if low.endswith(IMG_EXT):
            p = resolve_media(s, roots)
            if p and p not in imgs:
                imgs.append(p)
        elif low.endswith(VID_EXT):
            p = resolve_media(s, roots)
            if p and p not in vids:
                vids.append(p)
    if imgs and "<image>" not in user:
        user = (" ".join(["<image>"] * len(imgs)) + "\n" + str(user)).strip()
    if vids and "<video>" not in user:
        user = (" ".join(["<video>"] * len(vids)) + "\n" + str(user)).strip()
    return {
        "dataset": dataset,
        "image": imgs,
        "video": vids,
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human", "value": str(user).replace("Your answer should", "")},
            {"from": "gpt", "value": str(answer)},
        ],
    }


def convert(args):
    specs = ["Struct2D-Set", "pixmo-points", "ShareRobot", "RoboPoint", "EmbSpatial", "Robo2VLM-1", "robovqa", "Phys100k", "embodied_jsons"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with out.open("w", encoding="utf-8") as w:
        for name in specs:
            bases = [Path(args.data_root) / name]
            if name == "ShareRobot":
                bases.append(Path(args.data_root) / "embodied_jsons")
            roots = [str(b) for b in bases if b.exists()]
            files = []
            for b in bases:
                if b.exists():
                    try:
                        files += [p for p in b.rglob("*") if p.suffix in (".json", ".jsonl")]
                    except Exception:
                        pass
            files = sorted(set(files), key=lambda p: ("annotation" not in p.name.lower(), "qwen" not in p.name.lower(), str(p)))
            n = 0
            for f in files:
                for rec in iter_records(f):
                    try:
                        ex = normalize(rec, roots or [str(Path(args.data_root))], name)
                    except Exception:
                        ex = None
                    if ex:
                        w.write(jdump(ex) + "\n")
                        n += 1
                        if args.max_per_dataset and n >= args.max_per_dataset:
                            break
                if args.max_per_dataset and n >= args.max_per_dataset:
                    break
            counts[name] = n
    print("converted:", counts)
    print("output:", out)
    if sum(counts.values()) == 0:
        if getattr(args, "allow_dummy_data", False):
            with out.open("a", encoding="utf-8") as w:
                w.write(jdump(DUMMY_EX) + "\n")
            print("warning: no usable samples converted; wrote one dummy fallback sample so train can run")
        else:
            raise SystemExit("no usable samples converted")


def train(args):
    import torch
    from PIL import Image, ImageFile
    from torch.utils.data import Dataset
    from transformers import AutoConfig, AutoProcessor, Trainer, TrainingArguments, set_seed

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

    def _torch_empty_cache():
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _is_oom(e):
        s = (repr(e) + "\n" + str(e)).lower()
        return "out of memory" in s or "cuda error" in s and "memory" in s or "cublas_status_alloc_failed" in s

    def _log_bad_sample(err, sample=None, stage="sample"):
        obj = {"stage": stage, "err": repr(err)}
        if sample is not None:
            obj["sample"] = sample
        _append_jsonl(args.bad_samples, obj)

    def _log_bad_batch(text):
        _append_text(args.bad_batches, text)

    def _sanitize_tensor(v, fill=0.0):
        if torch.is_tensor(v) and torch.is_floating_point(v):
            try:
                if not torch.isfinite(v).all():
                    return torch.nan_to_num(v, nan=fill, posinf=fill, neginf=fill)
            except Exception:
                return torch.zeros_like(v)
        return v

    def _safe_processor_from_pretrained():
        kwargs = {"trust_remote_code": True}
        sig = inspect.signature(AutoProcessor.from_pretrained)
        # Many processor classes accept min_pixels/max_pixels through **kwargs; pass them, then retry without if rejected.
        kwargs.update({"min_pixels": args.min_pixels, "max_pixels": args.max_pixels})
        try:
            return AutoProcessor.from_pretrained(args.model_name_or_path, **kwargs)
        except TypeError:
            kwargs.pop("min_pixels", None); kwargs.pop("max_pixels", None)
            return AutoProcessor.from_pretrained(args.model_name_or_path, **kwargs)

    class D(Dataset):
        def __init__(self, path):
            self.rows = []
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    for line_no, x in enumerate(f, 1):
                        if not x.strip():
                            continue
                        try:
                            obj = _safe_json_loads(x)
                            if isinstance(obj, dict) and normalize_loaded_ex(obj):
                                self.rows.append(obj)
                            else:
                                _log_bad_sample("invalid normalized row", {"line_no": line_no, "row": obj}, "dataset-load")
                        except Exception as e:
                            _log_bad_sample(e, {"line_no": line_no, "raw": x[:2000]}, "dataset-load")
            except Exception as e:
                _log_bad_batch("failed to read data_path: " + repr(e) + "\n" + traceback.format_exc())
            if not self.rows:
                if args.allow_dummy_data:
                    self.rows = [DUMMY_EX]
                    _log_bad_batch("no usable training rows; inserted one dummy row")
                else:
                    raise RuntimeError("no usable training rows")
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            return self.rows[i]

    def normalize_loaded_ex(ex):
        try:
            cs = ex.get("conversations")
            if not isinstance(cs, list) or len(cs) < 3:
                return False
            for j in (0, 1, 2):
                if not isinstance(cs[j], dict) or _clean_text(cs[j].get("value")) is None:
                    return False
            ex.setdefault("image", [])
            ex.setdefault("video", [])
            if not isinstance(ex.get("image"), list):
                ex["image"] = []
            if not isinstance(ex.get("video"), list):
                ex["video"] = []
            ex["image"] = [p for p in ex["image"] if isinstance(p, str) and Path(p).exists()]
            ex["video"] = [p for p in ex["video"] if isinstance(p, str) and Path(p).exists()]
            return True
        except Exception:
            return False

    processor = _safe_processor_from_pretrained()
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "pad_token_id", None) is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        elif getattr(tok, "unk_token", None) is not None:
            tok.pad_token = tok.unk_token
        else:
            try:
                tok.add_special_tokens({"pad_token": "<|pad|>"})
            except Exception:
                pass
    if getattr(tok, "pad_token_id", None) is None:
        # Last resort. 0 is accepted by most tokenizers and prevents pad(None) crashes.
        try:
            tok.pad_token_id = 0
        except Exception:
            pass

    def _apply_chat_template(messages, add_generation_prompt):
        try:
            return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict) and x.get("type") == "text")
                parts.append(f"{role}: {content}")
            if add_generation_prompt:
                parts.append("assistant:")
            return "\n".join(parts)

    def msg(ex, with_answer, media=True):
        cs = ex.get("conversations", DUMMY_EX["conversations"])
        sys_msg = _clean_text(cs[0].get("value") if len(cs) > 0 and isinstance(cs[0], dict) else None) or SYSTEM
        usr = _clean_text(cs[1].get("value") if len(cs) > 1 and isinstance(cs[1], dict) else None) or DUMMY_EX["conversations"][1]["value"]
        ans = _clean_text(cs[2].get("value") if len(cs) > 2 and isinstance(cs[2], dict) else None) or "ok"
        content = []
        if media:
            for p in ex.get("image", []):
                if isinstance(p, str) and Path(p).exists():
                    content.append({"type": "image", "image": p})
            for p in ex.get("video", []):
                if isinstance(p, str) and Path(p).exists():
                    content.append({"type": "video", "video": p, "max_pixels": args.max_pixels, "fps": args.video_fps})
        content.append({"type": "text", "text": usr.replace("<image>", "").replace("<video>", "").strip()})
        m = [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}]
        if with_answer:
            m.append({"role": "assistant", "content": ans})
        return m

    def _load_images(paths):
        imgs = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB").copy())
            except Exception as e:
                _log_bad_sample(e, {"image": p}, "image-open")
        return imgs

    class Collator:
        def __init__(self):
            self.last_good = None
            self.dummy_uses = 0
        def _sanitize_tensors(self, item):
            for k, v in list(item.items()):
                item[k] = _sanitize_tensor(v)
            return item
        def _processor_encode(self, ex, media=True):
            messages_full, messages_prompt = msg(ex, True, media=media), msg(ex, False, media=media)
            text_full = _apply_chat_template(messages_full, add_generation_prompt=False)
            text_prompt = _apply_chat_template(messages_prompt, add_generation_prompt=True)
            if media and process_vision_info:
                imgs, vids = process_vision_info(messages_full)
                enc = processor(text=[text_full], images=imgs, videos=vids, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
                encp = processor(text=[text_prompt], images=imgs, videos=vids, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
            else:
                if media and ex.get("video") and not args.allow_text_fallback:
                    raise ValueError("video sample needs qwen_vl_utils")
                imgs = _load_images(ex.get("image", [])) if media else []
                enc = processor(text=[text_full], images=imgs or None, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
                encp = processor(text=[text_prompt], images=imgs or None, padding=False, truncation=True, max_length=args.model_max_length, return_tensors="pt")
            return enc, encp
        def encode(self, ex, allow_retry=True):
            try:
                enc, encp = self._processor_encode(ex, media=True)
            except Exception as e:
                _log_bad_sample(e, ex, "encode-media")
                if not args.allow_text_fallback or not allow_retry:
                    raise
                _torch_empty_cache()
                enc, encp = self._processor_encode(ex, media=False)
            item = {k: v.squeeze(0) if hasattr(v, "dim") and v.dim() > 0 and v.shape[0] == 1 else v for k, v in enc.items()}
            if "input_ids" not in item or "input_ids" not in encp:
                raise ValueError("processor output lacks input_ids")
            labels = item["input_ids"].clone()
            if labels.dim() != 1:
                labels = labels.reshape(-1)
                item["input_ids"] = item["input_ids"].reshape(-1)
                if "attention_mask" in item:
                    item["attention_mask"] = item["attention_mask"].reshape(-1)
            prompt_len = int(encp["input_ids"].reshape(-1).shape[-1])
            if prompt_len >= labels.shape[-1]:
                raise ValueError(f"no supervised tokens after truncation: prompt_len={prompt_len}, full_len={labels.shape[-1]}")
            labels[:prompt_len] = -100
            if (labels != -100).sum().item() == 0:
                raise ValueError("no supervised tokens after label masking")
            item["labels"] = labels
            return self._sanitize_tensors(item)
        def _dummy_item(self):
            try:
                return self.encode(DUMMY_EX, allow_retry=False)
            except Exception as e:
                _log_bad_batch("dummy encode failed: " + repr(e) + "\n" + traceback.format_exc())
                if self.last_good is not None:
                    return {k: v.clone() if torch.is_tensor(v) else v for k, v in self.last_good.items()}
                raise
        def __call__(self, batch):
            items = []
            for ex in batch:
                try:
                    item = self.encode(ex)
                    items.append(item)
                    self.last_good = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in item.items()}
                except Exception as e:
                    _log_bad_sample(e, ex, "encode")
                    _torch_empty_cache()
            if not items:
                if args.allow_dummy_batch:
                    self.dummy_uses += 1
                    _log_bad_batch(f"empty batch after filtering; inserted dummy fallback batch #{self.dummy_uses}")
                    items = [self._dummy_item()]
                else:
                    raise RuntimeError("empty batch after bad-sample filtering")
            maxlen = max(int(x["input_ids"].shape[-1]) for x in items)
            pad = getattr(tok, "pad_token_id", None)
            if pad is None:
                pad = 0
            out = {}
            for k in ("input_ids", "attention_mask", "labels"):
                vals = []
                for x in items:
                    if k not in x:
                        if k == "attention_mask":
                            v = torch.ones_like(x["input_ids"])
                        else:
                            raise KeyError(k)
                    else:
                        v = x[k]
                    v = v.reshape(-1)
                    fill = -100 if k == "labels" else (0 if k == "attention_mask" else pad)
                    vals.append(torch.nn.functional.pad(v, (0, maxlen - v.shape[-1]), value=fill))
                out[k] = torch.stack(vals)
            for k in items[0]:
                if k in out:
                    continue
                vals = [x[k] for x in items if k in x and torch.is_tensor(x[k])]
                if not vals:
                    continue
                try:
                    out[k] = torch.cat(vals, dim=0)
                except Exception:
                    try:
                        out[k] = torch.stack(vals)
                    except Exception as e:
                        _log_bad_batch(f"dropped non-stackable tensor key={k}: {repr(e)}")
            if (out["labels"] != -100).sum().item() == 0:
                if args.allow_dummy_batch:
                    _log_bad_batch("batch has no supervised tokens; replaced by dummy batch")
                    return self.__call__([DUMMY_EX])
                raise RuntimeError("empty batch supervision after bad-sample filtering")
            for k, v in list(out.items()):
                out[k] = _sanitize_tensor(v)
            return out

    set_seed(args.seed)
    try:
        cfg = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception as e:
        _log_bad_batch("config load failed: " + repr(e) + "\n" + traceback.format_exc())
        raise
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    def _load_model():
        kwargs = dict(config=cfg, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True)
        if args.attn_implementation:
            kwargs["attn_implementation"] = args.attn_implementation
        try:
            return AutoModel.from_pretrained(args.model_name_or_path, **kwargs)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            return AutoModel.from_pretrained(args.model_name_or_path, **kwargs)
        except Exception as e:
            if args.attn_implementation:
                _log_bad_batch("model load failed with attn_implementation; retrying without it: " + repr(e))
                kwargs.pop("attn_implementation", None)
                _torch_empty_cache()
                return AutoModel.from_pretrained(args.model_name_or_path, **kwargs)
            raise

    model = _load_model()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception as e:
            _log_bad_batch("gradient_checkpointing_enable failed: " + repr(e))
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    try:
        if hasattr(model, "resize_token_embeddings") and hasattr(tok, "__len__"):
            emb = getattr(model, "get_input_embeddings", lambda: None)()
            if emb is not None and len(tok) > emb.num_embeddings:
                model.resize_token_embeddings(len(tok))
    except Exception as e:
        _log_bad_batch("resize_token_embeddings skipped: " + repr(e))
    for n, p in model.named_parameters():
        ln = n.lower()
        if args.freeze_vision and ("visual" in ln or "vision" in ln) and "merger" not in ln:
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params:", trainable)
    if trainable == 0:
        _log_bad_batch("warning: no trainable params; training will be a no-op")

    class SafeTrainer(Trainer):
        def _log_bad_batch(self, msg):
            if self.is_world_process_zero():
                _log_bad_batch(str(msg))
        def _zero_loss(self, model):
            p = next((p for p in model.parameters() if p.requires_grad), None)
            if p is not None:
                return p.sum() * 0.0
            device = getattr(model, "device", self.args.device)
            return torch.zeros((), device=device, requires_grad=True)
        def _sanitize_params(self, model, where="params"):
            bad = 0
            with torch.no_grad():
                for p in model.parameters():
                    if p is not None and torch.is_tensor(p.data) and torch.is_floating_point(p.data):
                        try:
                            finite = torch.isfinite(p.data)
                            if not finite.all():
                                bad += int((~finite).sum().item())
                                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=0.0, neginf=0.0)
                        except Exception:
                            pass
            if bad:
                self._log_bad_batch(f"sanitized non-finite {where}: {bad}")
        def _sanitize_grads(self, model):
            bad = 0
            for p in model.parameters():
                g = p.grad
                if g is not None and torch.is_tensor(g) and torch.is_floating_point(g):
                    try:
                        finite = torch.isfinite(g)
                        if not finite.all():
                            bad += int((~finite).sum().item())
                            g.data = torch.nan_to_num(g.data, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        try:
                            p.grad = None
                        except Exception:
                            pass
            if bad:
                self._log_bad_batch(f"sanitized non-finite gradients: {bad}")
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            try:
                for k, v in list(inputs.items()):
                    inputs[k] = _sanitize_tensor(v)
                try:
                    out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
                except TypeError:
                    out = super().compute_loss(model, inputs, return_outputs=return_outputs)
                loss = out[0] if return_outputs else out
                if torch.is_tensor(loss) and not torch.isfinite(loss.detach()).all():
                    self._log_bad_batch("non-finite loss detected; replaced with zero-loss step")
                    z = self._zero_loss(model)
                    return (z, out[1] if isinstance(out, tuple) and len(out) > 1 else None) if return_outputs else z
                return out
            except Exception as e:
                self._log_bad_batch("compute_loss failed: " + repr(e) + "\n" + traceback.format_exc())
                _torch_empty_cache()
                z = self._zero_loss(model)
                return (z, None) if return_outputs else z
        def training_step(self, model, inputs, num_items_in_batch=None):
            self._sanitize_params(model, where="parameters-before-forward")
            try:
                for attempt in range(max(1, args.train_step_retries + 1)):
                    try:
                        try:
                            loss = super().training_step(model, inputs, num_items_in_batch)
                        except TypeError:
                            loss = super().training_step(model, inputs)
                        self._sanitize_grads(model)
                        if torch.is_tensor(loss) and not torch.isfinite(loss).all():
                            self._log_bad_batch("non-finite detached training loss after backward; gradients cleared")
                            model.zero_grad(set_to_none=True)
                            _torch_empty_cache()
                            return self._zero_loss(model).detach()
                        return loss
                    except Exception as e:
                        self._log_bad_batch(f"training_step attempt {attempt + 1} failed: {repr(e)}\n{traceback.format_exc()}")
                        model.zero_grad(set_to_none=True)
                        _torch_empty_cache()
                        if _is_oom(e):
                            time.sleep(1)
                        if attempt >= args.train_step_retries:
                            z = self._zero_loss(model)
                            self.accelerator.backward(z)
                            return z.detach()
            finally:
                self._sanitize_grads(model)
        def save_model(self, output_dir=None, _internal_call=False):
            output_dir = output_dir or self.args.output_dir
            try:
                return super().save_model(output_dir, _internal_call=_internal_call)
            except TypeError:
                try:
                    return super().save_model(output_dir)
                except Exception as e:
                    self._log_bad_batch("trainer.save_model failed: " + repr(e) + "\n" + traceback.format_exc())
            except Exception as e:
                self._log_bad_batch("trainer.save_model failed: " + repr(e) + "\n" + traceback.format_exc())
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                if hasattr(self.model, "save_pretrained"):
                    self.model.save_pretrained(output_dir, safe_serialization=True)
                return None
            except Exception as e:
                self._log_bad_batch("fallback model.save_pretrained failed: " + repr(e) + "\n" + traceback.format_exc())
                return None

    def make_training_args():
        base = dict(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            bf16=args.bf16,
            fp16=args.fp16,
            deepspeed=args.deepspeed or None,
            remove_unused_columns=False,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            report_to=[],
            gradient_checkpointing=True,
            logging_nan_inf_filter=True,
        )
        sig = inspect.signature(TrainingArguments.__init__).parameters
        if "eval_strategy" in sig:
            base["eval_strategy"] = "no"
        elif "evaluation_strategy" in sig:
            base["evaluation_strategy"] = "no"
        # Keep only supported names for older transformers.
        base = {k: v for k, v in base.items() if k in sig}
        return TrainingArguments(**base)

    targs = make_training_args()
    dataset = D(args.data_path)
    trainer_kwargs = dict(model=model, args=targs, train_dataset=dataset, data_collator=Collator())
    trainer_sig = inspect.signature(SafeTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = processor
    trainer = SafeTrainer(**trainer_kwargs)

    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    except Exception as e:
        _log_bad_batch("trainer.train failed; entering best-effort finalization: " + repr(e) + "\n" + traceback.format_exc())
        _torch_empty_cache()
        if not args.finish_on_train_exception:
            raise
    try:
        trainer.save_model(args.output_dir)
    except Exception as e:
        _log_bad_batch("final save_model failed: " + repr(e) + "\n" + traceback.format_exc())
    try:
        processor.save_pretrained(args.output_dir)
    except Exception as e:
        _log_bad_batch("processor.save_pretrained failed: " + repr(e) + "\n" + traceback.format_exc())
    print("finished best-effort training/finalization; check bad logs for skipped samples/batches:", args.bad_samples, args.bad_batches)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--data-root", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-per-dataset", type=int, default=0)
    c.add_argument("--allow-dummy-data", action="store_true")

    t = sub.add_parser("train")
    t.add_argument("--model-name-or-path", required=True)
    t.add_argument("--data-path", required=True)
    t.add_argument("--output-dir", required=True)
    t.add_argument("--deepspeed", default="")
    t.add_argument("--per-device-train-batch-size", type=int, default=8)
    t.add_argument("--gradient-accumulation-steps", type=int, default=4)
    t.add_argument("--learning-rate", type=float, default=5e-6)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--warmup-ratio", type=float, default=0.03)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--lr-scheduler-type", default="cosine")
    t.add_argument("--num-train-epochs", type=float, default=1)
    t.add_argument("--max-steps", type=int, default=-1)
    t.add_argument("--model-max-length", type=int, default=16384)
    t.add_argument("--min-pixels", type=int, default=50176)
    t.add_argument("--max-pixels", type=int, default=50176)
    t.add_argument("--video-fps", type=float, default=1.0)
    t.add_argument("--logging-steps", type=int, default=1)
    t.add_argument("--save-strategy", default="steps")
    t.add_argument("--save-steps", type=int, default=1000)
    t.add_argument("--save-total-limit", type=int, default=1)
    t.add_argument("--dataloader-num-workers", type=int, default=0)
    t.add_argument("--dataloader-pin-memory", action="store_true")
    t.add_argument("--bf16", action="store_true")
    t.add_argument("--fp16", action="store_true")
    t.add_argument("--bad-samples", default="/data/msz/point/bad/bad_samples.jsonl")
    t.add_argument("--bad-batches", default="/data/msz/point/bad/bad_batches.log")
    t.add_argument("--resume-from-checkpoint", default="")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--attn-implementation", default="eager")
    t.add_argument("--train-step-retries", type=int, default=1)
    t.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--allow-text-fallback", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--allow-dummy-batch", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--allow-dummy-data", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--finish-on-train-exception", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    convert(a) if a.cmd == "convert" else train(a)


if __name__ == "__main__":
    main()

