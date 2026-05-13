import os
import sys
import json
import importlib.util
from pathlib import Path

from transformers import AutoConfig


def _rank0_print(*xs):
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(*xs, flush=True)


def _import_py_file(py_file):
    py_file = Path(py_file)
    module_name = "local_qwen35_" + py_file.stem

    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scan_local_classes(model_dir):
    model_dir = Path(model_dir)

    py_files = []
    for name in [
        "configuration_qwen3_5.py",
        "modeling_qwen3_5.py",
        "configuration_qwen3.py",
        "modeling_qwen3.py",
        "configuration_qwen.py",
        "modeling_qwen.py",
    ]:
        p = model_dir / name
        if p.exists():
            py_files.append(p)

    py_files.extend([p for p in model_dir.glob("*.py") if p not in py_files])

    _rank0_print("[local-code] scanned python files:", [p.name for p in py_files])

    config_cls = None
    model_cls = None
    imported = []

    config_names = [
        "Qwen3_5Config",
        "Qwen35Config",
        "Qwen3Config",
        "QwenConfig",
    ]

    model_names = [
        "Qwen3_5ForConditionalGeneration",
        "Qwen35ForConditionalGeneration",
        "Qwen3ForConditionalGeneration",
        "QwenForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen35ForCausalLM",
        "Qwen3ForCausalLM",
        "QwenForCausalLM",
    ]

    for py in py_files:
        try:
            mod = _import_py_file(py)
            if mod is None:
                continue
            imported.append(py.name)

            for n in config_names:
                if hasattr(mod, n):
                    config_cls = getattr(mod, n)
                    _rank0_print(f"[local-code] found config class {n} in {py.name}")

            for n in model_names:
                if hasattr(mod, n):
                    model_cls = getattr(mod, n)
                    _rank0_print(f"[local-code] found model class {n} in {py.name}")

        except Exception as e:
            _rank0_print(f"[local-code] failed importing {py.name}: {repr(e)}")

    return config_cls, model_cls, imported


def load_qwen35_local_or_raise(model_dir, torch_dtype=None):
    """
    优先使用模型目录自带的本地 remote code 加载 Qwen3.5。

    成功条件：
      - 模型目录里存在 modeling_*.py / configuration_*.py
      - 能找到 Qwen3_5ForConditionalGeneration 或近似命名的模型类

    返回：
      model, config

    如果本地代码不存在或类不存在，直接 raise，让上层明确知道不能继续硬映射 Qwen2。
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")

    raw_cfg_path = model_dir / "config.json"
    raw_cfg = {}
    if raw_cfg_path.exists():
        raw_cfg = json.loads(raw_cfg_path.read_text(encoding="utf-8"))
        _rank0_print("[local-code] raw model_type =", raw_cfg.get("model_type"))
        _rank0_print("[local-code] raw architectures =", raw_cfg.get("architectures"))
        _rank0_print("[local-code] raw auto_map =", raw_cfg.get("auto_map"))

    # 把模型目录放到 sys.path 最前，支持 modeling 文件中的相对/同目录 import。
    sys.path.insert(0, str(model_dir))

    config_cls, model_cls, imported = _scan_local_classes(model_dir)

    if model_cls is None:
        raise RuntimeError(
            "No local Qwen3.5 model class found in model directory. "
            f"Imported files={imported}. "
            "Expected one of: Qwen3_5ForConditionalGeneration, "
            "Qwen3ForConditionalGeneration, Qwen3_5ForCausalLM, Qwen3ForCausalLM. "
            "This means the model directory probably does not contain usable remote code."
        )

    if config_cls is not None:
        _rank0_print("[local-code] loading config via local config class")
        config = config_cls.from_pretrained(str(model_dir), trust_remote_code=True)
    else:
        _rank0_print("[local-code] no local config class found; trying AutoConfig trust_remote_code=True")
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)

    # 关键：保留 Qwen3.5 原生模型类，不再映射到 Qwen2。
    if hasattr(config, "use_cache"):
        config.use_cache = False

    _rank0_print("[local-code] loading model via local model class:", model_cls)

    kwargs = {
        "config": config,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    try:
        model = model_cls.from_pretrained(str(model_dir), **kwargs)
    except TypeError:
        # 某些本地模型类不接受 low_cpu_mem_usage。
        kwargs.pop("low_cpu_mem_usage", None)
        model = model_cls.from_pretrained(str(model_dir), **kwargs)

    if hasattr(model, "config"):
        model.config.use_cache = False

    return model, config
