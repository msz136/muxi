# AGENTS.md — /data/msz (muxi/sft)

MACA C500 8-GPU vision-language SFT training for Qwen3-VL-8B-Instruct grounding model.
## Directory map

| Dir | Purpose |
|-----|---------|
| `point/` | **Active project**: SFT training, data conversion, chunked training |
| `ds_test/` | DeepSpeed method smoke tests (SFT/GRPO/DPO/OPD) |
| `dataset/` | Raw data: RoboPoint, PixMo-Points, ShareRobot, EmbSpatial, Robo2VLM-1, etc. |
| `models/` | Checkpoints: Qwen3-VL-8B-Instruct, Qwen3.5-4B/9B, qwen3-vl-32b |
| `models/venv/` | Python 3.12 venv |
| `metax_pkgs/` | MACA-adapted packages inventory + `pip-freeze.txt` |
| `point/data_expert/` | Converted JSONL mixes (v1/v2/v3, up to 1.5M rows) |
| `point/outputs/` | Training outputs (gitignored) |
| `point/bad/` | Bad-sample batch logs (gitignored) |
| `point/logs/` | Training run logs (gitignored by default) |
| `point/configs/` | DeepSpeed ZeRO-2 config |
| `opd_project/` | OPD/multi-teacher experiments and semantic-nav box-grounding data |
| `opd_project/data/semantic_nav_box_v1/` | Generated semantic navigation `<box>` grounding datasets |

## Entry points

- **Training**: `point/expert_sft.py {smoke,train}` — OOM-safe, NaN-safe Qwen3-VL training
- **Data prep**: `point/point_data_only.py` — converts raw datasets to unified `<point>` JSONL
- **Mix builder**: `point/build_mixes.py` — oversampling ratios for grounding data
- **Chunked training**: `point/run_chunked_sft.sh` — 15K-row chunks, 1 epoch each, sequential weight carry-forward
- **Semantic-nav box data**: `opd_project/scripts/build_semantic_nav_box_grounding.py` — builds RoboPoint point→box manifests
- **Remote crop labeling**: `opd_project/scripts/local_label_semantic_nav_remote_base64.py` — labels boxes with Qwen-122B without storing source images locally

## Environment (MACA platform)

```bash
# Required before any deepspeed launch:
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MACA NCCL workarounds (chunked training sets these):
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600
```

Key versions (from `metax_pkgs/pip-freeze.txt`):
- `torch==2.6.0+metax3.5.3.9`, `deepspeed==0.16.5+maca3.5.3.15`, `transformers>=4.51`

GPU monitor: `mx-smi` (not nvidia-smi). Python: `/opt/conda/bin/python3`.

## Quick start

```bash
# Smoke test (50 batches, auto batch-size probe):
cd point
deepspeed --num_gpus=8 expert_sft.py smoke \
  --data-path data_expert/expert_smoke_v1_local.jsonl \
  --output-dir outputs/smoke_test --smoke-batches 50 \
  --per-device-train-batch-size 8 --bf16

# Full SFT:
deepspeed --num_gpus=8 expert_sft.py train \
  --data-path data_expert/expert_mix_v1_shuffled.jsonl \
  --output-dir outputs/my_run --save-steps 600 --bf16
```

## Data format

JSONL, one dict per line:
```json
{"image": ["/path/to/img.jpg"], "video": [],
 "conversations": [
   {"from": "system", "value": "You are a helpful..."},
   {"from": "human", "value": "<image>\nPoint to the object."},
   {"from": "gpt", "value": "<point>[[450,320]]</point>"}
 ]}
```

Image paths: absolute or relative to dataset root. Missing images → collator skips sample.

### Semantic navigation box-grounding format

For semantic navigation, the target is a rectangle, not a point. The model is
given target object information (`object_name`, `relation`, `anchor_object`) and
must return only a `<box>` tag with normalized 0-1000 coordinates:

```json
{"dataset":"semantic_nav_box_grounding_full_object_ref_v1",
 "image":["/data/msz/dataset/RoboPoint/images/object_ref/example.png"],
 "video":[],
 "target":{
   "object_name":"white rectangular object",
   "relation":"in front of the highlighted object",
   "anchor_object":"highlighted object",
   "attributes":["white","rectangular","standing"],
   "box":[[257,659],[437,929]]
 },
 "conversations":[
   {"from":"system","value":"You are a semantic navigation grounding assistant..."},
   {"from":"human","value":"<image>\nTarget object information:\n{\"object_name\":\"white rectangular object\",\"relation\":\"in front of the highlighted object\",\"anchor_object\":\"highlighted object\"}\nReturn only the bounding box as <box>[[x1,y1],[x2,y2]]</box>."},
   {"from":"gpt","value":"<box>[[257,659],[437,929]]</box>"}
 ],
 "metadata":{"task_type":"box_grounding","prompt_mode":"obj_relation"}}
```

The generated training set also includes `obj_only` and `relation_only` prompt
variants for the same box.

## Semantic-nav box-grounding data (May 2026)

**Goal**: train the model for scenes where the input is an image plus target
object information such as `{object_name, relation, anchor_object}`, and the
output is the corresponding rectangular target region.

**Generation source**: RoboPoint `object_ref` samples from
`/data/msz/opd_project/data/prompt_pool_clean.jsonl`.

**Conversion**:
1. Keep `robopoint` rows with `object_ref`, relation in
   `on,left,right,inside,beside,front,behind,between`, and enough `gt_points`.
2. Convert each point cloud to a box using min/max x/y plus margin.
3. Crop the synthesized box in memory.
4. Ask Qwen-122B to label the cropped target region with `object_name`,
   `attributes`, `region_type`, and `confidence`.
5. Expand each accepted base row into three prompts: `obj_only`,
   `relation_only`, and `obj_relation`.

**Important**: source images are not downloaded to local disk for full labeling.
The final full run used a temporary read-only HTTP server on the remote machine
plus an SSH tunnel; images were read into memory, cropped with PIL, and only the
small crop was sent to Qwen-122B as base64. The temporary HTTP server and tunnel
were closed after generation.

**Final full object-ref dataset**:

| File | Rows | Notes |
|------|-----:|-------|
| `/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl` | 41,637 | Recommended training file |
| `/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_all.jsonl` | 41,655 | Includes 6 weak base labels expanded to 18 rows |
| `/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_base_annotations.jsonl` | 13,885 | One Qwen-122B label per base box |
| `/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_summary.json` | 1 | Dataset statistics |
| `/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_preview.png` | 1 | Visual QA sheet |

Summary:
- Base boxes: 13,885
- High-quality base boxes: 13,879
- Train rows: 41,655 total / 41,637 high-quality
- Region types: `object=11537`, `surface=1250`, `container=968`,
  `free_space=125`, `unclear=5`
- Relations: `on=5634`, `beside=2303`, `inside=1377`, `left=1257`,
  `right=1241`, `behind=1020`, `front=943`, `between=110`

Use the high-quality file first unless explicitly auditing weak labels.

## Training parameters

| Param | Default | Notes |
|-------|---------|-------|
| `--per-device-train-batch-size` | 6-8 | Auto-probes 8→6→4→2→1 |
| `--gradient-accumulation-steps` | 4 | Effective BS = 8×6×4 = 192 |
| `--learning-rate` | 5e-6 | Cosine schedule, 3% warmup |
| `--model-max-length` | 16384 | |
| `--min-pixels / --max-pixels` | 50176 / 50176 | |
| `--max-retry-per-batch` | 3 | OOM retries before skip |
| `--batch-timeout` | 60-120 | Seconds before skip |
| `--save-steps` | varies | Use ≥600 to enable step-417 NaN fix |
| `--tune-mm-vision` | false | Freezes vision encoder |
| `--tune-mm-mlp` | true | Trains merger/MLP |

## Known issues and fixes

### Step 417 NaN cascade (critical)

**Symptom**: Training stably produces `grad_norm=nan` at step 417, then `loss=0.0` from step 418+ forever. Confirmed across 8 separate fullSFT runs with identical behavior.

**Root cause**: Single corrupted batch at step 417 produces NaN during backward (not forward). Original NaN check in `training_step` only inspected forward loss value — missed backward NaN. NaN gradients poisoned all model parameters.

**Fix applied** (`expert_sft.py`, May 2026):
1. `_sanitize_params()` — checks & zeros NaN params before each forward
2. `_sanitize_grads()` — checks & zeros NaN grads after each backward
3. Both log to stdout and `bad/bad_batches.log` with step+count
4. New counters: `param_nan_count`, `grad_nan_count` reported at smoke/train end
5. All ranks now synchronize a bad-batch flag before backward. If any rank
   sees `loss=0`, NaN/Inf loss, or gives up after OOM retries, every rank runs
   the same zero-loss placeholder backward and the optimizer step is skipped.
6. `GradGuardCallback` checks DeepSpeed ZeRO's global grad norm immediately
   before `optimizer.step()`. If bf16 ZeRO-2 reports NaN/Inf grad norm, it
   clears ZeRO averaged gradients and skips the optimizer step before params or
   optimizer state can be poisoned.

**Verification**: Test run with `--save-steps 600` (so checkpoint at step 600, well past step 417). Train head metric `param_nan_count` / `grad_nan_count` should remain zero throughout.

### MACA NCCL instability

NCCL `all_reduce`, `all_gather`, `barrier` are unstable on MACA C500 hardware.

**Fixes in expert_sft.py**:
- `_patch_dist_ops()`: `dist.barrier` patched with `all_reduce` fallback
- `_nested_gather()`: overridden to skip `dist.all_gather` entirely (returns local tensor)

**Environment variables** (set in chunked script):
- `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`
- `TORCH_NCCL_BLOCKING_WAIT=1`

`distributed_broadcast_scalars` is patched at module import time before
`Trainer` imports cache it, and `OomSafeTrainer._nested_gather()` returns local
tensors to avoid MACA `all_gather` hangs during logging.

### Chunked training caveats
- Uses `smoke` subcommand (not `train`) because `--smoke-batches` controls per-chunk size
- Model weights carry forward by loading from previous chunk's `output_dir` as `--model-name-or-path`
- Optimizer/scheduler are NOT carried forward (reset each chunk)
- Epoch counter resets each chunk → no resume issue

## Remote access (Tailscale)

After reboot, restore SSH access with:

```bash
bash /data/msz/start-tailscale.sh
```

Full setup doc: `tailscale-setup.md`. Tailscale runs in userspace-networking mode (chroot-compatible, no kernel tun required). State persists in `/var/lib/tailscale/` — auth survives reboots.

## Git notes
- `*.log`, `outputs/`, `bad/`, `data_expert/` are gitignored
- Models and datasets are NOT in git (downloaded by `ds_test/scripts/deploy_datasets_models.sh`)
- Remote: `git@github.com:msz136/muxi.git` (branch main)
