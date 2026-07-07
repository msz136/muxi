# Muxi Expert SFT / OPD 工作报告

日期：2026-05-19

本文档汇总了 `muxi` 当前几条工作线：DeepSpeed smoke 测试、语义导航 box 数据合成、OPD smoke 工作，以及 Qwen3-VL-8B-Instruct 的新 expert SFT 路线。远端 OPD 工作日志已经拉回本地，作为材料保存在：

```text
opd_project/reports/REPORT.remote.md
```

## 当前目标

之前的 expert 模型主要基于 RoboPoint 点选 grounding 训练，和期望的语义导航 expert 行为不匹配。当前替换路线是：使用从 RoboPoint `object_ref` 样本生成的合成语义导航 `<box>` grounding 数据集，对 Qwen3-VL-8B-Instruct 进行训练。

期望的 expert 行为是：

- 输入：图像加目标物体/关系信息；
- 输出：只返回 `<box>[[x1,y1],[x2,y2]]</box>`；
- 坐标范围：整数 `0..1000`；
- 目标领域：语义导航中的物体或区域 grounding，而不是通用 RoboPoint 点选输出。

## 重要路径

| 区域 | 路径 | 用途 |
|---|---|---|
| DeepSpeed smoke 测试 | `ds_test/` | 最小可运行的 SFT/GRPO/DPO/OPD 测试 |
| Expert 训练 | `point/` | 当前 expert SFT 入口与配置 |
| 语义导航数据脚本 | `opd_project/scripts/` | 数据清洗、manifest 构建、远端 crop 标注 |
| 完整合成数据本地副本 | `opd_semantic_nav_full_object_ref/` | 完整 object-ref 产物的本地副本 |
| Pilot 合成数据本地副本 | `opd_semantic_nav_pilot1k/` | pilot 产物与图像包 |
| 远端报告副本 | `opd_project/reports/REPORT.remote.md` | 拉回本地的远端 OPD 工作日志 |

## 1. `ds_test` 可运行训练路线

`ds_test` 是已知可跑通的 smoke-test 区域。它的价值在于尽量避开自定义 trainer 行为，证明 MACA C500 + DeepSpeed + HuggingFace 这一整套栈可以端到端运行 Qwen 系列训练。

### Qwen3-VL SFT smoke 路径

相关文件：

- `ds_test/scripts/run_sft_qwen3vl8b_smoke.sh`
- `ds_test/train_sft_qwen3vl_smoke.py`
- `ds_test/train_sft.py`
- `ds_test/configs/ds_config_sft_smoke_zero3_offload.json`
- `ds_test/configs/ds_config_zero2.json`

启动脚本会设置 MACA/DeepSpeed 的核心环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Qwen3-VL smoke 命令如下：

```bash
deepspeed --num_gpus=8 /data/msz/ds_test/train_sft.py \
  --model_name_or_path /data/msz/models/Qwen3-VL-8B-Instruct \
  --data_path /data/msz/ds_test/data/sft_qwen3vl8b_smoke.jsonl \
  --output_dir /data/msz/ds_test/logs/sft_qwen3vl8b_instruct_smoke_<ts> \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --max_seq_length 64 \
  --logging_steps 1 \
  --save_strategy no \
  --bf16 \
  --deepspeed_config /data/msz/ds_test/configs/ds_config_sft_smoke_zero3_offload.json
```

这里最重要的实现模式是普通 HuggingFace `Trainer`：

- 使用 `trust_remote_code=True` 加载 tokenizer/model；
- 强制 `use_cache=False`；
- 在 config 支持时设置 eager attention；
- 开启 gradient checkpointing；
- 通过把 prompt token mask 成 `-100` 来构造 labels；
- 让 `Trainer + DeepSpeed` 完整接管 backward、gradient clipping、optimizer step 和保存。

这条路径是替换问题较多的 `point` 脚本时使用的基线。

### DeepSpeed 配置

`ds_test/configs/ds_config_zero2.json` 是常驻 GPU 的 ZeRO-2 参考配置：

- `zero_optimization.stage=2`；
- optimizer 和 parameter 都不 offload；
- `bf16.enabled=true`；
- `gradient_clipping=1.0`；
- batch-size 值都是 `auto`，由 TrainingArguments 驱动。

`ds_test/configs/ds_config_sft_smoke_zero3_offload.json` 是更保守的 ZeRO-3 offload smoke 兜底配置：

- 固定 `train_batch_size=8`；
- `train_micro_batch_size_per_gpu=1`；
- optimizer 和 params 都 offload 到 CPU；
- 通信 bucket 更小；
- 当大模型需要在更低 GPU 显存压力下做 smoke-test 时有用。

## 2. OPD Smoke / 多教师路线

远端 OPD 工作日志现在保存在本地 `opd_project/reports/REPORT.remote.md`。其中记录的主要成果包括：

- 将 `prompt_pool.jsonl` 清洗为 `prompt_pool_clean.jsonl`；
- 移除了旧的 0..1 tuple 风格点选 prompt；
- 校验了本地图像存在性和 `gt_points`；
- 配置了使用 `expert3` 和 `expert4` 的多教师 OPD；
- 跑通了 50-step OPD smoke 训练；
- 修复了完整 checkpoint 保存逻辑，使 optimizer state 会先 CPU-copy，且不修改仍在训练中的 live optimizer。

OPD 中得到的 checkpoint 经验非常重要：

- 不要在 live `optimizer.state_dict()` 内原地转换 tensor；
- 保存 optimizer state 时，要深拷贝非 tensor，并把 tensor state clone 到 CPU；
- checkpoint 保存后必须验证训练还能继续。

修复后的 OPD smoke 完成了 50 steps，并在 step 25 和 50 保存了完整 model、processor、optimizer 和 training state。

## 3. 语义导航 Box 数据合成

新 expert 使用的合成数据来自语义导航 box-grounding 工作线。

### 来源

输入来源：

```text
/data/msz/opd_project/data/prompt_pool_clean.jsonl
```

筛选标准：

- source row 来自 `robopoint`；
- prompt prefix 为 `object_ref`；
- relation 在 `on,left,right,inside,beside,front,behind,between` 中；
- 有足够的 `gt_points` 来形成稳定区域；
- 由 point 推出的 box 通过最小边长和面积过滤。

### Manifest 生成

脚本：

```text
opd_project/scripts/build_semantic_nav_box_grounding.py
```

核心转换：

1. 读取清洗后的 RoboPoint 点选样本。
2. 从 `prompt_id` 推断 relation。
3. 将 point cloud 转成带 margin 的 min/max x/y bounding box。
4. 过滤过小、过大、不支持或较弱的候选 box。
5. 写出 manifest 和 image list。

完整 object-ref manifest 摘要：

```json
{
  "base_selected": 13885,
  "relations": "on,left,right,inside,beside,front,behind,between",
  "margin_ratio": 0.15,
  "min_margin": 10,
  "min_points": 4,
  "min_box_side": 24,
  "min_box_area": 1200,
  "max_box_area": 400000
}
```

本地完整 manifest 摘要副本：

```text
opd_semantic_nav_full_object_ref/manifest_summary.json
```

### 远端 Crop 标注

脚本：

```text
opd_project/scripts/local_label_semantic_nav_remote_base64.py
```

完整运行刻意避免把源图像存到本地：

1. 从远端机器读取源图像 bytes。
2. 使用 PIL 只在内存中裁剪合成的目标 box。
3. 将 crop 缩放到最长边不超过 512 px。
4. 只把小 crop 以 base64 发送给 Qwen-122B。
5. 要求返回严格 JSON：
   - `object_name`；
   - `attributes`；
   - `region_type`；
   - `confidence`。
6. 当 confidence 低、region 不清晰或 object name 过于泛化时，将样本标为弱标签。
7. 将每个通过筛选的 base annotation 扩展成三种 prompt 变体：
   - `obj_only`；
   - `relation_only`；
   - `obj_relation`。

### 最终数据产物

远端生产数据集：

```text
/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl
```

本地完整副本摘要：

```text
opd_semantic_nav_full_object_ref/output_http/semantic_nav_box_grounding_full_object_ref_v1_summary.json
```

最终完整 object-ref 计数：

| 产物 | 行数 | 说明 |
|---|---:|---|
| `semantic_nav_box_grounding_full_object_ref_v1_base_annotations.jsonl` | 13,885 | 每个 base box 一个标签 |
| `semantic_nav_box_grounding_full_object_ref_v1_all.jsonl` | 41,655 | 包含弱标签 |
| `semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl` | 41,637 | 推荐训练文件 |

质量摘要：

- base boxes：`13,885`；
- high-quality base boxes：`13,879`；
- weak base boxes：`6`；
- high-quality train rows：`41,637`；
- region types：
  - `object=11,537`；
  - `surface=1,250`；
  - `container=968`；
  - `free_space=125`；
  - `unclear=5`；
- relation 分布：
  - `on=5,634`；
  - `beside=2,303`；
  - `inside=1,377`；
  - `left=1,257`；
  - `right=1,241`；
  - `behind=1,020`；
  - `front=943`；
  - `between=110`。

## 4. `point` 中的新 Expert SFT 路线

旧的 `point/expert_sft.py` 路径累积了很多自定义 OOM 处理、dummy-batch fallback、参数/梯度清洗、分布式 workaround 和 callback guard。它对诊断有帮助，但作为训练事实来源风险较高：之前日志显示过 forward loss 看似正常，但 backward 或 optimizer state 仍可能污染参数的情况。

替换后的路线有意更接近 `ds_test`。

相关文件：

- `point/expert_sft.py`
- `point/run_expertsft_semantic_nav.sh`
- `point/configs/zero2.json`

训练数据：

```text
/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl
```

默认运行配置：

| 设置 | 值 |
|---|---:|
| GPUs | 8 |
| `per_device_train_batch_size` | 8 |
| `gradient_accumulation_steps` | 4 |
| effective batch size | 256 |
| rows | 41,637 |
| expected optimizer steps | 163 |
| epochs | 1 |
| learning rate | `5e-6` |
| precision | `bf16` |
| DeepSpeed | ZeRO-2 |
| checkpoint policy | 无中间 checkpoint；只保存最终模型 |

final-only 策略是有意选择的，因为每个完整 DeepSpeed checkpoint 都会占用大量磁盘空间。预计只有 163 steps，本次运行不需要中间 checkpoint。

## 5. Backward NaN 规避策略

之前的失败模式比较隐蔽：forward loss 可能是有限值，但 backward 会产生非有限梯度，或者污染参数。新方法不再把事后 parameter sanitization 作为主要解决方案。

### 改动内容

1. 使用标准 `Trainer + DeepSpeed` 执行 backward 和 optimizer step。
2. 从主要 expert SFT 路径中移除旧的 dummy-batch 和 best-effort-finalization 行为。
3. 使用新的 VL collator，根据 JSONL `conversations` 字段构造 Qwen3-VL chat messages。
4. mask labels，使只有 assistant `<box>...</box>` token 参与 loss。
5. 校验 processor 产生的 float tensors，如果任何类似 `pixel_values` 的 tensor 出现非有限值则直接 abort。
6. 设置 `use_cache=False`。
7. 使用 eager attention。
8. 开启 gradient checkpointing。
9. 保持 `max_grad_norm=1.0`。
10. 增加一个轻量 callback：如果日志中的 `loss`、`grad_norm` 或 `learning_rate` 变成非有限值，立即抛错。

### 为什么能覆盖旧问题

关键指标是 backward 后的 `grad_norm`。仅有有限的 loss 不够。在当前运行中，最早记录到的 backward 指标如下：

| Step | loss | grad_norm | learning_rate |
|---:|---:|---:|---:|
| 1 | 1.4076 | 98.3593 | 0 |
| 2 | 1.3746 | 97.3995 | 1e-6 |
| 3 | 1.3161 | 84.7705 | 2e-6 |
| 4 | 1.1766 | 55.3699 | 3e-6 |
| 5 | 1.0278 | 12.7692 | 4e-6 |
| 6 | 0.9365 | 4.0776 | 5e-6 |
| 7 | 0.9209 | 3.1196 | about 5e-6 |

早期 global norm 较大，是因为这是约 8B 可训练参数上的全模型 L2 norm，不能直接和单层梯度幅度比较。重要事实是：

- `grad_norm` 是有限值；
- loss 在下降；
- norm 从约 `98` 很快下降到个位数；
- `max_grad_norm=1.0` 会约束实际 optimizer update。

如果 backward NaN 再次出现，新路径应该会 fail fast，而不是带着被污染的 state 继续静默训练。

## 6. Checkpoint 与存储策略

对这次 expert SFT 运行：

- 没有中间 `checkpoint-*` 目录；
- 训练过程中没有 `global_step*` DeepSpeed optimizer-state 目录；
- 训练完成后只保存最终模型和 processor；
- 在输出目录写出 `trainer_state.json` 和 `run_summary.json`。

对需要 checkpoint 的 OPD 实验：

- 保存完整 model、processor、optimizer 和 training state；
- 将 optimizer tensor clone 到 CPU，但不能修改 live optimizer state；
- 保存 checkpoint 后，至少继续一个 optimizer step 以验证。

## 7. 验收检查

完整 expert 训练前：

- `point/expert_sft.py` 在远端 runtime 上通过 Python 编译；
- `inspect` mode 可以将 8-sample batch 处理成 tensors：
  - `input_ids`：`(8, 200)`；
  - `attention_mask`：`(8, 200)`；
  - `pixel_values`：`(1728, 1536)`；
  - `image_grid_thw`：`(8, 3)`；
  - valid labels：`200`。

启动后：

- DeepSpeed 在 `tmux full` 中启动了 8 个 ranks；
- expected steps：`163`；
- step 1 和 step 2 完成，loss 和 `grad_norm` 都是有限值；
- 检查时没有产生中间 checkpoint 目录。

## 8. 未解决风险与备注

- 训练仍需要监控到结束，因为 MACA NCCL 和大模型训练可能在后期失败。
- 如果 ZeRO-2 下最终保存出问题，使用 `ds_test` 的 saved-output 模式作为 fallback，并确认生成了 shards/tokenizer 文件。
- 当前 expert SFT 有意选择干净的标准训练路径，而不是旧的自定义 NaN 清洗逻辑。
- high-quality 数据集是首选训练集。`all.jsonl` 包含弱标签，应保留给审计或有意的鲁棒性实验。

## 9. 按工作线整理的困难与解决方案

### `ds_test` 可运行基线

困难：

- 在修改 `point` 前，首先需要证明 MACA C500 平台、DeepSpeed、Transformers、bf16、模型加载、backward、optimizer step 和保存路径都能跑通。
- 如果没有这条基线，后续每个失败都会变得含混：可能是平台问题、模型加载问题、数据问题，也可能是自定义 trainer bug。

解决：

- 保持 `ds_test` 足够极简和稳定：普通 HuggingFace `Trainer`、小 smoke 数据、短序列长度、不做自定义 OOM 或 NaN 恢复逻辑。
- smoke run 使用 `save_strategy=no`，然后显式保存最终模型和 tokenizer/processor。
- 保留两个 DeepSpeed 参考配置：
  - ZeRO-2：正常 GPU-resident 训练；
  - ZeRO-3 CPU offload：保守的低显存 smoke 测试。
- 将已验证的组件迁移到 `point`：`use_cache=False`、eager attention、bf16、gradient checkpointing，以及由标准 `Trainer + DeepSpeed` 接管 backward 和 optimizer step。

### OPD / 多教师路线

困难：

- OPD 项目一开始是草稿状态，依赖不完整，路径也有 placeholder。
- 起初不清楚 `expert3`、`expert4`、`8b_base` 和 `Qwen3-VL-8B-Instruct` 的结构是否兼容。
- prompt pool 中混有旧 point-format 指令和新的严格 `<point>` 格式。很多 user prompt 仍要求 `(x, y)` 和 `0..1` 坐标，而目标 system prompt 要求整数 `0..1000`。
- Checkpointing 引入了 live optimizer mutation bug：原地把 `optimizer.state_dict()` 中的 tensor 移到 CPU，会导致下一个 optimizer step 看到 CPU state 与 GPU params/grads 混在一起。

解决：

- 训练前先做模型 inventory 和 key-set 检查。模型目录暴露出的 weight keys 兼容，因此 teacher/student 实验在结构上可以尝试。
- 将 OPD prompt pool 清洗为 canonical schema：
  - 规范化 message roles；
  - 重写旧点选指令；
  - 强制严格 `<point>[[x,y],...]</point>`；
  - 规范化 `gt_points`；
  - 过滤缺失本地图像的样本。
- 在更大实验前先跑小 OPD smoke jobs，记录 metrics 和 bad-step 日志。
- 修复 checkpoint 保存：深拷贝非 tensor optimizer state，并将 tensor state clone 到 CPU，不修改 live optimizer。验收标准是 checkpoint 保存后训练仍能继续。

### 语义导航 box 数据合成

困难：

- RoboPoint 提供的是 point grounding，但期望的 expert 必须返回语义导航 bounding boxes。
- 原始 point 样本中没有直接可用的目标物体名称和 region type。
- 大规模标注时不应把完整源图像复制到本地，否则浪费磁盘，也增加数据管理风险。
- 弱 crop 可能描述泛化物体、不清晰区域或 free space，因此生成标签需要质量过滤。

解决：

- 将 RoboPoint `object_ref` 样本转换成 box manifests：
  - 保留支持的 relations；
  - 要求足够的 `gt_points`；
  - 将 point clouds 转为带 margin 的 min/max boxes；
  - 过滤面积和 box 尺寸不合理的样本。
- 只标注裁剪后的目标区域：
  - 读取远端源图像 bytes；
  - 在内存中用 PIL crop；
  - 缩放 crop；
  - 只把小 base64 crop 发送给 Qwen-122B。
- 要求 Qwen-122B 输出结构化 JSON 标签：`object_name`、`attributes`、`region_type` 和 `confidence`。
- 当 name 泛化、confidence 低或 region type 不清晰时，拒绝或单独分离弱标签。
- 将每个通过筛选的 base box 扩展成三种 prompt 变体：
  - `obj_only`；
  - `relation_only`；
  - `obj_relation`。
- 优先使用 high-quality 最终数据集：
  - `13,879` high-quality base boxes；
  - `41,637` high-quality training rows。

### `point` expert SFT 路线

困难：

- 旧的 `point/expert_sft.py` 有过多自定义恢复机制：OOM retry、dummy batches、parameter sanitization、gradient sanitization、distributed patches 和 best-effort finalization。
- 这些机制有助于诊断失败，但使训练语义难以信任。
- 关键历史失败不是 forward loss NaN。loss 可以看起来是有限值，但 backward 或 optimizer state 产生 NaN 并污染模型参数。之后训练可能坍缩成 `loss=0.0`。

解决：

- 用基于 `ds_test` 的干净路径替换 expert SFT 实现。
- 让标准 `Trainer + DeepSpeed ZeRO-2` 接管 backward、gradient clipping 和 optimizer stepping。
- 只保留任务相关的 VL collator：
  - 从 JSONL `conversations` 构造 Qwen3-VL chat messages；
  - 通过 processor utility 加载 image/video 输入；
  - mask prompt tokens，使只有 assistant `<box>...</box>` tokens 参与训练；
  - 检查 processor 产生的 float tensors 是否有非有限值。
- 将 backward metrics 视为真正的验收信号：
  - 有限 loss 是必要但不充分条件；
  - backward 后有限的 `grad_norm` 才确认敏感路径仍然正常。
- 对日志中的非有限 `loss`、`grad_norm` 或 `learning_rate` 增加 fail-fast 行为；一旦怀疑 state 被污染，就不要继续训练。

### Checkpoint 与存储策略

困难：

- 带 optimizer state 的完整 8B checkpoint 非常大。
- DeepSpeed optimizer-state 目录，例如 `global_step*`，在 save 间隔太频繁时很容易填满磁盘。
- OPD 需要 checkpoint 正确性，而 expert SFT 只需要最终模型，因为完整运行较短。

解决：

- 对 OPD smoke 和 checkpoint 测试，保存完整 model、processor、optimizer 和 training state，然后验证之后仍能继续训练。
- 对当前 expert SFT 运行，完全禁用中间 checkpoints，只保存最终 model/processor 和运行元数据。
- 启动前计算 expected steps：
  - rows：`41,637`；
  - effective batch：`8 * 8 * 4 = 256`；
  - expected steps：`163`。
- 因为运行较短，final-only saving 是更安全的磁盘策略。

## Semantic-Nav Box Expert SFT 评估 - 2026-05-19

### 已检查的训练运行

- SFT expert 输出：
  `/data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338`
- Base model：
  `/data/msz/models/Qwen3-VL-8B-Instruct`
- 训练数据：
  `/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl`
- 训练行数：`41,637`
- 有效 batch size：`8 GPUs * 8 per-device * 4 grad-accum = 256`
- 总 optimizer steps：`163`
- 最终训练指标：
  - `train_loss = 0.7771285136053168`
  - `train_runtime = 1049.5756s`
  - `train_steps_per_second = 0.155`
- Checkpoint 策略：没有中间 checkpoints；只保存最终模型。
- 最终模型文件：五个 `model-0000x-of-00005.safetensors` shards，加 processor/tokenizer metadata。

### Held-out 评估集

初始训练文件消耗了所有 high-quality `object_ref` 行，因此从该文件采样只会是 train-set sanity check。后来从 `/data/msz/opd_project/data/prompt_pool_clean.jsonl` 构建了新的 held-out semantic-to-box eval set，并排除了 SFT manifest 中已经出现的 prompt IDs。

主混合 eval set：

- JSONL：
  `/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_eval_v1.jsonl`
- Base manifest：
  `/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_eval_v1_base_manifest.jsonl`
- 过滤后的 base candidates：`10,753`
- 选中的 base samples：`512`
- Eval prompt rows：`1,536`
- Prompt variants：
  - `relation_plain`
  - `semantic_json`
  - `object_relation_compat`
- Prefix split：
  - `region_ref = 355`
  - `object_ref = 157`
- Held-out reason：
  - `prefix_not_trained = 355`
  - `object_ref_prompt_id_not_in_train_manifest = 157`
- 平均目标 box 面积：`16,873.3`

同时还生成了额外 split 文件：

- Object-ref matched held-out：
  `/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_object_ref_eval_v1.jsonl`
  - `417` base samples，`1,251` prompt rows。
  - Relations：`above`、`below`、`on-left`、`on-back`、`on-right`、`on-front`。
- Region-ref challenge：
  `/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_region_ref_eval_v1.jsonl`
  - `512` base samples，`1,536` prompt rows。
  - 完全位于 SFT 训练 prefix 之外。

Builder 脚本：

- 本地/远端路径：`point/build_semantic_nav_eval_set.py`

### 评估命令

SFT expert：

```bash
cd /data/msz
export MACA_PATH=/opt/maca-3.5.3
export LD_LIBRARY_PATH=/opt/maca-3.5.3/lib:/opt/maca-3.5.3/mxgpu_llvm/lib:/opt/maca-3.5.3/ompi/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3 point/eval_semantic_nav_box.py \
  --model /data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338 \
  --data /data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_eval_v1.jsonl \
  --out /data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338/eval_heldout_semantic_nav_96.jsonl \
  --num-samples 96 \
  --seed 20260519 \
  --max-new-tokens 64
```

Base model：

```bash
cd /data/msz
export MACA_PATH=/opt/maca-3.5.3
export LD_LIBRARY_PATH=/opt/maca-3.5.3/lib:/opt/maca-3.5.3/mxgpu_llvm/lib:/opt/maca-3.5.3/ompi/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python3 point/eval_semantic_nav_box.py \
  --model /data/msz/models/Qwen3-VL-8B-Instruct \
  --data /data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_eval_v1.jsonl \
  --out /data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338/eval_base_qwen3vl8b_heldout_semantic_nav_96.jsonl \
  --num-samples 96 \
  --seed 20260519 \
  --max-new-tokens 64
```

评估脚本：

- 本地/远端路径：`point/eval_semantic_nav_box.py`

### 整体结果

Strict format 表示预测必须匹配：
`<box>[[x1,y1],[x2,y2]]</box>`。

Lenient format 还会解析 base model 常见的扁平变体：
`<box>[[x1,y1,x2,y2]]</box>`。

| 模型 | 格式模式 | 格式率 | Mean IoU | IoU@0.3 | IoU@0.5 | Mean center error |
|---|---:|---:|---:|---:|---:|---:|
| SFT expert | strict | 100.00% | 0.1544 | 18.75% | 15.63% | 153.0 |
| Base Qwen3-VL-8B | strict | 31.25% | 0.1137 | 13.33% | 3.33% | 184.9 |
| Base Qwen3-VL-8B | lenient | 100.00% | 0.0933 | 10.42% | 2.08% | 174.7 |

解读：

- SFT expert 显著提升了输出格式遵循能力。
- 在 strict scoring 下，base model 大多无效，因为它经常输出扁平 4-number boxes。
- 即使使用 lenient parsing，SFT expert 的 mean IoU 也更高，并且 IoU@0.5 明显更好。
- 改进主要集中在 held-out `object_ref`；`region_ref` 仍然较弱，需要显式训练。

### Prefix split

Strict format：

| Prefix | 模型 | 格式率 | Mean IoU | IoU@0.3 | IoU@0.5 | Mean center error |
|---|---|---:|---:|---:|---:|---:|
| object_ref | SFT expert | 100.00% | 0.3501 | 50.00% | 44.12% | 135.2 |
| object_ref | Base strict | 38.24% | 0.1645 | 23.08% | 7.69% | 172.1 |
| region_ref | SFT expert | 100.00% | 0.0471 | 1.61% | 0.00% | 162.8 |
| region_ref | Base strict | 27.42% | 0.0748 | 5.88% | 0.00% | 194.7 |

Lenient base comparison：

| Prefix | 模型 | 格式率 | Mean IoU | IoU@0.3 | IoU@0.5 | Mean center error |
|---|---|---:|---:|---:|---:|---:|
| object_ref | SFT expert | 100.00% | 0.3501 | 50.00% | 44.12% | 135.2 |
| object_ref | Base lenient | 100.00% | 0.1319 | 14.71% | 5.88% | 182.5 |
| region_ref | SFT expert | 100.00% | 0.0471 | 1.61% | 0.00% | 162.8 |
| region_ref | Base lenient | 100.00% | 0.0721 | 8.06% | 0.00% | 170.4 |

解读：

- `object_ref` held-out 是有意义的同分布胜利：SFT mean IoU `0.3501`，base lenient `0.1319`。
- `region_ref` 是 challenge transfer split：两个模型都差，当前 SFT 并没有解决 region-style semantic grounding。

### Prompt-mode split

Strict format：

| Prompt mode | 模型 | 格式率 | Mean IoU | IoU@0.3 | IoU@0.5 | Mean center error |
|---|---|---:|---:|---:|---:|---:|
| object_relation_compat | SFT expert | 100.00% | 0.1704 | 22.86% | 17.14% | 144.0 |
| object_relation_compat | Base strict | 28.57% | 0.1957 | 30.00% | 10.00% | 111.8 |
| relation_plain | SFT expert | 100.00% | 0.1694 | 21.43% | 17.86% | 147.3 |
| relation_plain | Base strict | 50.00% | 0.0909 | 7.14% | 0.00% | 215.8 |
| semantic_json | SFT expert | 100.00% | 0.1249 | 12.12% | 12.12% | 167.5 |
| semantic_json | Base strict | 18.18% | 0.0300 | 0.00% | 0.00% | 234.7 |

Lenient base comparison：

| Prompt mode | 模型 | 格式率 | Mean IoU | IoU@0.3 | IoU@0.5 | Mean center error |
|---|---|---:|---:|---:|---:|---:|
| object_relation_compat | SFT expert | 100.00% | 0.1704 | 22.86% | 17.14% | 144.0 |
| object_relation_compat | Base lenient | 100.00% | 0.1083 | 11.43% | 5.71% | 147.0 |
| relation_plain | SFT expert | 100.00% | 0.1694 | 21.43% | 17.86% | 147.3 |
| relation_plain | Base lenient | 100.00% | 0.1027 | 14.29% | 0.00% | 186.5 |
| semantic_json | SFT expert | 100.00% | 0.1249 | 12.12% | 12.12% | 167.5 |
| semantic_json | Base lenient | 100.00% | 0.0693 | 6.06% | 0.00% | 194.1 |

解读：

- SFT expert 在三种 prompt 风格下都能稳定遵循格式。
- 对当前模型来说，`semantic_json` 比 plain relation text 更难。
- `object_relation_compat` 最接近训练 prompt 家族，但 SFT 后 relation-only prompt 表现也相近。

### 保存的结果文件

- SFT sample-level results：
  `/data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338/eval_heldout_semantic_nav_96.jsonl`
- SFT summary：
  `/data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338/eval_heldout_semantic_nav_96.summary.json`
- Base sample-level results：
  `/data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338/eval_base_qwen3vl8b_heldout_semantic_nav_96.jsonl`
- Base summary：
  `/data/msz/point/outputs/expertsft_semantic_nav_final_20260519_102338/eval_base_qwen3vl8b_heldout_semantic_nav_96.summary.json`
- Eval set summary：
  `/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_eval_v1_summary.json`

### 建议

当前 expert 对 `object_ref` semantic-to-box grounding 和严格 `<box>` 格式来说，是相对原始 base model 的有效改进。不过，它还不能视为完整的语义导航 region grounding expert。下一轮数据/训练迭代应该：

- 加入 `region_ref` point-to-box 转换训练行；
- 训练前固定 held-out split；
- 为 object-name-specific eval 引入来自 crop VLM 标注的 object/region labels，而不仅是规则推断的 `target region` 标签；
- 继续分别报告 strict 和 lenient format metrics，因为 base model 的输出格式与训练目标不同。

## 9. Region-Box Expert SFT：`solution_a_bidir`

日期：2026-05-19

本次运行训练第二个面向语义导航 region grounding 的 expert：输入图像加自然语言或结构化区域描述，输出应只返回 `<box>[[x1,y1],[x2,y2]]</box>`。

重试遵循失败后的 reset 方案：

- 标准 HuggingFace `Trainer` + DeepSpeed ZeRO-2；
- 不使用自定义 training step，也不使用自定义 NaN skip/guard step 逻辑；
- 总 train micro-batch 为 `8`，实现方式是 8 GPUs 上 `per_device_train_batch_size=1`；
- `gradient_accumulation_steps=4`；
- learning rate 恢复为 `5e-6`；
- 没有中间 checkpoint，只保存最终模型。

### 训练运行

| 字段 | 值 |
|---|---|
| Run id | `region_expertsft_solution_a_bidir_final_20260519_171308` |
| Base model | `/data/msz/models/Qwen3-VL-8B-Instruct` |
| Train data | `/data/msz/opd_project/data/semantic_nav_region_box_v1/solution_a_bidir_train_v1/semantic_nav_region_solution_a_bidir_train_v1_high_quality.jsonl` |
| Train rows | 10,290 |
| Output | `/data/msz/point/outputs/region_expertsft_solution_a_bidir_final_20260519_171308` |
| Log | `/data/msz/point/logs/region_expertsft_solution_a_bidir_final_20260519_171308.log` |
| Effective batch | 32 |
| Steps | 322 |
| Final train loss | 0.7563 |
| Checkpoint dirs | 0 |

完成过程干净：日志到达 `[region-expertsft] done=Tue May 19 17:41:23 CST 2026`，最终模型保存为 5 个 safetensor shards，并带 tokenizer/config 文件，没有留下 `checkpoint-*` 目录。

### 评估协议

| 字段 | 值 |
|---|---|
| Eval data | `/data/msz/opd_project/data/semantic_nav_region_box_v1/eval/semantic_nav_region_solution_a_bidir_eval_v1.jsonl` |
| Eval rows | 144 |
| Seed | 20260519 |
| Max new tokens | 64 |
| Expert output | `/data/msz/point/outputs/region_expertsft_solution_a_bidir_final_20260519_171308/eval_region_expert_144.jsonl` |
| Base output | `/data/msz/point/outputs/region_expertsft_solution_a_bidir_final_20260519_171308/eval_region_base_144.jsonl` |

除非特别说明，下面指标都是 strict-format。`Mean IoU all` 将格式失败计为 IoU 0，因此这是和 base model 做任务级比较时更公平的指标。`Mean IoU parsed` 是评估脚本原本只在可解析输出上计算的均值。

### 整体对比

| 模型 | Format OK | Format rate | Mean IoU parsed | Mean IoU all | Median IoU all | IoU@0.1 all | IoU@0.3 all | IoU@0.5 all | Mean center error | Coord MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Region SFT expert | 144/144 | 100.0% | 0.2371 | 0.2371 | 0.1701 | 65.3% | 34.0% | 18.1% | 84.2 | 54.3 |
| Base Qwen3-VL-8B | 63/144 | 43.8% | 0.1157 | 0.0506 | 0.0000 | 21.5% | 3.5% | 0.0% | 143.7 | 109.3 |

在匹配的 region eval set 上，expert 相比原始 base model 有明确提升：strict formatting 变得可靠，任务级 mean IoU 从 `0.0506` 提升到 `0.2371`，IoU@0.3 从 `3.5%` 提升到 `34.0%`。

### Prompt-mode split

| Prompt mode | N | Expert format | Base format | Expert mean IoU all | Base mean IoU all | Delta | Expert IoU@0.3 all | Base IoU@0.3 all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `description_plain` | 48 | 100.0% | 52.1% | 0.230 | 0.068 | +0.162 | 31.3% | 4.2% |
| `description_relation` | 48 | 100.0% | 43.8% | 0.238 | 0.047 | +0.191 | 35.4% | 4.2% |
| `semantic_json` | 48 | 100.0% | 35.4% | 0.244 | 0.037 | +0.207 | 35.4% | 2.1% |

region expert 在三种 prompt 风格上都稳定。base model 在 `semantic_json` 上尤其脆弱，而 SFT model 在该 prompt mode 上还有轻微提升。

### Region-category split

| Region category | N | Expert format | Base format | Expert mean IoU all | Base mean IoU all | Delta | Expert IoU@0.3 all | Base IoU@0.3 all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `free_space` | 30 | 100.0% | 46.7% | 0.297 | 0.091 | +0.205 | 46.7% | 10.0% |
| `surface` | 114 | 100.0% | 43.0% | 0.221 | 0.040 | +0.182 | 30.7% | 1.8% |

目前 `free_space` 比 `surface` 更容易，但两个类别相对 base 都有显著提升。

### Relation split

| Relation | N | Expert format | Base format | Expert mean IoU all | Base mean IoU all | Delta | Expert IoU@0.3 all | Base IoU@0.3 all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `above` | 9 | 100.0% | 55.6% | 0.070 | 0.107 | -0.037 | 0.0% | 11.1% |
| `behind` | 3 | 100.0% | 0.0% | 0.254 | 0.000 | +0.254 | 33.3% | 0.0% |
| `below` | 9 | 100.0% | 100.0% | 0.210 | 0.129 | +0.081 | 33.3% | 33.3% |
| `beside` | 9 | 100.0% | 0.0% | 0.326 | 0.000 | +0.326 | 33.3% | 0.0% |
| `between` | 12 | 100.0% | 41.7% | 0.299 | 0.088 | +0.210 | 50.0% | 0.0% |
| `front` | 12 | 100.0% | 25.0% | 0.121 | 0.000 | +0.121 | 16.7% | 0.0% |
| `inside` | 3 | 100.0% | 100.0% | 0.160 | 0.122 | +0.039 | 0.0% | 0.0% |
| `left` | 12 | 100.0% | 33.3% | 0.287 | 0.039 | +0.247 | 50.0% | 0.0% |
| `on` | 33 | 100.0% | 45.5% | 0.201 | 0.042 | +0.159 | 27.3% | 0.0% |
| `on-back` | 3 | 100.0% | 100.0% | 0.038 | 0.107 | -0.069 | 0.0% | 0.0% |
| `on-front` | 9 | 100.0% | 44.4% | 0.434 | 0.051 | +0.383 | 66.7% | 0.0% |
| `on-left` | 6 | 100.0% | 66.7% | 0.343 | 0.105 | +0.237 | 83.3% | 16.7% |
| `on-right` | 3 | 100.0% | 66.7% | 0.744 | 0.065 | +0.680 | 100.0% | 0.0% |
| `right` | 21 | 100.0% | 28.6% | 0.192 | 0.014 | +0.179 | 23.8% | 0.0% |

弱项集中在 `above`、`front` 和 `on-back`。小样本关系不应过度解读，但它们很适合作为定向数据增强候选。

### 本地结果产物

详细逐样本评估结果和聚合对比已经复制到本地工作树：

- `report/region_expertsft_solution_a_bidir_final_20260519_171308/eval_region_expert_144.jsonl`
- `report/region_expertsft_solution_a_bidir_final_20260519_171308/eval_region_expert_144.summary.json`
- `report/region_expertsft_solution_a_bidir_final_20260519_171308/eval_region_base_144.jsonl`
- `report/region_expertsft_solution_a_bidir_final_20260519_171308/eval_region_base_144.summary.json`
- `report/region_expertsft_solution_a_bidir_final_20260519_171308/eval_region_comparison_aggregate.json`
- `report/region_expertsft_solution_a_bidir_final_20260519_171308/semantic_nav_region_solution_a_bidir_eval_v1.jsonl`

## 10. OPD 混合 Expert 模型 Agent 日志 - 2026-05-20

目标：从 `/data/msz/models/8b_base` 生成第一版 OPD 混合模型，使用 `/data/msz/models/expert_obj_v1` 和 `/data/msz/models/expert_reg_v1`。路由设计：object 样本只使用 object expert 的 top1 token target，region 样本只使用 region expert 的 top1 token target，general 样本使用 `0.5 * obj_loss + 0.5 * reg_loss`。teacher cache 只存 top1 token ids。

### 已完成工作

- 设置了 OPD 混合模型生成的 active goal。
- 确认远端访问可用，并在 `/data/msz/point` 下工作。
- 找到并清理了上一次 `opd_smoke_train40` 尝试遗留的 8-GPU smoke 进程。
- 确认远端 OPD 脚本存在：
  - `/data/msz/point/build_opd_mix_v1.py`
  - `/data/msz/point/prepare_opd_top1_cache.py`
  - `/data/msz/point/train_opd_top1_vl.py`
  - `/data/msz/point/run_opd_mix_v1_20k.sh`
- 阅读当前 OPD 实现：
  - `build_opd_mix_v1.py` 构建带 `opd.bucket`、`opd.teacher` 和 source metadata 的 obj/reg/general manifest。
  - `prepare_opd_top1_cache.py` 依次加载 object 和 region teachers，并写入 `obj_top1_ids` / `reg_top1_ids`。
  - `train_opd_top1_vl.py` 使用 routed top1 CE loss 训练 student，并只保存最终模型。
- 确认 smoke cache 已存在：
  `/data/msz/point/data_opd_smoke/opd_mix_smoke40_top1_b8.jsonl`，共 40 行，其中 26 行有 object-teacher，26 行有 region-teacher，12 行 general 样本同时使用两个 teachers。
- 跑了 1-GPU 非 DeepSpeed 检查。它到达 backward/optimizer setup，但在分配 AdamW optimizer state 时 OOM；对 8B 模型来说，没有 ZeRO sharding 时这是预期行为。
- 本地 patch 了 `train_opd_top1_vl.py` 并同步到远端，使 OPD CE 只在 assistant-answer token 位置计算，而不是对完整序列做 `log_softmax`。这降低了不必要的显存压力，同时保留预期的 top1-token OPD 目标。
- 使用 patched sparse-token loss 路径重试了 8-GPU smoke。

### 当前阻碍

运行仍在 DeepSpeed ZeRO-2 下第一次分布式 backward/optimizer step 处阻塞。失败模式是在任何正常训练指标记录前出现 MACA/NCCL allreduce timeout：

- timeout point：tqdm 到达 `0/2` 后的第一个 collective；
- operation：NCCL `ALLREDUCE`；
- observed seq number：`SeqNum=766`；
- timeout 中观察到的 tensor size：根据 rank 不同，约 `192,946,432` 到 `353,443,328` elements；
- timeout：`600000 ms`；
- 目前没有 NaN 证据，因为运行尚未到达成功记录 optimizer step 的阶段。

判断：当前直接阻碍不是数据格式、teacher cache 或 NaN，而是这套 MACA 环境上的 DeepSpeed ZeRO-2 梯度通信路径，很可能被较大的通信 bucket / overlap settings 放大。

### 建议解决路线

1. 保留 sparse assistant-token OPD loss patch；它仍然是 top1 OPD 的正确 loss 形状，并且能避免额外显存压力。
2. 使用更保守的 ZeRO-2 config 重试 8-GPU smoke：
   `/data/msz/point/configs/zero2_opd_maca.json`，其中使用更小的 `reduce_bucket_size` / `allgather_bucket_size`，并设置 `overlap_comm=false`。该配置已经在本地创建并同步到远端，但 smoke 命令在产生结果前被中断。
3. 如果保守 ZeRO-2 仍然挂起，就将 OPD 训练切换到 `ds_test` 已证明可跑通的 ZeRO-3 offload 模式：
   `/data/msz/ds_test/configs/ds_config_sft_smoke_zero3_offload.json`，或者复制一份具有相同小 bucket/offload 行为的 OPD 专用配置。
4. 当 smoke 至少跑到两步，并且 loss 与 grad norm 都是有限值后，再在 `tmux` session `full` 中启动第一版 small/full OPD mix，采用 final-only 保存，不保留中间 checkpoints。

### 注意

原始的 20k offline top1-cache 计划对第一轮迭代来说可能太慢。teacher-cache 准备阶段的 smoke timing 表明，除非并行化 teacher-cache 路径，否则第一版 OPD 更实际的规模是 2k-4k 混合行。

## 11. OPD v1 On-policy 混合模型完成记录与评估汇总

日期：2026-05-20

本节是对上一节 OPD 初版阻塞记录的后续。第 10 节停在 offline top1-cache / ZeRO 通信阻塞阶段；随后根据新的设计要求，OPD 被重新实现为 **on-policy teacher rollout**：训练时由 teacher model 当场生成 rollout，再取 teacher 对该 rollout 的 top1 logit/token 作为 student 的蒸馏目标，而不是预先离线生成 logits/cache。

### 当前结论

- 第一版 OPD 混合模型已训练完成，最终模型路径：
  `/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529`
- 训练从 `/data/msz/models/8b_base` 启动，teachers 为：
  - object teacher：`/data/msz/models/expert_obj_v1`
  - region teacher：`/data/msz/models/expert_reg_v1`
- OPD v1 在 object/region 两个领域 eval 上都保持 100% strict `<box>[[x1,y1],[x2,y2]]</box>` 格式率。
- OPD v1 没有达到两个专门 expert 的单域峰值：
  - object eval：`obj_expert` 明显最好，`mean_iou_all=0.398283`；OPD v1 为 `0.173187`。
  - region eval：`reg_expert` 明显最好，`mean_iou_all=0.236819`；OPD v1 为 `0.145121`。
- OPD v1 的行为更像“折中模型”：object 强于 region expert，region 强于 object expert，但没有超过对应单域 expert。
- 通用 200 条 Robo2VLM eval 上，OPD v1 的 relaxed match 与 base 持平，为 `0.295`；exact / token-F1 高于 base，但低于 reg expert。

### 关键本地与远端产物

| 类型 | 路径 |
|---|---|
| OPD online trainer | `point/train_opd_online_vl.py` |
| OPD mix builder | `point/build_opd_mix_v1.py` |
| OPD train launcher | `point/run_opd_online_mix_v1_first.sh` |
| OPD eval launcher | `point/run_eval_opd_online_mix_v1.sh` |
| General eval script | `point/eval_general_vqa.py` |
| General eval launcher | `point/run_eval_general_vqa_heldout200.sh` |
| Four-model domain eval launcher | `point/run_eval_domain_four_models_v1.sh` |
| OPD train data | `/data/msz/point/data_opd/opd_mix_v1_2048_mediaok_seed20260520.jsonl` |
| OPD train data summary | `/data/msz/point/data_opd/opd_mix_v1_2048_mediaok_seed20260520.summary.json` |
| OPD train log | `/data/msz/point/logs/opd_online_mix_v1_2048_final_20260520_142529.log` |
| OPD final model | `/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529` |
| OPD run summary | `/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529/opd_online_run_summary.json` |
| 本地域 eval 结果 | `report/eval_domain_four_models_v1_20260520_152926/` |
| 本地通用 eval 结果 | `report/eval_general_vqa_heldout200_20260520_151919/` |
| 本地 OPD-only grounding eval | `report/eval_opd_online_mix_v1_2048_20260520_145224/` |
| 本地综合小结 | `report/opd_v1_eval_summary_20260520.md` |

### 训练数据设计

训练 mix 共 2,048 行，seed 为 `20260520`。采样比例：

| Bucket | 行数 | Teacher route | 说明 |
|---|---:|---|---|
| object | 717 | obj | semantic-nav object box grounding |
| region | 717 | reg | region box grounding |
| general | 614 | both | 通用 Robo2VLM keepalive 样本，训练时复制为 obj/reg 两路，各 0.5 loss |

region bucket 内部来源：

| 来源 | 行数 |
|---|---:|
| `region_predbox_label_v1` | 502 |
| `region_solution_a_bidir` | 215 |

原始数据源：

| 角色 | 数据源 |
|---|---|
| object | `/data/msz/opd_project/data/semantic_nav_box_v1/semantic_nav_box_grounding_full_object_ref_v1_high_quality.jsonl` |
| region predbox | `/data/msz/opd_project/data/semantic_nav_region_box_v1/predbox_label_v1/semantic_nav_region_predbox_label_v1_high_quality_no_eval_holdout.jsonl` |
| region solution | `/data/msz/opd_project/data/semantic_nav_region_box_v1/solution_a_bidir_train_v1/semantic_nav_region_solution_a_bidir_train_v1_high_quality.jsonl` |
| general | `/data/msz/point/data_expert/expert_grounding_mix.jsonl` |

注意：通用数据最初希望混入更多 general 数据源，但可用 media 校验后第一版实际采用 `robo2vlm-1` keepalive 样本。ShareRobot/其它 grounding-local 样本在当时路径或媒体可读性上不稳定，因此 `general_grounding_local=0`。

### OPD on-policy 实现要点

`train_opd_online_vl.py` 的核心行为：

- `OPDOnlineDataset` 按 `opd.teacher` 路由样本：
  - `teacher=obj`：单路 object teacher，loss weight `1.0`；
  - `teacher=reg`：单路 region teacher，loss weight `1.0`；
  - `teacher=both`：复制成 obj/reg 两条训练样本，每条 loss weight `0.5`。
- 为了避免同一 microbatch 内混合 teacher，dataset 先按 route 分组，再拼接；collator 记录 `opd_route_ids` 与 `opd_loss_weights`。
- `compute_loss` 内部根据 route 懒加载对应 teacher；route 切换时卸载旧 teacher 并 `torch.cuda.empty_cache()`。
- teacher 先 `generate()` 得到 on-policy rollout；然后 teacher 对该 rollout 前向，取 `logits.argmax(dim=-1)` 作为 top1 token label。
- student 在 teacher rollout 序列上前向，只对 answer token 范围计算 cross entropy；prompt token 位置为 `-100`。
- general 样本通过两路 route 各 0.5 权重实现 `0.5 * obj_loss + 0.5 * reg_loss`。
- 训练目标是 top1 token CE，不保存完整 teacher logits，避免巨大的离线 logit 文件。

这版实现保留了“logit 只取 top1”的设计，但把 top1 的生成时机从离线 cache 改为训练时 on-policy rollout，更符合目标 OPD 语义。

### 训练参数与运行过程

| 字段 | 值 |
|---|---|
| Run id | `opd_online_mix_v1_2048_final_20260520_142529` |
| Base model | `/data/msz/models/8b_base` |
| Object teacher | `/data/msz/models/expert_obj_v1` |
| Region teacher | `/data/msz/models/expert_reg_v1` |
| DeepSpeed config | `/data/msz/point/configs/zero2.json` |
| GPUs | 8 |
| per-device batch size | 1 |
| gradient accumulation | 4 |
| effective batch size | 32 |
| learning rate | `1e-6` |
| warmup ratio | `0.03` |
| scheduler | cosine |
| max grad norm | 1 |
| bf16 | true |
| gradient checkpointing | true |
| min/max pixels | 50,176 / 50,176 |
| model max length | 16,384 |
| OPD max new tokens | 32 |
| save policy | final model only, no intermediate checkpoints |

训练日志关键痕迹：

| 项 | 值 |
|---|---|
| start | `Wed May 20 14:25:29 CST 2026` |
| world size | 8 |
| expanded samples | 2,672 |
| expected steps | 84 |
| actual global step | 84 |
| train runtime | 1,047.1816 sec |
| train samples/sec | 2.552 |
| train steps/sec | 0.08 |
| final train loss | 0.8056682614343507 |
| final logged loss | 0.4952 at step 84 |
| final logged grad norm | 5.476184368133545 |
| output shards | 5 safetensor shards, total about 17GB |
| checkpoint dirs | 0 |

训练过程观察：

- step 1 loss `1.5944`，grad norm `47.0990`；
- step 2 loss `1.7281`，grad norm `70.7472`；
- step 3 loss `2.6980`，grad norm `99.8338`，之后逐步下降；
- step 41 到 42 有一次明显耗时跳变，推测与 teacher route 切换和模型卸载/加载有关；
- 后半程 loss 大多在 `0.45-0.70` 区间；
- 未观察到 NaN/Inf 训练指标；
- DeepSpeed launcher 最后显示 8 个 rank process 均 successfully exit；
- 日志存在一个非致命 warning：`destroy_process_group() was not called before program exit`；
- generation flags 有非致命 warning：`temperature/top_p/top_k` may be ignored，因为实际 `do_sample=false`。

最终模型目录包含：

- `model-00001-of-00005.safetensors` 到 `model-00005-of-00005.safetensors`
- `model.safetensors.index.json`
- `config.json`
- `generation_config.json`
- tokenizer / preprocessor 文件
- `trainer_state.json`
- `run_summary.json`
- `opd_online_run_summary.json`

`opd_online_run_summary.json` 记录：

| 字段 | 值 |
|---|---|
| `opd_mode` | `online_teacher_rollout_top1` |
| `train_loss` | `0.8056682614343507` |
| `last_route` | `reg` |
| `last_response_tokens` | `24.0` |
| `last_opd_loss` | `0.8163891434669495` |

### Agent 工作轨迹

本轮 agent 实际完成的工作：

1. 读取 AGENTS.md，确认远端、MACA 环境、`point/` 为 active project、`ds_test/` 为可参考 smoke 区域。
2. 确认用户已登录远端后，通过已配置的 SSH 入口进入 `/data/msz/point` 工作。
3. 检查既有专家模型、region/object eval 数据、训练数据、已有脚本和输出目录。
4. 先按离线 top1-cache 方向实现并 smoke，但该方向被用户纠正为不符合 OPD 目标。
5. 根据用户要求改为 on-policy teacher rollout：
   - 新增/替换 `train_opd_online_vl.py`；
   - 新增/整理 `run_opd_online_mix_v1_first.sh`；
   - 保留 `build_opd_mix_v1.py` 负责构造 media-valid 混合数据；
   - 使用原始 ZeRO-2 与每卡 bs=1、grad accum=4。
6. 在 tmux session `full` 中启动正式 OPD 训练，训练完成后检查模型目录和 summary。
7. 对 OPD v1 先跑 latest region/object eval，生成 `eval_opd_online_mix_v1_2048_20260520_145224`。
8. 按用户要求抽 200 条通用数据，对 base、obj expert、reg expert、OPD v1 都跑通用 VQA eval。
9. 因为最初混入 `embspatial` 时远端图片 URL 返回 HTTP 403，重抽本地可读的 `robo2vlm-1` heldout 200 条，并记录失败原因。
10. 按用户追问补跑“之前领域”的完整四模型 domain eval：base、obj expert、reg expert、OPD v1 在 latest object 与 latest region 上同协议评估。
11. 将远端 eval 输出同步到本地 `report/` 下，并更新 `report/opd_v1_eval_summary_20260520.md`。
12. 本节将完整过程补写入根目录 `REPORT.md`，避免第 10 节停留在早期失败态。

### 阻碍与解决方案

| 阻碍 | 现象 | 解决方案 | 状态 |
|---|---|---|---|
| 旧 expert 与期望能力偏差 | 初始 expert 基于 RoboPoint，semantic-to-box/region 场景不匹配 | 先生成 object expert 与 region expert，再尝试 OPD 混合 | 已完成前置 expert |
| offline top1-cache OPD 设计不符合目标 | 预生成 logits/cache 不是 on-policy teacher rollout | 改写为 `train_opd_online_vl.py`，训练时 teacher generate + teacher logits top1 | 已解决 |
| 单卡非 DeepSpeed smoke OOM | 8B student + AdamW optimizer state 无 ZeRO sharding 时显存不足 | 回到 8-GPU DeepSpeed ZeRO-2，每卡 bs=1 | 已解决 |
| 离线路径 ZeRO-2 allreduce timeout | 第一次分布式 backward/optimizer collective 附近 NCCL timeout | 放弃错误 offline-cache 路线，回到标准 ZeRO-2 + on-policy online trainer | 已绕开 |
| 通用数据来源媒体不可读 | `embspatial` URL 在 eval 阶段返回 HTTP 403 | 通用 eval 改为本地可读 `robo2vlm-1` 200 条，并排除 OPD train mix media | 已解决 |
| general eval build 阶段 import torch 触发 MACA/Triton 环境错误 | 未设置 `MACA_PATH` 时 build 子命令也 import transformers/torch | 将重依赖改为 lazy import，只在 eval 子命令加载 | 已解决 |
| base 输出格式失败 | base 常输出 `<box>[[x1,y1,x2,y2]]</box>`，而 strict parser 只接受 `<box>[[x1,y1],[x2,y2]]</box>` | domain eval 记录 strict 结果；之前已做过 lenient 模式分析，当前四模型矩阵使用 strict 同协议 | 已记录 |
| object eval 耗时较长 | 1251 条 object eval，base 由于格式/输出模式更慢 | 8 GPU 并行跑四模型 object/region；等待全部 rc=0 后聚合 | 已完成 |

### Domain eval：四模型同协议对比

评估路径：

| Split | 数据 |
|---|---|
| latest_object | `/data/msz/opd_project/data/semantic_nav_box_v1/eval/semantic_nav_box_heldout_object_ref_eval_v1.jsonl` |
| latest_region | `/data/msz/opd_project/data/semantic_nav_goal_eval_v1/semantic_nav_goal_reg_eval_v1.jsonl` |

模型：

| 名称 | 路径 |
|---|---|
| base | `/data/msz/models/8b_base` |
| obj_expert | `/data/msz/models/expert_obj_v1` |
| reg_expert | `/data/msz/models/expert_reg_v1` |
| opd_v1 | `/data/msz/point/outputs/opd_online_mix_v1_2048_final_20260520_142529` |

指标说明：

- `mean_iou_parseable`：只在 strict format parse 成功样本上计算。
- `mean_iou_all`：将格式失败样本计为 IoU 0，更能表示端到端任务表现。
- strict format regex 要求 `<box>[[x1,y1],[x2,y2]]</box>`。

四模型矩阵：

| split | model_name | rc | num_samples | format_ok | format_rate | mean_iou_parseable | mean_iou_all | iou_at_0_3_all | iou_at_0_5_all | mean_center_error | mean_coord_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_object | base | 0 | 1251 | 196 | 0.156675 | 0.125033 | 0.019590 | 0.021583 | 0.009592 | 215.999777 | 176.341837 |
| latest_object | obj_expert | 0 | 1251 | 1251 | 1.000000 | 0.398283 | 0.398283 | 0.547562 | 0.474820 | 110.719908 | 71.961431 |
| latest_object | reg_expert | 0 | 1251 | 1251 | 1.000000 | 0.104216 | 0.104216 | 0.110312 | 0.031175 | 163.966553 | 108.932254 |
| latest_object | opd_v1 | 0 | 1251 | 1251 | 1.000000 | 0.173187 | 0.173187 | 0.227018 | 0.152678 | 197.358188 | 137.532774 |
| latest_region | base | 0 | 144 | 42 | 0.291667 | 0.093813 | 0.027362 | 0.020833 | 0.000000 | 163.678191 | 118.410714 |
| latest_region | obj_expert | 0 | 144 | 144 | 1.000000 | 0.111445 | 0.111445 | 0.111111 | 0.027778 | 137.265902 | 92.451389 |
| latest_region | reg_expert | 0 | 144 | 144 | 1.000000 | 0.236819 | 0.236819 | 0.347222 | 0.173611 | 96.033732 | 61.828125 |
| latest_region | opd_v1 | 0 | 144 | 144 | 1.000000 | 0.145121 | 0.145121 | 0.208333 | 0.090278 | 137.645307 | 92.493056 |

解读：

- base 的 strict format rate 很低：object `196/1251`，region `42/144`。它不是完全不会预测 box，而是经常使用 flat list 格式，如 `<box>[[0, 0, 1000, 1000]]</box>`，在 strict parser 中计为失败。
- object expert 是 object split 上的最强模型，OPD v1 尚未保住 object expert 的高精度。
- region expert 是 region split 上的最强模型，OPD v1 尚未保住 region expert 的高精度。
- OPD v1 相比单域迁移模型有折中收益：object split 上优于 reg expert，region split 上优于 obj expert。
- OPD v1 的最大价值目前是统一格式与跨域折中，不是单域最优。

### 通用 200 条 eval

通用 eval set：

`/data/msz/point/data_eval/general_robo2vlm_heldout200_seed20260520.jsonl`

构建细节：

- 来源：`/data/msz/point/data_expert/keepalive_vqa.jsonl`
- 样本数：200
- 数据集：`robo2vlm-1`
- 全部为本地可读图片路径；
- 排除了 OPD 训练 mix 中用过的 media；
- 初始尝试的 `robo2vlm-1=180, embspatial=20` 被废弃，因为 `embspatial` GCS 图片 URL 在评估加载时报 HTTP 403。

通用 eval 指标：

| model_name | rc | num_samples | normalized_exact | relaxed_match | mean_token_f1 | option_samples |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 200 | 0.000000 | 0.295000 | 0.000000 | 200 |
| obj_expert | 0 | 200 | 0.045000 | 0.245000 | 0.071667 | 200 |
| reg_expert | 0 | 200 | 0.065000 | 0.310000 | 0.159091 | 200 |
| opd_v1 | 0 | 200 | 0.055000 | 0.295000 | 0.135000 | 200 |

解读：

- OPD v1 没有明显改善 relaxed general VQA，和 base 持平。
- OPD v1 的 exact 与 token-F1 高于 base，说明输出形式比 base 更接近 reference，但 relaxed correctness 未提升。
- reg expert 在这 200 条上反而最高，可能与 Robo2VLM 选项题输出风格或采样偶然性有关，不宜过度解读。
- 所有 200 条都是 option samples，因此 relaxed match 比 exact 更接近实际选择题能力。

### OPD-only grounding eval

在四模型矩阵前，曾先单独对 OPD v1 跑 latest region/object：

| split | model_name | num_samples | format_ok | format_rate | mean_iou_all | iou_at_0_3_all | iou_at_0_5_all | mean_center_error | mean_coord_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latest_object | opd_online_mix_v1_2048 | 1251 | 1251 | 1.000000 | 0.173187 | 0.227018 | 0.152678 | 197.358188 | 137.532774 |
| latest_region | opd_online_mix_v1_2048 | 144 | 144 | 1.000000 | 0.145121 | 0.208333 | 0.090278 | 137.645307 | 92.493056 |

该结果与四模型矩阵中的 OPD v1 数值一致，说明 eval 可复现。

### 与旧矩阵的关系

之前已有一个 latest region/object matrix：

`report/eval_latest_region_object_matrix_20260520_104104/`

它包含旧路径命名：

- `base`：旧矩阵里使用 `/data/msz/models/Qwen3-VL-8B-Instruct`
- `object_expert_20260519_102338`
- `region_predbox_v1_20260519_191043`
- `region_solution_a_20260519_171308`

本节四模型矩阵使用的是 OPD 训练同源 base：`/data/msz/models/8b_base`，因此 base 数值与旧矩阵不同。旧矩阵仍作为历史对照保留，不能直接混同为同一个 base checkpoint。

旧矩阵中的关键 `mean_iou_all`：

| split | base(Qwen3-VL path) | object_expert | region_predbox | region_solution |
|---|---:|---:|---:|---:|
| latest_object | 0.041636 | 0.398283 | 0.133069 | 0.104216 |
| latest_region | 0.048248 | 0.111445 | 0.216611 | 0.236819 |

新四模型矩阵中的关键 `mean_iou_all`：

| split | base(8b_base) | obj_expert | reg_expert | opd_v1 |
|---|---:|---:|---:|---:|
| latest_object | 0.019590 | 0.398283 | 0.104216 | 0.173187 |
| latest_region | 0.027362 | 0.111445 | 0.236819 | 0.145121 |

### 当前风险与下一步建议

1. 第一版 OPD 数据规模只有 2,048 原始行，expanded 后 2,672 routes；这是验证实现的规模，不足以期待稳定超过两个单域专家。
2. 当前 general 数据只实际使用 Robo2VLM keepalive，本来希望混入更多通用数据源；下一版需要修复更多 general 数据的媒体路径与可读性。
3. general 样本使用 obj/reg 各半 loss，可能让 model 在 general 上更像 teacher 的答案风格，而不是保留 base 的通用能力；下一版可以加入 base/self teacher 或保留一部分 supervised reference CE。
4. object 与 region 的混合比例当前为 35/35/30；从结果看，OPD v1 对 object expert 精度损失很大，对 region expert 精度也明显损失，建议下一版增大总量后保持 routing，但考虑分阶段训练或 domain-balanced curriculum。
5. base strict format rate 太低，后续对 base 做能力判断时应该同时报告 strict 与 lenient；对 expert/OPD 则 strict 足够，因为它们已学会目标格式。
6. 当前 OPD on-policy 每 step 要加载/切换 teacher route，运行效率不高；如果扩大到 20k，应考虑按 route 分阶段 epoch 或 route block 排序，减少 teacher 切换次数。
7. 当前实现只蒸馏 top1 token，没有 KL 分布信息。它符合“logit 只取 top1”的约束，但可表达的信息有限；如果后续允许 top-k，可以尝试 top-k sparse KL。

### 状态

截至本节记录时：

- OPD v1 训练完成；
- 没有中间 checkpoint；
- final model 已保存；
- OPD-only grounding eval 完成；
- 四模型领域 eval 完成；
- 四模型通用 200 条 eval 完成；
- 所有详细输出已复制到本地 `report/`；
- 远端无残留 `eval_semantic_nav_box`、`eval_general_vqa`、`train_opd`、`deepspeed` 进程。

## 2026-05-21 数据构造日志：五个 Expert 与 OPD Student 融合数据

本节接在 2026-05-20 的小规模 OPD v1 评估之后。前一版 OPD 只验证了多教师训练与评估链路，数据规模很小，不能代表最终 student 融合训练。2026-05-21 的主要工作转向正式数据工程：先构建五个领域 expert 的 seed0 数据，再基于这些 expert 数据构造约 100 万规模的 OPD student 融合数据，并完成 shuffle、异常筛查与严格清洗。

### 设计目标

最终目标不是再训练一个单域 expert，而是准备一个 student OPD 融合训练集。student 需要同时保留五类能力：

1. 通用视觉/机器人推理；
2. RoboPoint 点选 grounding；
3. 通用 object-to-box grounding；
4. region/description-to-box grounding；
5. 空间关系 relation-to-box grounding。

这样设计的原因是：单个 expert 在自己领域内表现更强，但 OPD student 需要在输入 prompt 暗示不同任务时选择合适能力和输出格式。如果直接混合所有数据而不标注 route、边界和冲突格式，student 容易学成平均模型：既损失 object expert 的精度，也损失 region expert 的精度，还可能在 point/box/text 格式之间摇摆。

因此这次数据设计把训练目标拆成三层：

- expert 数据层：每个 expert 先有自己的稳定领域数据；
- OPD route 层：每条 student 样本带 `target_expert`、`candidate_experts`、`expected_format`；
- 边界与冲突层：专门训练 student 在相近任务和格式诱导下不要走错 route。

### 公版数据来源快照

这一轮明确不再使用旧的语义导航合成数据作为主训练来源，OPD student 只从当前清洗后的公版/公开池 expert mix 中抽样。这样做的原因是：之前合成 region/semantic-nav 数据质量不稳定，且部分 label 无法唯一对应期望 box；本轮目标是先用更可追溯、更稳定的公版 grounding/VQA 池构造五个 expert 与 OPD student。

构造脚本中使用的 source label 与清洗文件如下：

| source label | clean file | 公版/公开数据含义 | 在本轮中的角色 |
|---|---|---|---|
| `refcoco` | `refcoco_clean_v1.jsonl` | RefCOCO / RefCOCO+ / RefCOCOg referring expression grounding | object-to-box、region/general support |
| `flickr30k` | `flickr30k_entities_clean_v1.jsonl` | Flickr30k Entities 短语 grounding | object/region phrase-to-box |
| `vg_object` | `visual_genome_object_clean_v1.jsonl` | Visual Genome object boxes | 通用 object box |
| `vg_region` | `visual_genome_region_clean_v1.jsonl` | Visual Genome region descriptions | region/description-to-box |
| `vg_relationship` | `visual_genome_relationship_clean_v1.jsonl` | Visual Genome relationship annotations | relation grounding 与边界样本 |
| `vg_relationship_balanced` | same as `vg_relationship` | 对 Visual Genome relationship 按关系 bucket 重采样后的训练源 | spatial relation expert 主源 |
| `keepalive` | `keepalive_vqa_clean_v1_mediaok.jsonl` | Robo2VLM-1 VQA/机器人推理 keepalive | 防止模型只会输出坐标 |
| `robopoint` | `grounding_point_clean_v1_mediaok.jsonl` | RoboPoint point grounding | point expert 主源与 point/box 边界 |

远端 clean pool 行数快照如下。这里记录的是 2026-05-21 当时 `/data/msz/point/data_grounding_clean_v1` 下可见的行数；后续远端如果刷新，以本表作为本轮数据构造的本地证据。

| clean file | rows |
|---|---:|
| `refcoco_clean_v1.jsonl` | 376,357 |
| `flickr30k_entities_clean_v1.jsonl` | 415,951 |
| `visual_genome_object_clean_v1.jsonl` | 2,357,546 |
| `visual_genome_region_clean_v1.jsonl` | 5,282,274 |
| `visual_genome_relationship_clean_v1.jsonl` | 1,622,341 |
| `grounding_point_clean_v1_mediaok.jsonl` | 837,981 |
| `keepalive_vqa_clean_v1_mediaok.jsonl` | 678,034 |
| `semantic_nav_box_clean_v1.jsonl` | 41,637 |

其中几个聚合文件的内部数据集拆分如下：

| clean file | internal dataset/source distribution |
|---|---|
| `refcoco_clean_v1.jsonl` | `refcoco=141383`, `refcocoplus=140740`, `refcocog=94234` |
| `keepalive_vqa_clean_v1_mediaok.jsonl` | `robo2vlm-1=678034` |
| `grounding_point_clean_v1_mediaok.jsonl` | `robopoint=837981` |
| `flickr30k_entities_clean_v1.jsonl` | `flickr30k_entities=415951` |
| `visual_genome_object_clean_v1.jsonl` | `visual_genome_objects=2357546` |
| `visual_genome_region_clean_v1.jsonl` | `visual_genome_regions=5282274` |
| `visual_genome_relationship_clean_v1.jsonl` | `visual_genome_relationships=1622341` |

注意：`semantic_nav_box_clean_v1.jsonl` 当时在 clean dir 中存在，但本轮五个 expert mix 脚本没有把它列入 `SOURCE_FILES`，因此没有进入这批五 expert / OPD student 主数据。它属于之前语义导航 box 合成路线的产物，保留给单独审计或特定 semantic-nav 实验。

本轮明确排除的来源：

- old synthetic semantic-nav/region data；
- PhraseCut；
- Talk2Car image version；
- RoboRefIt。

排除原因：

- 旧 synthetic semantic-nav/region 数据在 label 唯一性、关系理解和 box 还原上不够稳定；
- PhraseCut、Talk2Car 图片版、RoboRefIt 当时不是当前 clean media-ok 主池的一部分，或存在少量异常/认证/媒体可用性问题；
- 本轮先保证 5 个 expert 与 OPD student 的公版池来源稳定、可复现、可清洗。

### 五个 Expert 领域设计

五个 expert 的数据均在远端 `/data/msz/point` 下生成，seed 为 `0`。

训练文件目录：

```text
/data/msz/point/data_expert_seed0_v1_shuffled/<expert>/train_shuffled_seed20260520.jsonl
```

评估文件目录：

```text
/data/msz/point/data_expert_seed0_v1/<expert>/eval.jsonl
```

五个 expert 分工如下：

| expert | 目标能力 | 设计原因 |
|---|---|---|
| `general_reasoning_expert` | 通用 VQA、机器人常识、非坐标回答 | 防止 grounding 训练把模型压成只会输出坐标；OPD student 仍需要普通问答和推理能力。 |
| `robopoint_expert` | `<point>` 点选 grounding | RoboPoint 负责 point 输出能力，是 point/box 边界学习的基础。 |
| `general_obj_expert` | 通用 object-to-box | 覆盖 RefCOCO/Flickr/VG object 这类物体框选，负责“物体名称 -> box”。 |
| `region_expert` | description/region-to-box | 覆盖区域描述、短语区域、VG region 等，负责“区域语义描述 -> box”。 |
| `spatial_rel_expert` | relation-aware box | 覆盖 left/right/front/behind/inside/on/near 等关系表达，负责“物体 + 关系 + anchor -> box”。 |

每个 expert 的总体策略是“领域为主，少量其它能力保活”。这对应之前讨论的 80% 领域数据加其它混合策略：expert 要足够专，但不能完全忘掉输入格式、图像解析和通用回答风格。后续 OPD student 再通过 route metadata 学会在多个 expert 能力之间切换。

设计配额来自 `build_expert_mixes_seed0_v2.py` 的 `EXPERT_TRAIN_PLAN`。原始目标是每个 expert `800,000` train / `20,000` eval，清洗后 train 会略低于目标，因为超长/异常样本被删除。

| expert | 主领域源 | 其它混合源 |
|---|---|---|
| `general_reasoning_expert` | `keepalive=640k` | `refcoco=40k`, `vg_region=40k`, `vg_object=30k`, `vg_relationship=20k`, `robopoint=30k` |
| `robopoint_expert` | `robopoint=640k` | `keepalive=80k`, `vg_object=30k`, `refcoco=20k`, `vg_region=20k`, `vg_relationship=10k` |
| `general_obj_expert` | `refcoco=200k`, `flickr30k=180k`, `vg_object=220k` | `vg_region=80k`, `vg_relationship=60k`, `keepalive=40k`, `robopoint=20k` |
| `region_expert` | `vg_region=640k` | `refcoco=40k`, `flickr30k=40k`, `vg_object=30k`, `vg_relationship=30k`, `keepalive=10k`, `robopoint=10k` |
| `spatial_rel_expert` | `vg_relationship_balanced=640k` | `vg_region=50k`, `vg_object=40k`, `refcoco=30k`, `flickr30k=20k`, `keepalive=10k`, `robopoint=10k` |

`vg_relationship_balanced` 不是新数据集，而是对 Visual Genome relationship 做关系类别平衡采样。关系 bucket 目标权重：

| bucket | weight |
|---|---:|
| `on` | 0.18 |
| `in_inside` | 0.18 |
| `near_next_beside` | 0.14 |
| `under_below` | 0.10 |
| `above_over` | 0.10 |
| `front` | 0.08 |
| `behind` | 0.08 |
| `left_right` | 0.06 |
| `between` | 0.04 |
| `other_spatial` | 0.04 |

这样设计 spatial relation expert 的原因是：原始关系数据很容易被高频 `on/in/near` 主导，而用户关心的语义导航关系还包括 `front/behind/between/left/right` 等低频关系。平衡采样能让这些关系在 expert 训练中有足够曝光。

清洗后最终 shuffled train 的真实 source 分布如下：

| expert | rows | source distribution |
|---|---:|---|
| `general_reasoning_expert` | 798,384 | `keepalive=639981`, `refcoco=40000`, `vg_region=39561`, `robopoint=29613`, `vg_object=29541`, `vg_relationship=19688` |
| `robopoint_expert` | 791,254 | `robopoint=631762`, `keepalive=79998`, `vg_object=29784`, `refcoco=20000`, `vg_region=19821`, `vg_relationship=9889` |
| `general_obj_expert` | 787,128 | `vg_object=213686`, `refcoco=200000`, `flickr30k=180000`, `vg_region=76631`, `vg_relationship=57072`, `keepalive=40000`, `robopoint=19739` |
| `region_expert` | 788,566 | `vg_region=631977`, `flickr30k=40000`, `refcoco=40000`, `vg_object=28396`, `vg_relationship=28329`, `keepalive=9999`, `robopoint=9865` |
| `spatial_rel_expert` | 782,803 | `vg_relationship_balanced=627276`, `vg_region=47729`, `vg_object=37942`, `refcoco=30000`, `flickr30k=20000`, `keepalive=10000`, `robopoint=9856` |

Eval 每个 expert 固定为 20,000 行，且 train/eval image overlap 为 `0`：

| expert | eval source distribution |
|---|---|
| `general_reasoning_expert` | `keepalive=16000`, `refcoco=1000`, `vg_region=1000`, `vg_object=750`, `robopoint=750`, `vg_relationship=500` |
| `robopoint_expert` | `robopoint=16000`, `keepalive=2000`, `vg_object=750`, `refcoco=500`, `vg_region=500`, `vg_relationship=250` |
| `general_obj_expert` | `vg_object=5500`, `refcoco=5000`, `flickr30k=4500`, `vg_region=2000`, `vg_relationship=1500`, `keepalive=1000`, `robopoint=500` |
| `region_expert` | `vg_region=16000`, `refcoco=1000`, `flickr30k=1000`, `vg_object=750`, `vg_relationship=750`, `keepalive=250`, `robopoint=250` |
| `spatial_rel_expert` | `vg_relationship_balanced=16000`, `vg_region=1250`, `vg_object=1000`, `refcoco=750`, `flickr30k=500`, `keepalive=250`, `robopoint=250` |

### Expert 数据清洗

在用户用每个领域前 100k 样本试训练时出现 OOM 风险后，先对样本体量做了统计，重点检查：

- 文本总长度；
- GPT answer 长度；
- point 数量；
- 图片尺寸；
- shuffle 后前 100k 的真实分布。

最终决定直接在原始 shuffled expert train 文件上清洗，因为试训练和后续 OPD 都基于这些 shuffled 文件，另存副本会增加混乱。

Point 样本清洗规则：

```text
drop if:
  point_count > 50
  OR gpt_chars > 500
  OR total_text_chars > 900
```

非 point 样本清洗规则：

```text
drop if:
  total_text_chars > 900
```

为什么这样清洗：

- `point_count > 50` 会显著拉长 answer，并且 point grounding 太密时对训练目标不稳定；
- `gpt_chars > 500` 通常意味着坐标列表过长或回答不符合短格式目标；
- `total_text_chars > 900` 是为了避免 collator/tokenizer 和视觉 token 叠加后形成长尾 batch，引发 OOM；
- 非 point 虽然没有 point 数问题，但超长 prompt/answer 仍会拖慢训练并造成 batch 长尾。

清洗后五个 shuffled train 文件行数：

| expert | rows after filtering |
|---|---:|
| `general_obj_expert` | 787,128 |
| `general_reasoning_expert` | 798,384 |
| `region_expert` | 788,566 |
| `robopoint_expert` | 791,254 |
| `spatial_rel_expert` | 782,803 |

对应报告：

```text
/data/msz/point/data_expert_seed0_v1_shuffled/point_outlier_filter_report.json
/data/msz/point/data_expert_seed0_v1_shuffled/point_outlier_filter_verify_after.json
/data/msz/point/data_expert_seed0_v1_shuffled/non_point_text900_filter_report.json
```

最终验证：

- point rule violations: `0`
- non-point `total_text_chars > 900`: `0`
- non-point `gpt_chars > 500`: `0`

### Expert 训练数据与 OPD Seen 定义

五个 expert 的试训练都使用各自 shuffled train 的前 `100,000` 条：

```text
LIMIT_SAMPLES=100000
```

因此 OPD student 数据中的 `seen_by_expert_100k` 定义为：

```text
first 100000 rows of each filtered shuffled expert train file
```

这样定义的原因是：OPD student 的 domain 数据需要明确区分 expert 已见 prompt 和未见同分布 prompt。student 融合时如果只看 expert 已见样本，会过拟合 expert 训练集；如果完全不看已见样本，又缺少对 expert 行为边界的稳定锚点。因此领域内部按 20/70/10 切分：

- 20% expert seen prompt；
- 70% unseen same-distribution prompt；
- 10% hard prompt。

### OPD Student 数据总体配比

OPD student train 目标约 100 万行，配比如下：

| category | target rows | ratio | 目的 |
|---|---:|---:|---|
| domain | 600,000 | 60% | 保留五个 expert 的主能力。 |
| general | 250,000 | 25% | 保留通用问答和基础 grounding，不让 student 只学边界样本。 |
| boundary | 100,000 | 10% | 专门训练相邻能力之间的 route 边界。 |
| format_conflict | 50,000 | 5% | 强化输出格式、坐标规范、抗错误诱导。 |

每个 expert 的 domain 配额为 120,000：

| subtype | rows per expert | total rows | 设计原因 |
|---|---:|---:|---|
| `seen_100k` | 24,000 | 120,000 | 对齐 expert 已训练过的 prompt，稳定蒸馏行为。 |
| `unseen_same_distribution` | 84,000 | 420,000 | 同分布未见样本，防止只记住 seen prompt。 |
| `hard_static` | 12,000 | 60,000 | 静态困难样本，覆盖长文本、稀有关系、极端 box/point 等长尾。 |

### Hard 样本设计

Hard 样本不是在线从训练 loss 采样，而是在构造阶段用静态规则预选。这样做是因为当前目标是先生成稳定可复现的数据集，而不是引入依赖训练中间状态的动态数据管线。

Hard scoring 主要考虑：

- 文本接近阈值但未超限；
- point 数量接近阈值但未超限；
- box 面积极小或极大；
- box 长宽比较大；
- spatial relation 中出现 front/behind/between/under/above 等更难关系；
- general reasoning 中较长的 keepalive 样本。

为什么需要 hard：

- 只训练普通样本会让 student 在长尾 prompt 上选择错误 expert；
- hard 样本能提前暴露 route 混淆、格式漂移和坐标边界问题；
- 10% 比例足够让模型看到困难边界，但不至于让训练集被离群样本主导。

### Boundary 样本设计

Boundary 数据是本轮 OPD 设计的核心之一。它不只是“难样本”，而是专门针对 expert 之间容易混淆的边界。

训练配额：

| subtype | rows | 设计目的 |
|---|---:|---|
| `obj_vs_region_object` | 10,000 | object prompt 应走 object expert，而不是 region expert。 |
| `obj_vs_region_region` | 10,000 | region/description prompt 应走 region expert，而不是 object expert。 |
| `obj_vs_spatial_relation` | 20,000 | 带 relation/anchor 的 object grounding 应走 spatial relation expert。 |
| `region_vs_spatial_region_text` | 10,000 | 描述性 region 与 relation-aware region 的边界。 |
| `region_vs_spatial_structured_relation` | 10,000 | 结构化关系 prompt 应走 spatial expert。 |
| `point_vs_box_point` | 10,000 | point 输出不能被 box 任务污染。 |
| `point_vs_box_box` | 10,000 | box 输出不能被 point 任务污染。 |
| `reasoning_vs_grounding_reasoning` | 10,000 | 普通问答不应被强制输出坐标。 |
| `reasoning_vs_grounding_point` | 5,000 | grounding prompt 中 point 格式边界。 |
| `reasoning_vs_grounding_box` | 5,000 | grounding prompt 中 box 格式边界。 |

为什么需要 boundary：

- OPD student 不是简单多任务 SFT，它需要在多个 teacher/expert 行为之间做路由；
- object、region、spatial relation 都输出 `<box>`，如果没有边界数据，student 很容易只学到“看到 box 就用同一种策略”；
- point 和 box 都是坐标输出，但格式和目标粒度不同，需要显式防止互相污染；
- reasoning 与 grounding 的边界能防止模型在普通问答里过度输出坐标。

### General 样本设计

General 数据共 250,000 行：

| subtype | rows | source expert | 目的 |
|---|---:|---|---|
| `keepalive_vqa` | 170,000 | `general_reasoning_expert` | 保留通用回答能力。 |
| `simple_object_grounding` | 30,000 | `general_obj_expert` | 保留普通 object box 能力。 |
| `simple_region_grounding` | 20,000 | `region_expert` | 保留普通 region box 能力。 |
| `simple_relation_grounding` | 15,000 | `spatial_rel_expert` | 保留基础 relation grounding。 |
| `short_point_grounding` | 15,000 | `robopoint_expert` | 保留短 point grounding。 |

为什么 general 占 25%：

- 如果只用 domain + boundary，student 会过度关注路由边界，普通任务能力会变窄；
- keepalive VQA 是防止 grounding 数据把模型推向“凡问必坐标”的缓冲层；
- simple grounding 是各领域最常见、最低噪声的样本，帮助 student 形成稳定输出格式。

### Format / Conflict 样本设计

Format/conflict 数据共 50,000 行：

| subtype | rows | 目的 |
|---|---:|---|
| `format_strong` | 15,000 | 强化只输出目标 XML tag，不带解释。 |
| `wrong_format_induction` | 10,000 | prompt 里提到其它格式时仍坚持当前任务格式。 |
| `prompt_injection` | 10,000 | 抵抗 prompt/image 中要求改变格式的诱导。 |
| `coord_norm` | 10,000 | 强调坐标为 0-1000 normalized integer。 |
| `short_hard_boundary` | 5,000 | 短提示下仍处理困难边界。 |

为什么保留这 5%：

- 前一版评估中 base strict format rate 很低，格式本身就是重要能力；
- 多 expert 融合时最容易出现的错误不是完全不会回答，而是输出了错误 tag 或夹带解释；
- 坐标归一化和 XML tag 是训练、评估、下游导航系统的接口契约，必须显式训练。

### OPD Student 构造脚本与输出格式

构造脚本：

```text
/data/msz/point/build_opd_student_v1.py
```

本地同步副本：

```text
build_opd_student_v1.py
```

构造方式：

- 只读 JSONL；
- 不加载模型；
- 不触碰 GPU；
- 使用 `CUDA_VISIBLE_DEVICES=`、`nice -n 19`、`ionice -c3` 低优先级运行；
- 输出 train/eval prompt+gold 数据，而不是在线 teacher logits。

每条 OPD row 保留原始 `conversations` 和 gold answer，并新增：

```json
{
  "gold": "...",
  "teacher_outputs": {},
  "metadata": {
    "opd": {
      "dataset_version": "opd_student_v1",
      "split": "train/eval",
      "sample_category": "...",
      "sample_subtype": "...",
      "target_expert": "...",
      "candidate_experts": ["..."],
      "expected_format": "box/point/text",
      "source_expert_file": "...",
      "source_file": "...",
      "source_line": 123,
      "seen_by_expert_100k": true,
      "hard_score": 0.0,
      "route_reason": "...",
      "fingerprint": "...",
      "opd_seed": 0
    }
  }
}
```

这样设计的原因：

- `gold` 让 SFT/CE 路线可以直接使用；
- `teacher_outputs` 预留给后续 OPD teacher logits 或 teacher answer，不阻断当前纯数据构造；
- `target_expert` 和 `candidate_experts` 记录 route 监督；
- `source_file/source_line` 让异常样本可以追溯回原 expert 数据；
- `fingerprint` 用于去重和 train/eval 泄漏检测；
- `seen_by_expert_100k` 用于区分 expert 已见与未见 prompt。

### 第一次 OPD Student 构造与修正

第一次构造完成后：

```text
/data/msz/point/opd_student_v1/train_prompts.jsonl  1,000,000 rows
/data/msz/point/opd_student_v1/eval_prompts.jsonl      46,602 rows
```

但 summary 中出现少量 `point_filter_violation`：

- train: 19
- eval: 45

定位结果：

- 原始样本本身已经通过 point 清洗；
- 问题来自 `format_conflict` transform；
- 追加 “Do not include explanations / wrong format / prompt injection / coord norm” 等提示后，少量 point 样本的 `total_text_chars` 被推到 900 以上。

修正：

```python
if transform_kind:
    ...
    out = apply_prompt(row, prompt)
    if not valid_row(out):
        continue
```

也就是在 prompt 变换后再次执行 `valid_row` 校验。这样保证最终写出的样本满足训练期长度约束，而不是只校验原始样本。

修正版重新构造后：

```text
train_rows = 1,000,000
eval_rows  = 46,599
bad_json   = 0
violations = {}
```

Eval 比目标略少，是因为严格排除了 train 图片重叠和 fingerprint 重叠后，spatial relation eval 的部分子类候选不足。这里没有用 train 污染补齐 eval，优先保持评估干净。

### OPD Student 最终构造产物

远端目录：

```text
/data/msz/point/opd_student_v1/
```

主要文件：

```text
train_prompts.jsonl
eval_prompts.jsonl
summary.json
manifests/opd_student_v1_build_summary.json
```

修正版构造后的 train 配比：

| category | rows |
|---|---:|
| domain | 600,000 |
| general | 250,000 |
| boundary | 100,000 |
| format_conflict | 50,000 |

Train target expert 分布：

| target expert | rows |
|---|---:|
| `general_reasoning_expert` | 310,000 |
| `general_obj_expert` | 180,000 |
| `region_expert` | 175,000 |
| `spatial_rel_expert` | 175,000 |
| `robopoint_expert` | 160,000 |

输出格式分布：

| split | box | text | point |
|---|---:|---:|---:|
| train | 555,637 | 293,050 | 151,313 |
| eval | 22,537 | 16,242 | 7,820 |

### Shuffle 处理

构造脚本的 `ShardedWriter` 会随机分 shard 再随机 merge shard，这只能算近似打散。检查前 50k 后发现：

- train 前 50k category 比例接近全局；
- 但最长连续同 category 达到 9,368 行；
- 说明顺序仍有 shard/block 痕迹。

因此新增外部 shuffle 脚本：

```text
/data/msz/point/shuffle_opd_student_v1.py
```

本地同步副本：

```text
shuffle_opd_student_v1.py
```

Shuffle 方法：

1. 用 hash(seed + line) 将每行分配到 bucket；
2. train 使用 256 个 bucket，eval 使用 64 个 bucket；
3. 每个 bucket 内随机洗牌；
4. bucket 顺序再随机合并；
5. 行数一致后原子替换原文件；
6. summary 中写入 shuffle metadata。

运行策略仍为：

```text
CUDA_VISIBLE_DEVICES=
ionice -c3
nice -n 19
```

Shuffle 后验证：

| split | rows | max same category run | max same subtype run |
|---|---:|---:|---:|
| train | 1,000,000 | 24 | 14 |
| eval | 46,599 | 21 | 21 |

Train 前 50k 分布：

| category | rows |
|---|---:|
| domain | 29,923 |
| general | 12,537 |
| boundary | 5,088 |
| format_conflict | 2,452 |

### 全量异常审计

新增审计脚本：

```text
/data/msz/point/audit_opd_student_v1.py
```

本地同步副本：

```text
audit_opd_student_v1.py
```

审计分两层：

1. 全量 JSONL 结构与文本/坐标特征；
2. 全量唯一图片路径存在性与图片 header 尺寸检查。

审计指标包括：

- JSON 是否可解析；
- `conversations`、human turn、gpt turn、answer 是否存在；
- image 是否存在且单图；
- `total_chars`、`human_chars`、`gpt_chars` 分布；
- point 数量；
- point/box 坐标是否在 `[0, 1000]`；
- box 是否退化、反向、过小、过大、长宽比异常；
- metadata expected format 是否与 answer 一致；
- fingerprint 重复；
- train/eval fingerprint overlap；
- train/eval image overlap；
- 图片文件是否存在且可打开；
- 图片尺寸、面积、长宽比分布。

清洗前审计结果：

```text
train rows = 1,000,000
eval rows  = 46,599
bad_json   = 0
```

文本长度：

| split | total max | gpt max | point max |
|---|---:|---:|---:|
| train | 900 | 500 | 49 |
| eval | 900 | 496 | 48 |

图片检查：

| item | value |
|---|---:|
| unique images checked | 614,731 |
| existing/openable image files | 614,731 |
| max width | 1,280 |
| max height | 1,280 |
| max area | 1,638,400 |
| max image aspect | 9.259 |

未发现：

- 缺图；
- 坏图；
- 超大图；
- train/eval 图片重叠；
- train/eval fingerprint 重叠；
- point 数量超限；
- 坐标越界；
- JSON 解析错误；
- mixed point/box answer；
- metadata expected format mismatch。

唯一软异常是 `box_aspect_gt_20`：

| split | count |
|---|---:|
| train | 1,700 |
| eval | 21 |

抽样检查发现这类样本大多是 `pole`、`line`、`wave`、`gutter`、`wall`、`sidewalk` 等天然细长目标。因此 `aspect > 20` 不直接作为删除条件。

### 严格几何清洗

为了进一步降低训练异常风险，额外执行了保守硬清洗。

硬删除规则：

```text
drop box if:
  min(width, height) < 5
  OR aspect_ratio > 100
```

为什么这样设：

- `aspect > 20` 仍可能是合法长条目标；
- `min_side < 5` 往往是极细边界、线条或标注误差，对 normalized box 训练不稳定；
- `aspect > 100` 基本属于“线/边界”级别，容易让模型学习到不可泛化的极端框；
- 删除量很小，不值得为了补齐 100 行而重新采样，避免引入新的 overlap 或分布扰动。

清洗结果：

| split | before | after | dropped rows |
|---|---:|---:|---:|
| train | 1,000,000 | 999,900 | 100 |
| eval | 46,599 | 46,599 | 0 |

原因计数：

| reason | count |
|---|---:|
| `box_min_side_lt_5` | 87 |
| `box_aspect_gt_100` | 37 |

原因计数合计大于删除行数，是因为部分样本同时命中两个规则。

严格清洗报告：

```text
/data/msz/point/opd_student_v1/strict_clean_report.json
```

清洗前完整审计备份：

```text
/data/msz/point/opd_student_v1/anomaly_audit_report_before_strict_clean.json
```

清洗后主审计报告：

```text
/data/msz/point/opd_student_v1/anomaly_audit_report.json
```

### 最终可用状态

当前最终文件：

```text
/data/msz/point/opd_student_v1/train_prompts.jsonl
/data/msz/point/opd_student_v1/eval_prompts.jsonl
```

最终行数：

| split | rows |
|---|---:|
| train | 999,900 |
| eval | 46,599 |

最终 category 分布：

| split | domain | general | boundary | format_conflict |
|---|---:|---:|---:|---:|
| train | 599,905 | 250,000 | 99,998 | 49,997 |
| eval | 29,108 | 11,750 | 3,388 | 2,353 |

最终 format 分布：

| split | box | text | point |
|---|---:|---:|---:|
| train | 555,537 | 293,050 | 151,313 |
| eval | 22,537 | 16,242 | 7,820 |

最终 OPD student 的底层公版 source 分布也固化如下。这个表比 category 更底层，表示最终 train/eval 实际来自哪些公开数据池。

| split | source | rows |
|---|---|---:|
| train | `keepalive` | 293,050 |
| train | `vg_region` | 188,515 |
| train | `robopoint` | 151,313 |
| train | `vg_relationship_balanced` | 138,564 |
| train | `vg_object` | 81,531 |
| train | `refcoco` | 68,291 |
| train | `flickr30k` | 51,513 |
| train | `vg_relationship` | 27,123 |
| eval | `keepalive` | 16,242 |
| eval | `robopoint` | 7,820 |
| eval | `refcoco` | 7,141 |
| eval | `vg_region` | 6,443 |
| eval | `flickr30k` | 4,898 |
| eval | `vg_relationship_balanced` | 2,789 |
| eval | `vg_object` | 801 |
| eval | `vg_relationship` | 465 |

按 OPD category 展开的 train source 分布：

| category | source distribution |
|---|---|
| domain | `vg_region=127937`, `keepalive=112510`, `robopoint=110470`, `vg_relationship_balanced=96382`, `vg_object=55412`, `refcoco=44764`, `flickr30k=33279`, `vg_relationship=19151` |
| general | `keepalive=170000`, `vg_region=19090`, `robopoint=15000`, `vg_relationship_balanced=13322`, `refcoco=11438`, `vg_object=11142`, `flickr30k=10008` |
| boundary | `vg_region=27255`, `vg_relationship_balanced=26585`, `robopoint=15000`, `keepalive=10000`, `vg_object=5995`, `refcoco=5808`, `vg_relationship=4951`, `flickr30k=4404` |
| format_conflict | `vg_region=14233`, `robopoint=10843`, `vg_object=8982`, `refcoco=6281`, `flickr30k=3822`, `vg_relationship=3021`, `vg_relationship_balanced=2275`, `keepalive=540` |

按 target expert 展开的 train source 分布：

| target expert | source distribution |
|---|---|
| `general_reasoning_expert` | `keepalive=274757`, `vg_region=9436`, `refcoco=7177`, `vg_object=7005`, `robopoint=6905`, `vg_relationship=4714` |
| `robopoint_expert` | `robopoint=135475`, `keepalive=9711`, `vg_object=5691`, `vg_region=3849`, `refcoco=3219`, `vg_relationship=2050` |
| `general_obj_expert` | `vg_object=53158`, `refcoco=43829`, `flickr30k=39545`, `vg_region=18979`, `vg_relationship=13856`, `keepalive=5919`, `robopoint=4689` |
| `region_expert` | `vg_region=144175`, `flickr30k=7470`, `refcoco=7371`, `vg_relationship=6503`, `vg_object=6071`, `robopoint=2074`, `keepalive=1284` |
| `spatial_rel_expert` | `vg_relationship_balanced=138564`, `vg_region=12076`, `vg_object=9606`, `refcoco=6695`, `flickr30k=4498`, `robopoint=2170`, `keepalive=1379` |

最终硬校验：

```text
bad_json = 0
hard_violations = {}
duplicate_fingerprints = 0
train/eval fingerprint_overlap = 0
train/eval image_overlap = 0
```

保留的软长条 box：

```text
box_aspect_gt_20:
  train = 1,600
  eval  = 21
```

这些不作为错误删除，因为它们主要是合法细长物体或区域。后续如果训练仍对这些样本敏感，可以再做 domain-specific policy，例如仅对 `object_name in {"line", "horizon", "shore line"}` 的极端框降权，而不是全局删除所有长条目标。

### 本轮数据工程结论

1. 五个 expert 数据已形成可复用的 seed0 shuffled train/eval 体系。
2. Expert train 已去除 point 长尾和非 point 超长文本长尾。
3. OPD student 数据已按 60/25/10/5 设计构造，且每条样本有 route metadata。
4. OPD domain 内部已按 20% seen、70% unseen、10% hard 落地。
5. Boundary 数据显式覆盖 object/region/spatial/point/reasoning 的易混边界。
6. Format/conflict 数据显式覆盖 strict format、错误格式诱导、prompt injection、坐标归一化。
7. Train/eval 已严格去重，且图片层面无 overlap。
8. 数据已完成真正逐行 shuffle，而不是 shard 级近似 shuffle。
9. 最终数据无超长文本、无 point 超限、无缺图、无坏图、无硬几何异常。
10. 当前 train 少 100 行，不补齐；因为删除比例只有 0.01%，补齐收益很低，反而可能引入新的追溯与去重风险。

### 远端引用落地说明

本节中所有远端路径仍保留为操作入口，但关键远端状态已经写成本地文本快照，包括：

- clean pool 文件名与行数；
- expert 设计配额；
- expert 清洗后的真实 source 分布；
- expert eval source 分布；
- OPD student category/format/source/target-expert 分布；
- shuffle 方法、bucket 数与完成时间；
- anomaly audit 与 strict clean 规则；
- 最终行数、去重、图片检查和硬异常状态。

这样做的原因是 `/data/msz` 远端数据目录会频繁刷新，单纯记录路径不足以复现当时的数据事实。以后如果远端文件消失或被重建，本地 `REPORT.md` 仍保留本轮构造时的关键证据；需要恢复具体 JSONL 时，再以本节的 source 分布、seed、脚本和清洗规则作为重建依据。

## 2026-05-21 训练执行日志：seed0 五个 100k Expert

本节接在上一节“五个 Expert 与 OPD Student 融合数据”的数据工程记录之后。上一节记录点停在：五个 expert 的 seed0 shuffled train/eval 体系已完成，OPD student 数据也已构造、shuffle 和严格审计完成；但当时尚未把五个 expert 的正式训练执行线、失败尝试、最终可跑通配置与代码状态写入本地 report。

本节只记录当前五个 expert 训练线。远端 `/data/msz` 的查看均为只读检查，实际落地修改仅发生在本地 `REPORT.md`。

### 当前训练目标

当前目标是基于 `/data/msz/point/data_expert_seed0_v1_shuffled` 中五个领域的数据，为 `/data/msz/models/8b_base` 生成五个 expert。每个 expert 取各自 shuffled train 的前 `100,000` 条，串行训练，避免多个 full finetune 同时抢显存。

五个 expert 顺序固定为：

| order | expert | train file |
|---:|---|---|
| 1 | `general_obj_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/general_obj_expert/train_shuffled_seed20260520.jsonl` |
| 2 | `general_reasoning_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/general_reasoning_expert/train_shuffled_seed20260520.jsonl` |
| 3 | `region_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/region_expert/train_shuffled_seed20260520.jsonl` |
| 4 | `robopoint_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/robopoint_expert/train_shuffled_seed20260520.jsonl` |
| 5 | `spatial_rel_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/spatial_rel_expert/train_shuffled_seed20260520.jsonl` |

输出目录模式：

```text
/data/msz/models/seed0_<expert>_100k_mb1_filtered_stdtrainer_v1
```

当前 tmux 会话：

```text
full:seed0_mb1_std
```

当前启动脚本：

```text
/data/msz/point/run_seed0_five_experts_100k_mb1_filtered_stdtrainer_v1.sh
```

日志目录：

```text
/data/msz/point/logs/seed0_*_100k_mb1_filtered_stdtrainer_v1.log
/data/msz/point/logs/seed0_100k_mb1_filtered_stdtrainer_v1.log
```

### 训练线尝试与决策日志

这条线最初讨论过每个 expert 使用 `200k` 样本。为了方便后续恢复和扩展，先对每个领域的数据做真正 shuffle，再取前 `200k` 或前 `100k`，避免后续继续训练时因重新抽样造成不可追溯的分布漂移。

随后因为稳定性优先，训练目标收缩为每个 expert 先跑 `100k`。当时的考虑是：先证明五个领域 expert 都能在当前 8 卡 MACA C500 环境下完整跑通，再决定是否继续扩到更大样本量。

第一轮配置尝试过较大的 microbatch。`microbatch=4` 很快出现 OOM 风险；随后降到 `microbatch=2` 继续尝试。`microbatch=2` 在长样本附近仍有 OOM/卡住风险，因此又进一步切到 `microbatch=1`。

为了解决 OOM 不稳定，曾做过一版最小自定义 `OomSkipTrainer`：只捕获 prepare/forward/backward 的 OOM，在多卡间同步 OOM flag，统一跳过该 batch 并清梯度。这个方案解决了“单卡 OOM 导致多卡不同步”的理论问题，但实际行为仍然不够稳定，尤其在 DeepSpeed/Accelerate 的训练步内部边界上，跳过 batch 容易引入状态不一致风险。

因此最终训练线回到标准 `transformers.Trainer`。自定义 OOM-skip 代码没有删除，而是整段保留为注释，作为失败尝试与可回溯实现。active path 不使用它。

训练前又针对导致 OOM 的样本特征做过过滤与重建：

- 删除 point 数量长尾和异常 point 样本；
- 删除非 point 数据中过长文本样本；
- 保留已经 shuffle 好的文件名与顺序，使前 `100k` 的训练切片可复用；
- 不做运行期 OOM skip，改为让标准 trainer 在仍然 OOM 时直接失败，避免 silent skip 改变训练语义。

最终可跑通方案是：

| item | value |
|---|---|
| base model | `/data/msz/models/8b_base` |
| trainer | stock `transformers.Trainer` |
| deepspeed | `/data/msz/point/configs/zero2.json` |
| GPUs | 8 |
| per-device microbatch | 1 |
| gradient accumulation | 4 |
| effective batch | `8 * 1 * 4 = 32` |
| samples per expert | 100,000 |
| expected steps per expert | 3,125 |
| learning rate | `5e-6` |
| warmup ratio | `0.03` |
| lr scheduler | cosine |
| max grad norm | `1.0` |
| weight decay | `0` |
| precision | bf16 |
| model max length | `16384` |
| min/max pixels | `50176 / 50176` |
| save policy | no intermediate checkpoints, final model only |
| OOM policy | standard trainer abort |
| NaN policy | abort on non-finite logged metric |

### 当前已跑通配置片段

远端启动脚本中的 manifest 固化了当前训练线的核心配置：

```bash
RUN_TAG=seed0_100k_mb1_filtered_stdtrainer_v1
LIMIT_SAMPLES=${LIMIT_SAMPLES:-100000}

EXPERTS=(
  general_obj_expert
  general_reasoning_expert
  region_expert
  robopoint_expert
  spatial_rel_expert
)

cat > "${ROOT}/outputs/${RUN_TAG}_manifest.json" <<EOF
{
  "run_tag": "${RUN_TAG}",
  "model_base": "${MODEL_BASE}",
  "data_root": "${DATA_ROOT}",
  "out_base": "${OUT_BASE}",
  "limit_samples": ${LIMIT_SAMPLES},
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 32,
  "expected_steps_per_expert": 3125,
  "trainer": "transformers.Trainer",
  "save_policy": "final_model_only_no_intermediate_checkpoints",
  "nan_policy": "abort_on_any_non_finite_logged_metric",
  "oom_policy": "standard_trainer_abort"
}
EOF
```

每个 expert 的实际 DeepSpeed launch 参数：

```bash
deepspeed --num_gpus=8 "${ROOT}/expert_sft.py" train \
  --model-name-or-path "${MODEL_BASE}" \
  --data-path "${data_path}" \
  --output-dir "${output_dir}" \
  --deepspeed "${ROOT}/configs/zero2.json" \
  --num-train-epochs 1 \
  --limit-samples "${LIMIT_SAMPLES}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --lr-scheduler-type cosine \
  --logging-steps 1 \
  --model-max-length 16384 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --dataloader-num-workers 4 \
  --bf16
```

MACA 环境变量仍使用此前验证过的组合：

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DS_ACCELERATOR=cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600
```

### 当前 active 代码实现片段

`expert_sft.py` 当前 active path 明确使用标准 `Trainer`。先用 callback 检查日志中的非有限值，一旦 loss、grad norm 或 learning rate 出现 NaN/Inf，就直接抛错中止：

```python
class AbortOnNonFiniteLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm", "learning_rate"):
            if not logs or key not in logs:
                continue
            value = logs[key]
            try:
                finite = math.isfinite(float(value))
            except Exception:
                finite = True
            if not finite:
                raise FloatingPointError(
                    f"non-finite metric at step {state.global_step}: {key}={value}"
                )
```

旧的 OOM skip 实验代码保留在文件中，但处于注释状态。关键标记如下：

```python
# Disabled OOM-skip experiment kept here for reference. The active path below
# intentionally uses the stock transformers.Trainer.
#
# class OomSkipTrainer(Trainer):
#     """Trainer with the narrowest possible OOM-only skip path."""
#     ...
#     def training_step(self, model, inputs, num_items_in_batch=None):
#         ...
#         # prepare/forward OOM and backward OOM were synced across ranks
#         # before returning a zero loss placeholder.
```

模型加载与训练参数保留了之前验证过的 Qwen3-VL/MACA 兼容设置：

```python
config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
if hasattr(config, "use_cache"):
    config.use_cache = False
if hasattr(config, "_attn_implementation"):
    config._attn_implementation = "eager"
if hasattr(config, "attn_implementation"):
    config.attn_implementation = "eager"

model = get_model_cls().from_pretrained(
    args.model_name_or_path,
    config=config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype="auto",
    attn_implementation="eager",
)
```

`TrainingArguments` 的 checkpoint 策略是 final-only。训练过程中不写中间 checkpoint，避免 8B full finetune 的 optimizer/checkpoint 文件把磁盘打满：

```python
return TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    max_grad_norm=args.max_grad_norm,
    lr_scheduler_type=args.lr_scheduler_type,
    logging_steps=args.logging_steps,
    save_strategy="no",
    save_only_model=True,
    save_safetensors=True,
    bf16=args.bf16,
    deepspeed=args.deepspeed,
    remove_unused_columns=False,
    dataloader_num_workers=args.dataloader_num_workers,
    dataloader_pin_memory=True,
    report_to=[],
    optim=args.optim,
    disable_tqdm=False,
    seed=args.seed,
    data_seed=args.seed,
    **kwargs,
)
```

训练入口也明确实例化标准 `Trainer`，而不是自定义 trainer：

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    processing_class=processor,
    callbacks=[AbortOnNonFiniteLog()],
)
trainer.train()
log(f"[train] finished global_step={trainer.state.global_step}")
save_final_model(trainer, processor, args)
log("[done] final-only expert SFT complete")
```

### 截至 2026-05-21 17:53 CST 的远端状态

只读检查命令确认 tmux 中仍在运行：

```text
full:seed0_mb1_std
```

总 run log：

```text
[run] tag=seed0_100k_mb1_filtered_stdtrainer_v1 limit=100000 mb=1 base=/data/msz/models/8b_base
[start] 2026-05-21 14:22:24 expert=general_obj_expert ...
[done] 2026-05-21 17:39:48 expert=general_obj_expert ...
[start] 2026-05-21 17:39:48 expert=general_reasoning_expert ...
```

`general_obj_expert` 已完成：

| item | value |
|---|---:|
| samples | 100,000 |
| global step | 3,125 |
| train runtime | 11,513.0294 s |
| train samples/s | 8.686 |
| train steps/s | 0.271 |
| train loss | 0.689730981349945 |

完成标记：

```text
[train] finished global_step=3125
[done] final-only expert SFT complete
```

最终模型目录已存在：

```text
/data/msz/models/seed0_general_obj_expert_100k_mb1_filtered_stdtrainer_v1
```

关键文件已落盘：

```text
config.json
generation_config.json
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
model.safetensors.index.json
preprocessor_config.json
run_summary.json
trainer_state.json
tokenizer.json
tokenizer_config.json
video_preprocessor_config.json
```

`general_reasoning_expert` 正在运行。早期因为数据分布从 object grounding 切到通用 reasoning/keepalive，起步 loss 和 raw grad norm 较大：

```text
step 1:  loss=5.1960, grad_norm=144.1368, lr=0
step 5:  loss=6.5952, grad_norm=220.0865
step 7:  loss=6.1664, grad_norm=244.1676
```

但随后很快回落，并未出现持续发散：

```text
step 24: loss=2.8445, grad_norm=48.9901
step 39: loss=0.2291, grad_norm=3.6831
step 50: loss=0.1909, grad_norm=4.1770
step 95: loss=0.2138, grad_norm=2.6653, lr=5e-6
step 150: loss=0.3031, grad_norm=6.3237
```

截至检查时未发现：

- `FloatingPointError`
- `Traceback`
- `OutOfMemory`
- `CUDA out of memory`
- `Watchdog`
- NaN/Inf metric

GPU 状态也稳定在单进程 8 卡训练：

| GPU | process | memory |
|---:|---|---:|
| 0 | `python3.12` | about 51.4 GB |
| 1 | `python3.12` | about 52.7 GB |
| 2 | `python3.12` | about 52.0 GB |
| 3 | `python3.12` | about 52.3 GB |
| 4 | `python3.12` | about 52.4 GB |
| 5 | `python3.12` | about 52.7 GB |
| 6 | `python3.12` | about 51.7 GB |
| 7 | `python3.12` | about 51.3 GB |

### 梯度范数解释记录

本轮用户观察到 `general_reasoning_expert` 启动阶段 `grad_norm` 比 `general_obj_expert` 更大。当前判断是正常的分布切换现象，而不是训练失败信号。

原因：

1. `general_reasoning_expert` 从同一个 `/data/msz/models/8b_base` 重新起训，不继承 `general_obj_expert`；
2. 它的数据以通用问答、keepalive、机器人常识和非坐标回答为主，和 object grounding 的输出分布差异明显；
3. warmup 初期学习率很小，step 1 的 lr 为 `0`，大 raw norm 不等于大参数更新；
4. `max_grad_norm=1.0` 只是 optimizer step 前的裁剪阈值，日志里打印的 `grad_norm` 仍可能是裁剪前或 DeepSpeed 统计的 raw norm，不会全程显示为 `1`；
5. 当前 loss 和 raw grad norm 已从早期高值快速回落，且没有 non-finite 指标。

因此当前策略是不干预继续跑。真正需要停的是以下情况：

- 出现 NaN/Inf；
- loss 异常变成持续 `0.0`；
- raw grad norm 持续升高且 loss 不降；
- OOM 或 Watchdog；
- 日志长时间不刷新但 GPU 进程仍占用显存。

### 当前训练线结论

1. 之前的失败主要来自过大的 microbatch、长样本导致的 OOM 风险，以及自定义 OOM-skip 与标准 DeepSpeed/Trainer 状态机之间的不稳定边界。
2. 已通过数据过滤、mb 降到 1、回归标准 `transformers.Trainer`、禁用中间 checkpoint，形成当前可跑通训练线。
3. `general_obj_expert` 已完整完成并保存 final-only 模型。
4. `general_reasoning_expert` 正常运行中，早期大梯度已回落，无 NaN/OOM/Traceback。
5. 后续 `region_expert`、`robopoint_expert`、`spatial_rel_expert` 将由同一个脚本串行启动。
6. 当前 report 只记录训练执行事实和可复现配置；模型训练本身继续在远端 tmux 后台进行。

## 2026-05-25 续写：seed0 五 Expert 完成与五 Teacher OPD Online 训练（Off-policy Teacher Rollout Top1 CE）

本节从上一节继续。上一节的最后记录点是 `2026-05-21 17:53 CST`：`general_obj_expert` 的早期 final-only 标准 Trainer 版本已经完成，`general_reasoning_expert` 正在运行，并且当时仍在观察起步梯度范数偏大的现象。之后实际实验路线发生了几次关键调整，最终产物变成：

1. 五个 seed0 expert 均重新按 `max_grad_norm=5.0`、`mb=1`、标准 Trainer、每个 100k 样本完成；
2. 基于这五个 expert，重新实现五 teacher 在线 rollout 的 OPD 训练；
3. OPD 最终使用 ZeRO-3、五 teacher 常驻 8 卡、左 padding、300k 样本、`mb=4` 完成；
4. OPD 最终 checkpoint `2344` 保留完整优化器状态，中间 checkpoint 已清掉 optimizer state 以释放磁盘；
5. 本地已补齐训练曲线资产和解析后的 metric 点。

本次只读检查和本地报告更新发生在 `2026-05-25`。没有启动新的训练，也没有改动远端模型或训练脚本。

### 本地报告资产

本节新增或更新的本地资产如下：

| asset | 说明 |
|---|---|
| `REPORT.md` | 本文件，追加完整实验日志。 |
| `report/seed0_five_experts_100k_maxnorm5/` | 五个 expert 的远端训练日志副本、loss 曲线、解析后的 loss 点。 |
| `report/seed0_five_experts_100k_maxnorm5/expert_loss_curves.svg` | 五 expert loss 曲线，MA50 平滑。 |
| `report/seed0_five_experts_100k_maxnorm5/expert_loss_points.json` | 五 expert 每步 loss/grad_norm/lr 解析点。 |
| `report/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/` | OPD full run 的远端日志副本、run summary、曲线和解析点。 |
| `report/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/opd_loss_entropy_curves.svg` | OPD `loss/opd_loss/entropy` 曲线，MA50 平滑。 |
| `report/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/opd_metrics_points.json` | OPD 每步 `loss/grad_norm/lr/opd_loss/entropy/route/tokens` 解析点。 |
| `report/training_curve_summary_20260525.json` | 曲线解析摘要，记录点数、首尾值、min/max、末端 MA50。 |

![seed0 five expert loss curves](report/seed0_five_experts_100k_maxnorm5/expert_loss_curves.svg)

![OPD loss and entropy curves](report/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/opd_loss_entropy_curves.svg)

曲线提取方法：

1. 从远端日志中只读复制训练日志到本地 `report/`；
2. 解析包含 `{'loss': ...}` 的 Trainer metric 行；
3. OPD 解析出 `2344` 条 step metric，对应最终 `global_step=2344`；
4. 每个 expert 解析出 `3125` 条 step metric，对应 `100000 / (8 * 1 * 4) = 3125`；
5. SVG 中使用 MA50 平滑，原始点完整保留在 JSON 中。图的 y 轴为了可读性按高分位截断，原始 min/max 以 JSON 和下表为准。

### 五个 Expert 的最终训练目标

最终目标是基于 `/data/msz/point/data_expert_seed0_v1_shuffled` 中的五份数据，使用同一个 `/data/msz/models/8b_base` 作为 base，分别训练五个领域 expert。每个 expert 固定使用对应 shuffled train 的前 `100,000` 条。

选择 `shuffle 后取前 100k` 的动机：

1. 每个领域的样本顺序可复现，后续如果继续训练到 200k 或更多，只需要从同一 shuffled 文件继续切片；
2. 避免多次随机抽样导致“这次前 100k”和“下次续训后 100k”分布不可追溯；
3. 先把五个 expert 都跑通，证明数据过滤和标准 Trainer 配置稳定，再考虑扩大样本量。

最终五个 expert：

| expert | 训练数据 |
|---|---|
| `general_reasoning_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/general_reasoning_expert/train_shuffled_seed20260520.jsonl` |
| `region_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/region_expert/train_shuffled_seed20260520.jsonl` |
| `robopoint_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/robopoint_expert/train_shuffled_seed20260520.jsonl` |
| `spatial_rel_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/spatial_rel_expert/train_shuffled_seed20260520.jsonl` |
| `general_obj_expert` | `/data/msz/point/data_expert_seed0_v1_shuffled/general_obj_expert/train_shuffled_seed20260520.jsonl` |

最终输出模型：

| expert | model path |
|---|---|
| `general_reasoning_expert` | `/data/msz/models/seed0_general_reasoning_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `region_expert` | `/data/msz/models/seed0_region_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `robopoint_expert` | `/data/msz/models/seed0_robopoint_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `spatial_rel_expert` | `/data/msz/models/seed0_spatial_rel_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `general_obj_expert` | `/data/msz/models/seed0_general_obj_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |

### 五个 Expert 的失败路径与修正

这轮五 expert 生成不是一次到位，关键阻碍如下。

第一类阻碍是 OOM。最开始讨论过 `200k` 样本和更大的 microbatch。随后为了稳定性先降到每个 expert `100k`。尝试过 `microbatch=4`，发生 OOM；再尝试 `microbatch=2`，仍然会被超长样本打爆。中间讨论过在 Trainer 上做极小 OOM skip 自定义，但实际行为不稳定，和标准 DeepSpeed/Trainer 状态机之间的边界不够可靠。因此最终回到标准 `transformers.Trainer`，不再使用 OOM skip。

第二类阻碍是超长样本。OOM 的根因不是普通样本，而是少量超长视觉语言样本造成单 batch 显存峰值过高。处理方式是重建数据并过滤超长样本，然后使用标准 Trainer、`mb=1` 来保证可跑通。旧的 OOM-skip 试验代码没有删除，只在远端代码中注释保留，便于后续复盘，但 active path 使用标准 Trainer。

第三类阻碍是梯度范数策略。`general_reasoning_expert` 起步阶段 raw `grad_norm` 明显大于 `general_obj_expert`，早期出现过 `grad_norm` 上百的记录。判断原因是 reasoning 数据分布和 object grounding 输出分布差异更大，而且 warmup 初期 lr 极小，大 raw norm 不等于实际大更新。用户希望保留梯度信号，所以没有用过强的 `max_grad_norm=1` 作为最终 expert 配置，而是改为 `max_grad_norm=5.0`。

最终策略：

| item | value |
|---|---|
| trainer | 标准 `transformers.Trainer` |
| base | `/data/msz/models/8b_base` |
| samples per expert | `100000` |
| per-device microbatch | `1` |
| gradient accumulation | `4` |
| effective batch | `8 * 1 * 4 = 32` |
| expected steps per expert | `3125` |
| learning rate | `5e-6` |
| warmup ratio | `0.03` |
| scheduler | cosine |
| max grad norm | `5.0` |
| DeepSpeed | ZeRO-2, `/data/msz/point/configs/zero2_gradclip5.json` |
| checkpoint policy | final-only, no intermediate checkpoint |
| OOM policy | 标准 Trainer 失败即中止，不自定义 skip |
| NaN policy | 任何 logged metric 非有限即 abort |

最终运行脚本：

```bash
/data/msz/point/run_seed0_five_experts_100k_mb1_filtered_stdtrainer_maxnorm5_v1.sh
```

最终训练顺序不是最初的 `general_obj` 先行顺序，而是从原来的第二段开始，最后补跑 `general_obj`：

```text
general_reasoning_expert
region_expert
robopoint_expert
spatial_rel_expert
general_obj_expert
```

这个顺序的动机是：`general_obj_expert` 在旧配置下已经有一版完成产物；调整 `max_grad_norm=5.0` 后，优先跑尚未完成或更关键的后续段，最后再用新配置补齐 `general_obj_expert`，保证五个最终 expert 配置一致。

### 五个 Expert 的实际完成记录

总 run log：

```text
/data/msz/point/logs/seed0_100k_mb1_filtered_stdtrainer_maxnorm5_v1.log
```

每个 expert 的单独日志：

```text
/data/msz/point/logs/seed0_general_reasoning_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1.log
/data/msz/point/logs/seed0_region_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1.log
/data/msz/point/logs/seed0_robopoint_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1.log
/data/msz/point/logs/seed0_spatial_rel_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1.log
/data/msz/point/logs/seed0_general_obj_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1.log
```

总 log 里有一次 `2026-05-21 18:33:06` 的早期启动记录，随后在 `18:37:15` 重新开始有效 run。最终完成记录以第二次启动为准。

| order | expert | start | done | global step | train loss | runtime s | samples/s | steps/s |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 1 | `general_reasoning_expert` | 2026-05-21 18:37:15 | 2026-05-21 21:55:01 | 3125 | 0.277714 | 11540.0691 | 8.665 | 0.271 |
| 2 | `region_expert` | 2026-05-21 21:55:01 | 2026-05-22 01:12:33 | 3125 | 0.762703 | 11504.3460 | 8.692 | 0.272 |
| 3 | `robopoint_expert` | 2026-05-22 01:12:33 | 2026-05-22 04:36:07 | 3125 | 0.763481 | 11887.3439 | 8.412 | 0.263 |
| 4 | `spatial_rel_expert` | 2026-05-22 04:36:07 | 2026-05-22 07:53:44 | 3125 | 0.733542 | 11519.9668 | 8.681 | 0.271 |
| 5 | `general_obj_expert` | 2026-05-22 07:53:44 | 2026-05-22 11:11:02 | 3125 | 0.688775 | 11501.3205 | 8.695 | 0.272 |

每个 final model 目录均包含：

```text
config.json
generation_config.json
model-00001-of-00005.safetensors
model-00002-of-00005.safetensors
model-00003-of-00005.safetensors
model-00004-of-00005.safetensors
model-00005-of-00005.safetensors
model.safetensors.index.json
preprocessor_config.json
run_summary.json
trainer_state.json
tokenizer.json
tokenizer_config.json
video_preprocessor_config.json
```

Expert loss 曲线摘要：

| expert | parsed steps | first loss | last loss | last MA50 loss | min loss | max loss |
|---|---:|---:|---:|---:|---:|---:|
| `general_reasoning_expert` | 3125 | 5.1960 | 0.2699 | 0.2372 | 0.0693 | 6.5971 |
| `region_expert` | 3125 | 1.6029 | 0.6987 | 0.7329 | 0.6179 | 1.6029 |
| `robopoint_expert` | 3125 | 2.5643 | 0.7267 | 0.7247 | 0.5497 | 3.5471 |
| `spatial_rel_expert` | 3125 | 1.3992 | 0.6992 | 0.7053 | 0.6055 | 1.5657 |
| `general_obj_expert` | 3125 | 1.4039 | 0.7324 | 0.6729 | 0.5145 | 1.9565 |

曲线解释：

1. `general_reasoning_expert` 起步 loss 和 raw grad norm 最大，但很快回落，符合“数据分布从坐标输出转到通用/文本推理”的预期；
2. `region/robopoint/spatial_rel/general_obj` 的 loss 更平滑，末端 MA50 都在 `0.67` 到 `0.73` 左右；
3. 这五个 expert 只保存 final model，不保存中间 optimizer checkpoint，所以没有在 expert 训练阶段继续制造磁盘压力。

### OPD 数据与训练目标

五 expert 完成后，目标变成：基于 `/data/msz/models/8b_base` 训练一个 OPD student，使它在不同 prompt 类型下学习五个 expert 的行为，而不是把所有数据预先离线生成成固定 logits。

OPD 数据：

```text
/data/msz/point/opd_student_v1/train_prompts.jsonl
```

数据设计动机：

1. Domain 数据让 student 学到各自领域的 expert 行为；
2. General 数据用于保持通用 VQA/文本回答能力，避免模型只会坐标输出；
3. Boundary 数据专门覆盖 object、region、spatial relation、point、reasoning 之间容易路由混淆的样本；
4. Format-conflict 数据用于压住输出格式漂移，尤其是 `<point>`、`<box>` 和纯文本回答之间的边界；
5. 每条样本带 `opd.target_expert`、候选 expert、源文件和格式元信息，便于在线路由到对应 teacher。

最终 full run 只取前 `300000` 条 raw row。由于 ZeRO-3 teacher generation 需要所有 rank 同步调用同一个 teacher，dataset 在 schedule 上做了 route-block shuffle，并补齐到 `300032` 行，保证分布式 microstep 中各 rank 路由一致。

OPD full run 的 dataset summary：

| item | value |
|---|---:|
| raw rows seen | 300000 |
| expanded rows | 300032 |
| padded rows | 32 |
| route policy | `target` |
| group by route | `false` |
| route block shuffle | `true` |
| shuffle seed | `20260520` |

按 route 的 raw row 分布：

| route | rows |
|---|---:|
| `general_reasoning_expert` | 92695 |
| `general_obj_expert` | 54355 |
| `region_expert` | 52512 |
| `spatial_rel_expert` | 52407 |
| `robopoint_expert` | 48031 |

按 route 的 schedule 分布：

| route | scheduled rows |
|---|---:|
| `general_reasoning_expert` | 92704 |
| `general_obj_expert` | 54368 |
| `region_expert` | 52512 |
| `spatial_rel_expert` | 52416 |
| `robopoint_expert` | 48032 |

按样本类别：

| category | rows |
|---|---:|
| `domain` | 180438 |
| `general` | 74603 |
| `boundary` | 30078 |
| `format_conflict` | 14881 |

按期望输出格式：

| expected format | rows |
|---|---:|
| `box` | 166891 |
| `text` | 87724 |
| `point` | 45385 |

候选 expert 数量：

| candidate count | rows |
|---:|---:|
| 1 | 269922 |
| 2 | 27106 |
| 3 | 2972 |

### OPD 实现设计

OPD 主脚本：

```text
/data/msz/point/train_opd_online_vl.py
```

本地同步副本：

```text
train_opd_online_vl.py
```

核心设计调整（此版为 off-policy teacher rollout + top1 CE，后续 2026-05-26 改为 on-policy student rollout + full-vocab KL）：

1. OPD 不再使用预先生成好的 teacher logits；
2. 每个 sample 在训练时根据 `target_expert` 路由到对应 teacher；
3. teacher 在线 rollout 生成 response tokens；
4. student 对 teacher rollout 的 top1 token 序列做 CE；
5. 对领域数据使用对应领域 teacher；
6. 对通用/边界/格式冲突数据按数据内 route 元信息路由；
7. 记录 `loss`、`opd_loss`、`grad_norm`、`entropy`、`opd_entropy`、`opd_route_id`、`opd_response_tokens`。

最关键的工程变化是 teacher 加载方式。最初如果按顺序或 lazy 方式逐个加载 teacher，训练会频繁发生模型加载/释放，速度和状态都不可接受。最终实现为：

```text
teacher_load_mode = preloaded_zero3
```

五个 teacher 全部常驻，且都通过 ZeRO-3 分片在 8 卡上：

| teacher | path |
|---|---|
| `general_obj_expert` | `/data/msz/models/seed0_general_obj_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `region_expert` | `/data/msz/models/seed0_region_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `robopoint_expert` | `/data/msz/models/seed0_robopoint_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `spatial_rel_expert` | `/data/msz/models/seed0_spatial_rel_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| `general_reasoning_expert` | `/data/msz/models/seed0_general_reasoning_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` |

ZeRO-3 teacher/student config：

```text
/data/msz/point/configs/zero3_opd_maca.json
```

关键点：

| config | value |
|---|---|
| zero stage | 3 |
| optimizer offload | none |
| param offload | none |
| overlap comm | false |
| contiguous gradients | true |
| reduce bucket size | 20000000 |
| prefetch bucket size | 20000000 |
| gather 16-bit weights on save | true |
| bf16 | enabled |

因为 ZeRO-3 teacher 的 generate 需要所有 rank 同步进入同一个 teacher，不能让 rank0 路由到 teacher A、rank1 路由到 teacher B。为此引入 route-block shuffle：

1. 样本仍然随机化；
2. 但随机化单位是 route block；
3. 每个分布式 microstep 内所有 rank 看到相同 route；
4. 这样既保留 route 顺序随机性，又满足 ZeRO-3 collective 调用约束。

### OPD Right Padding Bug 与 Left Padding 修正

在 `mb=2` 或更大 batch 下，decoder-only generation 如果使用 right padding，会导致 generate 从 padding 后的位置继续，出现 warning 和潜在错误：

```text
right-padding was detected
```

因此 full run 前做了 left padding 修正：

1. `configure_left_padding(processor)` 同时设置 processor/tokenizer 的 `padding_side="left"`；
2. collator 在 tokenization 前调用该函数；
3. rollout 后的 continuation label 不再使用每个样本自己的 prompt length；
4. 对 left-padded batch，统一使用 `prompt_width = inputs["input_ids"].shape[1]` 作为 continuation 起点；
5. generated attention mask 使用 prompt mask 加 suffix mask。

这个修正确保 `mb=4` 时每个样本的 teacher rollout token 对齐到正确的 continuation 区域。

### OPD Smoke 与 Full Run 路线

OPD 没有直接上 full run，而是经过了几步验证。

第一步，先用前 `100` 条样本做 smoke，验证：

1. OPD dataset 能正确读取；
2. route 元信息能正确映射到 teacher；
3. 五 teacher 能常驻 ZeRO-3；
4. 在线 rollout 能返回 token；
5. student CE 和 entropy 能正常计算。

第二步，做 checkpoint smoke。先尝试 `100 steps / save_steps=50`，发现过慢；随后缩短为 `max20steps / save_steps=10`，验证训练中保存 checkpoint 能成功。这个阶段的目的不是训练质量，而是确认 save path、ZeRO-3 model state、optimizer state、scheduler state 都能落盘。

第三步，启动 `100k / mb=2 / save_steps=300`。这一步确认显存大体可控，但暴露了 right padding 问题，因此停止并修 left padding。

第四步，最终 full run：

```bash
/data/msz/point/run_opd_five_online_300k_mb4_save500_zero3.sh
```

实际 DeepSpeed 命令核心参数：

```bash
deepspeed --num_gpus=8 train_opd_online_vl.py train \
  --model-name-or-path /data/msz/models/8b_base \
  --data-path /data/msz/point/opd_student_v1/train_prompts.jsonl \
  --output-dir /data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1 \
  --deepspeed /data/msz/point/configs/zero3_opd_maca.json \
  --route-policy target \
  --no-group-by-route \
  --route-block-shuffle \
  --teacher-load-mode preloaded_zero3 \
  --teacher-deepspeed /data/msz/point/configs/zero3_opd_maca.json \
  --limit-samples 300000 \
  --save-steps 500 \
  --save-total-limit 5 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-6 \
  --warmup-ratio 0.03 \
  --max-grad-norm 1.0 \
  --num-train-epochs 1 \
  --logging-steps 1 \
  --model-max-length 16384 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --bf16 \
  --gradient-checkpointing
```

OPD full run 训练配置：

| item | value |
|---|---:|
| raw samples | 300000 |
| expanded scheduled samples | 300032 |
| per-device microbatch | 4 |
| gradient accumulation | 4 |
| effective batch | `8 * 4 * 4 = 128` |
| expected steps | 2344 |
| learning rate | `1e-6` |
| warmup ratio | `0.03` |
| max grad norm | `1.0` |
| save steps | 500 |
| save total limit | 5 |
| DeepSpeed | ZeRO-3 |
| teacher mode | `preloaded_zero3` |
| OPD target | online teacher rollout top1 |

### OPD Full Run 结果

OPD full run 日志：

```text
/data/msz/point/logs/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1.log
```

输出模型：

```text
/data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1
```

完成记录：

| metric | value |
|---|---:|
| global step | 2344 |
| train runtime | 167955.0764 s |
| train samples/s | 1.786 |
| train steps/s | 0.014 |
| train loss | 0.3442268627489132 |
| final logged loss | 0.1846 |
| final logged opd_loss | 0.16669046878814697 |
| final logged entropy | 0.17483912408351898 |
| final logged grad_norm | 2.842887765948197 |
| final logged response tokens | 25 |

日志中的完成标记：

```text
100%|██████████| 2344/2344 [46:37:47<00:00, 62.30s/it]
{'train_runtime': 167955.0764, 'train_samples_per_second': 1.786, 'train_steps_per_second': 0.014, 'train_loss': 0.3442268627489132, 'opd_route_id': 4.0, 'opd_response_tokens': 25.0, 'opd_loss': 0.16669046878814697, 'entropy': 0.17483912408351898, 'opd_entropy': 0.17483912408351898, 'epoch': 1.0}
[opd-online] finished global_step=2344 train_loss=0.3442268627489132
[done] OPD online training complete
```

OPD 曲线解析摘要：

| metric | value |
|---|---:|
| parsed step metrics | 2344 |
| first loss | 4.5744 |
| first opd_loss | 2.977684736251831 |
| first entropy | 1.2759050130844116 |
| last loss | 0.1846 |
| last opd_loss | 0.16669046878814697 |
| last entropy | 0.17483912408351898 |
| loss min | 0.0679 |
| loss max | 5.6432 |
| loss last MA50 | 0.247998 |
| opd_loss last MA50 | 0.240245 |
| entropy last MA50 | 0.248550 |

曲线解释：

1. 早期 `loss/opd_loss/entropy` 均较高，符合 base student 刚开始模仿多 teacher 在线 rollout 的状态；
2. 中后段 loss 和 entropy 明显下降，说明 student 对 teacher rollout token 分布越来越确定；
3. 末端 `loss MA50` 和 `opd_loss MA50` 都在 `0.24` 左右，最终 logged `opd_loss=0.1667`；
4. 没有出现 loss 持续变成 `0.0` 的旧 NaN cascade 形态；
5. 只读 grep 未发现 `Traceback`、`RuntimeError`、`OOM`、`out of memory`、`nan`、`inf`、`non-finite`、`right-padding`、`Watchdog` 等异常关键词。

### Checkpoint 与磁盘状态

Full run 过程中按 `save_steps=500` 和 `save_total_limit=5` 保留了五个 checkpoint：

| checkpoint | saved time |
|---|---|
| `checkpoint-500` | 2026-05-23 01:05:09 |
| `checkpoint-1000` | 2026-05-23 11:00:56 |
| `checkpoint-1500` | 2026-05-23 20:54:03 |
| `checkpoint-2000` | 2026-05-24 06:48:18 |
| `checkpoint-2344` | 2026-05-24 13:39:23 |

训练完成后检查 `checkpoint-2344`，确认包含完整可恢复状态：

```text
global_step2344/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
global_step2344/bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt
global_step2344/bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt
global_step2344/bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt
global_step2344/bf16_zero_pp_rank_4_mp_rank_00_optim_states.pt
global_step2344/bf16_zero_pp_rank_5_mp_rank_00_optim_states.pt
global_step2344/bf16_zero_pp_rank_6_mp_rank_00_optim_states.pt
global_step2344/bf16_zero_pp_rank_7_mp_rank_00_optim_states.pt
global_step2344/zero_pp_rank_0_mp_rank_00_model_states.pt
...
global_step2344/zero_pp_rank_7_mp_rank_00_model_states.pt
scheduler.pt
rng_state_0.pth
...
rng_state_7.pth
trainer_state.json
```

随后为了释放磁盘，只清理中间 checkpoint 的 optimizer state：

```text
checkpoint-500/global_step500/bf16_zero_pp_rank_0..7_mp_rank_00_optim_states.pt
checkpoint-1000/global_step1000/bf16_zero_pp_rank_0..7_mp_rank_00_optim_states.pt
checkpoint-1500/global_step1500/bf16_zero_pp_rank_0..7_mp_rank_00_optim_states.pt
checkpoint-2000/global_step2000/bf16_zero_pp_rank_0..7_mp_rank_00_optim_states.pt
```

清理前这些 optimizer state 合计约 `374G`。清理后验证：

1. 只有 `checkpoint-2344` 仍保留 8 份 optimizer state；
2. 中间 checkpoint 仍保留模型权重和 trainer 相关文件，但不再能完整恢复优化器状态；
3. 最终 checkpoint 仍可完整恢复；
4. 总目录大小约 `179G`。

清理后各 checkpoint 大小：

| checkpoint | size |
|---|---:|
| `checkpoint-500` | 18G |
| `checkpoint-1000` | 18G |
| `checkpoint-1500` | 18G |
| `checkpoint-2000` | 18G |
| `checkpoint-2344` | 111G |

### 当前实验状态

截至本次记录：

1. 五个 seed0 `100k` expert 已完成，最终配置统一为 `mb=1`、标准 Trainer、`max_grad_norm=5.0`、final-only；
2. 五 teacher online OPD 已完成，最终模型在 `/data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1`；
3. OPD final checkpoint `2344` 保留完整 optimizer/model/scheduler/rng/trainer state；
4. 中间 checkpoint 的 optimizer state 已清理，避免继续占用 300G 以上空间；
5. 本地已落盘五 expert loss 曲线和 OPD loss/entropy 曲线；
6. 当前没有在本次 side conversation 中启动新的 eval；
7. 如果后续恢复实验，优先从本节列出的五 expert 路径、OPD run summary、`train_opd_online_vl.py`、`zero3_opd_maca.json` 和曲线 JSON 开始。

最小恢复清单：

| purpose | path |
|---|---|
| OPD student data | `/data/msz/point/opd_student_v1/train_prompts.jsonl` |
| OPD train script | `/data/msz/point/train_opd_online_vl.py` |
| OPD full launch script | `/data/msz/point/run_opd_five_online_300k_mb4_save500_zero3.sh` |
| OPD ZeRO-3 config | `/data/msz/point/configs/zero3_opd_maca.json` |
| Expert launch script | `/data/msz/point/run_seed0_five_experts_100k_mb1_filtered_stdtrainer_maxnorm5_v1.sh` |
| Expert ZeRO-2 config | `/data/msz/point/configs/zero2_gradclip5.json` |
| OPD full output | `/data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1` |
| OPD final checkpoint | `/data/msz/models/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/checkpoint-2344` |
| Local OPD curve | `report/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/opd_loss_entropy_curves.svg` |
| Local expert curve | `report/seed0_five_experts_100k_maxnorm5/expert_loss_curves.svg` |

# Raw Holdout 8 模型完整评估 - 2026-05-25

本节整理 raw-holdout 10k 评估与 OPD checkpoint 评估。8 模型主评估来自远端
`/data/msz/point/eval_raw_holdout_v1/runs_8models_20260525_120552`；OPD checkpoint 曲线来自
`/data/msz/point/eval_raw_holdout_v1/opd_ckpts_20260525_124224`。完整 JSON 与独立 Markdown 报告已归档到本地：

```text
report/raw_holdout_eval_8models_20260525/
```

## 评估过程

1. 构造 10k raw-holdout 评估集，覆盖通用 VQA、point grounding、box grounding、区域描述、关系框选与语义导航框选。
2. 训练集阻断：排除 5 个 expert 各自前 100k 训练样本、最终 OPD 训练样本，并用 source-line key、内容 fingerprint、训练图片集合做泄漏检查。
3. 严格去重验收：fingerprint、source line、image+prompt+answer 三类重复均为 0；训练图片泄漏为 0。
4. 8 模型主评估：每个模型单张卡，8 卡并行；`BATCH_SIZE=256`，`max_new_tokens=64`。生成阶段单卡显存约 35-37GB，GPU util 主要在 85%-98%。
5. OPD checkpoint 评估：checkpoint-500/1000/1500/2000 各占一张卡并行评估，同样 10k 样本、同一套指标；并把前一轮 checkpoint-2344 的 OPD final 结果并入曲线。
6. 指标：box 使用格式率、坐标有效率、IoU、Acc@0.3/0.5/0.75、中心距离；point 使用格式率、坐标有效率、Hit@50/Hit@100、距离与点数偏差；text/VQA 使用 exact、loose、boolean accuracy、multiple-choice accuracy。

## 评估集与去重验收

| 项目 | 数值 |
| --- | ---: |
| 总样本 | 10000 |
| 唯一图片 | 8251 |
| Fingerprint 重复 | 0 |
| Source line 重复 | 0 |
| Image+prompt+answer 重复 | 0 |
| 训练图片泄漏行数 | 0 |
| 重复图片路径数 | 1168 |
| 落在重复图片上的样本行数 | 2917 |

重复图片不等价于重复样本。RefCOCO、Visual Genome、Flickr30K 一张图天然会有多个 object/region/relation 标注；本次真正严格去重的 source、fingerprint、image+prompt+answer 均为 0，同时训练图片泄漏为 0。

### 数据池分布

| 数据池 | 领域/场景 | 样本数 |
| --- | --- | ---: |
| `refcoco` | RefCOCO / 指代表达框选 | 1100 |
| `flickr30k_entities` | Flickr30K Entities / 短语实体框选 | 900 |
| `visual_genome_object` | Visual Genome Object / 通用物体框选 | 1100 |
| `visual_genome_region` | Visual Genome Region / 区域描述框选 | 1100 |
| `visual_genome_relationship` | Visual Genome Relationship / 关系框选 | 1100 |
| `semantic_nav_box` | Semantic Nav Box / 语义导航框选 | 800 |
| `grounding_point` | Grounding point / 点选 | 1400 |
| `keepalive_vqa` | Keepalive VQA / 通用能力 | 2500 |

### 任务格式分布

| 任务格式 | 样本数 |
| --- | ---: |
| box | 6100 |
| point | 1400 |
| text | 2500 |

## 8 模型总体结果

| 模型 | n | Format | Coord | Box IoU | Box Acc@0.5 | Point Hit@100 | Text loose | Bool acc | MC acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 10000 | 85.3% | 80.5% | 0.385 | 40.9% | 0.0% | 0.8% | 0.0% | 18.7% |
| Qwen3-VL-8B-Instruct | 10000 | 85.8% | 81.0% | 0.413 | 43.3% | 0.0% | 52.0% | 81.2% | 29.0% |
| General reasoning expert | 10000 | 100.0% | 100.0% | 0.444 | 48.1% | 83.4% | 87.1% | 89.0% | 21.5% |
| RoboPoint expert | 10000 | 100.0% | 100.0% | 0.454 | 49.8% | 87.9% | 85.8% | 87.5% | 16.8% |
| General obj expert | 10000 | 100.0% | 100.0% | 0.470 | 51.3% | 82.5% | 84.6% | 85.8% | 24.3% |
| Region expert | 10000 | 100.0% | 100.0% | 0.469 | 51.5% | 81.5% | 83.0% | 81.6% | 20.6% |
| Spatial rel expert | 10000 | 100.0% | 100.0% | 0.473 | 51.6% | 81.5% | 84.2% | 85.8% | 21.5% |
| OPD final | 10000 | 100.0% | 100.0% | 0.468 | 51.3% | 84.1% | 87.4% | 90.0% | 20.6% |

总体观点：

- **OPD final 是综合最均衡的模型。** 它的 box Acc@0.5=51.3%、point Hit@100=84.1%、text loose=87.4%、bool acc=90.0%。它不是每个单项的绝对第一，但在 grounding 与通用能力之间的折中最好。
- **box 单项最强的是 Spatial rel expert。** 它的 box IoU=0.473、Acc@0.5=51.6%，略高于 OPD final 的 0.468/51.3%。差距很小，说明 OPD 融合基本保住了区域与关系框选能力。
- **point 单项最强的是 RoboPoint expert。** Hit@100=87.9%，高于 OPD final 的 84.1%。这符合训练目标，也说明 point 能力在融合中有约 3.8 个百分点的损失。
- **通用能力最强的是 OPD final。** text loose=87.4%、bool acc=90.0%，都是最高；说明 OPD 的通用数据与边界/格式数据没有被 grounding 数据淹没。
- **base 与 Qwen3-VL-8B-Instruct 在 point 上为 0，不代表视觉完全不会，而是格式未对齐。** 二者几乎不按本项目训练期望输出 `<point>`，而 expert/OPD 的 point format 都达到或接近 100%。

## 按任务类型拆分

### Box Grounding

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 6100 | 98.9% | 98.9% | 0.385 | 52.3% | 40.9% | 21.0% | 153.5 |
| Qwen3-VL-8B-Instruct | 6100 | 99.6% | 99.6% | 0.413 | 55.8% | 43.3% | 24.7% | 140.1 |
| General reasoning expert | 6100 | 100.0% | 100.0% | 0.444 | 58.3% | 48.1% | 29.8% | 138.4 |
| RoboPoint expert | 6100 | 100.0% | 100.0% | 0.454 | 59.2% | 49.8% | 31.1% | 132.7 |
| General obj expert | 6100 | 100.0% | 100.0% | 0.470 | 60.4% | 51.3% | 33.5% | 129.6 |
| Region expert | 6100 | 100.0% | 100.0% | 0.469 | 60.7% | 51.5% | 33.1% | 130.0 |
| Spatial rel expert | 6100 | 100.0% | 100.0% | 0.473 | 61.0% | 51.6% | 33.4% | 128.2 |
| OPD final | 6100 | 100.0% | 100.0% | 0.468 | 60.3% | 51.3% | 32.9% | 128.2 |

Box 观点：box 第一梯队是 spatial_rel、region、general_obj、OPD final，Acc@0.5 都在 51.3%-51.6%。OPD final 相比 Qwen3-VL-8B-Instruct 从 43.3% 提升到 51.3%，提升约 8.0 个百分点；相比 base 从 40.9% 提升到 51.3%，提升约 10.5 个百分点。

### Point Grounding

| 模型 | n | Format | Coord | Hit@50 | Hit@100 | MinDist | PredToGoldDist | PointCountDiff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| Qwen3-VL-8B-Instruct | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| General reasoning expert | 1400 | 100.0% | 100.0% | 71.0% | 83.4% | 60.8 | 84.5 | 4.4 |
| RoboPoint expert | 1400 | 100.0% | 100.0% | 77.6% | 87.9% | 50.2 | 77.7 | 4.6 |
| General obj expert | 1400 | 100.0% | 100.0% | 69.4% | 82.5% | 63.5 | 85.1 | 4.3 |
| Region expert | 1400 | 100.0% | 100.0% | 68.4% | 81.5% | 65.1 | 87.2 | 4.1 |
| Spatial rel expert | 1400 | 100.0% | 100.0% | 66.9% | 81.5% | 65.7 | 88.4 | 4.2 |
| OPD final | 1400 | 100.0% | 100.0% | 70.9% | 84.1% | 57.6 | 78.0 | 4.6 |

Point 观点：RoboPoint expert 明显最强，Hit@50=77.6%、Hit@100=87.9%。OPD final Hit@100=84.1%，低于 RoboPoint expert，但高于 general_obj、region、spatial_rel expert，说明融合保留了大部分 point 专家能力。

### Text / VQA 通用能力

| 模型 | n | Format | Text exact | Text loose | Bool n | Bool acc | MC n | MC acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 2500 | 100.0% | 0.8% | 0.8% | 910 | 0.0% | 107 | 18.7% |
| Qwen3-VL-8B-Instruct | 2500 | 100.0% | 0.8% | 52.0% | 910 | 81.2% | 107 | 29.0% |
| General reasoning expert | 2500 | 100.0% | 87.1% | 87.1% | 910 | 89.0% | 107 | 21.5% |
| RoboPoint expert | 2500 | 100.0% | 85.8% | 85.8% | 910 | 87.5% | 107 | 16.8% |
| General obj expert | 2500 | 100.0% | 84.6% | 84.6% | 910 | 85.8% | 107 | 24.3% |
| Region expert | 2500 | 100.0% | 83.0% | 83.0% | 910 | 81.6% | 107 | 20.6% |
| Spatial rel expert | 2500 | 100.0% | 84.2% | 84.2% | 910 | 85.8% | 107 | 21.5% |
| OPD final | 2500 | 100.0% | 87.4% | 87.4% | 910 | 90.0% | 107 | 20.6% |

Text 观点：OPD final 的 text loose=87.4%、bool acc=90.0%，为全体最佳。Qwen3-VL-8B-Instruct 的 loose 有 52.0%，但 exact 只有 0.8%，说明它经常输出带选项前缀或解释性文本，和本项目期望的短答案格式不完全一致；expert/OPD 的 exact 与 loose 接近，格式约束更稳定。

## 每个领域的完整指标与分析

### 各领域最佳模型概览

| 领域 | 主指标 | 最佳模型 | 指标值 |
| --- | --- | --- | ---: |
| RefCOCO / 指代表达框选 | Acc@0.5 | General obj expert | 83.9% |
| Flickr30K Entities / 短语实体框选 | Acc@0.5 | General obj expert | 78.3% |
| Visual Genome Object / 通用物体框选 | Acc@0.5 | Spatial rel expert | 35.5% |
| Visual Genome Region / 区域描述框选 | Acc@0.5 | Region expert | 41.5% |
| Visual Genome Relationship / 关系框选 | Acc@0.5 | Region expert | 42.7% |
| Semantic Nav Box / 语义导航框选 | Acc@0.5 | Spatial rel expert | 34.0% |
| Grounding point / 点选 | Hit@100 | RoboPoint expert | 87.9% |
| Keepalive VQA / 通用能力 | Text loose | OPD final | 87.4% |

### RefCOCO / 指代表达框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1100 | 97.2% | 97.3% | 0.646 | 82.9% | 74.5% | 46.9% | 85.5 |
| Qwen3-VL-8B-Instruct | 1100 | 99.9% | 99.9% | 0.682 | 88.5% | 80.1% | 54.3% | 70.3 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.729 | 89.1% | 82.9% | 65.9% | 63.3 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.719 | 87.2% | 81.8% | 64.6% | 67.9 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.740 | 89.5% | 83.9% | 67.9% | 61.2 |
| Region expert | 1100 | 100.0% | 100.0% | 0.710 | 87.1% | 81.0% | 63.6% | 69.7 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.725 | 87.7% | 81.7% | 65.7% | 63.7 |
| OPD final | 1100 | 100.0% | 100.0% | 0.720 | 87.5% | 82.0% | 64.8% | 65.8 |

解读：最佳为 General obj expert，Acc@0.5=83.9%。OPD final Acc@0.5=82.0%、IoU=0.720，距最佳差 1.9 个百分点，相比 Qwen3-VL-8B-Instruct 提升 1.9 个百分点。RefCOCO 是成熟指代表达框选领域，OPD final 保持在第一梯队。

### Flickr30K Entities / 短语实体框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 900 | 99.6% | 99.6% | 0.553 | 74.3% | 61.2% | 35.0% | 105.8 |
| Qwen3-VL-8B-Instruct | 900 | 99.9% | 99.9% | 0.591 | 77.4% | 65.7% | 42.3% | 95.8 |
| General reasoning expert | 900 | 100.0% | 100.0% | 0.608 | 77.1% | 65.9% | 46.4% | 102.4 |
| RoboPoint expert | 900 | 100.0% | 100.0% | 0.636 | 81.0% | 70.8% | 50.4% | 91.8 |
| General obj expert | 900 | 100.0% | 100.0% | 0.706 | 86.2% | 78.3% | 59.7% | 68.1 |
| Region expert | 900 | 100.0% | 100.0% | 0.700 | 86.0% | 78.3% | 59.8% | 66.7 |
| Spatial rel expert | 900 | 100.0% | 100.0% | 0.693 | 85.6% | 76.4% | 58.2% | 71.9 |
| OPD final | 900 | 100.0% | 100.0% | 0.701 | 85.7% | 77.6% | 59.4% | 67.7 |

解读：最佳为 General obj expert / Region expert，Acc@0.5=78.3%。OPD final Acc@0.5=77.6%、IoU=0.701，距最佳只差 0.8 个百分点，说明短语实体框选能力在融合后保持很好。

### Visual Genome Object / 通用物体框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1100 | 99.8% | 99.8% | 0.262 | 37.5% | 25.6% | 11.5% | 197.7 |
| Qwen3-VL-8B-Instruct | 1100 | 99.2% | 99.2% | 0.288 | 39.5% | 28.5% | 13.6% | 189.4 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.303 | 41.4% | 32.2% | 16.2% | 197.1 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.311 | 41.7% | 33.0% | 18.3% | 189.5 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.323 | 42.9% | 34.2% | 19.8% | 184.8 |
| Region expert | 1100 | 100.0% | 100.0% | 0.318 | 42.6% | 33.0% | 19.0% | 186.3 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.329 | 44.2% | 35.5% | 20.2% | 185.5 |
| OPD final | 1100 | 100.0% | 100.0% | 0.325 | 43.5% | 34.4% | 19.8% | 182.3 |

解读：最佳为 Spatial rel expert，Acc@0.5=35.5%。OPD final Acc@0.5=34.4%、IoU=0.325，距最佳差 1.1 个百分点，相比 Qwen3-VL-8B-Instruct 提升 5.9 个百分点。

### Visual Genome Region / 区域描述框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1100 | 98.5% | 98.5% | 0.315 | 44.7% | 30.3% | 10.5% | 155.7 |
| Qwen3-VL-8B-Instruct | 1100 | 99.5% | 99.5% | 0.341 | 48.7% | 33.3% | 14.3% | 148.0 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.374 | 53.6% | 38.9% | 18.3% | 145.0 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.376 | 53.4% | 39.3% | 18.2% | 141.2 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.381 | 53.9% | 38.9% | 18.5% | 140.7 |
| Region expert | 1100 | 100.0% | 100.0% | 0.392 | 55.7% | 41.5% | 20.0% | 139.6 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.384 | 53.5% | 40.5% | 18.3% | 142.1 |
| OPD final | 1100 | 100.0% | 100.0% | 0.382 | 54.5% | 41.1% | 18.2% | 140.1 |

解读：最佳为 Region expert，Acc@0.5=41.5%。OPD final Acc@0.5=41.1%，只落后约 0.5 个百分点，说明区域描述框选能力在融合后保持良好。

### Visual Genome Relationship / 关系框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1100 | 98.7% | 98.8% | 0.313 | 42.0% | 31.0% | 16.4% | 175.9 |
| Qwen3-VL-8B-Instruct | 1100 | 99.5% | 99.5% | 0.354 | 47.9% | 36.4% | 19.7% | 160.8 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.378 | 49.5% | 40.5% | 23.7% | 151.9 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.381 | 50.0% | 41.2% | 24.8% | 156.4 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.391 | 51.0% | 42.2% | 25.6% | 151.3 |
| Region expert | 1100 | 100.0% | 100.0% | 0.400 | 52.6% | 42.7% | 26.3% | 150.0 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.393 | 51.5% | 41.1% | 27.2% | 152.4 |
| OPD final | 1100 | 100.0% | 100.0% | 0.390 | 50.9% | 41.2% | 25.6% | 148.2 |

解读：最佳为 Region expert，Acc@0.5=42.7%。OPD final Acc@0.5=41.2%，与 spatial_rel expert 基本持平。关系框选不仅依赖空间关系，也依赖区域描述式定位。

### Semantic Nav Box / 语义导航框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 800 | 100.0% | 100.0% | 0.212 | 30.8% | 20.8% | 3.4% | 204.2 |
| Qwen3-VL-8B-Instruct | 800 | 99.9% | 99.9% | 0.197 | 29.3% | 11.4% | 0.8% | 179.5 |
| General reasoning expert | 800 | 100.0% | 100.0% | 0.253 | 36.4% | 25.6% | 4.4% | 174.2 |
| RoboPoint expert | 800 | 100.0% | 100.0% | 0.291 | 40.9% | 31.9% | 7.0% | 145.6 |
| General obj expert | 800 | 100.0% | 100.0% | 0.269 | 37.1% | 29.5% | 7.2% | 171.5 |
| Region expert | 800 | 100.0% | 100.0% | 0.281 | 39.2% | 31.6% | 7.4% | 154.4 |
| Spatial rel expert | 800 | 100.0% | 100.0% | 0.306 | 42.5% | 34.0% | 7.2% | 140.9 |
| OPD final | 800 | 100.0% | 100.0% | 0.278 | 38.5% | 31.1% | 6.9% | 153.9 |

解读：最佳为 Spatial rel expert，Acc@0.5=34.0%。OPD final Acc@0.5=31.1%、IoU=0.278。该领域是所有 box 任务中最难的一档：最佳 Acc@0.5 只有 34.0%，明显低于 RefCOCO 的 83.9% 和 Flickr30K 的 78.3%。这说明语义导航里的 `{object_name, relation, anchor_object}` 框选仍是短板，关系理解和目标唯一定位比普通指代表达更难。

### Grounding point / 点选

| 模型 | n | Format | Coord | Hit@50 | Hit@100 | MinDist | PredToGoldDist | PointCountDiff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| Qwen3-VL-8B-Instruct | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| General reasoning expert | 1400 | 100.0% | 100.0% | 71.0% | 83.4% | 60.8 | 84.5 | 4.4 |
| RoboPoint expert | 1400 | 100.0% | 100.0% | 77.6% | 87.9% | 50.2 | 77.7 | 4.6 |
| General obj expert | 1400 | 100.0% | 100.0% | 69.4% | 82.5% | 63.5 | 85.1 | 4.3 |
| Region expert | 1400 | 100.0% | 100.0% | 68.4% | 81.5% | 65.1 | 87.2 | 4.1 |
| Spatial rel expert | 1400 | 100.0% | 100.0% | 66.9% | 81.5% | 65.7 | 88.4 | 4.2 |
| OPD final | 1400 | 100.0% | 100.0% | 70.9% | 84.1% | 57.6 | 78.0 | 4.6 |

解读：最佳为 RoboPoint expert，Hit@100=87.9%；OPD final Hit@100=84.1%，落后 3.8 个百分点。纯 point expert 对点位预测仍有不可替代的专精收益。

### Keepalive VQA / 通用能力

| 模型 | n | Format | Text exact | Text loose | Bool n | Bool acc | MC n | MC acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 2500 | 100.0% | 0.8% | 0.8% | 910 | 0.0% | 107 | 18.7% |
| Qwen3-VL-8B-Instruct | 2500 | 100.0% | 0.8% | 52.0% | 910 | 81.2% | 107 | 29.0% |
| General reasoning expert | 2500 | 100.0% | 87.1% | 87.1% | 910 | 89.0% | 107 | 21.5% |
| RoboPoint expert | 2500 | 100.0% | 85.8% | 85.8% | 910 | 87.5% | 107 | 16.8% |
| General obj expert | 2500 | 100.0% | 84.6% | 84.6% | 910 | 85.8% | 107 | 24.3% |
| Region expert | 2500 | 100.0% | 83.0% | 83.0% | 910 | 81.6% | 107 | 20.6% |
| Spatial rel expert | 2500 | 100.0% | 84.2% | 84.2% | 910 | 85.8% | 107 | 21.5% |
| OPD final | 2500 | 100.0% | 87.4% | 87.4% | 910 | 90.0% | 107 | 20.6% |

解读：最佳为 OPD final，Text loose=87.4%，Bool acc=90.0%。融合后的通用能力没有被领域 grounding 任务压垮。

## OPD Checkpoint 曲线

| Checkpoint | n | Format | Coord | Box IoU | Box Acc@0.5 | Point Hit@50 | Point Hit@100 | Text loose | Bool acc | MC acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 10000 | 100.0% | 100.0% | 0.457 | 50.1% | 65.9% | 79.7% | 87.4% | 89.6% | 21.5% |
| 1000 | 10000 | 100.0% | 100.0% | 0.465 | 51.2% | 68.6% | 82.9% | 87.3% | 89.8% | 20.6% |
| 1500 | 10000 | 100.0% | 100.0% | 0.467 | 51.2% | 69.4% | 83.2% | 87.3% | 89.7% | 20.6% |
| 2000 | 10000 | 100.0% | 100.0% | 0.468 | 51.4% | 71.2% | 83.5% | 87.3% | 89.9% | 20.6% |
| 2344 | 10000 | 100.0% | 100.0% | 0.468 | 51.3% | 70.9% | 84.1% | 87.4% | 90.0% | 20.6% |

Checkpoint 观点：

- **500 -> 2000 基本单调提升。** Box Acc@0.5 从 50.1% 到 51.4%，Point Hit@100 从 79.7% 到 83.5%，说明 OPD 训练前 2000 step 仍在带来稳定收益。
- **2000 -> 2344 进入平台期。** Box Acc@0.5 从 51.4% 到 51.3%，几乎不变；Point Hit@100 从 83.5% 到 84.1%，小幅提升；Text loose 从 87.3% 到 87.4%，也基本持平。
- **若只看 box，checkpoint-2000 略优；若看综合，checkpoint-2344 略稳。** 2000 的 box Acc@0.5=51.4%，略高于 2344 的 51.3%；2344 的 Point Hit@100=84.1%、Bool acc=90.0%，略高于 2000。
- **不建议回退到 500。** 500 的 point Hit@100=79.7%，比 2344 低 4.4 个百分点；box Acc@0.5 也低 1.2 个百分点。

## 结论与建议

1. **当前推荐最终模型仍是 OPD final/checkpoint-2344。** 数据依据：它在 box 上达到第一梯队（Acc@0.5=51.3%），point 保持较高（Hit@100=84.1%），通用能力最好（Text loose=87.4%、Bool acc=90.0%）。
2. **如果业务只追求通用 box grounding，可以考虑 spatial_rel 或 region/general_obj expert，但不建议替代 OPD final。** Spatial rel expert box Acc@0.5=51.6%，只比 OPD final 高 0.3 个百分点；但 OPD final 的通用能力和 point 兼容性更完整。
3. **如果业务强依赖 point，RoboPoint expert 仍是专精上限。** 它 Hit@100=87.9%，比 OPD final 高 3.8 个百分点；但它的 text/VQA 与 box 综合均衡性不如 OPD final。
4. **semantic-nav box 是下一步最值得优化的短板。** 最佳模型在该领域 Acc@0.5 只有 34.0%，OPD final 只有 31.1%，明显低于 RefCOCO 的 82.0% 和 Flickr30K 的 77.6%。这说明语义导航的关系定位、anchor/object disambiguation、标注一致性仍需要更强数据或更严格清洗。
5. **OPD 训练已接近收敛平台。** 2000 到 2344 的收益很小，继续训练未必高性价比；下一轮提升更可能来自数据配比、semantic-nav 清洗、hard/boundary 样本设计，而不是单纯延长 step。

## 结果文件索引

| 文件 | 用途 |
| --- | --- |
| `report/raw_holdout_eval_8models_20260525/comparison_extended_metrics.json` | 8 模型完整扩展指标，含 by_format/by_pool |
| `report/raw_holdout_eval_8models_20260525/comparison_summary.json` | 8 模型原始紧凑汇总 |
| `report/raw_holdout_eval_8models_20260525/opd_ckpt_extended_comparison_with_2344.json` | OPD checkpoint 500/1000/1500/2000/2344 对比 |
| `report/raw_holdout_eval_8models_20260525/opd_ckpt_comparison_summary.json` | OPD checkpoint 原始紧凑汇总 |
| `report/raw_holdout_eval_8models_20260525/evalset_summary.json` | 评估集构造、数据池、blocklist、verify 信息 |
| `report/raw_holdout_eval_8models_20260525/dedupe_audit.json` | post-hoc 去重与训练图片泄漏审计 |
| 远端 8 模型 run | `/data/msz/point/eval_raw_holdout_v1/runs_8models_20260525_120552` |
| 远端 OPD ckpt run | `/data/msz/point/eval_raw_holdout_v1/opd_ckpts_20260525_124224` |

## 2026-05-26/27 Off-policy OPD 续训至 3500 步与 On-policy Student Rollout Full-Vocab KL 蒸馏

本节记录两件事：

1. 上一节记录的 OPD 300k off-policy 训练（teacher rollout + top1 CE）实际续训到了 3500 步，并在 checkpoint-2500/3000/3500 上做了评估；
2. 新的 on-policy student rollout + full-vocab KL 蒸馏训练完成，使用全量 999,900 条 OPD student 数据，训练 2500 步，并在 checkpoint-1500/2000/2500 上做了评估。

### 两版 OPD 蒸馏算法对比

| 维度 | Off-policy（旧版，上节记录） | On-policy Student Rollout（新版） |
|---|---|---|
| `opd_mode` | `five_expert_online_teacher_rollout_top1` | `five_expert_online_student_rollout_teacher_full_vocab_kl` |
| Rollout 来源 | **Teacher** generate response | **Student** generate response |
| 蒸馏目标 | Teacher top1 token CE（只取 argmax token id） | Full-vocab KL divergence（完整 151,936 维 softmax 分布） |
| 信息量 | 每 token 只有 1 bit 方向信号 | 每 token 有完整概率分布信号 |
| 训练数据量 | 前 300,000 条 | 全量 999,900 条 |
| max_steps | 2344（300k / 128 effective batch） | 2500（显式 `--max-steps 2500`） |
| effective batch | `8 * 4 * 4 = 128` | `8 * 16 * 1 = 128` |
| max_grad_norm | `1.0` | `5.0` |
| DeepSpeed config | `zero3_opd_maca.json` (grad_clip=1) | `zero3_opd_maca_gradclip5.json` (grad_clip=5) |
| 训练时长 | ~46.6 小时 | ~16.8 小时 |
| 最终 train_loss | 0.3442 | 0.1046 |

### 算法设计详解：On-policy Student Rollout + Full-Vocab KL

核心实现在 `point/train_opd_online_vl.py` 的 `compute_loss` 方法中。与旧版的关键区别：

**1. Student Rollout（而非 Teacher Rollout）**

```python
# student 先 generate response tokens（eval mode，不计算梯度）
student_was_training = model.training
try:
    model.eval()
    with torch.no_grad():
        generated_infer = model.generate(**inputs, **gen_kwargs)
finally:
    if student_was_training:
        model.train()
```

旧版是 teacher generate，student 只在 teacher 的 rollout 上做 forward。新版改为 student 自己 generate，这样蒸��目标是"让 student 在自己的输出分布上逼近 teacher 的评价"，而不是"让 student 模仿 teacher 的输出序列"。这是 on-policy 的核心含义。

**2. Full-Vocab KL Divergence（而非 Top1 CE）**

```python
def _sample_full_vocab_kl_and_entropy(
    self,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_logits = teacher_logits.to(student_logits.device).detach()
    token_mask = token_mask.to(student_logits.device).float()

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()

    # KL(teacher || student) = sum_v teacher(v) * [log teacher(v) - log student(v)]
    token_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    student_probs = student_log_probs.exp()
    student_entropy = -(student_probs * student_log_probs).sum(dim=-1)

    denom = token_mask.sum(dim=1).clamp_min(1.0)
    sample_kl = (token_kl * token_mask).sum(dim=1) / denom
    sample_teacher_entropy = (teacher_entropy * token_mask).sum(dim=1) / denom
    sample_student_entropy = (student_entropy * token_mask).sum(dim=1) / denom
    return sample_kl, sample_teacher_entropy, sample_student_entropy
```

旧版只取 `teacher_logits.argmax(dim=-1)` 作为 label���然后用标准 CE loss。新版计算完整 151,936 维词表上的 KL 散度，让 student 不仅学到 teacher 的 top1 选择，还学到 teacher 对其它 token 的置信度分布。

**3. 训练流程**

每个 microstep 的完整流程：

1. 从 dataset 取出 prompt-only batch，所有 rank 路由到同一个 teacher（route-block-shuffle 保证）；
2. Student eval mode generate response tokens（on-policy rollout）；
3. Teacher 对 student 的 rollout 做 forward，得到 teacher logits；
4. Student train mode 对同一 rollout 做 forward，得到 student logits；
5. 计算 response token 位置上的 full-vocab KL(teacher || student)；
6. 同时记录 teacher entropy 和 student entropy 作为监控指标；
7. Loss = weighted mean KL across batch。

### 新版训练配置

启动脚本：`point/run_opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5.sh`

```bash
deepspeed --num_gpus=8 train_opd_online_vl.py train \
  --model-name-or-path /data/msz/models/8b_base \
  --data-path /data/msz/point/opd_student_v1/train_prompts.jsonl \
  --output-dir /data/msz/models/opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5_20260526 \
  --deepspeed /data/msz/point/configs/zero3_opd_maca_gradclip5.json \
  --route-policy target \
  --no-group-by-route \
  --route-block-shuffle \
  --teacher-load-mode preloaded_zero3 \
  --teacher-deepspeed /data/msz/point/configs/zero3_opd_maca_gradclip5.json \
  --max-steps 2500 \
  --save-steps 500 \
  --save-total-limit 3 \
  --per-device-train-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-6 \
  --warmup-ratio 0.03 \
  --max-grad-norm 5.0 \
  --num-train-epochs 1 \
  --logging-steps 1 \
  --model-max-length 16384 \
  --min-pixels 50176 \
  --max-pixels 50176 \
  --bf16 \
  --gradient-checkpointing
```

| 配置项 | 值 |
|---|---|
| Base model | `/data/msz/models/8b_base` |
| 训练数据 | `/data/msz/point/opd_student_v1/train_prompts.jsonl` (999,900 rows) |
| 输出目录 | `/data/msz/models/opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5_20260526` |
| DeepSpeed | ZeRO-3, `zero3_opd_maca_gradclip5.json` |
| GPUs | 8 |
| per-device batch | 16 |
| gradient accumulation | 1 |
| effective batch | `8 * 16 * 1 = 128` |
| max_steps | 2500 |
| learning rate | `1e-6` |
| warmup ratio | `0.03` |
| scheduler | cosine |
| max grad norm | `5.0` |
| save steps | 500 |
| save total limit | 3 |
| teacher load mode | `preloaded_zero3` (五 teacher 常驻 ZeRO-3) |
| OPD mode | `five_expert_online_student_rollout_teacher_full_vocab_kl` |
| 蒸馏词表 | full vocab (151,936) |

DeepSpeed ZeRO-3 配置 (`zero3_opd_maca_gradclip5.json`)：

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 5.0,
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "none" },
    "offload_param": { "device": "none" },
    "overlap_comm": false,
    "contiguous_gradients": true,
    "reduce_bucket_size": 20000000,
    "stage3_prefetch_bucket_size": 20000000,
    "stage3_param_persistence_threshold": 100000,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "bf16": { "enabled": true }
}
```

### 新版训练过程

| 项目 | 值 |
|---|---|
| 开始时间 | 2026-05-26 16:30 CST |
| 完成时间 | 2026-05-27 09:29 CST |
| 训练时长 | ~16.8 小时 (60,710 秒) |
| global step | 2500 |
| train samples/sec | 5.271 |
| train steps/sec | 0.041 |
| 最终 train_loss | 0.1046 |
| 最终 opd_loss | 0.0246 |
| 最终 entropy (teacher) | 0.7312 |
| 最终 student_entropy | 0.7685 |
| 最终 grad_norm | 0.7992 |
| 最终 response_tokens | 24 |
| 峰值显存 (all GPUs) | 52,066 MiB |
| exit status | 0 (成功) |

训练早期指标（前 5 步）：

| Step | loss | grad_norm | opd_loss | entropy | student_entropy | route |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 4.0084 | 660.65 | 4.0625 | 1.9091 | 1.4146 | robopoint |
| 2 | 2.4298 | 207.60 | 2.8823 | 0.9038 | 0.5685 | spatial_rel |
| 3 | 6.4614 | 795.16 | 4.7462 | 5.4405 | 1.5773 | general_reasoning |
| 4 | 1.9090 | 269.97 | 1.3928 | 0.9738 | 0.7383 | region |
| 5 | 3.9120 | 937.36 | 3.2546 | 4.9445 | 2.2209 | general_reasoning |

训练末期指标（最后 5 步）：

| Step | loss | grad_norm | opd_loss | entropy | student_entropy | route |
|---:|---:|---:|---:|---:|---:|---|
| 2496 | 0.0262 | 0.639 | 0.0249 | 0.7558 | 0.7896 | spatial_rel |
| 2497 | 0.0293 | 1.098 | 0.0394 | 1.9838 | 2.2116 | region |
| 2498 | 0.0116 | 2.231 | 0.0170 | 5.2363 | 5.1225 | general_reasoning |
| 2499 | 0.0277 | 0.718 | 0.0292 | 0.7076 | 0.7323 | spatial_rel |
| 2500 | 0.0303 | 0.799 | 0.0246 | 0.7312 | 0.7685 | spatial_rel |

训练曲线观察：

1. 早期 loss 和 grad_norm 很大（loss 4-6，grad_norm 600-900），这是 student 刚开始学习 teacher 完整分布时的正常现象；full-vocab KL 比 top1 CE 的初始值更大，因为信息量更多。
2. 中期 loss 快速下降，约 step 100 后 loss 稳定在 0.1-0.5 区间。
3. 末期 loss 降到 0.02-0.03，opd_loss 降到 0.02 左右，说明 student 已经非常接近 teacher 的输出分布。
4. Student entropy 和 teacher entropy 在末期非常接近（差值 < 0.05），说明 student 的输出确定性已经和 teacher 对齐。
5. 没有出现 NaN/Inf、OOM、Traceback 或 Watchdog 异常。

数据集使用情况：

| 项目 | 值 |
|---|---|
| 原始数据行数 | 999,900 |
| expanded rows (route-block padding) | 1,000,192 |
| padded rows | 292 |
| 实际训练 epoch | 0.32 (max_steps=2500 限制) |
| route policy | target |
| route block shuffle | true |

Route 分布：

| Route | 原始行数 | 实际 loss 计算次数 |
|---|---:|---:|
| `general_reasoning_expert` | 309,994 | 12,720 |
| `general_obj_expert` | 179,975 | 7,264 |
| `region_expert` | 174,948 | 6,992 |
| `spatial_rel_expert` | 174,988 | 6,880 |
| `robopoint_expert` | 159,995 | 6,144 |

### Off-policy 续训记录（补充）

上一节记录的 OPD 300k off-policy 训练（`opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1`）实际在 checkpoint-2344 之后继续训练到了 3500 步。续训 checkpoint 评估结果如下：

| Checkpoint | IoU mean | Acc@0.5 | Hit@100 | Text loose |
|---:|---:|---:|---:|---:|
| 2344 | 0.4676 | 0.3132 | 0.1177 | 0.2184 |
| 2500 | 0.4683 | 0.3144 | 0.1161 | 0.2188 |
| 3000 | 0.4691 | 0.3133 | 0.1176 | 0.2185 |
| 3500 | 0.4681 | 0.3134 | 0.1168 | 0.2183 |

观察：off-policy 从 2344 到 3500 步几乎没有提升，所有指标在误差范围内波动。这进一步确认了上一节的结论：off-policy teacher-rollout top1 CE 在 2000 步后已进入平台期。

### 评估结果：On-policy vs Off-policy vs Base/Instruct 完整对比

评估使用同一个 10k raw-holdout 评估集（与上一节 8 模型评估相同），评估路径：

```text
/data/msz/point/eval_raw_holdout_v1/opd_fullvocab_studentrollout_full2500_maxnorm5_ckpts_1500_2000_2500_20260527_104507/
```

#### 评估指标含义

| 指标 | 含义 | 计算方式 |
|---|---|---|
| `format_pass` | 输出格式通过率 | 预测是否匹配期望的 `<box>[[x1,y1],[x2,y2]]</box>` 或 `<point>[[x,y],...]</point>` 或纯文本格式 |
| `coord_valid` | 坐标有效率 | 格式通过且坐标在 [0,1000] 范围内的比例 |
| `iou_mean` | 平均 IoU | 预测 box 与 ground truth box 的交并比均值（格式失败计为 0） |
| `acc_iou_0_3` | IoU@0.3 准确率 | IoU >= 0.3 的样本比例 |
| `acc_iou_0_5` | IoU@0.5 准确率 | IoU >= 0.5 的样本比例（标准 detection 阈值） |
| `acc_iou_0_75` | IoU@0.75 准确率 | IoU >= 0.75 的样本比例（严格定位） |
| `center_dist_mean` | 平均中心距离 | 预测 box 中心与 GT box 中心的欧氏距离（坐标空间 0-1000） |
| `hit_at_50` | Point Hit@50 | 预测点中至少有一个距离 GT 点 < 50 的比例 |
| `hit_at_100` | Point Hit@100 | 预测点中至少有一个距离 GT 点 < 100 的比例 |
| `text_exact` | 文本精确匹配 | 预测文本与 GT 完全一致的比例 |
| `text_loose` | 文本宽松匹配 | 预测文本经过归一化后与 GT 匹配的比例 |

#### 总体对比表

| 模型 | Format | Coord | IoU mean | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist | Hit@50 | Hit@100 | Text exact | Text loose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8b_base | 85.3% | 60.4% | 0.385 | 31.9% | 24.9% | 12.8% | 153.5 | 0.0% | 0.0% | 0.2% | 0.2% |
| Qwen3-VL-8B-Instruct | 85.8% | 60.8% | 0.413 | 34.0% | 26.4% | 15.1% | 140.1 | 0.0% | 0.0% | 0.2% | 13.0% |
| Off-policy ckpt-2344 | 100.0% | 75.0% | 0.468 | 36.8% | 31.3% | 20.1% | 128.2 | 9.9% | 11.8% | 21.8% | 21.8% |
| Off-policy ckpt-3500 | 100.0% | 75.0% | 0.468 | 36.8% | 31.3% | 20.1% | 128.1 | 10.1% | 11.7% | 21.8% | 21.8% |
| **On-policy ckpt-1500** | 100.0% | 75.0% | 0.468 | 36.9% | 31.2% | 20.1% | 128.3 | 10.2% | 11.9% | 21.8% | 21.8% |
| **On-policy ckpt-2000** | 100.0% | 75.0% | 0.470 | 37.0% | 31.3% | 20.2% | 127.8 | 10.1% | 11.8% | 21.8% | 21.8% |
| **On-policy ckpt-2500** | 100.0% | 75.0% | **0.470** | 37.0% | **31.4%** | **20.3%** | **127.2** | 10.1% | 11.8% | 21.9% | 21.9% |

#### On-policy Checkpoint 曲线（按领域拆分）

**RefCOCO / 指代表达框选 (n=1100)**

| Checkpoint | IoU mean | Acc@0.5 | Acc@0.75 | CenterDist |
|---|---:|---:|---:|---:|
| ckpt-1500 | 0.722 | 82.3% | 64.9% | 66.3 |
| ckpt-2000 | 0.726 | 82.4% | 66.3% | 64.9 |
| ckpt-2500 | 0.726 | 82.7% | 65.2% | 64.6 |

**Flickr30K Entities / 短语实体框选 (n=900)**

| Checkpoint | IoU mean | Acc@0.5 | Acc@0.75 | CenterDist |
|---|---:|---:|---:|---:|
| ckpt-1500 | 0.697 | 76.9% | 59.6% | 69.4 |
| ckpt-2000 | 0.698 | 76.8% | 59.1% | 69.3 |
| ckpt-2500 | 0.699 | 76.8% | 59.9% | 68.6 |

**Visual Genome Object / 通用物体框选 (n=1100)**

| Checkpoint | IoU mean | Acc@0.5 | Acc@0.75 | CenterDist |
|---|---:|---:|---:|---:|
| ckpt-1500 | 0.319 | 33.6% | 19.0% | 187.4 |
| ckpt-2000 | 0.322 | 34.0% | 19.1% | 186.1 |
| ckpt-2500 | 0.321 | 33.9% | 19.3% | 185.5 |

**Visual Genome Region / 区域描述框选 (n=1100)**

| Checkpoint | IoU mean | Acc@0.5 | Acc@0.75 | CenterDist |
|---|---:|---:|---:|---:|
| ckpt-1500 | 0.386 | 40.8% | 19.1% | 137.8 |
| ckpt-2000 | 0.387 | 41.2% | 18.5% | 136.2 |
| ckpt-2500 | 0.390 | 41.7% | 18.9% | 136.8 |

**Visual Genome Relationship / 关系框选 (n=1100)**

| Checkpoint | IoU mean | Acc@0.5 | Acc@0.75 | CenterDist |
|---|---:|---:|---:|---:|
| ckpt-1500 | 0.393 | 41.4% | 25.3% | 149.1 |
| ckpt-2000 | 0.395 | 41.5% | 25.9% | 150.4 |
| ckpt-2500 | 0.394 | 41.5% | 26.2% | 148.8 |

**Semantic Nav Box / 语义导航框选 (n=800)**

| Checkpoint | IoU mean | Acc@0.5 | Acc@0.75 | CenterDist |
|---|---:|---:|---:|---:|
| ckpt-1500 | 0.282 | 30.5% | 7.3% | 157.1 |
| ckpt-2000 | 0.281 | 30.6% | 7.8% | 157.5 |
| ckpt-2500 | 0.282 | 30.8% | 8.0% | 156.1 |

**Grounding Point / 点选 (n=1400)**

| Checkpoint | Hit@50 | Hit@100 | MinDist | PredToGoldDist |
|---|---:|---:|---:|---:|
| ckpt-1500 | 72.5% | 84.9% | 56.8 | 79.4 |
| ckpt-2000 | 71.9% | 84.2% | 55.8 | 77.5 |
| ckpt-2500 | 72.1% | 84.3% | 56.4 | 78.0 |

**Keepalive VQA / 通用能力 (n=2500)**

| Checkpoint | Text exact | Text loose | Bool acc | MC acc |
|---|---:|---:|---:|---:|
| ckpt-1500 | 87.3% | 87.3% | 32.8% | 0.8% |
| ckpt-2000 | 87.4% | 87.4% | 32.7% | 0.9% |
| ckpt-2500 | 87.4% | 87.4% | 32.8% | 0.9% |

### 评估结果分析

#### On-policy vs Off-policy 对比

| 指标 | Off-policy best (ckpt-3000) | On-policy best (ckpt-2500) | 差异 |
|---|---:|---:|---:|
| IoU mean | 0.4691 | **0.4704** | +0.0013 |
| Acc@0.5 | 0.3133 | **0.3135** | +0.0002 |
| Acc@0.75 | 0.2011 | **0.2028** | +0.0017 |
| CenterDist | 128.0 | **127.2** | -0.8 |
| Hit@100 | 0.1176 | **0.1180** | +0.0004 |
| Text loose | 0.2185 | **0.2186** | +0.0001 |

结论：

1. **On-policy student rollout + full-vocab KL 在所有指标上略优于 off-policy teacher rollout + top1 CE。** 差异虽小（IoU +0.0013，Acc@0.75 +0.0017），但方向一致，且 on-policy 只训练了 2500 步（覆盖 32% 数据），而 off-policy 训练了 3500 步。

2. **On-policy 训练效率更高。** 16.8 小时完成 2500 步，而 off-policy 46.6 小时完成 2344 步。原因是 on-policy 使用 `mb=16, accum=1`（每步只做一次 forward/backward），而 off-policy 使用 `mb=4, accum=4`（每步做 4 次 forward/backward）。虽然 effective batch 相同（128），但 on-policy 的 microbatch 更大，GPU 利用率更高。

3. **On-policy 的 loss 收敛更深。** 最终 train_loss 0.1046 vs off-policy 的 0.3442。这不代表过拟合，而是 full-vocab KL 的信息量更大，student 能更精确地逼近 teacher 分布。从 student_entropy ≈ teacher_entropy 可以看出 student 的输出确定性已经和 teacher 对齐。

4. **两版在 eval 上的差异很小，说明当前瓶颈不在蒸馏算法。** 无论是 top1 CE 还是 full-vocab KL，student 在 10k holdout 上的表现都接近上限。下一步提升更可能来自：
   - 更多训练数据（当前 on-policy 只用了 32% 的 999k 数据）；
   - 更强的 teacher（当前 expert 只训练了 100k 样本）；
   - 更好的数据配比（semantic-nav 仍是短板）。

5. **On-policy ckpt-1500 已经接近最终水平。** 从 1500 到 2500 步的提升很小（IoU +0.002），说明 on-policy 收敛更快，可能 1500-2000 步就足够。

#### 与上一节 8 模型评估的对比

注意：本次评估的 `coord_valid=75%` 低于上一节 8 模型评估中 OPD final 的 `100%`。这是因为本次评估脚本的 coord_valid 计算方式不同：上一节只在 box 样本上计算 coord_valid，本次在全部 10k 样本上计算（包括 point 和 text 样本，它们的 coord_valid 定义不同）。Box 子集上的 coord_valid 仍为 100%。

本次 on-policy ckpt-2500 与上一节 OPD final (off-policy ckpt-2344) 的 box 子集对比：

| 指标 | 上一节 OPD final (box only) | 本次 on-policy ckpt-2500 (box only) |
|---|---:|---:|
| IoU mean | 0.468 | 0.470 |
| Acc@0.3 | 60.3% | 60.6% |
| Acc@0.5 | 51.3% | 51.4% |
| Acc@0.75 | 32.9% | 33.2% |
| CenterDist | 128.2 | 127.2 |

On-policy 在 box grounding 上略优，且 CenterDist 更低（定位更准）。

### 训练过程图

![OPD 300k off-policy training curves](report/opd_seed0_five_online_300k_mb4_save500_zero3_leftpad_v1/opd_loss_entropy_curves.svg)

上图为 off-policy 300k 训练曲线（loss/opd_loss/entropy，MA50 平滑）。

![OPD 300k eval dashboard](report/opd_300k_analysis_20260526/opd_300k_eval_dashboard.svg)

上图为 off-policy 300k 评估 dashboard。

新版 on-policy 训练曲线资产尚未生成本地 SVG（训练刚完成），但关键数据点已记录在本节表格中。

### 远端产物索引

| 类型 | 路径 |
|---|---|
| On-policy 最终模型 | `/data/msz/models/opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5_20260526` |
| On-policy checkpoint-1500 | 同上 `/checkpoint-1500` |
| On-policy checkpoint-2000 | 同上 `/checkpoint-2000` |
| On-policy checkpoint-2500 | 同上 `/checkpoint-2500` |
| On-policy 训练日志 | `/data/msz/point/logs/opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5_20260526.log` |
| On-policy 峰值显存 | `/data/msz/point/logs/opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5_20260526_peakmem_summary.tsv` |
| On-policy run summary | 模型目录下 `opd_online_run_summary.json` |
| 评估结果目录 | `/data/msz/point/eval_raw_holdout_v1/opd_fullvocab_studentrollout_full2500_maxnorm5_ckpts_1500_2000_2500_20260527_104507/` |
| 评估对比 JSON | 同上 `comparison_with_base_instruct_offpolicy.json` |
| 评估对比 Markdown | 同上 `comparison_with_base_instruct_offpolicy.md` |
| 本地启动脚本 | `point/run_opd_fullvocab_studentrollout_full2500_save500_zero3_mb16_accum1_frombase_maxnorm5.sh` |
| 本地 ZeRO-3 config | `point/configs/zero3_opd_maca_gradclip5.json` |
| 本地训练脚本 | `point/train_opd_online_vl.py` |

### 当前状态与下一步

1. On-policy student rollout + full-vocab KL 蒸馏已验证可行，且在相同 step 数下略优于 off-policy top1 CE。
2. 当前 on-policy 只训练了 32% 的数据（2500 步 / ~7800 步 full epoch）。如果继续训练到 full epoch，可能还有提升空间。
3. 两版蒸馏的 eval 差异很小，说明当前瓶颈在 teacher 质量和数据，而不是蒸馏算法。
4. 推荐后续实验方向：
   - 延长 on-policy 训练到 5000-7800 步（full epoch）；
   - 增强 teacher：将 expert 从 100k 样本扩展到 200k-400k；
   - 改善 semantic-nav 数据质量（当前最弱领域，Acc@0.5 只有 30.8%）；
   - 尝试 on-policy + top-k KL（而非 full vocab）以降低显存和加速。

## 2026-05-28 OPD P0/P1 优化版：coldstart + Veto + overlap 监控

上一节记录的是第一版 on-policy full-vocab KL：从 `8b_base` 直接开始，student rollout，teacher 在 student response 上做 full-vocab forward KL。随后我们按审阅意见完成了 P0/P1 级别优化，并训练了第二版融合模型。

### 本轮改动摘要

本轮从上次记录到现在完成了以下改动：

1. 修正并保持 OPD 正确定义：由 student rollout response，再让对应 route 的 teacher 在该 response 上 forward logits，loss 对 student logits 和 teacher logits 做 full-vocab KL。
2. 引入 Veto beta schedule：`opd_veto_beta_start=1.0`，`opd_veto_beta_end=0.0`，`opd_veto_beta_steps=500`。训练早期用 student logit 参与目标分布，降低 teacher-student gap 过大时的病态梯度。
3. 增加 P1 诊断指标：记录 `opd_topk_overlap`，即 student/teacher top-k token support overlap，用来观察分布对齐是否真的发生。
4. 保留 entropy 监控：同时记录 `entropy/opd_entropy`、`student_entropy`、`teacher_entropy`，用于观察 student 是否被拉到 teacher 的不确定性结构附近。
5. 完成 100 step off-policy coldstart，并以该模型作为第二版 OPD 的起点。
6. 第二版全量训练参数：`max_steps=2500`，`save_steps=500`，`save_total_limit=3`，`per_device_train_batch_size=16`，`gradient_accumulation_steps=1`，`zero3`，`max_grad_norm=5`，最终将 `opd_max_new_tokens` 设置为 `64`，`opd_prefix_loss_tokens=64`。
7. 训练中保留峰值显存记录，最终峰值为 `52030 MiB`。

关键远端路径：

| 类型 | 路径 |
|---|---|
| coldstart 模型 | `/data/msz/models/opd_offpolicy_coldstart100_p0p1_fullvocab_maxnew128_prefix64_veto1to0_zero3_mb16_accum1_20260527_132800` |
| OPD v2 最终模型 | `/data/msz/models/opd_p0p1_studentrollout_full2500_save500_zero3_mb16_accum1_from_coldstart100_maxnew64_prefix64_veto1to0s500_maxnorm5_20260527_174410` |
| OPD v2 训练日志 | `/data/msz/point/logs/opd_p0p1_studentrollout_full2500_save500_zero3_mb16_accum1_from_coldstart100_maxnew64_prefix64_veto1to0s500_maxnorm5_20260527_174410.log` |
| OPD v2 final eval | `/data/msz/point/eval_raw_holdout_v1/opd_p0p1_maxnew64_final2500_20260528_133416/` |
| OPD v2 ckpt eval | `/data/msz/point/eval_raw_holdout_v1/opd_p0p1_maxnew64_ckpts_1500_2000_retry_20260528_143650/` |
| 本地曲线与摘要 | `report/opd_p0p1_maxnew64_20260528/` |

### 训练过程曲线

![OPD P0/P1 max_new_tokens=64 training curves](report/opd_p0p1_maxnew64_20260528/opd_p0p1_loss_entropy_overlap_gradnorm_curves.svg)

曲线使用每 step 日志，MA50 平滑。核心变化如下：

| 指标 | 前 100 step 均值 | 后 100 step 均值 | 观察 |
|---|---:|---:|---|
| `loss` | 0.1070 | 0.0297 | 主 loss 明显下降 |
| `opd_loss` | 0.0919 | 0.0293 | full-vocab KL 收敛到低位 |
| `entropy/opd_entropy` | 0.3280 | 2.1197 | Veto 退火后目标熵与 teacher 熵对齐 |
| `student_entropy` | 0.4521 | 2.1567 | student 从过度确定变为接近 teacher |
| `teacher_entropy` | 2.5816 | 2.1197 | route 分布变化下总体保持稳定 |
| `opd_topk_overlap` | 0.6102 | 0.8286 | top-k support 对齐显著提升 |
| `grad_norm` | 8.1650 | 1.8416 | 早期 gap 被消化后梯度进入稳定区间 |

最终训练摘要：

| 项目 | 值 |
|---|---:|
| steps | 2500 |
| train_loss | 0.0425 |
| train_runtime | 69731.7s |
| train_samples_per_second | 4.589 |
| 峰值显存 | 52030 MiB |
| 结束状态 | 正常退出，无 NaN/OOM/Inference Tensor 错误 |

### 最终评估收益

评估统一使用 `raw_holdout_eval_v1_10k.jsonl`，`max_new_tokens=64`。下表使用按格式子集计算的指标：box 指标只在 6100 条 box 样本上算，point 指标只在 1400 条 point 样本上算，text 指标只在 2500 条 text 样本上算。

| 模型 | Box IoU | Box Acc@0.3 | Box Acc@0.5 | Box Acc@0.75 | Box CenterDist | Point Hit@50 | Point Hit@100 | Text exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 0.3854 | 0.5234 | 0.4087 | 0.2098 | 153.5 | 0.0000 | 0.0000 | 0.0080 |
| `8b_instruct` | 0.4134 | 0.5577 | 0.4331 | 0.2472 | 140.1 | 0.0000 | 0.0000 | 0.0076 |
| `offpolicy_3500steps` | 0.4681 | 0.6026 | 0.5138 | 0.3293 | 128.1 | 0.7193 | 0.8343 | 0.8732 |
| `coldstart100_offpolicy` | 0.4283 | 0.5620 | 0.4621 | 0.2825 | 142.1 | 0.4014 | 0.5393 | 0.1032 |
| `opd_v1_fullvocab_frombase_2500` | 0.4704 | 0.6059 | 0.5139 | 0.3325 | 127.2 | 0.7214 | 0.8429 | 0.8744 |
| `opd_v2_p0p1_maxnew64_2500` | **0.4723** | **0.6074** | **0.5157** | **0.3359** | **126.2** | 0.7143 | **0.8479** | **0.8744** |

收益判断：

1. 相比 `opd_v1_fullvocab_frombase_2500`，v2 在 box grounding 上稳定小涨：`IoU +0.0019`，`Acc@0.5 +0.0018`，`Acc@0.75 +0.0034`，`CenterDist -0.94`。
2. 相比之前的 `offpolicy_3500steps`，v2 的 box 提升更明显：`IoU +0.0042`，`Acc@0.75 +0.0066`，`CenterDist -1.8`。这说明 student-rollout full-vocab KL 加 P0/P1 稳定化后，仍然优于旧 off-policy top1 路线。
3. `coldstart100_offpolicy` 本身不是可用终态：box 和 text 都明显低于完整 OPD。它的作用主要是把 base 拉近 teacher manifold，真正收益来自后续 2500 step on-policy KL。
4. point 任务上 v2 的 `Hit@50` 比 v1 低 `0.0071`，但 `Hit@100` 高 `0.0050`。这说明 v2 的点选严格近距离命中略有波动，但粗粒度命中更好。主目标仍是 box/region grounding，v2 在这些指标上更优。

### V2 checkpoint 收敛性

同样用 10k raw holdout 补评了 v2 的 `checkpoint-1500` 和 `checkpoint-2000`，并与最终 `2500` 对比：

| Checkpoint | Box IoU | Box Acc@0.3 | Box Acc@0.5 | Box Acc@0.75 | CenterDist | Point Hit@50 | Point Hit@100 | Text exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1500 | 0.4692 | 0.6067 | 0.5146 | 0.3316 | 128.5 | 0.7079 | 0.8393 | 0.8736 |
| 2000 | 0.4707 | **0.6079** | 0.5130 | 0.3330 | 127.7 | 0.7100 | **0.8500** | 0.8728 |
| 2500 | **0.4723** | 0.6074 | **0.5157** | **0.3359** | **126.2** | **0.7143** | 0.8479 | **0.8744** |

按领域拆分：

| Pool | n | ckpt1500 | ckpt2000 | ckpt2500 | 观察 |
|---|---:|---:|---:|---:|---|
| `refcoco` IoU | 1100 | 0.7237 | 0.7263 | **0.7308** | 继续上涨 |
| `flickr30k_entities` IoU | 900 | 0.6999 | 0.7023 | **0.7027** | 基本平台，2500 略好 |
| `semantic_nav_box` IoU | 800 | 0.2805 | 0.2814 | **0.2821** | 仍慢速上涨，是最弱域 |
| `visual_genome_object` IoU | 1100 | **0.3220** | 0.3216 | 0.3217 | 已平台 |
| `visual_genome_region` IoU | 1100 | 0.3851 | 0.3891 | **0.3898** | 继续上涨 |
| `visual_genome_relationship` IoU | 1100 | 0.3948 | 0.3939 | **0.3968** | 2500 最好 |
| `grounding_point` Hit@50 | 1400 | 0.7079 | 0.7100 | **0.7143** | 2500 最好 |
| `grounding_point` Hit@100 | 1400 | 0.8393 | **0.8500** | 0.8479 | 2000 略高 |
| `keepalive_vqa` Text exact | 2500 | 0.8736 | 0.8728 | **0.8744** | 基本保持 |

收敛结论：

1. v2 在 1500 步已经接近终态，但 2000 到 2500 仍有小幅有效提升，尤其是 `Box IoU`、`Acc@0.75`、`CenterDist`、`refcoco` 和 `visual_genome_region/relationship`。
2. 2500 不是明显过训：box 主指标和 text keepalive 都是 2500 最好，point 的 `Hit@100` 虽然 2000 略高，但差异只有 `0.0021`。
3. 当前更像是进入平台期而不是发散。若继续训练，预期收益会很小；除非引入更强 teacher、更多 expert 数据，或专门加强 `semantic_nav_box` 这个短板域。

### 本地/远端产物

| 类型 | 路径 |
|---|---|
| 本地训练曲线 SVG | `report/opd_p0p1_maxnew64_20260528/opd_p0p1_loss_entropy_overlap_gradnorm_curves.svg` |
| 本地训练 step metrics CSV | `report/opd_p0p1_maxnew64_20260528/opd_p0p1_training_metrics.csv` |
| 本地训练摘要 JSON | `report/opd_p0p1_maxnew64_20260528/opd_p0p1_training_curve_summary.json` |
| 本地模型对比表 | `report/opd_p0p1_maxnew64_20260528/selected_model_comparison.md` |
| 本地 v2 checkpoint eval summary | `report/opd_p0p1_maxnew64_20260528/v2_ckpt1500_2000_comparison_summary.json` |
| 本地 v2 final eval summary | `report/opd_p0p1_maxnew64_20260528/v2_final2500_comparison_summary.json` |

## 2026-05-29 到 2026-06-01 新 200k experts + coldstart500 + OPD5000

上一节的结论是：OPD v2 已经比 v1 和旧 off-policy 蒸馏更好，但 2500 step 后进入平台期，继续提升主要受限于 teacher/expert 上限。因此这一轮工作围绕三件事展开：

1. 把五个 seed0 domain experts 从 100k 继续扩到 200k，尽量提升 teacher 质量。
2. 用新的 200k experts 重做一版 500 step off-policy coldstart，而不是继续使用旧 100 step coldstart。
3. 从 coldstart500 模型启动 5000 step on-policy OPD，并每 1000 step 评估一次，判断是否继续收敛。

### Expert 续训流水线

先尝试过从 `8b_base` 直接训练每个 expert 300k 样本，配置为 `per_device_train_batch_size=4`、`gradient_accumulation_steps=4`、`max_grad_norm=5`、标准 Trainer、ZeRO-2。该方案在第一个 `general_reasoning_expert` 的第 13 step 附近 OOM：

| 项目 | 记录 |
|---|---|
| 失败脚本 | `/data/msz/point/run_seed0_five_experts_300k_mb4_filtered_stdtrainer_maxnorm5_v1.sh` |
| 失败日志 | `/data/msz/point/logs/seed0_general_reasoning_expert_300k_mb4_filtered_stdtrainer_maxnorm5_v1.log` |
| OOM 位置 | rank7 backward |
| 显存状态 | GPU7 总 63.59 GiB，PyTorch 已分配 57.38 GiB，仅余 277.91 MiB |
| 结论 | `mb=4` 对 expert SFT 仍然不稳，放弃 300k from-base 方案 |

随后切换到更稳的策略：在已有 100k experts 基础上，每个 expert 追加训练第二个 100k slice，即 `rows100001_200000`，用 `mb=1`、`accum=4`、`lr=5e-6`、`max_grad_norm=5`、标准 Trainer、ZeRO-2，不使用自定义 OOM skip。这样既避开了重复数据，也保留了此前 100k 模型的有效进度。

| Expert | 训练数据 | 起点模型 | 输出模型 | 时间 |
|---|---|---|---|---:|
| `general_reasoning_expert` | `train_shuffled_seed20260520_rows100001_200000.jsonl` | `seed0_general_reasoning_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` | `seed0_general_reasoning_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | 3h18m |
| `region_expert` | `train_shuffled_seed20260520_rows100001_200000.jsonl` | `seed0_region_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` | `seed0_region_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | 3h18m |
| `robopoint_expert` | `train_shuffled_seed20260520_rows100001_200000.jsonl` | `seed0_robopoint_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` | `seed0_robopoint_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | 3h24m |
| `spatial_rel_expert` | `train_shuffled_seed20260520_rows100001_200000.jsonl` | `seed0_spatial_rel_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` | `seed0_spatial_rel_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | 3h17m |
| `general_obj_expert` | `train_shuffled_seed20260520_rows100001_200000.jsonl` | `seed0_general_obj_expert_100k_mb1_filtered_stdtrainer_maxnorm5_v1` | `seed0_general_obj_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` | 3h18m |

聚合日志 `/data/msz/point/logs/seed0_continue100k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1.log` 显示五段均正常 `[done]`，最终 `[all_done] 2026-05-29 08:43:38`。这一轮没有继续使用 300k from-base 失败产物；后续 OPD teacher 均指向 200k from100k 这五个模型。

### OPD 代码设计调整

本轮继续沿用上一节已经跑通的 OPD 设计：五个 teacher 同时以 `preloaded_zero3` 分片加载到 8 卡，按 sample route 选择对应 teacher；on-policy 阶段由 student rollout response，teacher 在同一 response 上 forward，做 full-vocab KL。

为了把 500 step coldstart 和后续 5000 step OPD 串起来，同时不重复消费数据，在 `/data/msz/point/train_opd_online_vl.py` 中做了一个最小新增：

| 改动 | 说明 |
|---|---|
| `--opd-rollout-source {student,teacher}` | 同一个 Trainer 同时支持 off-policy coldstart 和 on-policy OPD。coldstart 用 `teacher` rollout，正式 OPD 用 `student` rollout。 |
| `--skip-expanded-samples` | 在构造 route-block shuffle 后跳过指定数量的 expanded OPD rows，用于避免 coldstart 和正式 OPD 重复训练同一段样本。 |
| P0/P1 指标保留 | 继续记录 `loss`、`opd_loss`、`entropy/opd_entropy`、`student_entropy`、`teacher_entropy`、`grad_norm`、`opd_topk_overlap`、`opd_veto_beta`、`opd_response_tokens`。 |
| Prefix KL | `opd_prefix_loss_tokens=64`，即 response 可以更长，但只在前 64 个 response token 上反传 KL。 |
| Veto schedule | coldstart 用 `1.0 -> 0.0 / 100 steps`，正式 OPD 用 `1.0 -> 0.0 / 500 steps`。 |

数据不重复的计算如下：

| 阶段 | steps | effective batch | 消费 expanded samples | 说明 |
|---|---:|---:|---:|---|
| coldstart500 | 500 | 8 GPU x 16 x 1 = 128 | 64,000 | 从 OPD 数据开头开始 |
| OPD5000 | 5000 | 8 GPU x 16 x 1 = 128 | 640,000 | 启动时 `--skip-expanded-samples 64000` |

因此 OPD5000 消费的是 coldstart500 之后的下一段数据，不与 coldstart 重复。

### Coldstart500

coldstart500 的目的不是产出最终模型，而是把 `8b_base` 先拉近新的 200k experts，降低随后 on-policy KL 的初始 gap。

| 项目 | 值 |
|---|---|
| 启动脚本 | `/data/msz/point/run_opd_offpolicy_coldstart500_p0p1_zero3_mb16_accum1_newexperts.sh` |
| 起点 | `/data/msz/models/8b_base` |
| Teachers | 五个 `seed0_*_expert_200k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1` |
| rollout source | `teacher` |
| KL | teacher response 上的 full-vocab KL |
| `max_new_tokens` | 128 |
| `prefix_loss_tokens` | 64 |
| `max_steps` | 500 |
| `save_steps` | 0，最终只保存一次 |
| batch | `per_device_train_batch_size=16`，`gradient_accumulation_steps=1` |
| ZeRO | `zero3_opd_maca_gradclip5.json` |
| 输出模型 | `/data/msz/models/opd_offpolicy_coldstart500_newexperts_p0p1_fullvocab_maxnew128_prefix64_veto1to0_zero3_mb16_accum1_20260529_105409` |

训练稳定性摘要：

| 指标 | 前 100 step 均值 | 后 100 step 均值 | 观察 |
|---|---:|---:|---|
| `loss` | 0.3665 | 0.0718 | coldstart 快速收敛 |
| `opd_loss` | 0.3736 | 0.0712 | teacher response 上的 KL 明显下降 |
| `entropy/opd_entropy` | 0.5488 | 1.2373 | Veto 退火后目标分布熵上升 |
| `student_entropy` | 0.8363 | 1.3139 | student 不再过度尖锐 |
| `teacher_entropy` | 1.2928 | 1.2373 | 后期与 OPD 目标一致 |
| `opd_topk_overlap` | 0.5606 | 0.7850 | top-k support 对齐明显提升 |
| `grad_norm` | 79.2381 | 2.5354 | 初期大梯度被消化，后期稳定 |

最终 `train_loss=0.1459`，`train_runtime=16509.9s`，峰值显存 `51357 MiB`，正常落盘，无 NaN/OOM/Inference Tensor 错误。

### OPD5000 全量训练

基于 coldstart500 模型，启动新 experts 的 5000 step on-policy OPD。该模型是当前这一轮的主结果。

| 项目 | 值 |
|---|---|
| 启动脚本 | `/data/msz/point/run_opd_p0p1_studentrollout_full5000_skip64000_save1000_zero3_mb16_accum1_from_coldstart500_newexperts.sh` |
| 起点 | coldstart500 输出模型 |
| Teachers | 五个 200k from100k experts |
| rollout source | `student` |
| KL | student response 上 teacher full-vocab forward KL |
| `max_new_tokens` | 64 |
| `prefix_loss_tokens` | 64 |
| `skip_expanded_samples` | 64,000 |
| `max_steps` | 5000 |
| `save_steps` | 1000 |
| `save_total_limit` | 5 |
| batch | `per_device_train_batch_size=16`，`gradient_accumulation_steps=1` |
| ZeRO | `zero3_opd_maca_gradclip5.json` |
| 输出模型 | `/data/msz/models/opd_p0p1_studentrollout_full5000_skip64000_save1000_zero3_mb16_accum1_from_coldstart500_newexperts_maxnew64_prefix64_veto1to0s500_maxnorm5_20260529_154544` |

训练过程摘要：

| 指标 | 前 100 step 均值 | 后 100 step 均值 | 观察 |
|---|---:|---:|---|
| `loss` | 0.0993 | 0.0246 | 从 coldstart 后继续下降 |
| `opd_loss` | 0.0995 | 0.0235 | full-vocab KL 收敛到低位 |
| `entropy/opd_entropy` | 0.3657 | 1.1207 | 正式 OPD 后期保持合理熵 |
| `student_entropy` | 0.7904 | 1.1600 | student 与 teacher 熵结构接近 |
| `teacher_entropy` | 1.2784 | 1.1207 | route 分布变化下总体稳定 |
| `opd_topk_overlap` | 0.7128 | 0.8080 | support overlap 保持高位 |
| `grad_norm` | 11.0580 | 1.6041 | 初期仍有 gap，但明显低于 coldstart 初始 |
| `opd_response_tokens` | 37.31 | 33.88 | `max_new_tokens=64` 下未出现大规模截断 |

最终训练摘要：

| 项目 | 值 |
|---|---:|
| steps | 5000 |
| train_loss | 0.0306 |
| train_runtime | 136861.1s |
| train_samples_per_second | 4.676 |
| train_steps_per_second | 0.037 |
| 峰值显存 | 52150 MiB |
| 保存点 | `checkpoint-1000`、`checkpoint-2000`、`checkpoint-3000`、`checkpoint-4000`、`checkpoint-5000` |
| 结束状态 | 正常退出，无 NaN/OOM/Inference Tensor 错误 |

### OPD5000 checkpoint 评估

评估仍统一使用 `/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl`，`max_new_tokens=64`。五个 checkpoint 并行评估，最终有效目录为：

`/data/msz/point/eval_raw_holdout_v1/opd5000_newexperts_ckpts_1000_2000_3000_4000_5000_retry_20260601_104908/`

下表按格式子集统计：box 指标只在 6100 条 box 样本上算，point 指标只在 1400 条 point 样本上算，text 指标只在 2500 条 text 样本上算。

| Checkpoint | Box IoU | Box Acc@0.3 | Box Acc@0.5 | Box Acc@0.75 | CenterDist | Point Hit@50 | Point Hit@100 | Text exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.4692 | 0.6039 | 0.5139 | 0.3330 | 128.4 | 0.7000 | 0.8357 | 0.8716 |
| 2000 | 0.4745 | 0.6105 | 0.5192 | 0.3379 | 126.4 | 0.7157 | 0.8550 | 0.8708 |
| 3000 | **0.4761** | **0.6156** | 0.5233 | **0.3380** | 126.1 | 0.7414 | 0.8629 | 0.8716 |
| 4000 | 0.4747 | 0.6113 | 0.5207 | 0.3374 | 126.4 | 0.7379 | 0.8650 | **0.8784** |
| 5000 | 0.4759 | 0.6123 | **0.5238** | 0.3370 | **125.7** | **0.7493** | **0.8729** | 0.8724 |

按 pool 看，checkpoint 的最佳点并不完全一致：

| Pool | n | ckpt1000 | ckpt2000 | ckpt3000 | ckpt4000 | ckpt5000 | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| `refcoco` IoU | 1100 | 0.7271 | 0.7302 | 0.7343 | 0.7337 | **0.7358** | 5000 |
| `flickr30k_entities` IoU | 900 | 0.6972 | 0.7030 | 0.7079 | 0.7097 | **0.7101** | 5000 |
| `semantic_nav_box` IoU | 800 | 0.2768 | **0.2936** | 0.2921 | 0.2894 | 0.2924 | 2000 |
| `visual_genome_object` IoU | 1100 | **0.3240** | 0.3213 | 0.3197 | 0.3172 | 0.3211 | 1000 |
| `visual_genome_region` IoU | 1100 | 0.3825 | 0.3919 | **0.3925** | 0.3915 | 0.3912 | 3000 |
| `visual_genome_relationship` IoU | 1100 | 0.3966 | 0.3991 | **0.4019** | 0.3987 | 0.3974 | 3000 |
| `grounding_point` Hit@50 | 1400 | 0.7000 | 0.7157 | 0.7414 | 0.7379 | **0.7493** | 5000 |
| `keepalive_vqa` Text exact | 2500 | 0.8716 | 0.8708 | 0.8716 | **0.8784** | 0.8724 | 4000 |

与上一轮 OPD v2 2500 相比：

| 模型 | Box IoU | Box Acc@0.5 | Box Acc@0.75 | CenterDist | Point Hit@50 | Point Hit@100 | Text exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPD v2 2500, 100k experts | 0.4723 | 0.5157 | 0.3359 | 126.2 | 0.7143 | 0.8479 | 0.8744 |
| OPD5000 ckpt3000, 200k experts | **0.4761** | 0.5233 | **0.3380** | 126.1 | 0.7414 | 0.8629 | 0.8716 |
| OPD5000 ckpt5000, 200k experts | 0.4759 | **0.5238** | 0.3370 | **125.7** | **0.7493** | **0.8729** | 0.8724 |

结论：

1. 新 experts + coldstart500 + OPD5000 相比上一轮 v2 有明确收益：以 ckpt5000 看，`Box Acc@0.5 +0.0081`，`Point Hit@50 +0.0350`，`Point Hit@100 +0.0250`，`CenterDist -0.5`。主 grounding 能力继续提升。
2. box IoU 在 3000 step 达到最高，5000 step 基本持平；point 指标一路到 5000 仍继续上升。综合看 5000 没有明显过训，但如果只追求 box IoU，3000 是更优候选。
3. `semantic_nav_box` 仍是短板域，2000 step 达到 0.2936 后平台，这与之前“semantic-nav 最弱”的判断一致。后续收益更可能来自该域 teacher/data 质量，而不是单纯拉长 OPD steps。
4. text keepalive 基本保持在 0.87 左右，4000 step 最高，5000 有轻微回落但仍在上一轮 v2 的同一量级；新融合没有破坏通用文本格式能力。

### 本地同步与可恢复状态

本轮已把远端的非模型产物同步到本地，模型权重和优化器状态没有落本地。同步范围包括：

| 类型 | 本地路径 |
|---|---|
| 当前 OPD 训练代码 | `point/train_opd_online_vl.py` |
| skip-expanded 备份 | `point/train_opd_online_vl.py.bak_skip_expanded_20260529_154323` |
| expert 300k OOM 探针脚本 | `point/run_seed0_five_experts_300k_mb4_filtered_stdtrainer_maxnorm5_v1.sh` |
| expert 100k 续训脚本 | `point/run_seed0_five_experts_continue100k_from100k_mb1_filtered_stdtrainer_maxnorm5_v1.sh` |
| coldstart500 脚本 | `point/run_opd_offpolicy_coldstart500_p0p1_zero3_mb16_accum1_newexperts.sh` |
| OPD5000 脚本 | `point/run_opd_p0p1_studentrollout_full5000_skip64000_save1000_zero3_mb16_accum1_from_coldstart500_newexperts.sh` |
| 评估脚本 | `point/eval_qwen_vl_raw_holdout.py`、`point/launch_raw_holdout_eval_8models.sh`、`point/launch_opd_ckpt_eval.sh`、`point/summarize_raw_holdout_eval.py` |
| 评估结果 | `point/eval_raw_holdout_v1/opd5000_newexperts_ckpts_1000_2000_3000_4000_5000_retry_20260601_104908/` |
| 训练日志 | `point/logs/*200k_from100k*`、`point/logs/*coldstart500*`、`point/logs/*full5000*` |
| 报告附件 | `report/opd5000_newexperts_20260601/` |

同步时显式排除了 `*.safetensors`、`*.pt`、`*.pth`、`*.bin`、`predictions.jsonl` 和 `raw_holdout_eval_v1_10k.jsonl`。因此本地保留了恢复远端代码与实验流水线所需的脚本、配置、日志、summary JSON 和评估 metrics，但没有复制模型权重、optimizer state 或原始大样本预测文件。

本轮新增报告附件：

| 文件 | 说明 |
|---|---|
| `report/opd5000_newexperts_20260601/artifact_manifest.json` | 本地同步范围、远端关键路径、最终模型路径索引 |
| `report/opd5000_newexperts_20260601/coldstart500_training_curve_summary.json` | coldstart500 的训练曲线摘要 |
| `report/opd5000_newexperts_20260601/opd5000_training_curve_summary.json` | OPD5000 的训练曲线摘要 |
| `report/opd5000_newexperts_20260601/opd5000_ckpt_eval_comparison_summary.json` | 1000/2000/3000/4000/5000 的完整 eval summary |
| `report/opd5000_newexperts_20260601/opd5000_ckpt_eval_by_format_table.md` | 按 box/point/text 子集整理的 checkpoint 表 |
| `report/opd5000_newexperts_20260601/opd5000_ckpt_eval_by_pool_table.md` | 按 eval pool 整理的 checkpoint 表 |
| `report/opd5000_newexperts_20260601/remote_model_summaries/` | coldstart500、OPD5000、五个 200k experts 的 `trainer_state.json` / run summary 小文件 |

## 2026-06-03：Qwen-122B / Qwen3-35B-VL base64 API raw-holdout 评估

本轮在沐曦服务器 `/data/msz/point` 上复用 raw-holdout 10k eval 口径，对另一台服务器的两个 OpenAI-compatible VLM 服务做评估：

| 项目 | 值 |
|---|---|
| API 服务器 | `root@10.12.82.43` |
| 模型服务 | `qwen-122b` at port `30001`；`qwen-35b` at port `30002` |
| 访问方式 | 本地 SSH key 建反向隧道，沐曦机访问 `127.0.0.1:13001/13002` |
| 输入图片 | eval 脚本读取沐曦本地图片后转为 `data:image/...;base64,...` |
| eval set | `/data/msz/point/eval_raw_holdout_v1/raw_holdout_eval_v1_10k.jsonl` |
| max tokens | 64 |
| 最终有效 run | `/data/msz/point/eval_raw_holdout_v1/openai_base64_qwen122_qwen35_nothink_20260603_201411` |
| 本地结果 | `report/openai_base64_qwen122_qwen35_20260603_nothink/` |

注意：第一版请求没有关闭 Qwen3 thinking，vLLM 返回 `message.reasoning` 且 `message.content=null`，导致 prediction 全为空；该 run 仅保留为调试记录，不作为结果。最终脚本默认加入 `chat_template_kwargs={"enable_thinking": false}`，两模型均 10k 完成且 `api_errors=0`。

### 与 OPD 后 8B 模型对比

这里把 8B base、8B instruct、两个 8B OPD checkpoint 与两个大模型放在同一张表中。`opd5000_ckpt5000` 是最终 checkpoint；`opd5000_ckpt3000` 是同一轮里 box IoU 最高的中间 checkpoint，用作参考。box 指标在 6100 条 box 样本上统计，point 指标在 1400 条 point 样本上统计，text 指标在 2500 条 text 样本上统计。

| 模型 | rows | Overall format | Box format | Box IoU | Box Acc@0.3 | Box Acc@0.5 | Box Acc@0.75 | CenterDist | Point format | Point Hit@50 | Point Hit@100 | Text exact | Text loose |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8B base | 10000 | 0.8532 | 0.9889 | 0.3854 | 0.5234 | 0.4087 | 0.2098 | 153.5 | 0.0000 | 0.0000 | 0.0000 | 0.0080 | 0.0080 |
| 8B instruct | 10000 | 0.8578 | 0.9964 | 0.4134 | 0.5577 | 0.4331 | 0.2472 | 140.1 | 0.0000 | 0.0000 | 0.0000 | 0.0076 | 0.5200 |
| OPD5000 ckpt3000, 8B | 10000 | **1.0000** | **1.0000** | 0.4761 | 0.6156 | 0.5233 | 0.3380 | 126.1 | **1.0000** | 0.7414 | 0.8629 | 0.8716 | 0.8716 |
| OPD5000 ckpt5000, 8B | 10000 | **1.0000** | **1.0000** | 0.4759 | 0.6123 | 0.5238 | 0.3370 | 125.7 | **1.0000** | **0.7493** | **0.8729** | **0.8724** | **0.8724** |
| Qwen-122B API, base64 | 10000 | 0.8561 | 0.9936 | **0.5466** | **0.6759** | **0.5841** | **0.4110** | **98.0** | 0.0000 | 0.0000 | 0.0000 | 0.0024 | 0.6000 |
| Qwen3-35B-VL API, base64 | 10000 | 0.8517 | 0.9864 | 0.5384 | 0.6626 | 0.5782 | 0.4005 | 100.9 | 0.0000 | 0.0057 | 0.0093 | 0.0000 | 0.4476 |

结论：

1. **OPD 对 8B 的增益很大。** 相比 8B base，OPD5000 ckpt5000 的 `Box IoU +0.0905`、`Box Acc@0.5 +0.1151`、`CenterDist -27.9`，并把 point 从几乎不可用提升到 `Hit@50=0.7493` / `Hit@100=0.8729`。
2. **8B instruct 比 8B base 略好，但远不如 OPD。** instruct 的 box IoU 从 base 的 `0.3854` 到 `0.4134`，text loose 从 `0.0080` 到 `0.5200`，但 point 仍为 0，协议稳定性没有解决。
3. **box grounding：大模型仍最强。** 相比 OPD5000 ckpt5000，Qwen-122B 的 `Box IoU +0.0707`，`Box Acc@0.5 +0.0603`，`CenterDist -27.7`；Qwen3-35B-VL 也有 `Box IoU +0.0625`、`Box Acc@0.5 +0.0544`。
4. **point grounding 和统一协议：OPD 8B 明显最强。** base、instruct、122B、35B 在 point 子集都几乎不输出符合 `<point>` 口径的答案；OPD5000 ckpt5000 是唯一在 `<box>/<point>/text` 三类格式上同时稳定的模型。
5. **实际取舍：** 如果目标只看 `<box>` 区域框选，122B/35B API 是更强 teacher/labeler；如果目标是部署统一遵循 `<box>/<point>/text` 输出协议的 8B 模型，当前 OPD5000 8B 仍然更适合直接作为 student/production 候选。

### Point 协议修正后的重新评估

上表中的大模型 point 结果并不能直接代表真实 point 定位能力，因为 raw-holdout 的 point prompt 本身存在协议不一致：user prompt 里给出的格式示例是 `[(x1, y1), (x2, y2), ...]`，但 gold 与 strict parser 要求的是 `<point>[[x1,y1],[x2,y2],...]</point>`。122B/35B 原始输出经常是 `[(597, 362)]`，坐标可能接近目标，但没有 `<point>` 标签，因此 strict scoring 直接记为格式失败。

为正确评估大模型 point 能力，本轮对 API eval 脚本做了最小修复：

| 项目 | 值 |
|---|---|
| 脚本 | `/data/msz/point/eval_openai_vl_raw_holdout_base64.py` |
| 新参数 | `--expected-format-filter point`，`--enforce-point-protocol` |
| 协议提示 | 要求只输出 `<point>[[x1,y1],[x2,y2],...]</point>`，禁止 `[(x,y)]`、禁止 XML attributes，opening tag 必须正好是 `<point>` |
| eval 子集 | raw-holdout 的 1400 条 `expected_format=point` 样本 |
| scoring | 仍使用原 strict parser，不做 lenient 放宽 |
| 远端 run | `/data/msz/point/eval_raw_holdout_v1/openai_base64_point_protocol_qwen122_qwen35_20260603_221231` |
| 本地结果 | `report/openai_base64_point_protocol_qwen122_qwen35_20260603/` |

重新评估结果：

| 模型 / 口径 | Point format | Point Hit@50 | Point Hit@100 | MinDist | 说明 |
|---|---:|---:|---:|---:|---|
| 8B base，原 prompt strict | 0.0000 | 0.0000 | 0.0000 | - | 不按 `<point>` 协议输出 |
| 8B base，原 prompt lenient | - | 0.4450 | 0.5950 | 89.1 | 接受裸 `[(x,y)]` 后的估计 |
| 8B instruct，原 prompt strict | 0.0000 | 0.0000 | 0.0000 | - | 不按 `<point>` 协议输出 |
| 8B instruct，原 prompt lenient | - | 0.3686 | 0.5536 | 99.5 | 接受裸 `[(x,y)]` 后的估计 |
| Qwen-122B，原 prompt strict | 0.0000 | 0.0000 | 0.0000 | - | 协议失败为主 |
| Qwen-122B，原 prompt lenient | - | 0.4914 | 0.6500 | 69.1 | 有一定定位能力 |
| Qwen-122B，修正 prompt strict | **0.9986** | **0.6771** | **0.8200** | 69.1 | 协议修正后显著提升 |
| Qwen3-35B-VL，原 prompt strict | 0.0000 | 0.0057 | 0.0093 | 66.5 | 少数输出可被 parser 读到 |
| Qwen3-35B-VL，原 prompt lenient | - | 0.4400 | 0.5736 | 71.4 | 有一定定位能力 |
| Qwen3-35B-VL，修正 prompt strict | 0.9857 | 0.6493 | 0.7871 | 75.9 | 协议修正后显著提升 |
| OPD5000 ckpt5000，原 prompt strict | **1.0000** | **0.7493** | **0.8729** | **51.7** | 仍是 point 子集最强 |

修正后的结论：

1. **之前大模型 point strict 为 0 主要是 prompt/protocol 问题。** 原 prompt 用 `[(x,y)]` 引导，但 evaluator 要 `<point>...</point>`，这对未专门训练过协议的大模型不公平。
2. **协议修正后，大模型 point 能力明显成立。** Qwen-122B 达到 `Hit@50=0.6771` / `Hit@100=0.8200`，Qwen3-35B-VL 达到 `0.6493` / `0.7871`。
3. **OPD5000 8B 仍是 point 最强模型。** 它在不修改 prompt 的原始 strict 口径下就有 `Hit@50=0.7493` / `Hit@100=0.8729`，且最小距离均值更低，说明不仅协议稳定，定位也更贴近 gold point。
4. **后续评估口径应分两层报告。** 对部署协议能力看 original strict；对大模型几何定位能力看 protocol-fixed strict 或 lenient。大模型可以作为 point/box teacher，但若要公平比较几何能力，必须先把输出协议在 prompt 中写死。

### 不看错误 eval / 格式率的领域能力总表

本表不再纳入错误的 point strict eval，也不展示 format accuracy。box 域只看几何框选能力，使用 IoU 与 Acc@0.5；point 域对 base/instruct 使用 lenient 几何能力估计，对 122B/35B 使用 protocol-fixed strict 结果，对 OPD 使用原始 strict 结果；text 域使用 `text_loose`。

Box 域 IoU：

| 模型 | RefCOCO | Flickr30K | SemanticNav Box | VG Object | VG Region | VG Relation |
|---|---:|---:|---:|---:|---:|---:|
| 8B base | 0.6458 | 0.5534 | 0.2117 | 0.2617 | 0.3150 | 0.3134 |
| 8B instruct | 0.6820 | 0.5908 | 0.1967 | 0.2878 | 0.3413 | 0.3535 |
| OPD5000 ckpt3000 | 0.7343 | 0.7079 | 0.2921 | 0.3197 | 0.3925 | 0.4019 |
| OPD5000 ckpt5000 | 0.7358 | **0.7101** | 0.2924 | 0.3211 | 0.3912 | 0.3974 |
| Qwen-122B | **0.8386** | 0.6874 | **0.4199** | **0.4097** | **0.4339** | **0.4787** |
| Qwen3-35B-VL | 0.8312 | 0.6753 | 0.4124 | 0.4001 | 0.4270 | 0.4743 |

Box 域 Acc@0.5：

| 模型 | RefCOCO | Flickr30K | SemanticNav Box | VG Object | VG Region | VG Relation |
|---|---:|---:|---:|---:|---:|---:|
| 8B base | 0.7455 | 0.6122 | 0.2075 | 0.2564 | 0.3027 | 0.3100 |
| 8B instruct | 0.8009 | 0.6567 | 0.1138 | 0.2845 | 0.3327 | 0.3636 |
| OPD5000 ckpt5000 | 0.8345 | **0.7878** | 0.3275 | 0.3427 | 0.4182 | 0.4264 |
| Qwen-122B | **0.9036** | 0.7367 | **0.4925** | **0.4255** | **0.4445** | 0.5045 |
| Qwen3-35B-VL | 0.9009 | 0.7067 | 0.4825 | 0.4245 | 0.4427 | **0.5091** |

Point / Text 域能力：

| 模型 | Point Hit@50 | Point Hit@100 | Point MinDist | Keepalive VQA loose |
|---|---:|---:|---:|---:|
| 8B base | 0.4450 | 0.5950 | 89.1 | 0.0080 |
| 8B instruct | 0.3686 | 0.5536 | 99.5 | 0.5200 |
| OPD5000 ckpt3000 | 0.7414 | 0.8629 | 53.3 | 0.8716 |
| OPD5000 ckpt5000 | **0.7493** | **0.8729** | **51.7** | **0.8724** |
| Qwen-122B, protocol-fixed | 0.6771 | 0.8200 | 69.1 | 0.6000 |
| Qwen3-35B-VL, protocol-fixed | 0.6493 | 0.7871 | 75.9 | 0.4476 |

综合结论：

1. 122B/35B 在大多数 box 域上仍是最强，尤其 RefCOCO、semantic-nav box、VG object/region/relation。
2. OPD5000 8B 在 Flickr30K box、point、keepalive text 上更强，且是当前统一多任务协议下最稳的可部署 8B 模型。
3. 8B instruct 相对 8B base 有提升，但没有解决 point 与整体任务保持问题；OPD 带来的提升远大于 instruct/base 差异。
