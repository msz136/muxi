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
