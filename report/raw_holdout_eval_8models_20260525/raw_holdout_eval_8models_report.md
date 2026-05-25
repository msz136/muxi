# Raw Holdout 8 模型完整评估报告

生成时间：2026-05-25

本报告整理本轮 raw-holdout 评估与 OPD checkpoint 评估。8 模型主评估来自远端 `/data/msz/point/eval_raw_holdout_v1/runs_8models_20260525_120552`；OPD checkpoint 曲线来自 `/data/msz/point/eval_raw_holdout_v1/opd_ckpts_20260525_124224`。本地归档目录为 `C:\Users\mousongzhe\muxi\report\raw_holdout_eval_8models_20260525`。

## 1. 评估过程

1. 构造 10k raw-holdout 评估集，覆盖通用 VQA、point grounding、box grounding、区域描述、关系框选与语义导航框选。
2. 训练集阻断：排除 5 个 expert 各自前 100k 训练样本、最终 OPD 训练样本，并用 source-line key、内容 fingerprint、训练图片集合做泄漏检查。
3. 严格去重验收：fingerprint、source line、image+prompt+answer 三类重复均为 0；训练图片泄漏为 0。
4. 8 模型主评估：每个模型单张卡，8 卡并行；`BATCH_SIZE=256`，`max_new_tokens=64`。生成阶段单卡显存约 35-37GB，GPU util 主要在 85%-98%。
5. OPD checkpoint 评估：checkpoint-500/1000/1500/2000 各占一张卡并行评估，同样 10k 样本、同一套指标；并把前一轮 checkpoint-2344 的 OPD final 结果并入曲线。
6. 指标：box 使用格式率、坐标有效率、IoU、Acc@0.3/0.5/0.75、中心距离；point 使用格式率、坐标有效率、Hit@50/Hit@100、距离与点数偏差；text/VQA 使用 exact、loose、boolean accuracy、multiple-choice accuracy。

## 2. 评估集与去重验收

| 项目 | 数值 |
| --- | --- |
| 总样本 | 10000 |
| 唯一图片 | 8251 |
| Fingerprint 重复 | 0 |
| Source line 重复 | 0 |
| Image+prompt+answer 重复 | 0 |
| 训练图片泄漏行数 | 0 |
| 重复图片路径数 | 1168 |
| 落在重复图片上的样本行数 | 2917 |

重复图片不等价于重复样本。RefCOCO、Visual Genome、Flickr30K 一张图天然会有多个 object/region/relation 标注；本次真正严格去重的 source、fingerprint、image+prompt+answer 均为 0，同时训练图片泄漏为 0。

### 2.1 数据池分布

| 数据池 | 领域/场景 | 样本数 |
| --- | --- | --- |
| refcoco | RefCOCO / 指代表达框选 | 1100 |
| flickr30k_entities | Flickr30K Entities / 短语实体框选 | 900 |
| visual_genome_object | Visual Genome Object / 通用物体框选 | 1100 |
| visual_genome_region | Visual Genome Region / 区域描述框选 | 1100 |
| visual_genome_relationship | Visual Genome Relationship / 关系框选 | 1100 |
| semantic_nav_box | Semantic Nav Box / 语义导航框选 | 800 |
| grounding_point | Grounding point / 点选 | 1400 |
| keepalive_vqa | Keepalive VQA / 通用能力 | 2500 |

### 2.2 任务格式分布

| 任务格式 | 样本数 |
| --- | --- |
| box | 6100 |
| point | 1400 |
| text | 2500 |

## 3. 8 模型总体结果

| 模型 | n | Format | Coord | Box IoU | Box Acc@0.5 | Point Hit@100 | Text loose | Bool acc | MC acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 10000 | 85.3% | 80.5% | 0.385 | 40.9% | 0.0% | 0.8% | 0.0% | 18.7% |
| Qwen3-VL-8B-Instruct | 10000 | 85.8% | 81.0% | 0.413 | 43.3% | 0.0% | 52.0% | 81.2% | 29.0% |
| General reasoning expert | 10000 | 100.0% | 100.0% | 0.444 | 48.1% | 83.4% | 87.1% | 89.0% | 21.5% |
| RoboPoint expert | 10000 | 100.0% | 100.0% | 0.454 | 49.8% | 87.9% | 85.8% | 87.5% | 16.8% |
| General obj expert | 10000 | 100.0% | 100.0% | 0.470 | 51.3% | 82.5% | 84.6% | 85.8% | 24.3% |
| Region expert | 10000 | 100.0% | 100.0% | 0.469 | 51.5% | 81.5% | 83.0% | 81.6% | 20.6% |
| Spatial rel expert | 10000 | 100.0% | 100.0% | 0.473 | 51.6% | 81.5% | 84.2% | 85.8% | 21.5% |
| OPD final | 10000 | 100.0% | 100.0% | 0.468 | 51.3% | 84.1% | 87.4% | 90.0% | 20.6% |

### 3.1 总体观点

- **OPD final 是综合最均衡的模型。** 它的 box Acc@0.5=51.3%、point Hit@100=84.1%、text loose=87.4%、bool acc=90.0%。它不是每个单项的绝对第一，但在 grounding 与通用能力之间的折中最好。
- **box 单项最强的是 Spatial rel expert。** 它的 box IoU=0.473、Acc@0.5=51.6%，略高于 OPD final 的 0.468/51.3%。差距很小，说明 OPD 融合基本保住了区域与关系框选能力。
- **point 单项最强的是 RoboPoint expert。** Hit@100=87.9%，高于 OPD final 的 84.1%。这符合训练目标，也说明 point 能力在融合中有约 3.8 个百分点的损失。
- **通用能力最强的是 OPD final。** text loose=87.4%、bool acc=90.0%，都是最高；说明 OPD 的通用数据与边界/格式数据没有被 grounding 数据淹没。
- **base 与 Qwen3-VL-8B-Instruct 在 point 上为 0，不代表视觉完全不会，而是格式未对齐。** 二者几乎不按本项目训练期望输出 `<point>`，而 expert/OPD 的 point format 都达到或接近 100%。

## 4. 按任务类型拆分

### 4.1 Box Grounding

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 6100 | 98.9% | 98.9% | 0.385 | 52.3% | 40.9% | 21.0% | 153.5 |
| Qwen3-VL-8B-Instruct | 6100 | 99.6% | 99.6% | 0.413 | 55.8% | 43.3% | 24.7% | 140.1 |
| General reasoning expert | 6100 | 100.0% | 100.0% | 0.444 | 58.3% | 48.1% | 29.8% | 138.4 |
| RoboPoint expert | 6100 | 100.0% | 100.0% | 0.454 | 59.2% | 49.8% | 31.1% | 132.7 |
| General obj expert | 6100 | 100.0% | 100.0% | 0.470 | 60.4% | 51.3% | 33.5% | 129.6 |
| Region expert | 6100 | 100.0% | 100.0% | 0.469 | 60.7% | 51.5% | 33.1% | 130.0 |
| Spatial rel expert | 6100 | 100.0% | 100.0% | 0.473 | 61.0% | 51.6% | 33.4% | 128.2 |
| OPD final | 6100 | 100.0% | 100.0% | 0.468 | 60.3% | 51.3% | 32.9% | 128.2 |

观点：box 第一梯队是 spatial_rel、region、general_obj、OPD final，Acc@0.5 都在 51.3%-51.6%。OPD final 相比 Qwen3-VL-8B-Instruct 从 43.3% 提升到 51.3%，提升约 8.0 个百分点；相比 base 从 40.9% 提升到 51.3%，提升约 10.5 个百分点。

### 4.2 Point Grounding

| 模型 | n | Format | Coord | Hit@50 | Hit@100 | MinDist | PredToGoldDist | PointCountDiff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| Qwen3-VL-8B-Instruct | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| General reasoning expert | 1400 | 100.0% | 100.0% | 71.0% | 83.4% | 61.6 | 85.0 | 4.4 |
| RoboPoint expert | 1400 | 100.0% | 100.0% | 77.6% | 87.9% | 44.8 | 63.2 | 4.6 |
| General obj expert | 1400 | 100.0% | 100.0% | 69.4% | 82.5% | 62.4 | 85.0 | 4.3 |
| Region expert | 1400 | 100.0% | 100.0% | 68.4% | 81.5% | 67.0 | 90.8 | 4.1 |
| Spatial rel expert | 1400 | 100.0% | 100.0% | 66.9% | 81.5% | 68.7 | 92.4 | 4.2 |
| OPD final | 1400 | 100.0% | 100.0% | 70.9% | 84.1% | 57.6 | 78.0 | 4.6 |

观点：RoboPoint expert 明显最强，Hit@50=77.6%、Hit@100=87.9%。OPD final Hit@100=84.1%，低于 RoboPoint expert，但高于 general_obj、region、spatial_rel expert，说明融合保留了大部分 point 专家能力。

### 4.3 Text / VQA 通用能力

| 模型 | n | Format | Text exact | Text loose | Bool n | Bool acc | MC n | MC acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 2500 | 100.0% | 0.8% | 0.8% | 910 | 0.0% | 107 | 18.7% |
| Qwen3-VL-8B-Instruct | 2500 | 100.0% | 0.8% | 52.0% | 910 | 81.2% | 107 | 29.0% |
| General reasoning expert | 2500 | 100.0% | 87.1% | 87.1% | 910 | 89.0% | 107 | 21.5% |
| RoboPoint expert | 2500 | 100.0% | 85.8% | 85.8% | 910 | 87.5% | 107 | 16.8% |
| General obj expert | 2500 | 100.0% | 84.6% | 84.6% | 910 | 85.8% | 107 | 24.3% |
| Region expert | 2500 | 100.0% | 83.0% | 83.0% | 910 | 81.6% | 107 | 20.6% |
| Spatial rel expert | 2500 | 100.0% | 84.2% | 84.2% | 910 | 85.8% | 107 | 21.5% |
| OPD final | 2500 | 100.0% | 87.4% | 87.4% | 910 | 90.0% | 107 | 20.6% |

观点：OPD final 的 text loose=87.4%、bool acc=90.0%，为全体最佳。Qwen3-VL-8B-Instruct 的 loose 有 52.0%，但 exact 只有 0.8%，说明它经常输出带选项前缀或解释性文本，和本项目期望的短答案格式不完全一致；expert/OPD 的 exact 与 loose 接近，格式约束更稳定。

## 5. 每个领域的完整指标与分析

| 领域 | 主指标 | 最佳模型 | 指标值 |
| --- | --- | --- | --- |
| RefCOCO / 指代表达框选 | Acc@0.5 | General obj expert | 83.9% |
| Flickr30K Entities / 短语实体框选 | Acc@0.5 | General obj expert | 78.3% |
| Visual Genome Object / 通用物体框选 | Acc@0.5 | Spatial rel expert | 35.5% |
| Visual Genome Region / 区域描述框选 | Acc@0.5 | Region expert | 41.5% |
| Visual Genome Relationship / 关系框选 | Acc@0.5 | Region expert | 42.7% |
| Semantic Nav Box / 语义导航框选 | Acc@0.5 | Spatial rel expert | 34.0% |
| Grounding point / 点选 | Hit@100 | RoboPoint expert | 87.9% |
| Keepalive VQA / 通用能力 | Text loose | OPD final | 87.4% |

### 5.1 RefCOCO / 指代表达框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1100 | 97.2% | 97.3% | 0.646 | 82.9% | 74.5% | 46.9% | 85.5 |
| Qwen3-VL-8B-Instruct | 1100 | 99.9% | 99.9% | 0.682 | 88.5% | 80.1% | 54.3% | 70.3 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.729 | 89.1% | 82.9% | 65.9% | 63.3 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.719 | 87.2% | 81.8% | 64.6% | 67.9 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.740 | 89.5% | 83.9% | 67.9% | 61.2 |
| Region expert | 1100 | 100.0% | 100.0% | 0.710 | 87.1% | 81.0% | 63.6% | 69.7 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.725 | 87.7% | 81.7% | 65.7% | 63.7 |
| OPD final | 1100 | 100.0% | 100.0% | 0.720 | 87.5% | 82.0% | 64.8% | 65.8 |

解读：按 Acc@0.5，最佳为 **General obj expert**（83.9%）。OPD final 在该领域 Acc@0.5=82.0%、IoU=0.720；距最佳差 1.9 个百分点，相比 Qwen3-VL-8B-Instruct 提升 1.9 个百分点，相比 base 提升 7.5 个百分点。
这是较成熟的通用指代表达/短语实体框选领域，模型整体表现显著高于 Visual Genome 和 semantic-nav；OPD final 基本保持在第一梯队。

### 5.2 Flickr30K Entities / 短语实体框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 900 | 99.6% | 99.6% | 0.553 | 74.3% | 61.2% | 35.0% | 105.8 |
| Qwen3-VL-8B-Instruct | 900 | 99.9% | 99.9% | 0.591 | 77.4% | 65.7% | 42.3% | 95.8 |
| General reasoning expert | 900 | 100.0% | 100.0% | 0.608 | 77.1% | 65.9% | 46.4% | 102.4 |
| RoboPoint expert | 900 | 100.0% | 100.0% | 0.636 | 81.0% | 70.8% | 50.4% | 91.8 |
| General obj expert | 900 | 100.0% | 100.0% | 0.706 | 86.2% | 78.3% | 59.7% | 68.1 |
| Region expert | 900 | 100.0% | 100.0% | 0.700 | 86.0% | 78.3% | 59.8% | 66.7 |
| Spatial rel expert | 900 | 100.0% | 100.0% | 0.693 | 85.6% | 76.4% | 58.2% | 71.9 |
| OPD final | 900 | 100.0% | 100.0% | 0.701 | 85.7% | 77.6% | 59.4% | 67.7 |

解读：按 Acc@0.5，最佳为 **General obj expert**（78.3%）。OPD final 在该领域 Acc@0.5=77.6%、IoU=0.701；距最佳差 0.8 个百分点，相比 Qwen3-VL-8B-Instruct 提升 11.9 个百分点，相比 base 提升 16.3 个百分点。
这是较成熟的通用指代表达/短语实体框选领域，模型整体表现显著高于 Visual Genome 和 semantic-nav；OPD final 基本保持在第一梯队。

### 5.3 Visual Genome Object / 通用物体框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1100 | 99.8% | 99.8% | 0.262 | 37.5% | 25.6% | 11.5% | 197.7 |
| Qwen3-VL-8B-Instruct | 1100 | 99.2% | 99.2% | 0.288 | 39.5% | 28.5% | 13.6% | 189.4 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.303 | 41.4% | 32.2% | 16.2% | 197.1 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.311 | 41.7% | 33.0% | 18.3% | 189.5 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.323 | 42.9% | 34.2% | 19.8% | 184.8 |
| Region expert | 1100 | 100.0% | 100.0% | 0.318 | 42.6% | 33.0% | 19.0% | 186.3 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.329 | 44.2% | 35.5% | 20.2% | 185.5 |
| OPD final | 1100 | 100.0% | 100.0% | 0.325 | 43.5% | 34.4% | 19.8% | 182.3 |

解读：按 Acc@0.5，最佳为 **Spatial rel expert**（35.5%）。OPD final 在该领域 Acc@0.5=34.4%、IoU=0.325；距最佳差 1.2 个百分点，相比 Qwen3-VL-8B-Instruct 提升 5.9 个百分点，相比 base 提升 8.7 个百分点。
该领域的 OPD final 处在第一梯队但不是单项最佳，说明融合模型牺牲了一点专精上限来换取跨任务稳定性。

### 5.4 Visual Genome Region / 区域描述框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1100 | 98.5% | 98.5% | 0.315 | 44.7% | 30.3% | 10.5% | 155.7 |
| Qwen3-VL-8B-Instruct | 1100 | 99.5% | 99.5% | 0.341 | 48.7% | 33.3% | 14.3% | 148.0 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.374 | 53.6% | 38.9% | 18.3% | 145.0 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.376 | 53.4% | 39.3% | 18.2% | 141.2 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.381 | 53.9% | 38.9% | 18.5% | 140.7 |
| Region expert | 1100 | 100.0% | 100.0% | 0.392 | 55.7% | 41.5% | 20.0% | 139.6 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.384 | 53.5% | 40.5% | 18.3% | 142.1 |
| OPD final | 1100 | 100.0% | 100.0% | 0.382 | 54.5% | 41.1% | 18.2% | 140.1 |

解读：按 Acc@0.5，最佳为 **Region expert**（41.5%）。OPD final 在该领域 Acc@0.5=41.1%、IoU=0.382；距最佳差 0.5 个百分点，相比 Qwen3-VL-8B-Instruct 提升 7.8 个百分点，相比 base 提升 10.8 个百分点。
区域描述框选上 region_expert 最好，符合其训练目标；OPD final 只落后约 0.5 个百分点，说明区域能力在融合后保持良好。

### 5.5 Visual Genome Relationship / 关系框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1100 | 98.7% | 98.8% | 0.313 | 42.0% | 31.0% | 16.4% | 175.9 |
| Qwen3-VL-8B-Instruct | 1100 | 99.5% | 99.5% | 0.354 | 47.9% | 36.4% | 19.7% | 160.8 |
| General reasoning expert | 1100 | 100.0% | 100.0% | 0.378 | 49.5% | 40.5% | 23.7% | 151.9 |
| RoboPoint expert | 1100 | 100.0% | 100.0% | 0.381 | 50.0% | 41.2% | 24.8% | 156.4 |
| General obj expert | 1100 | 100.0% | 100.0% | 0.391 | 51.0% | 42.2% | 25.6% | 151.3 |
| Region expert | 1100 | 100.0% | 100.0% | 0.400 | 52.6% | 42.7% | 26.3% | 150.0 |
| Spatial rel expert | 1100 | 100.0% | 100.0% | 0.393 | 51.5% | 41.1% | 27.2% | 152.4 |
| OPD final | 1100 | 100.0% | 100.0% | 0.390 | 50.9% | 41.2% | 25.6% | 148.2 |

解读：按 Acc@0.5，最佳为 **Region expert**（42.7%）。OPD final 在该领域 Acc@0.5=41.2%、IoU=0.390；距最佳差 1.5 个百分点，相比 Qwen3-VL-8B-Instruct 提升 4.8 个百分点，相比 base 提升 10.2 个百分点。
关系框选上 region_expert 最好，说明当前关系样本不仅依赖空间关系，也依赖区域描述式定位；OPD final 与 spatial_rel expert 接近，融合没有明显损伤关系能力。

### 5.6 Semantic Nav Box / 语义导航框选

| 模型 | n | Format | Coord | IoU | Acc@0.3 | Acc@0.5 | Acc@0.75 | CenterDist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 800 | 100.0% | 100.0% | 0.212 | 30.8% | 20.8% | 3.4% | 204.2 |
| Qwen3-VL-8B-Instruct | 800 | 99.9% | 99.9% | 0.197 | 29.3% | 11.4% | 0.8% | 179.5 |
| General reasoning expert | 800 | 100.0% | 100.0% | 0.253 | 36.4% | 25.6% | 4.4% | 174.2 |
| RoboPoint expert | 800 | 100.0% | 100.0% | 0.291 | 40.9% | 31.9% | 7.0% | 145.6 |
| General obj expert | 800 | 100.0% | 100.0% | 0.269 | 37.1% | 29.5% | 7.2% | 171.5 |
| Region expert | 800 | 100.0% | 100.0% | 0.281 | 39.0% | 31.6% | 8.3% | 166.3 |
| Spatial rel expert | 800 | 100.0% | 100.0% | 0.306 | 43.5% | 34.0% | 8.9% | 149.4 |
| OPD final | 800 | 100.0% | 100.0% | 0.278 | 38.1% | 31.1% | 7.2% | 163.7 |

解读：按 Acc@0.5，最佳为 **Spatial rel expert**（34.0%）。OPD final 在该领域 Acc@0.5=31.1%、IoU=0.278；距最佳差 2.9 个百分点，相比 Qwen3-VL-8B-Instruct 提升 19.8 个百分点，相比 base 提升 10.4 个百分点。
该领域是所有 box 任务中最难的一档：最佳 Acc@0.5 只有 34.0%，OPD final 为 31.1%。这说明语义导航里的 `{object_name, relation, anchor_object}` 框选仍是短板，关系理解和目标唯一定位比普通指代表达更难。

### 5.7 Grounding point / 点选

| 模型 | n | Format | Coord | Hit@50 | Hit@100 | MinDist | PredToGoldDist | PointCountDiff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| Qwen3-VL-8B-Instruct | 1400 | 0.0% | 0.0% | 0.0% | 0.0% | - | - | 8.7 |
| General reasoning expert | 1400 | 100.0% | 100.0% | 71.0% | 83.4% | 61.6 | 85.0 | 4.4 |
| RoboPoint expert | 1400 | 100.0% | 100.0% | 77.6% | 87.9% | 44.8 | 63.2 | 4.6 |
| General obj expert | 1400 | 100.0% | 100.0% | 69.4% | 82.5% | 62.4 | 85.0 | 4.3 |
| Region expert | 1400 | 100.0% | 100.0% | 68.4% | 81.5% | 67.0 | 90.8 | 4.1 |
| Spatial rel expert | 1400 | 100.0% | 100.0% | 66.9% | 81.5% | 68.7 | 92.4 | 4.2 |
| OPD final | 1400 | 100.0% | 100.0% | 70.9% | 84.1% | 57.6 | 78.0 | 4.6 |

解读：最佳为 **RoboPoint expert**，Hit@100=87.9%；OPD final Hit@100=84.1%，落后 3.8 个百分点。RoboPoint expert 的优势非常明确，说明纯 point expert 对点位预测仍有不可替代的专精收益。

### 5.8 Keepalive VQA / 通用能力

| 模型 | n | Format | Text exact | Text loose | Bool n | Bool acc | MC n | MC acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 2500 | 100.0% | 0.8% | 0.8% | 910 | 0.0% | 107 | 18.7% |
| Qwen3-VL-8B-Instruct | 2500 | 100.0% | 0.8% | 52.0% | 910 | 81.2% | 107 | 29.0% |
| General reasoning expert | 2500 | 100.0% | 87.1% | 87.1% | 910 | 89.0% | 107 | 21.5% |
| RoboPoint expert | 2500 | 100.0% | 85.8% | 85.8% | 910 | 87.5% | 107 | 16.8% |
| General obj expert | 2500 | 100.0% | 84.6% | 84.6% | 910 | 85.8% | 107 | 24.3% |
| Region expert | 2500 | 100.0% | 83.0% | 83.0% | 910 | 81.6% | 107 | 20.6% |
| Spatial rel expert | 2500 | 100.0% | 84.2% | 84.2% | 910 | 85.8% | 107 | 21.5% |
| OPD final | 2500 | 100.0% | 87.4% | 87.4% | 910 | 90.0% | 107 | 20.6% |

解读：最佳为 **OPD final**，Text loose=87.4%。OPD final 的 text loose=87.4%、bool acc=90.0%，说明融合后的通用能力没有被领域 grounding 任务压垮。

## 6. OPD Checkpoint 曲线

| Checkpoint | n | Format | Coord | Box IoU | Box Acc@0.5 | Point Hit@50 | Point Hit@100 | Text loose | Bool acc | MC acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | 10000 | 100.0% | 100.0% | 0.457 | 50.1% | 65.9% | 79.7% | 87.4% | 89.6% | 21.5% |
| 1000 | 10000 | 100.0% | 100.0% | 0.465 | 51.2% | 68.6% | 82.9% | 87.3% | 89.8% | 20.6% |
| 1500 | 10000 | 100.0% | 100.0% | 0.467 | 51.2% | 69.4% | 83.2% | 87.3% | 89.7% | 20.6% |
| 2000 | 10000 | 100.0% | 100.0% | 0.468 | 51.4% | 71.2% | 83.5% | 87.3% | 89.9% | 20.6% |
| 2344 | 10000 | 100.0% | 100.0% | 0.468 | 51.3% | 70.9% | 84.1% | 87.4% | 90.0% | 20.6% |

### 6.1 Checkpoint 观点

- **500 -> 2000 基本单调提升。** Box Acc@0.5 从 50.1% 到 51.4%，Point Hit@100 从 79.7% 到 83.5%，说明 OPD 训练前 2000 step 仍在带来稳定收益。
- **2000 -> 2344 进入平台期。** Box Acc@0.5 从 51.4% 到 51.3%，几乎不变；Point Hit@100 从 83.5% 到 84.1%，小幅提升；Text loose 从 87.3% 到 87.4%，也基本持平。
- **若只看 box，checkpoint-2000 略优；若看综合，checkpoint-2344 略稳。** 2000 的 box Acc@0.5=51.4%，略高于 2344 的 51.3%；2344 的 Point Hit@100=84.1%、Bool acc=90.0%，略高于 2000。
- **不建议回退到 500。** 500 的 point Hit@100=79.7%，比 2344 低 4.4 个百分点；box Acc@0.5 也低 1.2 个百分点。

## 7. 结论与建议

1. **当前推荐最终模型仍是 OPD final/checkpoint-2344。** 数据依据：它在 box 上达到第一梯队（Acc@0.5=51.3%），point 保持较高（Hit@100=84.1%），通用能力最好（Text loose=87.4%、Bool acc=90.0%）。
2. **如果业务只追求通用 box grounding，可以考虑 spatial_rel 或 region/general_obj expert，但不建议替代 OPD final。** Spatial rel expert box Acc@0.5=51.6%，只比 OPD final 高 0.3 个百分点；但 OPD final 的通用能力和 point 兼容性更完整。
3. **如果业务强依赖 point，RoboPoint expert 仍是专精上限。** 它 Hit@100=87.9%，比 OPD final 高 3.8 个百分点；但它的 text/VQA 与 box 综合均衡性不如 OPD final。
4. **semantic-nav box 是下一步最值得优化的短板。** 最佳模型在该领域 Acc@0.5 只有 34.0%，OPD final 只有 31.1%，明显低于 RefCOCO 的 82.0% 和 Flickr30K 的 77.6%。这说明语义导航的关系定位、anchor/object disambiguation、标注一致性仍需要更强数据或更严格清洗。
5. **OPD 训练已接近收敛平台。** 2000 到 2344 的收益很小，继续训练未必高性价比；下一轮提升更可能来自数据配比、semantic-nav 清洗、hard/boundary 样本设计，而不是单纯延长 step。

## 8. 本地/远端文件索引

| 文件 | 用途 |
| --- | --- |
| `comparison_extended_metrics.json` | 8 模型完整扩展指标，含 by_format/by_pool |
| `comparison_summary.json` | 8 模型原始紧凑汇总 |
| `opd_ckpt_extended_comparison_with_2344.json` | OPD checkpoint 500/1000/1500/2000/2344 对比 |
| `opd_ckpt_comparison_summary.json` | OPD checkpoint 原始紧凑汇总 |
| `evalset_summary.json` | 评估集构造、数据池、blocklist、verify 信息 |
| `dedupe_audit.json` | post-hoc 去重与训练图片泄漏审计 |
| 远端 8 模型 run | `/data/msz/point/eval_raw_holdout_v1/runs_8models_20260525_120552` |
| 远端 OPD ckpt run | `/data/msz/point/eval_raw_holdout_v1/opd_ckpts_20260525_124224` |
