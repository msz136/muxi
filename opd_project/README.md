# OPD Pointing Expert Fusion for Qwen3-VL-Instruct

## 项目概述

本项目对 **Qwen3-VL-8B-Instruct** 进行单个 Pointing 专家的 On-Policy Distillation (OPD) 融合。
目标：在不引入额外任务奖励的前提下，通过教师模型的 token 级 log-prob 蒸馏，将 pointing 能力注入学生模型，
同时保持学生模型原有的通用 VLM 能力不退化。

---

## 1. 方案总体设计

### 1.1 为什么选择 OPD 而非 SFT/RL

| 方案 | 优势 | 劣势 |
|------|------|------|
| SFT (监督微调) | 简单直接 | 分布偏移：离线数据无法覆盖学生在线生成的分布，容易过拟合教师的表面模式 |
| RL (PPO/GRPO) | 在线探索 | 需要设计 pointing 奖励函数（IoU/距离），奖励稀疏且噪声大 |
| **OPD (在线蒸馏)** | 在线生成 + 教师软标签 | 需要教师推理服务，计算成本较高 |

**选择 OPD 的核心理由：**

1. **分布对齐**：学生在自己的策略下采样，教师对这些采样给出 token 级 log-prob，KL 散度直接作用于学生当前分布，避免 SFT 的 train-test mismatch。
2. **无需手工奖励**：pointing 任务的奖励设计（坐标距离、IoU）存在离散化噪声和阈值敏感性，OPD 用教师 soft label 替代，信号更平滑。
3. **能力保持**：KL penalty 天然约束学生不偏离教师太远，配合 replay buffer 可有效防止通用能力遗忘。
4. **已有基础设施**：项目已有 slime OPD rollout 实现（`on_policy_distillation.py`），可直接复用。

### 1.2 整体架构

```
┌───────────��─────────────────────────────────────────────┐
│                    OPD Training Loop                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐    prompt     ┌──────────┐                │
│  │ Pointing │ ──────────▶  │ Student  │ ── sample ──┐  │
│  │ Prompt   │              │ (Qwen3-VL│              │  │
│  │ Pool     │              │  8B-Inst)│              │  │
│  └──────────┘              └──────────┘              │  │
│                                                      ▼  │
│  ┌──────────┐    input_ids  ┌──────────┐   response │  │
│  │ Teacher  │ ◀──────────── │ Log-prob │ ◀──────────┘  │
│  │ (Pointing│              │ Server   │                 │
│  │  Expert) │ ──────────▶  │ (sglang) │                │
���  └──────────┘  teacher_lp   └──────────┘                │
│                                                          │
│  Loss = KL(student_policy || teacher_policy)             │
│  Advantage = GRPO(reward=0, kl_penalty=β·KL)            │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  Replay Buffer: 20% 通用 VLM 数据 (防遗忘)              │
└─────────────────────────────────────────────────────────┘
```

### 1.3 教师模型选择

**教师模型：已训练好的 Pointing Expert（grounding multicontract 阶段产出）**

具体 checkpoint：`qwen3vl-8b-spatial_v3+qwen_v4-wudi` 或 `8b_grounding_multicontract`

选择理由：
- 该模型经过两阶段 grounding 训练（bbox warmup → multicontract），pointing 能力已充分收敛
- 与学生同架构（Qwen3-VL-8B），log-prob 空间对齐，蒸馏效率高
- 已验证在 RoboPoint/RefSpatial 等 benchmark 上表现良好

### 1.4 学生模型

**学生模型：Qwen3-VL-8B-Instruct（原始预训练权重）**

或可选：ACE-Brain-0-8B（已融合其他专家的版本），取决于目标是：
- 从零注入 pointing → 用原始 Instruct
- 在已有多专家基础上补充 pointing → 用 ACE-Brain

---

## 2. 训练配置

### 2.1 OPD 超参数

```yaml
# OPD 核心参数
teacher_server_url: "http://<teacher-host>:30000/v1/completions"
kl_penalty_coeff: 0.05          # β，控制 KL 惩罚强度
reward_baseline: 0.0            # 纯蒸馏，无外部奖励
advantage_estimator: "grpo"     # 组内相对优势
group_size: 4                   # 每个 prompt 采样 4 条 response

# 采样参数
temperature: 0.7                # 学生采样温度（鼓励探索）
max_new_tokens: 256             # pointing 回答通常较短
top_p: 0.95

# 训练参数
learning_rate: 1e-6             # OPD 用更小 LR，避免震荡
batch_size: 32                  # prompt 数/batch
gradient_accumulation: 4
total_steps: 2000               # ~8K unique prompts × 多轮
warmup_ratio: 0.05

# 模型配置
freeze_vision: true             # 视觉编码器冻结
tune_mlp: true                  # visual merger 可训练
tune_llm: true                  # LLM 可训练
max_pixels: 50176               # 与教师训练一致
```

### 2.2 为什么这样设置

- **kl_penalty_coeff=0.05**：过大会让学生过度模仿教师（丧失泛化），过小则学不到 pointing 能力。0.05 是 slime 框架验证过的起点。
- **temperature=0.7**：比 greedy 高，让学生探索更多 pointing 表达方式，教师对这些探索给出软标签引导。
- **learning_rate=1e-6**：比 SFT 的 5e-6 低一个量级，因为 OPD 的梯度信号来自 KL，方差较大，需要更保守的步长。
- **freeze_vision=true**：pointing 能力主要在语言侧（坐标生成），视觉特征已足够好。
- **group_size=4**：GRPO 需要组内对比，4 是计算效率和信号质量的平衡点。

---

## 3. 数据集设计（核心）

### 3.1 训练 Prompt Pool 设计

OPD 不需要标注答案（教师在线提供），只需要 **高质量的 prompt（图片 + 问题）**。

#### 3.1.1 Prompt 来源与配比

| 数据集 | 样本数 | 权重 | 任务类型 | 选择理由 |
|--------|--------|------|----------|----------|
| **RoboPoint** | ~50K | 30% | 机器人场景 affordance pointing | 核心目标场景，多视角，空间推理强 |
| **RefSpatial-3D** | ~20K | 15% | 3D 场景空间定位 | 补充深度/3D 理解，与机器人场景互补 |
| **RefSpatial-Sim** | ~15K | 10% | 仿真环境空间引用 | 增加 sim 场景多样性 |
| **PixMo-Points** | ~30K | 15% | 自然图像多点标注 | 泛化到非机器人场景，防止过拟合 |
| **PacoLVis** | ~25K | 10% | 部件级视觉定位 | 细粒度 pointing（物体部件） |
| **Grasp-Anything** | ~20K | 10% | 抓取区域 pointing | 直接对应机器人操作 |
| **ShareRobot-Affordance** | ~10K | 5% | 机器人 affordance | 补充 affordance 语义 |
| **通用 VLM QA (replay)** | ~20K | 5% | 通用视觉问答 | 防止通用能力遗忘 |

**总计：~190K prompts，有效训练约 8K unique prompts/epoch（采样）**

#### 3.1.2 数据集选取原则

1. **任务覆盖度**：从单点 → 多点 → 区域 → 轨迹，覆盖 pointing 的完整语义谱
2. **场景多样性**：机器人（RoboPoint, Grasp）+ 室内3D（RefSpatial）+ 自然图像（PixMo, PacoLVis）
3. **难度梯度**：简单单物体定位 → 多物体关系推理 → 3D 空间推理
4. **格式统一**：所有 prompt 统一为 `<image>\n{question}` 格式，答案格式为 `<point>[[x,y],...]</point>`
5. **Replay 防遗忘**：5% 通用 VLM 数据混入，确保 OPD 不破坏基础能力

#### 3.1.3 Prompt 格式规范

```json
{
  "prompt_id": "robopoint_001",
  "images": ["s3://path/to/image.jpg"],
  "messages": [
    {
      "role": "system",
      "content": "Your task is to locate several points in the given image according to the task descriptions. Your answer should be formatted as \"<point>[[x1, y1], [x2, y2],...]</point>\". The point coordinates are normalized to integers between 0 and 1000."
    },
    {
      "role": "user",
      "content": "<image>\nPoint to the location where I should place the cup to avoid blocking the laptop screen."
    }
  ],
  "metadata": {
    "source": "robopoint",
    "task_type": "affordance_pointing",
    "num_points": 2,
    "difficulty": "medium"
  }
}
```

#### 3.1.4 为什么不用 Box 数据

本项目聚焦 **pointing（点坐标）** 专家，不混入 box 数据，原因：
- Box 和 point 的输出格式不同（`<box>` vs `<point>`），混合训练会引入格式冲突
- 已有独立的 bbox warmup 阶段处理 box 能力
- 单一专家聚焦单一能力，后续通过 WUDI merge 融合

### 3.2 Replay Buffer 设计

```yaml
replay:
  source: "cambrian_general_qa"  # 通用 VLM QA 数据
  ratio: 0.05                    # 每 batch 5% 为 replay
  strategy: "uniform"            # 均匀采样
  purpose: "防止通用 VLM 能力退化"
```

---

## 4. 评估集设计（核心）

### 4.1 评估维度与指标

| 维度 | Benchmark | 指标 | 目的 |
|------|-----------|------|------|
| **Pointing 精度** | RoboPoint-Test | Acc@50/100/150 (像素距离) | 核心能力：坐标预测准确性 |
| **空间推理** | ViewSpatial-Bench | Accuracy (5700 QA) | 多视角空间理解 |
| **3D 定位** | RefSpatial-3D-Test | Point Acc@threshold | 3D 场景中的定位能力 |
| **Affordance** | Where2Place | Success Rate | 机器人场景 affordance |
| **通用 VLM** | MMBench / MME | Score | 确认通用能力未退化 |
| **格式正确性** | 自建 Format-Check | Format Acc (%) | 输出格式 `<point>...</point>` 合规率 |

### 4.2 评估集详细说明

#### 4.2.1 RoboPoint-Test（主评估）

- **来源**：RoboPoint 官方 test split
- **规模**：~5K 样本
- **指标**：
  - `Acc@50`：预测点与 GT 距离 ≤ 50（归一化坐标，1000 尺度）的比例
  - `Acc@100`：距离 ≤ 100
  - `Acc@150`：距离 ≤ 150
  - `Mean Distance`：平均欧氏距离
- **选择理由**：直接对应训练目标，机器人场景，多视角，是 pointing 能力的黄金标准

#### 4.2.2 ViewSpatial-Bench（空间泛化）

- **来源**：arXiv:2505.21500，ScanNet + MS-COCO 场景
- **规模**：5,700+ QA pairs，1,000+ 3D 场景
- **指标**：5 类空间任务的准确率
- **选择理由**：
  - 测试 pointing 能力是否泛化到空间理解
  - 包含 egocentric 和 allocentric 视角
  - 与训练数据分布不同（OOD 泛化测试）

#### 4.2.3 RefSpatial-3D-Test（3D 推理）

- **来源**：RefSpatial 数据集 test split
- **规模**：~2K 样本
- **指标**：Point Accuracy，Reasoning Accuracy
- **选择理由**：测试 3D 空间中的 pointing 推理能力，比 2D 更具挑战性

#### 4.2.4 通用能力回归测试

- **MMBench**：综合 VLM 能力（感知、推理、知识）
- **MME**：感知 + 认知双维度
- **目的**：确认 OPD 没有破坏学生模型的通用能力
- **阈值**：相比 baseline 下降不超过 2%

#### 4.2.5 格式正确性检查（自建）

```python
# 评估输出格式合规率
def check_format(response: str) -> bool:
    pattern = r'<point>\[\[(\d+,\s*\d+)(,\s*\[\d+,\s*\d+\])*\]\]</point>'
    return bool(re.search(pattern, response))
```

- **规模**：从各评估集中抽取 500 样本
- **指标**：格式正确率（目标 > 95%）
- **选择理由**：OPD 可能导致格式漂移，需要专门监控

### 4.3 评估频率

| 评估集 | 频率 | 说明 |
|--------|------|------|
| RoboPoint-Test (subset 500) | 每 100 steps | 快速反馈，early stopping 依据 |
| RoboPoint-Test (full) | 每 500 steps | 完整评估 |
| ViewSpatial-Bench | 每 500 steps | 泛化监控 |
| MMBench/MME | 训练前后各一次 | 回归测试 |
| Format-Check | 每 100 steps | 格式监控 |

---

## 5. 训练流程

### Phase 1: 环境准备
1. 部署教师模型 sglang 服务（pointing expert）
2. 准备 prompt pool（统一格式化）
3. 运行 baseline 评估（学生原始能力）

### Phase 2: OPD 训练
1. 从 prompt pool 采样 batch
2. 学生模型生成 response（temperature=0.7）
3. 教师模型返回 token-level log-probs
4. 计算 KL penalty，GRPO 优势估计
5. 策略梯度更新学生模型
6. 每 100 steps 评估 + checkpoint

### Phase 3: 后处理
1. 选择最优 checkpoint（RoboPoint-Test Acc@100 最高）
2. 运行完整评估 suite
3. 可选：WUDI merge 回 ACE-Brain 主线

---

## 6. 项目文件结构

```
opd_project/
├── README.md                    # 本文档
├── configs/
│   ├── opd_pointing.yaml        # OPD 训练主配置
│   ├── teacher_server.yaml      # 教师模型服务配置
│   └── eval_config.yaml         # 评估配置
├── data/
│   ├── prepare_prompt_pool.py   # Prompt pool 构建脚本
│   ├── dataset_registry.py      # 数据集注册
│   └── replay_sampler.py        # Replay buffer 采样
├── training/
│   ├── launch_teacher.sh        # 启动教师 sglang 服务
│   ├── run_opd.sh               # OPD 训练启动脚本
│   └── opd_reward.py            # OPD reward function (slime 接口)
├── evaluation/
│   ├── eval_pointing.py         # Pointing 精度评估
│   ├── eval_format.py           # 格式正确性检查
│   ├── eval_general.sh          # 通用能力回归测试
│   └── run_all_eval.sh          # 一键评估
└── merge/
    └── merge_to_acebrain.py     # WUDI merge 回主线
```

---

## 7. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 通用能力退化 | 5% replay buffer + MMBench 回归监控 |
| 格式漂移 | Format-Check 每 100 steps + 格式 reward 可选加入 |
| KL 震荡 | 小 LR (1e-6) + gradient clipping (1.0) |
| 教师服务不稳定 | sglang 多副本 + 重试机制 |
| 过拟合少数 prompt | 大 prompt pool (190K) + 均匀采样 |

---

## 8. 与现有工作的关系

- **继承**：slime OPD 框架（`on_policy_distillation.py`）的 reward function 接口
- **继承**：grounding multicontract 训练产出的教师模型
- **继承**：数据格式规范（`<point>[[x,y]]</point>`，坐标 0-1000 归一化）
- **产出**：pointing 专家 checkpoint，可通过 WUDI merge 融入 ACE-Brain 主线
- **区别**：不同于 SFT grounding 训练，OPD 是在线蒸馏，学生自主探索 + 教师引导
