# 🎥 ReAgent-V 方法与模块全景笔记

> **核心理念：**  
> ReAgent-V 是一个 **奖励驱动的多智能体视频理解框架（Reward-Driven Multi-Agent Framework for Video Understanding）**。  
> 它通过「**实时奖励生成 + 多视角反思 + 工具增强推理**」实现了大模型在视频任务中的自我纠错与自进化。  
> 框架由两个主要智能体组成：
> - **Target Agent**：执行推理、生成答案；
> - **Critic Agent**：动态评估推理质量并生成奖励反馈。  
> 二者形成“**推理 → 奖励 → 反思 → 再学习**”的自闭环优化体系。

---

## 🚀 整体框架概览

ReAgent-V 将传统单次推理的 LVLM 推进为一个 **动态反思的推理系统**。  
流程分为三阶段：

| 阶段 | 名称 | 核心功能 | 关键公式/策略 |
|------|------|-----------|---------------|
| **Stage 1** | Entropy-Calibrated Frame Selection | 熵校准帧选择，过滤冗余帧 | \(ECRS_i = \frac{s_i \cdot H_i}{\sum H_k}\) |
| **Stage 2** | Tool-Augmented Reasoning | 调用外部工具完成多模态推理 | 动态选择 T′ ⊂ T |
| **Stage 3** | Evaluation & Multi-Perspective Reflection | Critic 打分 + 三视角反思修正 | Conservative / Neutral / Aggressive |

---

## 🧩 模型组成

| 模块 | 功能定位 | 输入 | 输出 | 训练方式 |
|------|-----------|------|------|-----------|
| **Target Agent** | 视频理解与推理生成 | 视频关键帧 F + 任务 Q + 工具输出 R | 回答 A 及置信度 p | SFT → DPO → GRPO |
| **Critic Agent** | 实时评估与奖励生成 | (Q, F, A₀) | 奖励 E = {score + feedback} | SFT + 自蒸馏 |
| **Frame Selector** | 提取信息量最高的帧 | 全视频 V + Query Q | 帧集 F | 无需训练 |
| **Tool Factory** | 外部多模态工具集 | 任务类型 | 工具结果 R | 模块化调用 |
| **Reflection Module** | 三视角反思与合并 | (A₀, E) | A_final | Prompt-based |

---

## 📼 Stage 1 — Entropy-Calibrated Frame Selection

### 🔍 核心目标
在保持语义完整的前提下最小化输入帧数，提高推理效率。

### 🧮 算法要点
1. **语义相似度**：  
   \( s_i = \frac{e_i^\top q}{\|e_i\|\|q\|} \)
2. **信息熵**：  
   \( H_i = \tfrac13 \sum_{c∈{R,G,B}} -\sum_j p_{j,c}^{(i)} \log_2 p_{j,c}^{(i)} \)
3. **ECRS 打分**：  
   \( ECRS_i = \frac{s_i · H_i}{\sum H_k} \)
4. **迭代筛选**：  
   阈值 \(ECRS_i > k · α^m · τ\)，逐轮收紧选帧。

✅ **效果**：减少 30–40 % 推理时延，准确率提升 1–3 %。

---

## 🧠 Stage 2 — Tool-Augmented Reasoning

### 🔧 机制
Target Agent 根据任务类型自主调用工具 T′：

| 工具类别 | 功能 |
|-----------|------|
| OCR / ASR | 文本与语音转录 |
| Grounding DINO | 目标检测 + 语言对齐 |
| Scene Graph / Action Detector | 场景关系 + 动作识别 |
| Caption / VQA Model | 描述生成 + 问答 |
| CLIP | 语义匹配与检索 |

生成的工具结果 R 与帧 F 、问题 Q 共同输入模型：
\[
A₀ = f_\theta(Q, F, R)
\]

✅ **意义**：外部感知增强，提升模型推理可解释性与稳定性。

---

## 💬 Stage 3 — Evaluation & Multi-Perspective Reflection

### 1️⃣ Critic Agent 评估
- 生成结构化报告 E：
  - 分项得分：视觉匹配、时序准确、语言精度、推理特异性；
  - 文字化反馈。
- 若 A₀ 不满意：提出子问题 {qᵢ} + 更新工具 → 获得 R_update。

### 2️⃣ Target Agent 反思三策略

| 视角 | 修改范围 | 特征 |
|------|-----------|------|
| **Conservative (tᶜ)** | 仅修正最终结论 | 稳定可靠 |
| **Neutral (tⁿ)** | 调整场景实体 / 语义 | 平衡 |
| **Aggressive (tᵃ)** | 重写推理链 | 探索性强 |

每个视角生成 \(A^{(t)}, p^{(t)}\)。  
若全部 p > 0.6 → 合并一致部分，否则取最高 p 答案。

### 3️⃣ 数据回流
高分样本 → 缓存至 SFT/DPO/GRPO 训练集，实现在线自增强。

---

## 🧠 联合闭环训练

ReAgent-V 的两个核心模块并不是独立训练的 reward model + policy，而是构成一个 **reward-driven co-training framework**：

```
   ┌────────────┐          ┌──────────────┐
   │ Target Agent│ ───────▶│ Critic Agent │
   │(生成推理结果)│          │(实时评估反馈)│
   └────────────┘◀─────────└──────────────┘
        ↑                          ↓
     SFT / GRPO / DPO           奖励引导数据筛选
```

* **Critic Agent** 在推理时生成结构化评估报告（reward signal）。
* **Target Agent** 利用这些反馈进行 **多视角反思 (reflection)**，并在后续再训练时（SFT / DPO / GRPO）优化参数。
* 两者形成一个持续改进的闭环。

---

### 🧩 Target Agent 的训练

#### 1️⃣ 初始阶段：Supervised Fine-Tuning (SFT)

* **初始化模型**：从开源 LVLM (如 LLaVA-Video-7B、Qwen2-VL、Qwen2.5-VL、InternVL-2.5) 加载。
* **训练数据**：

  * 通用视频理解集（Video-R1-260k, VideoMME, LVBench, LongBench, EgoSchema等）；
  * 自生成的高质量推理轨迹（由 ReAgent-V 框架中 Critic 高评分样本筛选得到）。
* **目标函数**：最小化标准语言建模损失：
  [
  \mathcal{L}*{\text{SFT}} = -\sum_t \log P*\theta(a_t | a_{<t}, v, q)
  ]
  其中 (v) 为关键帧集合 (F)，(q) 为视频问题，(a_t) 为生成的 token。

✅ **目的**：让 Target Agent 学会生成结构化 reasoning、调用工具、以及输出带置信度的回答。

---

#### 2️⃣ 奖励驱动的二次训练阶段

SFT 之后，Target Agent 会进一步通过两种方式优化：

##### (a) **Direct Preference Optimization (DPO)**

* 从 Critic Agent 生成的评估报告 E 中提取“正/负样本对”：
  [
  (x, y_{\text{chosen}}, y_{\text{rejected}})
  ]
  其中 chosen 是高 reward 的回答、rejected 是低 reward 的回答。
* **损失函数**：
  [
  \mathcal{L}*{\text{DPO}} = -\log\sigma\left(\beta(\log P*\theta(y_{\text{chosen}}|x) - \log P_\theta(y_{\text{rejected}}|x))\right)
  ]
  使模型更倾向生成高评分回答。

##### (b) **Group Relative Policy Optimization (GRPO)**

* 类似 PPO，但以 Critic-Agent 奖励为目标。
* 每一组回答 ({A_1, A_2, …, A_k}) 计算相对奖励：
  [
  \hat{A}*i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}
  ]
  然后优化：
  [
  \mathcal{L}*{\text{GRPO}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta(A_i|x)}{\pi_{\text{old}}(A_i|x)} \hat{A}*i,
  \text{clip}\left(\frac{\pi*\theta(A_i|x)}{\pi_{\text{old}}(A_i|x)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right)\right]
  ]

✅ **结果**：
Target Agent 从 Critic 的动态 reward 中学习更合理的推理轨迹与反思模式。

---

#### 3️⃣ Reflection Prompt 适配

SFT / GRPO / DPO 训练中会混入三类专门模板：

| 模板类型             | 内容                       | 作用    |
| ---------------- | ------------------------ | ----- |
| **Conservative** | 仅调整最终结论句式                | 稳定性   |
| **Neutral**      | 更新 scene entity / object | 语义修正  |
| **Aggressive**   | 重写 reasoning chain       | 创造性反思 |

> 这些模板通过 prompt 工程嵌入训练，使模型在后续推理阶段具备三视角反思能力。

---

### 🧩 Critic Agent 的训练

Critic Agent 的目标是**生成奖励报告与评分维度**，即学习如何“像人一样评估推理质量”。

#### 1️⃣ 初始化阶段

* 起点模型：同样来自大型 LVLM（如 Qwen2.5-VL-7B 或 LLaVA-Video-7B）。
* 输入形式：
  [
  (Q, F, A_0) \to \text{E = Evaluation Report}
  ]
  包含：

  * 分项得分（Visual Alignment, Temporal Accuracy, Reasoning Specificity, Linguistic Precision等）；
  * 结构化反馈文本。

---

#### 2️⃣ 奖励标注来源与训练方式

Critic Agent 的训练数据来自三类源：

| 数据来源        | 内容                        | 获取方式                        |
| ----------- | ------------------------- | --------------------------- |
| **人工评审子集**  | 人工标注的视频问答正确性评分            | 从公开 benchmark 抽样标注          |
| **模板化评分规则** | 对齐基准规则（如视觉匹配、时序一致性）       | 自动生成 pseudo-label           |
| **在线反射数据**  | Target 生成结果 + reward 推理报告 | 自闭环采样（online bootstrapping） |

训练采用 **监督式奖励建模 (SFT)**：
[
\mathcal{L}*{\text{critic}} = |\hat{R} - R*{\text{label}}|*2^2 + \text{CE}(E*{\text{text}}, E_{\text{label}})
]
其中：

* (\hat{R})：Critic 预测奖励；
* (R_{\text{label}})：人工或模板评分；
* (E_{\text{text}})：Critic 输出的文字化评语；
* (E_{\text{label}})：参考解释文本。

---

#### 3️⃣ Online Self-Improvement

推理过程中，Critic Agent 的评分结果会被用于筛选高质量样本：

* 高 reward 样本 → 存入「positive replay buffer」；
* 低 reward 样本 → 用作 DPO 的 rejected pair。

这些样本在之后再次用于 Critic 自蒸馏（self-training）：
[
\mathcal{L}*{\text{self}} = |\hat{R}*{\text{new}} - R_{\text{old}}|_2^2
]
这样 Critic 会逐步对 Target 的输出分布进行自适应调整。

---

#### 🔄 四、Target 与 Critic 的交替优化策略

| 阶段            | Target Agent                | Critic Agent          | 优化目标                      |
| ------------- | --------------------------- | --------------------- | ------------------------- |
| **1. 预训练阶段**  | SFT (video captioning + QA) | SFT (reward labeling) | 初始化语言与视觉能力                |
| **2. 推理阶段**   | 生成 A₀ → 多视角反思               | 评估并生成 reward E        | 形成在线反馈                    |
| **3. 训练闭环阶段** | 使用 E 更新 via DPO / GRPO      | 使用高置信 E 自蒸馏           | 共同提升 reasoning & judgment |

这个训练循环可以持续运行，直到 reward variance 收敛。

---

### ✅ 训练总结

| 模块               | 模型类型                                     | 训练目标                | 训练方式                                   |
| ---------------- | ---------------------------------------- | ------------------- | -------------------------------------- |
| **Target Agent** | 基于 LVLM 的生成模型 (Qwen2.5-VL / LLaVA-Video) | 学会多步推理与反思           | SFT → DPO → GRPO (Reward-guided)       |
| **Critic Agent** | 基于 LVLM 的奖励模型                            | 输出结构化 reward + 反馈报告 | SFT + Self-Training on online feedback |
| **交互方式**         | 实时反思、动态打分、数据自筛选                          | ——                  | Reward-driven Co-Training Loop         |

---

> 🧩 **一句话总结：**
> ReAgent-V 的 Target Agent 通过 **Critic Agent 的实时奖励反馈** 实现自我反思与再学习；
> Critic Agent 又通过 **Target Agent 的输出分布** 持续自蒸馏，
> 共同构成了一个 **Reward-Driven Multi-Agent Online Learning Framework**。

---

## 🧩 模块间依赖关系

| 模块 | 是否训练 | 主要输入 | 主要输出 | 关键作用 |
|------|-----------|-----------|-----------|-----------|
| **Frame Selector** | ❌ | 视频 V + 问题 Q | 关键帧 F | 高效信息提取 |
| **Tool Factory** | ❌ | Q + 任务类型 | 工具结果 R | 多模态增强 |
| **Target Agent** | ✅ | F + Q + R | 答案 A | 视频推理 |
| **Critic Agent** | ✅ | Q + F + A | 奖励 E | 评估与指导 |
| **Reflection Module** | ❌ (Prompt) | A₀ + E | A_final | 自修正与融合 |

---

## 🧭 框架闭环总结

> **核心循环：**
>
> 1️⃣ Frame Selector 提取关键信息；  
> 2️⃣ Target Agent 结合工具推理；  
> 3️⃣ Critic Agent 生成奖励 E；  
> 4️⃣ Target Agent 三视角反思修正；  
> 5️⃣ 高奖励样本 → 再训练；  
> 形成“**推理-奖励-反思-再学习**”闭环。

---

## ⚙️ 局限与展望

| 问题 | 描述 | 未来方向 |
|------|------|-----------|
| 计算成本 | 双 Agent 推理较慢 | 模块共享 / 轻量化 critic |
| 奖励一致性 | 多维打分标准化难 | 统一 reward 标度 |
| 长时视频 | 仍依赖 clip 级选帧 | 引入时间建模与记忆机制 |
| 泛化性 | 需大规模预训练数据 | 结合 self-play 与 sim2real 数据 |

---

## 🧩 一句话总结

> **ReAgent-V = “Video Reasoner + Online Reward Critic + Multi-View Reflection”**  
> 通过推理-奖励-反思-再学习的协同机制，实现视频理解模型的自我改进与泛化提升，  
> 是向 **自适应、多模态、可解释 Video Agent** 迈出的关键一步。

---