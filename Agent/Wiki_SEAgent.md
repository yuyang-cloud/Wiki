# 🧩 SEAgent 方法与模块全景笔记

> **核心理念：**
> SEAgent 构建了一个三模块闭环体系——**Actor Model、World State Model、Curriculum Generator**——使计算机智能体能够通过与软件环境的交互、自主评估与任务生成，实现“自我进化式学习（Self-Evolution）”，完全摆脱人工标注与人工奖励依赖。

---

![SEAgent](https://github.com/yuyang-cloud/Wiki/tree/main/Assets/SEAgent.jpg)

---

## 🚀 整体框架概览

SEAgent 的训练目标是：
让 Computer Use Agent (CUA) 在不同软件中通过自主探索获得可验证的奖励，从而通过强化学习自我进化。

### 模型组成

| 模块                                | 功能定位      | 输入             | 输出                     | 训练方式                                  |
| --------------------------------- | --------- | -------------- | ---------------------- | ------------------------------------- |
| **Actor Model (π)**               | 执行动作与任务探索 | GUI截图 + 任务指令 I | 动作分布 π(a|s,I)          | Reinforcement Fine-Tuning (GRPO + AI) |
| **World State Model (M_state)**   | 环境理解与奖励生成 | GUI轨迹 H + 截图序列 | 状态描述 C + 成败判断 J + 奖励 r | LVLM 微调（监督学习）                         |
| **Curriculum Generator (M_task)** | 任务演化与知识积累 | Jₚ + Cₚ + Uₚ   | 新任务 Iₚ₊₁ + 新知识 Uₚ₊₁    | Prompt-driven LLM 生成                  |

三者构成闭环：

> Curriculum Generator → Actor Model → Environment → World State Model → Curriculum Generator

形成完整的自进化循环。

---

## 🎮 Actor Model（行动者模型）

### ✅ 功能与定位

* 负责执行软件操作任务（点击、输入、拖拽等）；
* 模拟人类用户与 GUI 界面的操作行为；
* 生成动作轨迹 (H = {(s_0, a_0), (s_1, a_1), \ldots})，供奖励模型评估。

### ⚙️ 架构与实现

* 初始化模型：**UI-TARS-7B-DPO**；
* 输入：任务指令 I + GUI 截图状态 s_t；
* 输出：动作概率分布 π(a|s_t, I)；
* 行动类型包括：

  * 点击点预测；
  * 文本输入；
  * 拖拽路径；
  * 选择控件。

### 🎓 训练机制

采用 **Reinforcement Fine-Tuning (RFT)**：

* 奖励来源：World State Model 的逐步评估信号；

* 损失函数：

  [
  L = L_{\text{GRPO}} + \gamma L_{\text{AI}} \quad (\gamma=0.2)
  ]

* 训练细节：

  * 学习率：2×10⁻⁵；
  * 批大小：16；
  * 梯度累积：8；
  * 优化器：AdamW + cosine decay；
  * 设备：8×A100 80GB；
  * 训练步数：约 1000；
  * 分三阶段进化（Phase 1–3）。

### 🌍 作用

* 在每个阶段执行任务集 Iₚ；
* 根据 WSM 的奖励信号强化成功策略；
* 跨阶段累积技能，逐步扩展可操作的软件范围。

---

## 🌐 World State Model（世界状态模型）

### ✅ 功能与定位

* 负责评估 Actor 的每步操作；
* 输出 GUI 状态变化、任务成功与否以及逐步奖励；
* 充当强化学习的“Reward Model”。

### ⚙️ 架构

* 基于 **大型视觉语言模型 (LVLM)** 微调（如 Qwen2.5-VL）；
* 输入：GUI截图序列 + 动作描述；
* 输出：

  1. GUI状态变化 caption；
  2. 成功/失败标注 (aᵀ / aᶠ)；
  3. 奖励信号；
  4. GUI变化解释 C。

### 🎓 训练机制

* 数据来源：

  * OSWorld、AgentRewardBench；
  * 来自 UI-TARS 与 SEAgent 早期阶段的轨迹；
* 训练任务：

  * GUI Caption；
  * 成败判断；
  * 错误步定位；
* 奖励标注逻辑：

  * 成功轨迹：所有动作 aᵀ；
  * 冗余轨迹：有效步为 aᵀ；
  * 失败轨迹：错误步为 aᶠ，之前步为 aᵀ；
* 训练方式：

  * 监督微调（SFT）；
  * 学习多帧历史依赖；
  * 联合训练 caption + 判断任务可显著提升准确率；
* 结果：

  * 精度接近 GPT-4o；
  * 冻结后作为奖励模型使用。

---

## 🧮 Curriculum Generator（课程生成器）

### ✅ 功能与定位

* 生成、评估并演化任务；
* 管理“软件知识记忆”（Software Guidebook U）；
* 是 SEAgent 的自进化核心。

### ⚙️ 实现机制

* 基于 **Qwen2.5-72B**；
* 无梯度更新，纯 Prompt 驱动；
* 输入：

  * 当前任务 Iₚ；
  * 来自 WSM 的状态变化描述 Cₚ；
  * 成败判断 Jₚ；
  * 当前软件知识库 Uₚ；
* 输出：
  [
  U_{p+1}, I_{p+1} = M_{\text{task}}(U_p, I_p, J_p, C_p)
  ]
* 动态任务生成示例：

  ```
  Draw a rectangle. → Draw a green rectangle → Draw a green rectangle with 50% transparency
  ```



### 🧩 训练与执行逻辑

1. **Phase 0：任务初始化**

   * WSM 对 GUI 元素进行密集 caption；
   * Curriculum Generator 生成初始任务 I₀ 与知识 U₀；
2. **Phase p：任务演化**

   * 收集上一阶段评估结果 (Jₚ, Cₚ)；
   * 更新知识库 Uₚ → Uₚ₊₁；
   * 生成更复杂的任务 Iₚ₊₁；
3. **特性**

   * 课程式进化（Curriculum Learning）；
   * 支持跨软件泛化；
   * 不依赖人工监督。

---


## 🧩 SEAgent 模块训练顺序与依赖关系

整个 SEAgent 的自进化学习系统遵循严格的分阶段训练流程，其核心思想是：

> **先让“判断者”学会准确评估 → 再让“行动者”基于正确奖励学习 → 最后由“课程生成器”驱动持续进化。**

---

### 🧠 **阶段 1：World State Model 预训练（Reward Model Pretraining）**

**目的：**
在 Actor 学习之前，必须先获得一个具备高可信度的奖励评估器。

**训练数据：**
来自 OSWorld、AgentRewardBench、UI-TARS 产生的 GUI 操作轨迹数据。
每条轨迹包含：

* GUI 界面截图序列；
* 操作序列；
* 任务成功/失败标签；
* 状态变化说明（state captions）。

**训练目标：**

1. GUI 状态理解（captioning）；
2. 任务完成与否判断；
3. 冗余或错误步骤定位；
4. 输出逐步的 aᵀ / aᶠ 标注。

**训练方式：**

* 在 LVLM（如 Qwen2.5-VL）上进行 supervised fine-tuning；
* 学习历史截图间的时序依赖与多步推理；
* 最终模型具备 step-level reward estimation 能力。

**结果：**

* 微调完成的 **World State Model (WSM)** 拥有对 GUI 任务的精准理解能力；
* 后续阶段中完全冻结，不再更新；
* 成为整个系统的“奖励与评估核心”。

---

### ⚙️ **阶段 2：闭环强化学习（Actor + World State Model）**

**核心机制：**
冻结的 WSM 提供逐步奖励信号；
Actor Model 基于此进行强化学习更新。

#### **(1) 初始化**

* **Actor Model** ← 使用 UI-TARS-7B-DPO 作为初始策略；
* **World State Model (冻结)** ← 提供 step-level reward；
* **Curriculum Generator (冻结)** ← 基于 prompt 的任务生成器；
* 初始任务集 ( I_0 ) 与初始软件知识库 ( U_0 ) 由 Curriculum Generator 基于 GUI caption 生成。

#### **(2) 执行与评估**

1. Actor 执行任务 ( I_p )，生成轨迹 ( H = {(s_0,a_0),…} )；
2. WSM 分析整个轨迹，输出：

   * 每步正确/错误标签（aᵀ, aᶠ）；
   * GUI 状态变化描述 Cₚ；
   * 奖励信号；
3. 这些评估结果作为强化学习的监督信号。

#### **(3) Actor 强化学习更新**

* 使用 **Group Relative Policy Optimization (GRPO)** 强化成功动作；
* 使用 **Adversarial Imitation (AI)** 惩罚失败动作；
* 损失函数：
  [
  L = L_{GRPO} + \gamma L_{AI}, \quad \gamma=0.2
  ]
* 得到更新后的策略 πₚ₊₁。

---

### 🧩 **阶段 3：Curriculum Generator 驱动任务进化**

**作用：**
引导 Agent 学习“新的任务”，逐步扩展探索空间。

**实现方式：**

* 基于大型语言模型 **Qwen2.5-72B**；
* 并非梯度训练，而是 **prompt-driven 生成机制**。

**输入：**

* 上一轮任务指令 ( I_p )；
* 来自 WSM 的评估反馈 ( J_p, C_p )；
* 当前知识库（软件指南书）( U_p )。

**输出：**
[
U_{p+1}, I_{p+1} = M_{task}(U_p, I_p, J_p, C_p)
]
即：

* 更新后的知识记忆 Uₚ₊₁；
* 新生成的更具挑战性的任务集 Iₚ₊₁。

**功能：**

* 通过 prompt engineering 自动整合 GUI 状态变化描述；
* 动态设计难度递增任务；
* 实现课程式学习（Curriculum Learning）。

---

### 🔁 **循环迭代机制（Self-Evolution Loop）**

1. **初始化：**
   Curriculum Generator 生成初始任务；
2. **执行任务：**
   Actor 执行动作；
3. **状态评估：**
   World State Model 提供奖励信号；
4. **策略更新：**
   Actor 进行强化学习；
5. **任务进化：**
   Curriculum Generator 生成下一批更复杂任务；
6. **重复** 至收敛或任务多样性充分。

---

### 🧩 最终系统关系总结表

| 模块 | 训练阶段 | 梯度 | 主要输入 | 主要输出 | 关键作用 |
|---|---|---|---|---|---|
| **World State Model (WSM)** | 阶段 1 | ✅ | GUI 轨迹 + 状态标签 | 逐步奖励 \(r_t\)、成败 \(J\)、状态描述 \(C\) | **奖励与评估（冻结使用）** |
| **Actor Model (\(\pi\))** | 阶段 2 | ✅ | GUI 截图 + 任务指令 \(I\) + 奖励 \(r_t\) | 动作分布 \(\pi(a\mid s,I)\)、轨迹 \(H\) | **执行与学习（GRPO + AI）** |
| **Curriculum Generator (M\_task)** | 阶段 3 | ❌ | \(J_p, C_p, U_p\) | 新任务 \(I_{p+1}\)、新知识 \(U_{p+1}\) | **任务进化与知识积累** |

---


### 🔁 自进化循环流程图

```
             ┌──────────────────────────────┐
             │  Curriculum Generator (LLM)  │
             │  ────────────────            │
             │  - 生成任务 Iₚ               │
             │  - 更新知识 Uₚ               │
             └──────────────┬───────────────┘
                            │ 任务 Iₚ
                            ▼
             ┌──────────────────────────────┐
             │      Actor Model (πₚ)        │
             │  - 执行动作 aₜ               │
             │  - 强化学习更新              │
             └──────────────┬───────────────┘
                            │ 执行动作 aₜ
                            ▼
             ┌──────────────────────────────┐
             │       Computer UI 环境        │
             │  - 返回 GUI 状态 Sₜ₊₁        │
             └──────────────┬───────────────┘
                            │ 状态 Sₜ₊₁
                            ▼
             ┌──────────────────────────────┐
             │    World State Model (VLM)   │
             │  - 输出奖励 rₜ               │
             │  - 状态变化描述 Cₚ           │
             │  - 成败判断 Jₚ               │
             └──────────────┬───────────────┘
                            │ (rₜ, Jₚ, Cₚ)
                            ▼
             ┌──────────────────────────────┐
             │     Curriculum Generator     │
             │  - 更新知识 Uₚ₊₁             │
             │  - 生成任务 Iₚ₊₁             │
             └──────────────┬───────────────┘
                            ▼
                       Phase p+1 开始
```

---

📘 **一句话总结：**

> SEAgent 的训练流程是一个“从奖励模型出发的自我进化体系”：
> **先训练一个可靠的评估者（WSM） → 冻结它来监督行动者（Actor） → 再由语言模型（Curriculum Generator）设计新任务，驱动持续自进化。**

---

## 📊 关键实验与结果

* **Reward Model 准确率**：

  * Caption + Judgment 联合训练提升 6.4%；
* **任务生成性能**：

  * 在 Celestia 软件上显著超越基线方法；
  * Guidebook-based Curriculum 生成策略在探索多样性上优于 NNetNav；
* **整体性能提升**：

  * 在 OSWorld 与 ScienceBoard 上均实现成功率翻倍；
  * 尤其在 OOD 应用中提升显著。

---

## 🧭 总结

> **SEAgent 的核心创新点：**

1. 以 **World State Model** 替代传统 critic；
2. 通过 **GRPO + AI** 形成无人工监督的强化学习机制；
3. 利用 **LLM 驱动的 Curriculum Generator** 实现任务的自进化；
4. 形成“**评估—强化—任务生成**”的三阶段自闭环体系。

> **最终实现：**
>
> * 自主生成任务；
> * 自主奖励；
> * 自主进化；
>   实现真正意义上的“**Self-Evolving Computer Use Agent**”。
