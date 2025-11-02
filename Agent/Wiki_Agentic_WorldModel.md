# 🧩 **Agentic World Model Framework**

> **核心理念：**
> 本框架构建了一个由 **Action Agent、Critic Agent、Curriculum Agent、World Model Environment、World Memory** 组成的闭环体系。
> 它通过「任务生成 → 动作规划 → 视频生成 → 状态评估 → 课程进化」形成自我进化式的视频世界模型学习机制，
> 实现从**语言任务理解 → 动作轨迹生成 → 视频生成与评估 → 新任务探索**的多智能体协同循环。

---

![WorldAgent](https://github.com/yuyang-cloud/Wiki/blob/main/Assets/WorldAgent.jpg?raw=true)

---

## 🚀 整体框架概览

### 🎯 系统目标

让智能体能够：

1. **从自然语言任务 (Text Goal)** 出发生成多步动作轨迹；
2. 利用 **World Model (Video Generator)** 模拟执行效果；
3. 由 **Critic Agent** 自动评价动作与视频的合理性、任务完成度；
4. 最终通过 **Curriculum Agent** 自动演化任务难度，实现持续自我学习。

---

### 🧩 模块组成概览表

| 模块                             | 功能定位      | 输入                                                       | 输出                         | 训练方式                               |
| ------------------------------ | --------- | -------------------------------------------------------- | -------------------------- | ---------------------------------- |
| **Action Agent (π)**           | 动作生成与策略探索 | Text Goal + Reference Image + (History via World Memory) | 动作轨迹（text形式）               | **SFT + GRPO/DPO 强化微调**            |
| **Critic Agent (M₍critic₎)**   | 视频理解与奖励评估 | 动作轨迹 + 生成视频序列                                            | 状态描述 + 成功/失败标签 + 奖励分数      | **SFT + Pairwise Reward Modeling** |
| **Curriculum Agent (M₍curr₎)** | 任务生成与课程进化 | 历史任务 + 状态描述 + 成败反馈                                       | 新任务指令（Goal）                | **LLM Prompt生成（无梯度）**              |
| **World Model Environment**    | 视频生成模拟器   | 动作轨迹文本                                                   | 视频序列（world state）          | 冻结生成模型（如 Wan2.2, StreamDiT 等）      |
| **World Memory**               | 状态存储与检索模块 | Key=任务、动作、结果；Value=状态描述/视频摘要                             | 历史信息检索 (Retrieval Context) | 向量数据库或KV存储（外部模块）                   |

---

## 🎮 Action Agent（行动智能体）

### ✅ 功能与定位

* 根据 **任务描述 (Text Goal)** 与 **参考图像 (Reference Image)** 生成动作轨迹；
* 轨迹以自然语言形式（或结构化 JSON）表示，为 World Model 提供可执行条件；
* 模拟人类在环境中完成目标的多步规划与执行。

### ⚙️ 架构与实现

* **模型基座**：Qwen2.5-VL / Qwen3-VL / MLLM 系列；
* **输入**：

  ```
  {
    "goal": "Cook an omelette.",
    "reference_image": <kitchen scene>,
    "context": WorldMemory.retrieve(goal)
  }
  ```
* **输出**：

  ```
  [
    {"step": 1, "action": "Crack two eggs into a bowl"},
    {"step": 2, "action": "Whisk eggs thoroughly"},
    {"step": 3, "action": "Pour mixture into heated pan"}
  ]
  ```

### 🎓 训练机制

1. **SFT阶段**：

   * 使用已有的 CoT + Action 轨迹数据进行监督微调；
   * 学习动作分解与语义规划；
2. **GRPO/DPO阶段**：

   * 固定 Critic Agent；
   * 生成多条轨迹 → World Model 生成对应视频；
   * 由 Critic 评估打分 → 奖励信号驱动 Action Agent 优化；
3. **损失函数**：
   [
   L = L_{GRPO} + \lambda L_{DPO}
   ]
   其中 ( L_{GRPO} ) 用于正向偏好强化，( L_{DPO} ) 控制生成一致性。

---

## 🧠 Critic Agent（评估智能体）

### ✅ 功能与定位

* 接收 **动作轨迹** 与 **生成视频**，评估任务完成度；
* 输出多维度评估指标与奖励信号；
* 作为强化学习的“Reward Model”。

### ⚙️ 评估维度

1. **Action Following** — 视频是否忠实执行了动作轨迹；
2. **Temporal Consistency** — 时间顺序是否合理；
3. **Object Interaction** — 与关键物体交互是否正确；
4. **Goal Achievement** — 最终结果是否完成任务目标；
5. **Visual Realism / Physics** — 物理合理性与视觉连贯度。

### ⚙️ 实现方式

* 模型基座：**Qwen3-VL / LLaVA-Next / InternVL2**
* 输入：

  ```
  (goal_text, action_text, generated_video)
  ```
* 输出：

  ```
  {
    "state_caption": "The person successfully pours eggs into the pan.",
    "success": true,
    "scores": {
      "action_following": 0.92,
      "temporal_consistency": 0.88,
      "goal_completion": 0.95
    },
    "reward": 0.91
  }
  ```

### 🎓 训练机制

1. **SFT阶段：**

   * 使用 VideoVerse / VideoCriticBench 等标注数据；
   * 任务：多维度评分 + Caption + 成功/失败分类；
2. **偏好建模阶段 (Pairwise Reward Modeling)：**

   * 对同一任务的多种视频生成结果进行两两比较；
   * 训练 Critic 输出偏好奖励，稳定 Action 优化；
3. **冻结使用：**

   * 作为奖励模型，后续不再更新；
   * 输出 reward signal 给 Action Agent。

---

## 📚 Curriculum Agent（课程生成智能体）

### ✅ 功能与定位

* 负责生成并演化探索任务；
* 随着训练阶段推进，任务逐步复杂化；
* 结合历史失败/成功任务，实现课程式自进化。

### ⚙️ 任务生成阶段

| Phase   | 任务类型    | 示例                                                         |
| ------- | ------- | ---------------------------------------------------------- |
| Phase 1 | 原子动作生成  | “Pick up the cup.”                                         |
| Phase 2 | 组合动作生成  | “Pour water into the cup and stir.”                        |
| Phase 3 | 多阶段复杂任务 | “Make coffee, pour it into a cup, and clean up the table.” |

### ⚙️ 实现机制

* 基于 **大语言模型 (LLM)** prompt 驱动；
* 输入：

  ```
  past_task, state_caption, reward_feedback
  ```
* 输出：

  ```
  new_goal, curriculum_phase
  ```
* 目标：逐步构建从简单到复杂的任务树，指导系统探索更多样的行为。

---

## 💾 World Memory（世界记忆模块）

### ✅ 功能与定位

* 存储任务执行的历史上下文；
* 为 Action Agent 提供 retrieval 支撑；
* 记录 (goal, action, video summary, reward) 对齐信息。

### ⚙️ 实现

* 使用向量数据库（如 FAISS / Milvus / LanceDB）；
* key: goal 或 scene embedding；
* value: 状态描述 + 动作轨迹 + 视频摘要。

---

## 🌍 World Model Environment（世界模型环境）

### ✅ 功能

* 模拟动作轨迹执行；
* 将文本动作轨迹 → 生成视频；
* 输出 video latent/state 给 Critic Agent。

### ⚙️ 可选实现

* 采用现有视频生成模型：

  * **Wan2.2 / StreamDiT / CausVid / SkyReels-V2**
* 支持条件控制（text/action tokens）。

---

## 🧩 模块训练顺序与依赖关系

| 阶段                            | 训练目标           | 模块                         | 是否冻结               | 说明                               |
| ----------------------------- | -------------- | -------------------------- | ------------------ | -------------------------------- |
| **Phase 1: SFT 预训练**          | 建立基础格式理解与输出一致性 | Action Agent, Critic Agent | ❌                  | Action 学动作规划格式，Critic 学视频理解与评分   |
| **Phase 2: GRPO / DPO 强化阶段**  | 奖励引导策略优化       | Action Agent               | Critic 冻结          | Critic 评估多样轨迹 → 奖励信号强化 Action 策略 |
| **Phase 3: Curriculum 自进化阶段** | 任务难度递增，体系自我演化  | Curriculum Agent           | Action/Critic 共同参与 | Curriculum 输出新任务，驱动闭环自学习         |

---

## 🔁 自进化闭环循环机制

```
   ┌──────────────────────────────┐
   │     Curriculum Agent         │
   │ - 生成任务 Goal              │
   │ - 调整任务复杂度 Phase       │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │      Action Agent (π)        │
   │ - 输入 Goal + Ref Image      │
   │ - 输出动作轨迹 (Text)         │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │   World Model Environment    │
   │ - 根据轨迹生成视频           │
   │ - 输出 World State           │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │        Critic Agent          │
   │ - 评估视频与动作匹配度        │
   │ - 输出状态描述+奖励           │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │        World Memory          │
   │ - 存储任务/轨迹/状态          │
   │ - 支持检索复用               │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │     Curriculum Agent         │
   │ - 综合状态与奖励生成新任务     │
   └──────────────────────────────┘
```

---

## 📘 一句话总结

> 该框架实现了一个“**由智能体驱动的世界模型自进化体系**”：
> 先训练会生成（Action），再训练会评估（Critic），最后引入课程探索（Curriculum）实现**长期的任务成长与自学习**。
> 它融合了 **Video Generation + Multi-Agent Reinforcement + Curriculum Learning**，
> 形成一个可持续扩展的视频世界智能系统。

---

