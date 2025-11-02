# 🎥 VideoVerse: How Far is Your T2V Generator from a World Model  
### 📄 Paper Summary + Critic Agent Training Plan Based on VideoVerse

---

## 🧩 一、论文核心概述

**论文标题：** *VideoVerse: How Far is Your T2V Generator from a World Model*  
**作者团队：** Zeqing Wang et al.（2025）  
**目标：** 从“世界模型（World Model）”视角重新定义 Text-to-Video (T2V) 模型的评估体系。  

---

### 🎯 研究动机

现有 T2V 评估基准（如 VBench、EvalCrafter）主要衡量：
- 画面质量（FID、FVD）
- 文本语义对齐  

但这些指标**无法反映模型是否真正理解世界规律与因果逻辑**。  
因此，作者提出 **VideoVerse** —— 首个用于衡量 T2V 模型“世界模型能力”的系统化基准。

---

### 🧠 基准设计：十个维度 × 动静双视角

| 类别 | 维度 | 说明 |
|------|------|------|
| **动态 (Dynamic)** | Event Following | 是否能生成正确的事件顺序 |
|  | Mechanics | 是否符合物理规律 |
|  | Interaction | 物体交互是否合理 |
|  | Material Properties | 材质属性是否正确（如融化、反弹） |
|  | Camera Control | 摄像机控制是否符合指令 |
| **静态 (Static)** | Natural Constraints | 是否遵守自然规律（如水流方向） |
|  | Common Sense | 是否符合常识与社会知识 |
|  | Attribution Correctness | 属性是否正确（颜色、形状等） |
|  | 2D Layout | 平面布局是否合理 |
|  | 3D Depth | 景深与前后关系是否真实 |

这些维度构成一个“**世界理解 → 动态逻辑 → 视觉物理一致性**”的完整评估体系:contentReference[oaicite:0]{index=0}。

---

### 🏗️ 数据与 Prompt 构建

- 共计 **300 个高质量 Prompt**，覆盖 10 个评估维度与 815 个事件。
- 数据来自三大语义领域：
  1. **Daily Life**（日常场景）  
  2. **Scientific Experiment**（科学实验，检验物理常识）  
  3. **Science Fiction**（科幻测试，评估跨域泛化）

每个 prompt 均包含显式与隐式事件描述（hidden semantics），例如：
> “Pour vinegar and then add baking soda”  
模型需隐式生成 “气泡反应” 才算符合世界规律。

---

### ⚙️ 评估协议

采用 **VLM 驱动的问答式评估**：
- Binary QA：每个维度下若干 Yes/No 问题；
- Event Following：通过 LCS（Longest Common Subsequence）匹配生成事件顺序；
- 综合得分：
  \[
  S(V) = LCS(V) + \sum_i \sum_j I(Eval(V, q_{i,j})=Yes)
  \]

主评估器为 **Gemini 2.5 Pro**（与人类标注一致率 > 90%）:contentReference[oaicite:1]{index=1}。

---

### 📊 实验结果与结论

- **闭源模型（Veo-3, Sora-2）** 在“世界模型维度”得分显著领先；
- **开源模型（Wan2.2, OpenSora2.0, SkyReels-V2）** 虽在画面质量接近，但在因果推理与物理规律维度上仍存在明显差距；
- **Qwen2.5-VL 32B** 可作为评估器替代 Gemini，结果高度相关（Spearman r=0.71）:contentReference[oaicite:2]{index=2}。

> 🔍 结论：T2V 模型距离“世界模型”仍有较大鸿沟，缺乏稳定的因果推理与物理认知能力:contentReference[oaicite:3]{index=3}。

---

## 🧩 二、如何基于 VideoVerse 实现 Critic Agent 训练

> 目标：训练一个 **Critic Agent**，输入 *(text prompt + generated video)*，输出多维度评分（event following, interaction, mechanics 等），作为世界理解判别器。

---

### 🏗️ Step 1. 数据与仓库准备

- 克隆官方仓库与数据集：
  ```bash
  git clone https://github.com/Zeqing-Wang/VideoVerse
  git lfs install
  git clone https://huggingface.co/datasets/NNaptmn/VideoVerse


* 数据结构：

  ```
  prompt/prompts_of_VideoVerse.json     # 所有prompt与问题模板
  eval_videos/                          # 存放生成视频
  eval_res/                             # 存放VLM评估输出
  ```

---

### 🧠 Step 2. 生成训练监督（标签蒸馏）

利用 VideoVerse 官方评估脚本，使用 **Gemini 2.5 Pro** 或 **Qwen3-VL** 作为“强师”产生标签：

```bash
# 跑VLM评估，输出eval_res.json
python scripts/eval_with_Gemini_like_video_url.py
# 或使用开源VLM
python scripts/eval_with_other_vlm.py
```

得到每条样本的维度问题答案（Yes/No）与事件顺序预测。

---

### 🧱 Step 3. 构建 SFT 训练集

合并官方 prompt 与评估结果，生成如下 JSONL：

```json
{
  "video_path": "eval_videos/sample.mp4",
  "t2v_prompt": "A person pours vinegar into baking soda.",
  "dimension": "mechanics",
  "question": "Does the mixture bubble due to a chemical reaction?",
  "target": "Yes"
}
```

每个样本可包含多个维度标签（binary）与一个事件链标签（event sequence）。

---

### ⚙️ Step 4. 模型微调（以 Qwen3-VL 为例）

```bash
python train_critic_qwen3vl.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train_file data/critic_sft/train.jsonl \
  --video_frames 16 --video_fps 8 \
  --lora --lr 1e-5 --epochs 2
```

* **输入：** Prompt + 视频帧（8~16帧采样）
* **输出：** 多维度评分 (0–1) 或 二元判断 (Yes/No)
* **损失：**

  * Binary：交叉熵 (Yes/No)
  * Event Following：LCS 分数回归或序列生成

---

### 🧩 Step 5. Critic 推理与评估

微调后，Critic Agent 可以：

1. **输入**：`text prompt + generated video`
2. **输出**：

   ```json
   {"event_following": 0.82, "mechanics": 0.76, "interaction": 0.69, ...}
   ```
3. **验证**：可直接重用 VideoVerse 的 `cal_acc.py` 脚本，计算与强师或人工一致率。

---

### 📈 Step 6. 训练方案总结

| 模块   | 功能                          | 备注          |
| ---- | --------------------------- | ----------- |
| 数据   | VideoVerse prompt + VLM评估结果 | 监督信号来源      |
| 模型   | Qwen3-VL / Qwen2.5-VL       | 多模态理解基底     |
| 训练方式 | SFT（LoRA可选）                 | 可多任务训练      |
| 输出   | 多维度评分（event, mechanics 等）   | 支持回归或分类     |
| 验证   | 使用官方 `cal_acc.py`           | 与Gemini评估对齐 |

---

## 🧩 三、整体闭环流程

```mermaid
graph TD
A[VideoVerse Prompts + Videos] --> B[VLM Teacher (Gemini/QwenVL)]
B --> C[Distilled Evaluation Labels]
C --> D[SFT Fine-tune Qwen3-VL Critic Agent]
D --> E[Critic Agent Scoring]
E --> F[RLHF / Video Quality Control / World Model Alignment]
```

---

## ✅ 总结

> VideoVerse 为 T2V 世界模型能力提供了首个系统评估基准。
> 你可以借助它的数据与脚本，蒸馏出“世界理解”监督信号，
> 微调 Qwen3-VL 形成 Critic Agent ——
> 一个能独立评估视频生成中**因果、物理与常识一致性**的智能评估器。

