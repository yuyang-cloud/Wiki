# LTX-Video 训练与推理输入输出设定总结

## 🧠 总体概述

LTX-Video 模型在设计上采用 **多时长混合训练 + 固定 token 数 + 高压缩 Video-VAE**，
使得它在训练时以 8–10s 的短视频片段为核心单元，推理时默认生成 5s（120帧）视频，
但可扩展到更长时长或通过“分段接力”方式生成分钟级视频。

---

## 🧩 训练阶段（Training）

### 🎞 视频片段长度
- 训练数据中的 clip 时长分布集中在 **5–10 秒**；
- 平均长度约为 **8 秒**；
- 这一长度与 VAE 的时间压缩比（×8）匹配，能在保持动态一致性的同时控制 token 数量。

### ⏱ 帧率 (FPS)
- 模型统一采用 **24 FPS**；
- 因此每个 clip 约对应：
  - 8s × 24 = **192 帧**
  - 10s × 24 = **240 帧**

### 🧮 Token / 压缩结构
- Video-VAE 的压缩率为 **32×32×8**（空间×空间×时间）；
- 即 **每 8 帧 → 1 latent block**；
- 模型一次输入约 24–30 个时间步（192 ÷ 8 = 24）；
- Latent 空间通道数为 128，总压缩比约 **1:192**。

### ⚙️ 多时长混训
- 同时训练多种组合（如 5s、8s、10s）；
- 使用 **0–20% token dropping** 保持每批样本 token 数一致；
- 有助于模型泛化到“未见过的时长或分辨率”。

---

## 🚀 推理阶段（Inference）

### 🧾 默认配置
- **分辨率**：768×512  
- **时长**：5 秒  
- **帧率**：24 FPS（共 120 帧）  
- **去噪步数**：40 步  
- **运行时间**：在 H100 上 2 秒内完成生成（超实时）

### ⏩ 可扩展时长
- 因为训练时混合了多时长片段，
  推理时可在显存允许下生成更长片段（10s 左右）；
- 时长扩展受限于 token 数量（与帧数线性增长）。

### 🔁 自回归式长视频生成
若需生成分钟级长视频，推荐使用 **分段生成 + 首帧条件接力**：
1. 将长视频拆分为多个 8–10s 片段；
2. 第一段正常生成；
3. 对于后续片段：
   - 提取上一段的 **末帧 latent** 作为 **首帧条件**；
   - 设置该帧 token 的时间步 `t ≈ 0`；
   - 其他帧的时间步 `t = 1`；
   - 这样模型会延续前段的内容，保证连续性；
4. 最后再进行片段拼接。

---

## 📊 关键参数汇总表

| 阶段 | 时长 / 帧数 | FPS | 输入结构 | 说明 |
|------|-------------|------|-----------|-------|
| 训练 | 8–10s（192–240帧） | 24 | 每8帧→1 latent block (32×32×8 压缩) | 多时长混训 + token drop |
| 推理 | 5s（120帧） | 24 | 同上 | 默认生成配置 |
| 可扩展生成 | ≤10s | 24 | 同上 | 显存允许即可 |
| 长视频生成 | 分段拼接 | 24 | 上段末帧→首帧条件 | 自回归接力生成更长视频 |

---

## 🧩 总结结论

- LTX-Video 训练输入通常为 **8–10s @ 24FPS（≈192–240帧）**；
- 推理默认输出 **5s @ 24FPS（120帧）**；
- 高压缩 Video-VAE（32×32×8）保证每 8 帧合并为 1 latent；
- 通过混训与 token-drop 策略，模型对更长视频具备一定外推能力；
- 若需生成更长视频，最佳策略为：
  > **分段生成 + 上段末帧条件接力**  
  实现自回归式长视频生成。

---



# SkyReels-V2 训练与推理设定总结

## 📘 简介

**SkyReels-V2** 是由 SkyworkAI 发布的高性能长视频生成模型（2025），核心思想是：
- 通过 **Diffusion Forcing** 框架实现任意时长连续视频生成；
- 在训练中混合 **部分预测帧 + 部分真实帧**，减少 exposure bias；
- 在推理中通过 **滑动窗口 (sliding window)** 与 **重叠帧 (overlap)** 实现无缝长视频生成；
- 同时融合 **SFT → RL → Diffusion Forcing → 高分辨率微调** 的多阶段训练流程。

目标是突破传统 5–10s 视频生成的限制，实现**分钟级可扩展视频生成**。

---

## 🧠 训练阶段（Training Settings）

| 项目 | 设定 | 说明 |
|------|------|------|
| **训练流程** | 多阶段：Pretraining → SFT → RL → Diffusion Forcing → High-Res Fine-tune | 逐步提升模型的语义理解、动作一致性和画质精度。 |
| **训练框架** | **Diffusion Forcing** | 每个 token 分配独立噪声水平，模型学习如何在部分噪声帧存在时恢复整个序列。 |
| **目标** | 减少 **exposure bias** | 模型在训练时已见到“自己生成的帧”作为输入，因此推理时 rollout 更稳定。 |
| **FoPP 调度** | Frame-oriented Probability Propagation | 在每次训练中随机选取帧 f 与时间步 t，使噪声均匀分布在视频时间轴上。 |
| **强化学习阶段 (RL)** | 使用 motion-quality reward 模型 + DPO | 对多个候选视频打分，优化生成的运动流畅性和物理合理性。 |
| **数据集** | 多源高质量视频（网络电影、纪录片、艺术片等） | 通过 **SkyCaptioner-V1** 自动生成层次化 caption；并执行 shot segmentation 与筛选。 |
| **分辨率组合** | 多分辨率混训（540P、720P） | 后期 fine-tune 阶段提升到高分辨率。 |
| **训练片段长度** | 可变（非固定 8–10s） | 训练时允许多种长度组合，支持在更长时间尺度上建模运动。 |

---

## 🚀 推理阶段（Inference Settings）

| 项目 | 默认 / 推荐设定 | 说明 |
|------|----------------|------|
| **输出分辨率** | 540P 模型：544×960<br>720P 模型：720×1280 | 不同分辨率版本分别训练。 |
| **帧率 (FPS)** | **24 FPS** | 与训练阶段保持一致。 |
| **基础帧窗口 (base_num_frames)** | 540P 模型：97 帧<br>720P 模型：121 帧 | 表示单个滑窗生成的基础帧数。 |
| **去噪步数 (steps)** | 每窗口 30 步 | 每个滑窗的 diffusion 步数。 |
| **异步/同步模式** | - 同步：`ar_step = 0`<br>- 异步：`ar_step > 0` | 异步模式（如 `ar_step=5`）允许更长时长生成。 |
| **滑动窗口生成机制** | 当 `num_frames > base_num_frames` 时自动启用 | 模型按滑窗方式逐段生成长视频。 |
| **重叠帧数 (overlap_history)** | 典型值：17 | 每个窗口与上一个窗口之间重叠帧数，用于平滑过渡。 |
| **噪声注入 (addnoise_condition)** | 推荐值：20 | 在接力生成时给前序帧注入少量噪声，防止长链误差积累。 |
| **长视频生成方式** | 滑动窗口 + Overlap 拼接 | 例如：第1段生成帧 [1..97]，第2段生成 [81..177]，重叠17帧后拼接。 |
| **总步数计算** | `steps_total = base_steps + (num_blocks - 1) * ar_step` | 异步模式下每个块的步数累加。 |
| **实现接口** | HuggingFace Diffusers Pipeline | 参数包括 `num_frames`, `base_num_frames`, `ar_step`, `overlap_history`, `addnoise_condition`, `fps` 等。 |

---

## ⚙️ 推理机制细节

### 🔹 滑动窗口 (Sliding Window)
- 每次生成固定数量帧（如 97 帧），后续窗口重叠一部分历史帧（如 17 帧）；
- 模型在生成新片段时使用前一段的末尾帧作为条件；
- 最终通过帧拼接（或过渡平滑）实现连续视频。

### 🔹 异步生成 (Asynchronous Generation)
- 通过设置 `ar_step > 0`，模型在时间上“分块”预测；
- 允许在保持显存占用可控的同时扩展生成时长；
- 例如：`causal_block_size=4`, `ar_step=5` 时，可同时处理多帧依赖。

### 🔹 条件噪声注入 (addnoise_condition)
- 在自回归生成的每个滑窗开头，对已知帧 latent 注入轻微噪声；
- 提高模型的鲁棒性，防止长链生成导致“冻结”或漂移。

---

## 🧩 对比 LTX-Video

| 对比项 | **LTX-Video** | **SkyReels-V2** |
|---------|---------------|-----------------|
| 训练片段时长 | 固定 8–10s | 可变，支持更长片段 |
| 帧率 | 24 FPS | 24 FPS |
| 架构核心 | Holistic Latent Diffusion + 高压缩 Video-VAE | Diffusion Forcing + 滑窗自回归 |
| 生成方式 | 固定时长生成（5–10s） | 滑窗拼接，支持任意长 |
| 长视频策略 | 分段 + 首帧条件 | 滑动窗口 + overlap_history |
| 强化学习 | 无 | 有（motion reward + DPO） |
| 目标场景 | 短视频高画质 | 长视频连贯叙事 |

---

## 📊 参数示例（以 540P 模型为例）

```python
# SkyReels-V2 推理参数示例（Diffusers）
pipe = SkyReelsV2Pipeline.from_pretrained("SkyworkAI/SkyReels-V2-540p")
video = pipe(
    prompt="A cinematic view of a spaceship flying through a nebula",
    num_frames=250,            # 总帧数
    base_num_frames=97,        # 单滑窗帧数
    overlap_history=17,        # 重叠帧
    addnoise_condition=20,     # 条件噪声注入
    ar_step=5,                 # 异步步长
    causal_block_size=4,       # 时序块大小
    fps=24,                    # 帧率
    height=544, width=960,
    steps=30                   # 每滑窗去噪步数
)
```



# 🌀 Stable Video Infinity 训练与推理输入输出设定总结

---

## 🧠 总体概述

Stable Video Infinity (SVI) 采用 **误差回收微调 (Error-Recycling Fine-Tuning, ERFT)** 策略，
在训练阶段显式注入历史误差、让模型学会自我修复；
推理阶段则基于“**跨片段条件 (Cross-Clip Conditioning)**”连续生成无限长度视频，
并支持 **交互式 prompt 切换 (Interactive Prompt Switching)**。

训练以 **5s–8s clip** 为核心单位，通过 **LoRA 微调** 在短视频上模拟长时误差分布；
推理时可连续拼接 clip，无需额外模块即可生成分钟级乃至无限长视频。

---

## 🧩 训练阶段（Training）

### 🎞 视频片段长度
- 每个训练样本为短视频 clip；
- clip 时长：**约 5–8 秒**；
- 平均帧数：**≈81 帧（16 FPS）**；
- 与 Diffusion Transformer 的时间压缩率（×8）匹配，可在有限 token 下保持时序连续。

### ⏱ 帧率 (FPS)
- **16 FPS**；
- 对应 5s → 80 帧；
- 经过 Video-VAE 编码后得到时间压缩 latent（每 8 帧 → 1 token block）。

### 🧮 Video-VAE / Token 压缩结构
- Video-VAE 压缩率：**32×32×8**（空间×空间×时间）；
- 因此：
  - 每 8 帧 → 1 latent token；
  - 每个 clip 对应约 **10 latent 时间步**；
  - Latent 分辨率为 **480×832**；
- 通道数 128，总压缩比约 **1:192**。

---

### ⚙️ 数据构造与误差注入
SVI 在训练时引入三类 clip：
1. **Consistent clips**：单一 prompt 的稳定场景；
2. **Creative clips**：多 prompt 连续场景（自动生成 prompt stream）；
3. **Conditional clips**：带语音 / 骨架等模态条件。

每个 clip 被随机注入三种误差：
\[
\tilde{X}_{vid} = X_{vid} + I_{vid}E_{vid},\quad
\tilde{X}_{noi} = X_{noi} + I_{noi}E_{noi},\quad
\tilde{X}_{img} = X_{img} + I_{img}E_{img}
\]
- 注入概率 \( I_* = 0.5 \)；
- 误差来自上一轮的 **replay memory**；
- 模型因此学会在含误差输入下预测修复方向。

---

### 🧩 模型结构与训练配置

| 参数 | 数值 | 描述 |
|-------|-------|------|
| Base model | Wan / CogVideoX DiT | 基础视频扩散架构 |
| LoRA rank | 128 | 低秩适配维度 |
| LoRA α | 128 | 缩放因子 |
| Learning rate | 2e-5 | Adam 优化器 |
| Batch size | 32–64 | clip 级训练 |
| Max epochs | 10 | 轻量微调 |
| Distributed | DeepSpeed Stage 2 | 多 GPU 训练 |
| Gradient checkpoint | ✅ | 节省显存 |
| Gradient clipping | 1.0 | 稳定训练 |

---

### 🎯 训练目标函数
模型优化“误差回收速度”：
\[
L_{SVI} = \mathbb{E}\|\,u(\tilde{X}_t, \tilde{X}_{img}, C, t; \theta) - V_t^{rcy}\,\|^2
\]
其中：
- \( \tilde{X}_t = t\tilde{X}_{vid} + (1-t)\tilde{X}_{noi} \)
- \( V_t^{rcy} = X_{vid} - \tilde{X}_{noi} \)
- 方向始终指向干净 latent；
- 仅 LoRA 层参与训练（无须全模型微调）。

---

## 🚀 推理阶段（Inference）

### 🧾 默认配置
- **分辨率**：480×832  
- **帧率**：16 FPS  
- **clip 时长**：约 5 秒（≈81 帧）  
- **采样步数**：50 步 ODE 采样  
- **推理耗时**：<2s / clip（A100 单卡）

---

### ♾️ 无限长度生成（Infinite-Length Generation）
SVI 在推理时无需额外模块即可连续生成：
\[
X_{clip}^{(i+1)} \leftarrow f(X_{clip}^{(i)}[-K:], prompt_{i+1})
\]
- 上一 clip 的最后 K 帧（通常 K=5）作为下一段条件；
- prompt 可连续变化；
- 误差不会累积导致漂移；
- 支持数百秒甚至无限生成。

---

### 🔁 交互式 Prompt-Switch / Streaming Generation

#### 🧱 实现逻辑
1. **Prompt Stream Buffer**
   - 管理一系列时间连续的 prompt；
   - 每个 prompt 控制一个 5s clip；
   - 支持实时更新。

2. **Cross-Clip Conditioning**
   - 使用上一段 clip 的最后几帧 latent 作为条件；
   - 保持角色 / 物体一致性；
   - 场景平滑过渡。

3. **Anchor Frame Caching**
   - 保留 anchor latent 以维持长时身份一致；
   - 对突变 prompt 可启用轻噪声重采样。

#### 🧩 示例伪代码
```python
init_image = load_image("dog_sleeping.png")
clip = generate_video(init_image, prompt="dog wakes up")

for next_prompt in ["dog runs", "dog jumps", "dog barks"]:
    last_frames = clip[-5:]  # 条件帧
    clip_next = generate_video(last_frames, prompt=next_prompt)
    clip = concatenate(clip, clip_next)
