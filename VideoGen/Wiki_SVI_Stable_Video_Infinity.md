# 🌀 Stable Video Infinity (SVI)
### Infinite-Length Video Generation with Error Recycling

---

## 🌈 一、直观理解

### 🔍 核心问题：Train–Test Hypothesis Gap
- **训练时假设输入是干净的（error-free）**
- **推理时输入却是模型自己生成的、有误差的帧**
- → 导致误差在时间上累积、放大，视频逐渐崩坏（blur、drift、color shift）

### ⚙️ 解决思路：Error-Recycling Fine-Tuning (ERFT)
> “让模型在训练时暴露、感知并修复自己的错误。”

SVI通过在训练阶段**主动注入历史误差**，让模型学习如何**识别并修正自身生成的偏差**，最终实现：
- 无限长度（∞）视频生成；
- 无漂移（anti-drift）；
- 可交互剧情控制（interactive prompt switching）。

---

## 🔁 二、直观流程图（Closed-Loop Error Recycling）
      ┌──────────────────────────────────────────┐
      │       Clean video latent X_vid, X_img    │
      └──────────────────────────────────────────┘
                            │
                            ▼
            Inject recycled errors (E_vid, E_noi, E_img)
                            │
                            ▼
     Predict velocity: u(𝑋̃_t, 𝑋̃_img, C, t; θ)  ← Diffusion Transformer
                            │
                            ▼
       Compute bidirectional errors via 1-step integration
                            │
                            ▼
        Save errors → Replay Memory (B_vid, B_noi)
                            │
                            ▼
       Sample errors → Inject again (closed-loop recycling)
                            │
                            ▼
         Model learns to self-correct under noisy inputs


---

## 🧠 三、核心思想与方法结构

| 模块 | 作用 | 关键内容 |
|------|------|-----------|
| **Error-Recycling Fine-Tuning** | 模拟真实推理误差 | 注入历史误差打破“干净假设” |
| **Bidirectional Error Curation** | 高效计算误差 | 用单步双向积分近似ODE误差 |
| **Error Replay Memory** | 存储与重采样误差 | 时间步对齐采样，跨帧复用 |
| **Error-Recycled Objective** | 学习修复方向 | 预测指向干净latent的速度 |
| **Cross-Clip Conditioning** | 推理阶段连续生成 | 上一clip的末尾帧作为下一clip条件 |

---

## ⚗️ 四、具体方法详解

### 1️⃣ Error Injection（错误注入）
对视频latent、噪声和参考图像随机注入历史误差：
\[
\tilde{X}_{vid} = X_{vid} + I_{vid}E_{vid},\quad 
\tilde{X}_{noi} = X_{noi} + I_{noi}E_{noi},\quad 
\tilde{X}_{img} = X_{img} + I_{img}E_{img}
\]
其中 \(I_*\) 控制注入概率（约0.5），误差E来源于历史bank。

中间状态：
\[
\tilde{X}_t = t\tilde{X}_{vid} + (1-t)\tilde{X}_{noi}
\]
输入DiT网络预测速度：
\[
\hat{V}_t = u(\tilde{X}_t, \tilde{X}_{img}, C, t; \theta)
\]

---

### 2️⃣ Bidirectional Error Curation（双向误差整理）

通过单步积分近似计算误差：
\[
E_{vid} = \hat{X}_{vid} - X_{vid}^{rcy},\quad
E_{noi} = \hat{X}_{noi}^{img} - X_{noi}^{rcy},\quad
E_{img} = \text{Uniform}_T(E_{vid})
\]

三类误差：
| 类型 | 模拟阶段 | 作用 |
|------|-----------|------|
| **E_vid** | 中间漂移 | 校正预测偏差 |
| **E_noi** | 起点条件误差 | 修复历史积累 |
| **E_img** | 跨片段输入误差 | 防止场景崩坏（最关键） |

---

### 3️⃣ Error Replay Memory（误差重放记忆）

构建两类记忆库：
\[
B_{vid} = \{B_{vid,n}\}_{n=1}^{N_{test}}, \quad
B_{noi} = \{B_{noi,n}\}_{n=1}^{N_{test}}
\]

- 每个时间步保存误差样本（最多500条）
- 若bank满，则替换最相似误差（L2距离最小）
- 使用跨GPU聚合策略（federated-style gather）加速更新

训练时重采样误差：
\[
E_{vid} = \text{Unif}(B_{vid,n}),\;
E_{noi} = \text{Unif}(B_{noi,n}),\;
E_{img} = \text{Unif}_T(B_{vid})
\]

---

### 4️⃣ Optimization（优化目标）

误差回收目标函数：
\[
L_{SVI} = \mathbb{E}_{\tilde{X},C,t}
\|\,u(\tilde{X}_t, \tilde{X}_{img}, C, t; \theta) - V_t^{rcy}\,\|^2
\]
其中：
\[
V_t^{rcy} = X_{vid} - \tilde{X}_{noi}
\]
表示“指向干净latent的速度向量”。

---

## 🧬 五、训练设定（Training Setup）

| 参数 | 数值 | 描述 |
|-------|-------|------|
| LoRA rank | 128 | 低秩适配维度 |
| LoRA α | 128 | 缩放因子 |
| Learning rate | 2e-5 | Adam优化器 |
| Max epochs | 10 | 轻量微调即可收敛 |
| Training samples | 300–6000 | 短视频clip（约81帧） |
| Frame size | 480×832 | VAE latent分辨率 |
| Gradient checkpoint/offload | ✅ | 节省显存 |
| Distributed training | DeepSpeed Stage 2 | 多GPU支持 |

训练数据包括：
- **Consistent Track**：单prompt视频；
- **Creative Track**：自动生成prompt流；
- **Conditional Track**：语音、骨架等模态输入。

---

## 🎬 六、推理设定（Inference Setup）

### 🔹 无限长度生成
推理时，无需任何额外模块：
\[
X_{clip}^{(i+1)} \leftarrow f(X_{clip}^{(i)}[-K:], prompt_{i+1})
\]
- 上一clip最后K帧作为条件输入；
- prompt可自由切换；
- 模型自动保持时序一致性。

---

### 🔹 交互式 (Interactive) / Prompt-Switch 生成

支持实时prompt切换，实现连续剧情：

```python
init_image = load_image("dog_sleeping.png")
clip = generate_video(init_image, prompt="dog wakes up")

for next_prompt in ["dog runs", "dog jumps", "dog barks"]:
    last_frames = clip[-5:]  # 用作条件
    clip_next = generate_video(last_frames, prompt=next_prompt)
    clip = concatenate(clip, clip_next)
