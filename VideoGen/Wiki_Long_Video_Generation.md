# 🌊 Long Video Generation 五大主线机制

---

## 1️⃣ Diffusion Forcing

### 🧠 基本原理
传统 Diffusion Video Model 在所有帧上使用**相同噪声水平**（同步扩散），即每一帧被同等程度加噪 → 同步去噪。  
**Diffusion Forcing** 则为 **每个 token / 帧 分配独立噪声强度**，使得同一时刻不同帧处于不同扩散阶段。  

- 噪声=0 → 完全可见（unmasked）  
- 噪声较高 → 部分遮蔽（partial mask）  
- 模型学习利用“干净帧”去引导“被噪声掩盖帧”的恢复  

因此：
> **Diffusion Forcing = 可变噪声掩蔽 + 条件去噪训练**  
模型学会在任意噪声组合下恢复部分帧 → 在推理时即可连续地扩展视频、逐步展开未来帧。

核心目标：让模型**学会条件依赖**与**分布对齐**，避免全局同步扩散的“全帧重置”限制。

---

### 🧩 训练方式
- **数据处理**  
  - 视频切为 8～16 秒 clip（通常 64～128 帧 @8FPS）。  
  - 对每帧独立采样噪声比例 `t_i ~ U(0,1)`。  
  - 形成带不同噪声强度的“混合片段”，输入模型。  

- **损失函数**  
  \[
  L = \mathbb{E}_t[||\epsilon_\theta(x_t, t) - \epsilon||^2]
  \]
  每帧拥有独立噪声 `t_i`，模型学习条件重建。  

- **输入/输出**  
  - 输入：不同噪声水平的视频片段  
  - 输出：预测完整 clean 帧或噪声残差  

---

### ⚙️ 推理方式
- 使用“**时间步滚动 + 局部噪声调度**”：  
  1. 部分帧设低噪（条件帧）；  
  2. 新帧采高噪去噪；  
  3. 滑动更新窗口继续前进。  
- 模型支持任意噪声组合 → 可流式生成任意长度视频。

---

## 2️⃣ Self-Forcing

### 🧠 基本原理
传统自回归模型在训练时依赖真实帧（teacher forcing），  
推理时却必须使用自身生成帧 → 产生 **train-test mismatch**。

**Self-Forcing**：在训练中部分使用自己生成的帧作为输入，  
逼迫模型在“自生成轨迹”上学习稳定生成。

---

### 🧩 训练方式
- 每个 clip（8s，64帧）部分步骤由模型 roll-out；  
- 其他步骤仍用 ground-truth；  
- Loss = denoising loss + rollout loss + 一致性正则。  

输入：真实/生成混合帧 + prompt  
输出：下一帧预测。  

---

### ⚙️ 推理方式
- 起始输入为 prompt + 初始帧；  
- 模型逐帧自回归生成；
- 可结合 KV-Cache 流式输出。  

优点：消除暴露偏差、生成更稳。  

---

## 3️⃣ Error-Recycling Fine-Tuning

### 🧠 基本原理
传统长视频生成面临 **误差累积 (error accumulation)** 与 **分布偏移 (train-test gap)**。  
SVI 提出 **Error-Recycling Fine-Tuning (ERFT)**：  
> 在训练中主动注入模型生成误差，让模型学习“识别与修复自身错误”，实现稳定的无限时长生成。

核心机制：  
- 模型通过闭环误差回收（Error Recycling Loop）实现自稳；
- 训练阶段注入历史误差；
- 推理阶段自动纠错、连续生成；
- 可交互式切换 prompt（Interactive Generation）。

---

### 🧩 训练方式
- **训练单元**：5–8s 短视频 clip（≈81帧 @16FPS）  
- **输入结构**：  
  \[
  \tilde{X}_{vid} = X_{vid} + I_{vid}E_{vid},\quad 
  \tilde{X}_{noi} = X_{noi} + I_{noi}E_{noi},\quad 
  \tilde{X}_{img} = X_{img} + I_{img}E_{img}
  \]
  随机注入误差 (prob=0.5)，模拟真实推理漂移；
- **误差来源**：Replay Memory（记录过往推理误差）；  
- **优化目标**：预测“误差回收速度”  
  \[
  L_{SVI} = \|u(\tilde{X}_t, \tilde{X}_{img}, C, t; \theta) - V_t^{rcy}\|^2
  \]
- **调优方式**：LoRA 微调（rank=128），10 epoch 收敛。  

---

### ⚙️ 推理方式
- **默认配置**：480×832, 16FPS, 每 clip ≈5s；  
- **无限长度生成**：  
  \[
  X_{clip}^{(i+1)} \leftarrow f(X_{clip}^{(i)}[-K:], prompt_{i+1})
  \]
  上一段末帧作为条件，prompt 可连续切换；  
- **交互式生成 (Interactive Prompt Switching)**：  
  通过 Prompt Stream Buffer 实现动态剧情；  
- 无需额外模块，模型可自我修正漂移，稳定生成数百秒级视频。

---

### 🧩 输入/输出结构
| 阶段 | 时长 | 输入 | 输出 | 特性 |
|------|------|------|------|------|
| 训练 | 5–8s | 含误差 latent + 条件 | 修复速度向量 | Error Recycling 训练 |
| 推理 | clip 级（≈5s） | 上段末帧 + prompt | 连续视频 | 无限时长生成 |
| 交互式 | 动态 prompt 流 | 上段末帧 + 动态 caption | 动态剧情视频 | 实时控制 |

---

### ⚙️ 关键特性
- ✅ 模型学会纠正自身生成误差（error-aware）  
- ✅ 无需 streaming module，即可无限时长生成  
- ✅ 可交互 prompt，支持剧情过渡  
- ✅ 长视频一致性维持优于 Causal / Streaming 系方法  

---

## 4️⃣ Causal Autoregressive Diffusion (带 KV-Cache)

### 🧠 基本原理
将视频 Transformer 的注意力结构改为**因果单向注意力**，仅关注过去帧。  
模型逐块（block）生成未来帧，并使用 **KV 缓存复用历史上下文**。

---

### 🧩 训练方式
- 视频划分为若干 block（如每16帧一块）；  
- 块内局部双向，块间强制因果依赖；  
- Teacher（双向） → Student（单向）蒸馏；  
- 同时使用 **分布匹配蒸馏 (DMD)**。  

输入：加噪视频块 + caption  
输出：预测噪声或clean帧  

---

### ⚙️ 推理方式
1. 初始化前若干帧；  
2. 每次生成一个 block；  
3. 历史帧 KV 缓存复用；  
4. 块级自回归扩展。  

优点：推理线性复杂度，可流式扩展。  
缺点：长期漂移仍存在。

---

## 5️⃣ Streaming Diffusion (StreamDiT)

### 🧠 基本原理
通过「滑动缓冲区（Moving Buffer）」机制仅关注最近 N 帧，实现低延迟流式生成。  
每一步预测下一帧并更新缓冲区。

---

### 🧩 训练方式
- 视频切成连续滑动窗口（如16帧）；  
- 每窗口采样噪声预测未来帧；  
- 损失由三部分构成：  
  - Flow Matching Loss  
  - Denoising Loss  
  - Buffer Boundary Consistency  

输入：历史 N 帧  
输出：未来 1～k 帧（常为1帧）

---

### ⚙️ 推理方式
- 维持缓冲区并逐帧滑动；
- 每步生成新帧并追加；
- 低延迟、可实时播放；
- 长期一致性略受限。

---



# 🧭 总结对照表

| 方法 | 核心思想 | 训练要点 | 推理方式 | 典型代表 |
|------|-----------|-----------|-----------|-----------|
| **Diffusion Forcing** | 每帧独立噪声，引导局部恢复 | per-frame 噪声采样 + 去噪监督 | 局部低噪滚动生成 | Diffusion Forcing (Wan / SkyReels) |
| **Self-Forcing** | 混合真实与自生成历史 | rollout 强迫监督 | 自回归连续生成 | Self-Forcing, Self-Forcing++ |
| **Error-Recycling Fine-Tuning** | 闭环误差回收，自修正生成 | Error Injection + Replay Memory + LoRA | Cross-Clip Conditioning + Interactive Generation | Stable Video Infinity |
| **Causal Autoregressive** | 单向注意力 + KV 缓存 | 块级因果注意力 + 蒸馏 / DMD | 块级自回归 + 缓存复用 | CausVid, Ca²-VDM, LongLive |
| **Streaming Diffusion** | 滑窗缓冲区实时预测 | Flow Matching + Buffer Consistency | 滑动窗口预测 | StreamDiT |


# 🧭 更合理的分类法：四个正交维度

---

## A. 架构（Architecture / Attention Topology）

- **Causal Autoregressive（因果单向）**  
  只关注过去帧；通过 **KV-Cache** 进行块级或帧级自回归生成。  
  代表作：CausVid、Ca²-VDM  
- **Bidirectional（双向全局/局部）**  
  全片或局部窗口的双向注意力（多数 DiT 基座如 Wan）。  
- **Streaming Transformer（流式/滑窗）**  
  模型结构原生支持移动缓冲区、动态序列输入；用于实时生成。  
  代表作：StreamDiT  

---

## B. 训练机制（Training Mechanism / Curriculum）

- **Diffusion Forcing（逐帧异噪）**  
  每帧分配独立噪声强度，学习“部分帧可见 → 引导被掩帧”；  
  支持任意噪声组合下的局部去噪与流式生成。  

- **Self-Forcing（自强迫）**  
  训练时混合真实帧与模型生成帧（rollout），缓解 train-test mismatch。  

- **Error-Recycling Fine-Tuning（ERFT / Stable Video Infinity）**  
  显式注入历史误差并建立 replay memory，让模型学会“自纠错”与误差回收。  

- （可组合项）  
  - 蒸馏 / 分布匹配蒸馏 (DMD)  
  - Flow Matching  
  - Curriculum schedule  

---

## C. 推理/运行策略（Inference Regime）

- **Streaming / Sliding-Window**  
  保持固定缓冲区，逐帧滚动更新；低延迟流式输出。  
- **Autoregressive Blocks**  
  块级自回归推理（block-by-block），复用历史 KV-Cache。  
- **Cross-Clip Conditioning**  
  利用上一段末帧或 anchor latent 作为下一段条件，实现连续生成。  
- **Prompt Stream Buffer**  
  动态管理 prompt 序列，实现交互式剧情生成。  

---

## D. 长期一致性与记忆（Long-Horizon Consistency & Memory）

- **Frame/Feature Anchoring**：关键帧特征缓存；维持角色与背景一致性。  
- **Identity Caching**：保存人物身份 embedding。  
- **Replay Memory**：记录生成误差或特征偏移，用于训练或推理阶段修复。  
- **局部-全局混合窗口**：局部流式 + 全局稀疏注意力平衡短期平滑与长期结构。  

---

# 📊 机制对照表

| 机制 | 类型（训练/架构/推理） | 控制对象 | 何时使用 | 优势 | 潜在代价 | 代表 |
|------|-------------------------|-----------|-----------|-----------|-----------|-----------|
| **Diffusion Forcing** | 训练机制 | per-frame 噪声分配 | 需学习局部可见→缺失引导 | 支持局部更新、流式扩展 | 噪声调度复杂 | Wan / SkyReels |
| **Self-Forcing** | 训练机制 | 历史分布（真实+生成） | 降低 exposure bias | 长时更稳 | rollout 成本较高 | Self-Forcing / ++ |
| **Error-Recycling (ERFT)** | 训练机制 | 生成误差分布 | 长视频稳定 / 交互生成 | 自纠错能力强，长期稳定 | 需误差缓存 | Stable Video Infinity |
| **Causal Autoregressive + KV** | 架构/推理 | 注意力拓扑 / 缓存 | 块级自回归 / 低延迟生成 | 线性复杂度，流式扩展 | 长期漂移 | CausVid / Ca²-VDM |
| **Streaming Diffusion** | 架构/推理 | 滑窗缓冲区 | 实时/在线生成 | 低延迟、显存低 | 窗口外一致性弱 | StreamDiT |

> 💡 这些机制**可叠加组合**：  
> 例如 “Causal+KV 架构 + Self-Forcing 训练 + Cross-Clip 推理”，  
> 或 “双向 DiT + Diffusion Forcing 训练 + ERFT 微调 + Prompt-Switch 推理”。  

---
