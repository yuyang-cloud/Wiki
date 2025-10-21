# 🧩 DMD / Self-Forcing 训练流程总览

> **核心思想**  
> DMD（Distribution Matching Distillation）是一种基于 **样本分布对齐** 的蒸馏方法。  
> 它不直接让学生模型拟合教师模型的输出值，而是让学生生成的样本在样本空间上**沿着教师给出的分布梯度方向更新**，  
> 从而实现分布级别的匹配。  
>
> 训练包含三个核心模块：
> - `real_score`（teacher，冻结）  
> - `fake_score`（student critic，可训练）  
> - `generator`（student 生成器，可训练）  
>
> 三者形成一个分布匹配闭环：
> ```
>          ┌─────────────────────────────┐
>          │         real_score          │
>          │       (teacher, frozen)     │
>          └──────────────┬──────────────┘
>                         │
>                         │ 提供真实分布梯度 s_real
>                         │
>          ┌──────────────▼──────────────┐
>          │        fake_score           │
>          │     (critic, student)       │
>          │    学习预测 generator 的分布   │
>          └──────────────┬──────────────┘
>                         │
>                         │ 计算分布差 grad = s_fake - s_real
>                         │
>          ┌──────────────▼──────────────┐
>          │         generator           │
>          │     (student generator)     │
>          │  生成样本 x，并沿 -grad 更新   │
>          └──────────────┬──────────────┘
>                         │
>                         │ x_target = x - grad
>                         │ 使用 MSE(x, x_target) 训练生成器
>                         │
>          └────── teacher 方向反馈回路 ──────┘
> ```
> - **critic 阶段**：训练 fake_score，使其输出的分布更贴近 generator 的真实 inference 分布；  
> - **generator 阶段**：根据 real_score 与 fake_score 的分布差，计算 grad，将生成样本 \(x\) 沿 \(-grad\) 移动得到 \(x_{target}\)，并用 MSE(x, x_target) 进行更新。


---

## 一、总体训练框架 (`train()`)

### 🔁 外层训练循环

```python
while step < max_iters:
    TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)

    if TRAIN_GENERATOR:
        generator_loss.backward()
        generator_optimizer.step()
    else:
        critic_loss.backward()
        critic_optimizer.step()

    step += 1
```

- 每隔 `dfake_gen_update_ratio` 步更新一次 generator；  
  其余时间训练 critic（fake_score）。
- 形成「**critic 校准方向 → generator 沿方向更新**」的稳定蒸馏闭环。
- 使用 `generator_ema` 维护生成器的滑动平均版本用于验证或推理。

---

## 二、Self-Forcing 轨迹生成 (`_run_generator`)

`_run_generator()` 负责在训练阶段**模拟真实推理过程**：
- 从噪声出发生成视频或图像 latent；
- 使用 `SelfForcingTrainingPipeline.inference_with_trajectory()`：
  - 以 block 为单位；
  - 通过 KV cache 滚动；
  - 在指定 timestep 停止；
- 输出：
  - `pred_image_or_video`：生成的 latent；
  - `gradient_mask`：屏蔽首帧梯度；
  - `denoised_timestep_from/to`：轨迹的时间步范围。

---

## 三、`generator_loss` — 分布匹配蒸馏（DMD）

### 🎯 目标
让生成器输出的分布 \( p_\text{student}(x_0) \)  
逐步逼近 teacher 的分布 \( p_\text{teacher}(x_0) \)。

---

### 🚀 步骤

#### 1️⃣ 获取当前生成样本
```python
x = generator(noise, conditional_dict)
```
代表 student 当前生成的样本（latent）。

#### 2️⃣ 随机采样时间步并加噪
```python
x_t = scheduler.add_noise(x, ε, t)
```

#### 3️⃣ 计算 teacher 与 student 的 score
```python
s_real = real_score(x_t, t)  # teacher (frozen)
s_fake = fake_score(x_t, t)  # student critic
```

#### 4️⃣ 得到 KL 梯度方向
\[
grad = s_\text{fake} - s_\text{real}
\]
代表 student 分布相对 teacher 分布的偏移方向。

#### 5️⃣ 构造 teacher 引导下的理想样本
\[
x_\text{target} = x - grad
\]
表示把当前样本沿 teacher 提示的方向（即 -grad）移动一步，  
得到更接近 teacher 分布的样本。

#### 6️⃣ 用 MSE 约束生成器
```python
dmd_loss = 0.5 * F.mse_loss(x, x_target)
```
即：
\[
\mathcal{L}_\text{DMD} = \tfrac{1}{2}\|x - x_\text{target}\|^2
= \tfrac{1}{2}\|s_\text{fake} - s_\text{real}\|^2
\]
使生成器的输出 \(x\) 向 \(x_\text{target}\) 靠近，从而沿着 teacher 的 KL 方向下降。

---

### 🧭 直觉解释

| 变量 | 含义 |
|------|------|
| `grad = s_fake - s_real` | 表示 **student 分布相对于 teacher 分布的偏移方向** |
| `x_target = x - grad` | 表示 **teacher 引导下的理想样本**（沿 teacher 指示方向更新） |
| `MSE(x, x_target)` | 让 generator 的输出样本 \(x\) 沿 teacher 的 KL 梯度方向靠近 \(x_\text{target}\) |

> ✅ 总结一句话：  
> DMD 不是让 student 学会 teacher 的梯度，  
> 而是让 student 生成的样本沿 teacher 的梯度方向前进。

---

### 🧠 与 Score Matching 的区别

| 方法 | 优化目标 | 作用空间 | 缺点 |
|------|-----------|-----------|------|
| **Score Distillation** | \(\| s_\text{fake} - s_\text{real} \|^2\) | 梯度空间 | 只能逼近局部 score，不保证分布对齐 |
| **DMD** | \(\| x - (x - grad) \|^2\) | 样本空间 | 直接在分布层面最小化 KL |

DMD 的 loss 本质上是样本空间的 KL 梯度下降近似：
\[
\nabla_x D_{KL}(p_\text{student}\|p_\text{teacher}) \approx s_\text{fake} - s_\text{real}
\]
从而实现显式的 **Distribution Matching**。

---

## 四、`critic_loss` — 训练 fake_score（学生评分器）

### 🎯 目标
校准 fake_score，让它能在与生成器一致的时间步上预测正确的去噪目标（x₀ 或 noise）。

---

### 🚀 步骤

1️⃣ 用当前 generator 生成样本（不反传梯度）：
```python
with torch.no_grad():
    x = _run_generator(...)
```

2️⃣ 采样时间步 \(t_c\)，并加噪：
```python
x_t = scheduler.add_noise(x, ε, t_c)
```

3️⃣ 让 fake_score 预测：
```python
x_pred = fake_score(x_t, conditional_dict, t_c)
```

4️⃣ 根据类型计算去噪损失：
```python
if loss_type == "flow":
    loss = flow_loss(x_pred, ...)
else:
    noise_pred = scheduler.convert_x0_to_noise(x_pred, x_t, t_c)
    loss = 0.5 * MSE(noise_pred, ε)
```

> - 只更新 `fake_score` 的参数；
> - 不影响 generator；
> - 目标是让 fake_score 学会正确的去噪方向（校准其 score 场）。

---

### 📘 理解

- **critic_loss** 校准 student critic 的 score 场；
- **generator_loss** 使用校准后的 critic 给出 KL 方向；
- 两者交替更新，使系统稳定收敛。

---

## 五、训练闭环总结

| 模块 | 是否更新 | 损失 | 主要公式 | 作用 |
|------|-----------|------|-----------|------|
| **generator** | ✅ | \( \mathcal{L}_\text{DMD} = \tfrac{1}{2}\|x - (x - grad)\|^2 \) | \( grad = s_\text{fake} - s_\text{real} \) | 沿 teacher 的 KL 方向更新生成分布 |
| **fake_score (critic)** | ✅ | \( \mathcal{L}_\text{critic} = 0.5\|\epsilon_\text{pred} - \epsilon\|^2 \) | Denoising Loss | 校准学生的 score 场 |
| **real_score** | ❌ | - | - | 冻结的 teacher，用于提供方向参考 |

---

### 🔄 训练交替伪代码

```python
for step in range(max_iters):
    TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)

    if TRAIN_GENERATOR:
        # Teacher–Student distribution matching
        generator_loss.backward()
        generator_optimizer.step()
    else:
        # Calibrate student critic
        critic_loss.backward()
        critic_optimizer.step()

    step += 1
```

---

## 六、关键思想与优点

1. **Self-Forcing Backward Simulation**  
   - 模拟真实推理轨迹（KV cache 滚动、local attention 约束）；  
   - 解决 train/inference mismatch。

2. **Two-Stage Distillation**  
   - critic 先校准方向；  
   - generator 再沿方向前进。

3. **Distribution-Level Alignment**  
   - 不对齐数值输出；  
   - 对齐样本分布，显式最小化 KL。

4. **数值稳定性与可扩展性**  
   - block-wise rollout 控制显存；  
   - gradient_mask 与 timestep_schedule 保持稳定。

---

## 七、核心公式汇总

\[
\begin{aligned}
grad &= s_\text{fake}(x_t) - s_\text{real}(x_t) \\
x_\text{target} &= x - grad \\
\mathcal{L}_\text{DMD} &= \tfrac{1}{2}\|x - x_\text{target}\|^2 \\
\mathcal{L}_\text{critic} &= \tfrac{1}{2}\|\epsilon_\text{pred} - \epsilon\|^2 \\
\nabla_x D_{KL}(p_\text{student}\|p_\text{teacher}) &\approx s_\text{fake} - s_\text{real}
\end{aligned}
\]

---

## 八、总结一句话

> DMD / Self-Forcing 的训练实质是：  
> - 先让 student critic（fake_score）学会预测正确的 score；  
> - 再让 generator 沿着 teacher 给出的**分布**方向（real_score - fake_score）更新；  
> - 从而在**样本空间**上显式执行分布级的 KL 下降。  
>
> **grad 表示分布偏离方向，x_target 表示更贴近 teacher 的样本，MSE(x, x_target) 让生成器沿 teacher 的 KL 梯度下降一步。**
