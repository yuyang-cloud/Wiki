# LongLive Streaming Train 调用链与数据流

核心思想
- 串流式生成：长视频被切成固定大小的训练块（chunk_size=21），块内再按 num_frame_per_block=3 逐步去噪。
- 缓存复用：跨块复用自注意力 KV cache 与 Cross-Attn cache，块末用 context_noise 回写，保持上下文连贯。
- 渐进增长：每次只新增 f_new∈[min_new_frame, chunk_size] 帧，剩余用上一块尾部重叠，保证每块恒定 21 帧。
- 梯度掩码：仅对“新增帧”回传梯度（gradient_mask），重叠帧不回传，稳定且省显存。
- Prompt 切换（可选）：DMDSwitch 在 switch_frame_index 切换到第二段 prompt，仅重建必要缓存，并用≤21帧 recache 平滑过渡。

参与模块
- Trainer：主循环与梯度累积，触发一次/多次前后向
- StreamingTrainingModel：管理序列状态、chunk 切分、重叠与梯度掩码
- Pipeline（无切换/切换版）：分块时序去噪、KV/CrossAttn 缓存读写、context 回写、可选切换与 recache
- Generator（student）：生成段落帧
- FakeScore（critic）：判别器路径的去噪/对齐损失
- Scheduler：timestep 的加噪/去噪调度器
- KV/CrossAttn Cache：跨块复用的注意力缓存

──────────────────────────────────────────────────────────────────────────────
流程图（一次训练步，含梯度累积）
```

            ┌────────────────────────────────────────────────────┐
            │                    Trainer.train                   │
            │  - 设定 TRAIN_GENERATOR                            │
            │  - zero_grad（按需）                               │
            └───────────────┬────────────────────────────────────┘
                            │   for accumulation_step in 1..G
                            ▼
           ┌────────────────────────────────────────────────────────┐
           │       fwdbwd_one_step_streaming(train_generator?)      │
           │  - 若无激活序列/序列结束 → start_new_sequence          │
           │  - 生成一个 chunk（生成器路径可要求梯度/判别器无梯度）  │
           └───────────────┬────────────────────────────────────────┘
                           │
     ┌─────────────────────┴───────────────────────┐
     │                                             │
     ▼                                             ▼
┌───────────────────────┐                   ┌────────────────────────┐
│   生成器子步（可跳过） │                   │    判别器子步（每步）   │
│ requires_grad=True     │                   │ requires_grad=False    │
├───────────────────────┤                   ├────────────────────────┤
│ StreamingTrainingModel │                   │ StreamingTrainingModel  │
│ .generate_next_chunk   │                   │ .generate_next_chunk    │
│  - 计算 f_new/f_overlap│                   │  - 同上，但无梯度       │
│  - 采样 noise_new      │                   │                         │
│  - _generate_chunk →   │                   │ chunk:[B,F,C,H,W]       │
│   Pipeline.generate_   │                   │ chunk_info（mask等）    │
│   chunk_with_cache     │                   │                         │
│  - 拼接 full_chunk=F   │                   │ compute_critic_loss:    │
│  - gradient_mask:新帧  │                   │  - 采样 timestep t      │
│ chunk:[B,F,C,H,W]      │                   │  - add_noise(chunk, t)  │
│ chunk_info（mask/ts）   │                   │  - fake_score 前向       │
│ compute_generator_loss │                   │  - denoising_loss       │
│ backward (accumulate)  │                   │ backward (accumulate)   │
└───────────────────────┘                   └────────────────────────┘

                            ▼
            ┌────────────────────────────────────────────────────┐
            │                 累积完成（一次训练步）              │
            │  - clip_grad(generator/fake_score)                 │
            │  - optimizer.step()（各自）                        │
            │  - generator EMA（可选）                           │
            │  - 日志/可视化/保存 checkpoint                     │
            └────────────────────────────────────────────────────┘
```
──────────────────────────────────────────────────────────────────────────────




目录
- 0. 术语与张量约定
- 1. 一次训练步的总览
- 2. Trainer.train 主循环（串流路径）
- 3. Trainer.fwdbwd_one_step_streaming（一次“生成器/判别器”子步）
- 4. Trainer.start_new_sequence（启动新序列）
- 5. StreamingTrainingModel 状态与 setup_sequence
- 6. StreamingTrainingModel.generate_next_chunk（产生训练块）
- 7. StreamingTrainingModel._generate_chunk 与管线
  - 7.1 无切换：StreamingTrainingPipeline.generate_chunk_with_cache
  - 7.2 中途切换：StreamingSwitchTrainingPipeline.generate_chunk_with_cache
  - 7.3 切换后的 Recache：_recache_after_switch
- 8. 生成器损失 compute_generator_loss
- 9. 判别器损失 compute_critic_loss
- 10. 累积、优化器 step 与 EMA
- 11. 可视化与 Checkpoint
- 12. 端到端时序与调用链汇总
- 13. 调试要点与常见坑

---

## 0. 术语与张量约定

- B：batch_size
- F：本次“训练块”的帧数（chunk_frames，固定等于 chunk_size，通常 21）
- C,H,W：VAE latent 通道数与空间分辨率（C 是 latent 通道，例如 16 或 4；H,W 为 latent 空间尺寸）
- T：文本 token 数（编码器输出长度）
- d：注意力通道/embedding 维度
- num_frame_per_block：块内再细分的“帧块”大小（生成网络一次处理的帧数，默认 3）
- frame_seq_length：每帧对应的 token 序列长度（Wan 模型为 1560）
- denoising_step_list：扩散时步序列（最后一步 0 在训练时通常已被剔除）
- exit_flag：选择在每个 block 的哪一个时步“退出”（作为该 block 的最终输出）
- gradient_mask：对 chunk 中哪些“新帧”回传梯度的掩码

典型张量形状：
- 噪声输入 noise：[B, F, C, H, W]
- 生成器输出（denoised_pred/x0）：[B, f_block, C, H, W]
- chunk 输出 output：[B, F, C, H, W]
- KV cache（每个 Transformer block）：
  - k/v（自注意力缓存）：[B, kv_len, d]，kv_len ≈ (local_attn_size + slice_last_frames) * frame_seq_length
- CrossAttn cache（每个 Transformer block）：
  - k/v（跨注意力缓存）：[B, T, d]，以及 is_init 标志
- timestep（训练时步张量）：[B, f_block] 或 [B* f_block] 视实现而定

---

## 1. 一次训练步的总览

- 单个训练步由多个“梯度累积子步”组成（gradient_accumulation_steps 次）。
- 每个子步会：
  - 生成器路径（按设定间隔可跳过）：产生一个 chunk，计算生成器 loss 并后向。
  - 判别器路径：产生一个 chunk（无梯度）并计算判别器 loss 后向。
- 累积完成后，对生成器/判别器分别 clip_grad_norm，并 optimizer.step()；可选做 EMA 更新。
- 日志、可视化与 checkpoint 保存按频率进行。

---

## 2. Trainer.train 主循环（串流路径）

入口：`trainer/distillation.py::Trainer.train`

关键步骤（streaming_training=True 时）：
1. 计算本步是否训练生成器：`TRAIN_GENERATOR = (step % dfake_gen_update_ratio == 0)`
2. 根据 TRAIN_GENERATOR 分别 zero_grad：
   - generator_optimizer.zero_grad(set_to_none=True)（有需要才清）
   - critic_optimizer.zero_grad(set_to_none=True)
3. 梯度累积循环（accumulation_step in [1..gradient_accumulation_steps]）：
   - 若训练生成器：`fwdbwd_one_step_streaming(train_generator=True)`
   - 训练判别器：`fwdbwd_one_step_streaming(train_generator=False)`
4. 累积结束：
   - 若训练生成器：
     - `grad_norm_g = model.generator.clip_grad_norm_(max_grad_norm_generator)`
     - `generator_optimizer.step()`
     - 若有 EMA：`generator_ema.update(model.generator)`
   - 判别器：
     - `grad_norm_d = model.fake_score.clip_grad_norm_(max_grad_norm_critic)`
     - `critic_optimizer.step()`
5. `self.step += 1`，记录日志/可视化/保存/GPU 内存输出等。

数据流（每个累积子步）：
- 输入：当前串流序列状态（见 §5）、必要时抽样的噪声 noise [B, f_new, C, H, W]
- 生成器路径输出：chunk [B, F, C, H, W]、chunk_info（含 gradient_mask、denoise 时步范围等）、loss_g
- 判别器路径输出：loss_d

---

## 3. Trainer.fwdbwd_one_step_streaming（一次“生成器/判别器”子步）

入口：`trainer/distillation.py::Trainer.fwdbwd_one_step_streaming(train_generator: bool)`

职责：
- 确保当前“串流序列”已就绪；若没有，调用 `start_new_sequence()`。
- 若当前序列不可继续（达到 temp_max_length 或剩余帧不足最小新帧数），重启新序列。
- 分支执行：
  - 生成器路径：
    - 通过 `StreamingTrainingModel.generate_next_chunk(requires_grad=True)` 产生 chunk 与信息；
    - `compute_generator_loss(chunk, chunk_info)` 得到 loss_g；
    - `loss_g.backward()`（按累积步数比例缩放）。
  - 判别器路径：
    - 通过 `StreamingTrainingModel.generate_next_chunk(requires_grad=False)` 产生 chunk（无梯度）；
    - `compute_critic_loss(chunk.detach(), chunk_info)` 得到 loss_d；
    - `loss_d.backward()`（累积）。

输出：
- 返回日志字典（loss 值、各类统计），供上层合并与日志。

张量形状：
- chunk：[B, F, C, H, W]（F 固定为 chunk_size）
- gradient_mask：[B, F]（True 表示本块中新生成的帧；False 表示重叠帧）

---

## 4. Trainer.start_new_sequence（启动新序列）

入口：`trainer/distillation.py::Trainer.start_new_sequence`

职责：
- 从 dataloader 取一个 batch（文本/图像等），编码得到：
  - conditional_dict（正向文本条件）
  - unconditional_dict（可用于 CFG 或无条件路径）
- DMDSwitch：若配置开启，准备第二段 `switch_conditional_dict` 与 `switch_frame_index`。
- 在多卡下同步“临时最大长度 temp_max_length”（从 possible_max_length 中随机挑选、broadcast 保持一致）。
- 调用 `StreamingTrainingModel.setup_sequence(conditional_dict, unconditional_dict, initial_latent=?, switch_conditional_dict=?, switch_frame_index=?, temp_max_length=?)` 完成底层序列与缓存初始化。
- 设置 `streaming_active = True`。

注意：
- i2v 模式可传入 initial_latent 作为首帧 latent。
- previous_frames 清空；KV 与 CrossAttn Cache 初始化并清零。

---

## 5. StreamingTrainingModel 状态与 setup_sequence

类：`model/streaming_training.py::StreamingTrainingModel`

关键配置：
- chunk_size（默认 21）
- max_length / possible_max_length / min_new_frame（用于控制序列增长与最小新增帧数，默认 18）
- num_frame_per_block（Wan 默认 3）
- frame_seq_length（Wan = 1560）
- inference_pipeline：可能是
  - 无切换：`pipeline/streaming_training.py::StreamingTrainingPipeline`
  - 切换：`pipeline/streaming_switch_training.py::StreamingSwitchTrainingPipeline`

状态（`self.state`）：
- current_length：当前序列长度（已生成帧数）
- conditional_info：包含 conditional_dict、unconditional_dict，以及可选的 switch_info（switch_conditional_dict、switch_frame_index）
- has_switched：是否已经进行过 prompt 切换
- previous_frames：[B, P, C, H, W]，上一块的尾部若干帧（用于和下一块拼接以固定 F）
- temp_max_length：本序列临时上限（各 rank 同步）

`setup_sequence(...)` 主要做：
- 初始化或清零 KV/CrossAttn 缓存（见管线的 `_initialize_kv_cache/_initialize_crossattn_cache/clear_kv_cache`）
- 写入 conditional_info；记录 temp_max_length
- 可选：用 initial_latent 做一次“上下文预热”（timestep=context_noise），仅更新缓存，不参与 loss
- 重置 `current_length=0`、`previous_frames=None`

---

## 6. StreamingTrainingModel.generate_next_chunk（产生训练块）

入口：`model/streaming_training.py::StreamingTrainingModel.generate_next_chunk(requires_grad: bool)`

职责：
- 根据 `current_length` 与 `min_new_frame`、`chunk_size` 决定：
  - 新增帧数 f_new ∈ [min_new_frame, chunk_size]（步长通常为 num_frame_per_block=3）
  - 重叠帧数 f_overlap = chunk_size - f_new（若 previous_frames 为空或长度不足，自动截断）
- 采样 noise_new：[B, f_new, C, H, W]
- 生成新增部分帧：
  - 调用 `_generate_chunk(noise_chunk=noise_new, chunk_start_frame=current_length, requires_grad=requires_grad)` 得到 out_new：[B, f_new, C, H, W]
- 和重叠部分拼接成固定大小 full_chunk：
  - full_chunk = concat(previous_frames_tail[f_overlap], out_new) → [B, F=chunk_size, C, H, W]
- “首帧重编码”（与真实推理对齐）：对 full_chunk 的第一帧按图像 latent 的路径重新编码一次（只在需要时处理），保持训练时“第一帧来自图像编码”的语义一致性
- 构造 gradient_mask：[B, F]，仅对 out_new 对应的位置为 True
- 递增 `current_length += f_new`，更新 `previous_frames = full_chunk[:, -21:, ...]`（最新尾部）

返回：
- chunk = full_chunk：[B, F, C, H, W]
- chunk_info：字典，包含
  - chunk_start_frame（整数）
  - new_frames_generated=f_new（整数）
  - gradient_mask：[B, F]
  - denoised_timestep_from/to（由管线返回）
  - 若为 Switch 模型，还可能包含 switch 时的记录

---

## 7. StreamingTrainingModel._generate_chunk 与管线

入口：`model/streaming_training.py::_generate_chunk(...)`

职责：
- 根据 chunk_start_frame 判断是否命中切换点，选择本块要使用的 conditional_dict
- 委托“串流生成管线”生成块（含多步时序、缓存读写、可选切换）：
  - 无切换：`StreamingTrainingPipeline.generate_chunk_with_cache(...)`
  - 切换：`StreamingSwitchTrainingPipeline.generate_chunk_with_cache(...)`

输出：
- generated_chunk：[B, f_new, C, H, W]
- denoised_timestep_from/to（可选）

### 7.1 无切换：StreamingTrainingPipeline.generate_chunk_with_cache

入口：`pipeline/streaming_training.py::StreamingTrainingPipeline.generate_chunk_with_cache(...)`

核心逻辑：
- 将 f_new 切分为多个 frame-block（大小 num_frame_per_block，默认 3）：
  - all_num_frames = [3, 3, ..., 3]，sum = f_new
- 生成并同步 exit_flags（每个 block 的“退出步”索引）：
  - rank0 采样，broadcast 保持一致
- 对每个 block 做多步时序去噪：
  - 循环 `current_timestep ∈ denoising_step_list`：
    - 非退出步：`torch.no_grad()` 下前向，得到 denoised_pred，并用 scheduler.add_noise 预测下一步的 noisy_input
    - 退出步：依据 `requires_grad` 决定是否开图，做一次前向得到最终 denoised_pred
  - 写入 output 的相应切片
  - 用 context_noise 再“回写”一次缓存（no_grad），保持 KV/CrossAttn 上下文为后续块服务
- 计算并返回 denoised_timestep_from/to（当 `same_step_across_blocks` 为 True 时根据 exit_flags[0] 推算；否则为 None）

关键张量：
- 输入 noisy_input：[B, f_block, C, H, W]
- 退出步 denoised_pred：[B, f_block, C, H, W]
- output 聚合：[B, f_new, C, H, W]
- context_noisy、context_timestep 用于缓存更新

### 7.2 中途切换：StreamingSwitchTrainingPipeline.generate_chunk_with_cache

入口：`pipeline/streaming_switch_training.py::StreamingSwitchTrainingPipeline.generate_chunk_with_cache(...)`

与无切换版的差异：
- 新增参数：
  - switch_frame_index：切换帧（相对本次 f_new 的局部坐标）
  - switch_conditional_dict：切换后的条件
  - switch_recache_frames：[B, ≤21, C, H, W]，外部提供的 recache 上下文（通常就是“上一块的最后 21 帧”）
- 逻辑：
  - 当 `local_start_frame >= switch_frame_index` 且尚未切换：
    - `_recache_after_switch(...)`：重置 KV 与 CrossAttn 缓存，并用最近 21 帧做一次“上下文重建”（仅更新缓存，不产生输出）
    - 之后 `cond_in_use = switch_conditional_dict`，`using_second=True`
  - 梯度开启位置从 `switch_frame_index` 开始（即切换前帧不回传梯度），对应变量 `start_gradient_frame_index = switch_frame_index`
  - 其余与无切换版一致（block 内时序循环、退出步选择、context 回写）

关键张量：
- noisy_input、denoised_pred、output 同无切换版
- 切换触发时，`_recache_after_switch` 里会基于 [B, ≤21, C, H, W] 的 recache 片段更新缓存

### 7.3 切换后的 Recache：_recache_after_switch

入口：`pipeline/streaming_switch_training.py::StreamingSwitchTrainingPipeline._recache_after_switch(...)`

职责：
- 重置缓存：
  - KV：对每个 block 的 k/v 张量 `.zero_()`
  - CrossAttn：对每个 block 的 k/v `.zero_()`，`is_init=False`
- 选择 recache 帧序列：
  - 若提供了 `switch_recache_frames`：拼接当前已输出段取最后 21 帧；
  - 否则：基于 `local_start_frame` 或 `current_start_frame` 从已输出中截取最后 ≤21 帧
- 构造 block-wise causal mask（local_attn_size=21），准备 `context_timestep`
- `torch.no_grad()` 下调一次生成器，仅用于更新 KV/CrossAttn 缓存（`current_start` 会回溯 `num_recache_frames * frame_seq_length`）
- 最后再次清空 CrossAttn cache 的 k/v 并 `is_init=False`（保证后续在新 prompt 上重建 cross-attn）

张量：
- frames_to_recache：[B, R, C, H, W]，R ≤ 21
- context_timestep：[B, R]（全为 context_noise）

---

## 8. 生成器损失 compute_generator_loss

入口：`model/streaming_training.py::StreamingTrainingModel.compute_generator_loss(chunk, chunk_info)`

职责：
- 取出当前应使用的 conditional_dict / unconditional_dict（切换后使用第二段）
- 从 chunk_info 读出：
  - gradient_mask：[B, F]，只对“新增帧”的位置为 True
  - denoised_timestep_from/to（若 same_step_across_blocks=True）
- 调用基础模型的 distribution matching 损失（如 DMD/SID）：
  - `base_model.compute_distribution_matching_loss(chunk, conditional_dict, unconditional_dict, gradient_mask, denoised_timestep_from/to, ...)`
  - 仅“新增帧”参与监督，降低不稳定性与显存
- 返回标量 loss_g 与日志字典

张量：
- chunk：[B, F, C, H, W]
- gradient_mask：[B, F] → 在损失里通常会展平/广播到 token 维度

---

## 9. 判别器损失 compute_critic_loss

入口：`model/streaming_training.py::StreamingTrainingModel.compute_critic_loss(chunk, chunk_info)`

职责：
- 防御性处理：
  - 若 `chunk.requires_grad`，先 `.detach()`，避免跨子步/跨块图持有
  - 清理缓存中可能带有的 grad（`_clear_cache_gradients`），并可适当 empty_cache
- 采样训练时步：
  - `t = base_model._get_timestep(...)` 并做 shift / clamp
- 加噪：`noisy_chunk = scheduler.add_noise(chunk, randn_like, t)` → [B, F, C, H, W]
- 前向 fake_score：
  - 得到 x0 或 flow 预测（与模型实现有关），换算为噪声/flow_pred
- 计算 denoising loss：
  - `base_model.denoising_loss_func(...)`，可将 gradient_mask 传入，仅对新增帧位置计算
- 返回标量 loss_d 与日志字典

---

## 10. 累积、优化器 step 与 EMA

- 每个累积子步只做 backward，不更新参数。
- 累积完成后：
  - 生成器：
    - `clip_grad_norm_(max_grad_norm_generator)`
    - `generator_optimizer.step()`
    - 若配置有 EMA：`generator_ema.update(model.generator)`
  - 判别器：
    - `clip_grad_norm_(max_grad_norm_critic)`
    - `critic_optimizer.step()`
- 递增 `self.step`，记录日志/可视化/保存

---

## 11. 可视化与 Checkpoint

- 可视化：定期用固定 prompts/长度生成视频；Switch 模型会在中点切换 prompt。
- 保存：
  - FSDP：使用 FULL_STATE_DICT 与 FULL_OPTIM_STATE_DICT 保存 generator/fake_score 与优化器状态，按 step 滚动清理旧 ckpt
  - LoRA：只保存 LoRA adapter 权重

---

## 12. 端到端时序与调用链汇总

一次训练步（含梯度累积）：
1) `Trainer.train`
   - 设定 TRAIN_GENERATOR、zero_grad
   - for accumulation_step in 1..G:
     - 若 TRAIN_GENERATOR：`fwdbwd_one_step_streaming(True)`
       - 若无序列或不能继续：`start_new_sequence` → `StreamingTrainingModel.setup_sequence`
       - `StreamingTrainingModel.generate_next_chunk(requires_grad=True)`
         - 计算 f_new/f_overlap，采样 `noise_new:[B, f_new, C,H,W]`
         - `_generate_chunk(...)` →
           - Switch 否：`StreamingTrainingPipeline.generate_chunk_with_cache`
           - Switch 是：`StreamingSwitchTrainingPipeline.generate_chunk_with_cache`
             - 触发切换则 `_recache_after_switch(...)`
         - 拼接成 `chunk:[B, F, C,H,W]`，构造 `gradient_mask:[B,F]`
       - `compute_generator_loss(chunk, chunk_info)` → backward
     - `fwdbwd_one_step_streaming(False)`（判别器）
       - （必要时）重启序列逻辑同上
       - `StreamingTrainingModel.generate_next_chunk(requires_grad=False)` → `chunk.detach()`
       - `compute_critic_loss(chunk, chunk_info)` → backward
   - 累积完成 → clip grad → 两个 optimizer.step() → EMA（生成器）
   - step++、日志/可视化/保存

---

## 13. 调试要点与常见坑

- 梯度范围：
  - 无切换：block 的最后一步才可能建图；`requires_grad=False` 时整块不建图
  - 切换：从 `switch_frame_index` 起才建图，切换前帧不回传梯度
- 重叠帧与掩码：
  - 每个训练块固定 F=chunk_size；通过重叠 P=F-f_new 和新增 f_new 拼合
  - 只对新增帧回传梯度：`gradient_mask:[B,F]`，防止重复计算/不稳定
- 缓存一致性：
  - 每个 block 末尾用 context_noise 回写一次缓存，保证下一个 block/下一块能在连贯上下文中生成
  - 切换时必须清 KV/CrossAttn，并用 ≤21 帧做 recache 平滑过渡
- 显存：
  - 判别器路径务必 `.detach()`，并清理缓存持有的梯度，避免跨子步/跨块图残留导致 OOM
- 多卡一致性：
  - exit_flags、f_new 的随机性需在 rank0 采样并 broadcast
  - temp_max_length 在序列启动时需广播一致
- 时步返回：
  - 仅当 `same_step_across_blocks=True` 时，管线返回 denoised_timestep_from/to；否则应按 None 处理或在损失内自行处理
