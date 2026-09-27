# CosyVoice2 Token-to-Wave WavTTS Research Plan

## 1. 背景与目标

当前 WavTTS 的核心思路是直接在 raw waveform 空间建模，训练时使用 flow matching loss 和 multi-scale mel-spectrogram loss，推理时通过 ODE solver 从噪声生成目标语音。原始结构中的文本条件由 text encoder 提供。

本研究计划将 WavTTS 的输入条件从文本替换为 CosyVoice2 的离散 speech token，构建一个 token-to-wave 模块：

```text
prompt waveform + CosyVoice2 semantic tokens -> target waveform
```

更准确地说，目标不是简单地把 text encoder 换成 token encoder，而是探索一个语义 token 条件的 raw waveform decoder。它可以作为 CosyVoice2 token language model 后面的 waveform-native decoder，尝试减少对 mel、codec acoustic token、传统 vocoder 或额外 acoustic decoder 的依赖。

如果第一阶段 flow matching 版本有效，再进一步研究将 flow decoder 替换或蒸馏为 MeanFlow / few-step flow 版本，以降低 NFE 和推理延迟。

## 2. 推荐论文主线

建议主线定义为：

```text
Robust Semantic Token-to-Waveform Generation with Flow Matching
```

或者更具体：

```text
Prompt-Conditioned Semantic Token-to-Waveform Generation in Raw Waveform Space
```

核心研究问题：

1. CosyVoice2 的离散语义 token 是否足够驱动高质量 raw waveform 生成？
2. prompt waveform 是否可以稳定提供说话人、音色和风格信息？
3. token-to-wave flow decoder 是否比 text-conditioned WavTTS 或传统 token decoder 更鲁棒？
4. 在 token 条件已经提供较强内容先验时，flow / MeanFlow 是否可以用更少采样步数生成高质量语音？

## 3. 方法设计

### 3.1 基础结构

第一版建议保持 WavTTS 的 waveform patchification、DiT backbone、flow matching loss 和 multi-scale mel loss，仅替换条件侧：

```text
CosyVoice2 tokens
  -> token embedding
  -> token encoder
  -> rate adaptor / alignment module
  -> patch-level condition
  -> WavTTS DiT flow decoder
```

输入输出：

```text
input:
  prompt waveform
  prompt token sequence
  target token sequence

output:
  generated target waveform
```

训练时可以优先使用 ground-truth speech tokens，即从真实目标语音中提取 CosyVoice2 token。推理时再接入 CosyVoice2 language model 预测的 token，用来测试真实级联系统的效果。

### 3.2 不建议简单替换 text encoder

不要只把 token 当作普通文本 token 送进原 text encoder。原因是：

1. text token 与 speech token 的时间结构不同；
2. speech token 通常有固定帧率或更强的局部连续性；
3. waveform patch 序列长度远大于 token 序列长度；
4. token 到 waveform 的对齐方式会显著影响稳定性和音质。

更好的做法是显式加入 rate adaptor：

```text
semantic token rate -> waveform patch rate
```

可选方案：

1. Linear upsampling：简单、稳定，适合作为第一版 baseline。
2. Duration-aware upsampling：利用 token 时长或重复计数，适合做强版本。
3. Cross-attention：让 waveform patches attend 到 token sequence，表达能力强，但训练成本更高。
4. Hybrid：先粗 upsample，再通过 cross-attention 修正局部对齐。

第一阶段建议从 Linear upsampling 或 Cross-attention 里选一个最小可行版本，不要一开始做过复杂。

### 3.3 Prompt conditioning

prompt waveform 应该继续保留，因为 CosyVoice2 semantic tokens 主要承担内容信息，音色、录音环境和部分风格信息更适合从 prompt waveform 中提取。

可以设计两个条件分支：

```text
semantic token encoder -> content condition
prompt waveform encoder -> speaker/style condition
```

融合方式：

1. concat 后线性投影；
2. adaptive layer norm / modulation；
3. prompt waveform cross-attention；
4. speaker embedding + local prompt acoustic features。

为了论文表达清楚，可以把它写成 content-style disentanglement：

```text
semantic tokens provide linguistic/content constraints;
prompt waveform provides acoustic identity and style.
```

## 4. 关键创新点候选

### 4.1 Semantic Token-to-Wave Decoder

将 WavTTS 从 text-to-wave TTS 改造成 semantic-token-to-wave decoder。

潜在贡献：

1. 直接从语义 token 生成 raw waveform；
2. 减少 mel / acoustic latent / codec decoder 依赖；
3. 将 CosyVoice2 的 token language model 和 WavTTS 的 waveform-native generation 连接起来；
4. 为 token-based TTS 提供一个新的 waveform decoder 选择。

风险：

如果只是替换输入条件，创新性不足。必须通过对齐模块、鲁棒训练、推理速度或质量提升来支撑论文贡献。

### 4.2 Token Error Robustness

这是最推荐强化的创新点。

真实推理时 token 来自 language model，必然存在预测错误。decoder 如果对 token 错误过敏，会产生内容错读、重复、跳词、爆音或音质下降。因此可以在训练中模拟 token 噪声：

1. token dropout；
2. token replacement；
3. span masking；
4. token duplication；
5. token deletion；
6. ground-truth token 与 predicted token 混合训练。

实验可以比较：

```text
clean ground-truth token
predicted token
corrupted token at 5% / 10% / 20%
```

评价指标：

1. WER / CER；
2. speaker similarity；
3. UTMOS / DNSMOS / MOS；
4. failure rate；
5. corruption-performance curve。

这个方向的论文性比单纯替换模块更强，因为它解决了 token-based speech generation 的实际问题。

### 4.3 Few-Step / One-Step Token-to-Wave Generation

当 flow matching 版本有效后，可以进一步替换为 MeanFlow 或 few-step flow：

```text
semantic tokens + prompt waveform -> 1-4 NFE waveform generation
```

目标不是简单使用 MeanFlow，而是证明：

1. token condition 已经提供强内容先验；
2. waveform decoder 可以用更少 NFE 生成稳定语音；
3. 与原始 ODE solver 相比，RTF 明显下降；
4. 与传统 vocoder / codec decoder 相比，质量和速度有竞争力。

建议不要第一阶段直接上 MeanFlow。否则失败时难以判断问题来自 token condition、alignment、waveform modeling，还是 MeanFlow 训练本身。

### 4.4 Streaming Token-to-Wave

如果目标偏应用或系统方向，可以考虑流式版本：

```text
past prompt + current token chunk + limited future tokens -> waveform chunk
```

可研究内容：

1. causal token encoder；
2. chunk-wise waveform generation；
3. limited lookahead；
4. overlap-add；
5. 首包延迟和 RTF；
6. chunk boundary artifact。

这个方向与 CosyVoice2 的流式设计比较契合，但工程复杂度更高，建议作为后续版本。

## 5. 实验路线

### Stage 1: 最小可行版本

目标：证明 token-to-wave flow decoder 可以训起来。

改动：

1. 提取 CosyVoice2 ground-truth semantic tokens；
2. 新增 token dataset 字段；
3. 新增 token encoder；
4. 将 token condition 接入 WavTTS DiT；
5. 保持原始 flow matching loss 和 multi-scale mel loss。

实验：

1. 使用 ground-truth tokens 训练和推理；
2. 与原始 WavTTS text condition 对比；
3. 检查生成语音是否内容稳定、音色可控、无明显爆音。

### Stage 2: 对齐与条件融合

目标：找到稳定的 token-to-wave conditioning 方式。

比较：

1. token mean pooling；
2. linear upsampling；
3. cross-attention；
4. duration-aware upsampling；
5. prompt waveform condition ablation。

重点观察：

1. 内容准确率；
2. 语速和停顿；
3. 长句稳定性；
4. speaker similarity；
5. 训练收敛速度。

### Stage 3: 鲁棒训练

目标：让 decoder 对 predicted token 和 corrupted token 更稳。

训练策略：

1. token dropout；
2. token replacement；
3. span masking；
4. duplication / deletion；
5. ground-truth token 与 predicted token 混合。

实验：

1. ground-truth token inference；
2. predicted token inference；
3. controlled token corruption；
4. 不同噪声率下的 WER、MOS、speaker similarity 曲线。

### Stage 4: MeanFlow / Few-Step Decoder

目标：降低 NFE 和 RTF。

比较：

1. original flow, 16 / 32 / 64 NFE；
2. few-step flow, 2 / 4 / 8 NFE；
3. MeanFlow, 1 / 2 / 4 NFE；
4. CosyVoice2 原始 decoder / vocoder baseline。

指标：

1. RTF；
2. NFE；
3. WER / CER；
4. speaker similarity；
5. MOS / UTMOS；
6. 显存与吞吐。

## 6. Baseline 设计

至少需要以下 baseline：

1. WavTTS 原始 text-conditioned 模型；
2. CosyVoice2 原始系统；
3. CosyVoice2 semantic token + 常规 vocoder / decoder；
4. token-to-wave flow decoder；
5. token-to-wave flow decoder with robustness training；
6. token-to-wave MeanFlow / few-step decoder。

如果资源有限，优先做：

```text
WavTTS text
CosyVoice2 original
token-to-wave flow
token-to-wave flow + robustness training
```

## 7. 评价指标

内容准确性：

1. WER；
2. CER；
3. insertion / deletion / substitution breakdown。

音质和自然度：

1. MOS；
2. CMOS；
3. UTMOS；
4. DNSMOS；
5. multi-scale mel reconstruction loss，仅作辅助分析。

说话人保持：

1. speaker similarity；
2. speaker verification accept rate；
3. prompt-target embedding cosine similarity。

效率：

1. RTF；
2. NFE；
3. latency；
4. GPU memory；
5. throughput。

鲁棒性：

1. token corruption rate vs WER；
2. token corruption rate vs speaker similarity；
3. predicted token vs ground-truth token gap；
4. long-form generation failure rate。

## 8. 预期风险

### 8.1 创新性不足

风险：

```text
text encoder -> token encoder
```

可能被认为只是模块替换。

缓解：

1. 强调 semantic token-to-wave decoder；
2. 加入显式 token-to-wave alignment；
3. 做 token error robustness；
4. 加入 few-step / MeanFlow 推理加速；
5. 和强 baseline 做系统比较。

### 8.2 token 信息不足

如果 CosyVoice2 semantic token 主要编码内容，缺少细粒度韵律、音高或能量信息，生成语音可能过平或风格不稳定。

缓解：

1. 更强 prompt waveform encoder；
2. 加入 prosody token 或 acoustic side condition；
3. 使用 duration-aware alignment；
4. 使用 prompt cross-attention 保留局部声学细节。

### 8.3 predicted token 误差传播

ground-truth token 效果好，不代表真实推理有效。

缓解：

1. 必须测试 predicted token；
2. 使用 token noise augmentation；
3. 分析不同 token 错误类型对 waveform 的影响。

### 8.4 MeanFlow 变量过多

如果第一版直接引入 MeanFlow，失败原因难以定位。

缓解：

1. 先完成 flow matching token-to-wave；
2. 再做 MeanFlow；
3. 所有加速实验都与同样条件的 flow baseline 对比。

## 9. 推荐实现顺序

1. 跑通 CosyVoice2 token 提取脚本，保存到训练 metadata。
2. 新增 token dataset loading 逻辑。
3. 实现 token embedding + token encoder。
4. 实现最简单的 token-to-patch rate adaptor。
5. 替换或旁路原 text encoder condition。
6. 训练小规模 sanity check。
7. 与原 WavTTS text condition 做快速对比。
8. 加入 prompt waveform condition ablation。
9. 加入 token corruption training。
10. 大规模训练和完整指标评估。
11. 在稳定 flow 版本上尝试 MeanFlow / few-step 版本。

## 10. 阶段性结论标准

### 可以继续推进的信号

1. ground-truth token 条件下 WER 明显低于 text-conditioned WavTTS；
2. speaker similarity 不低于原始 WavTTS；
3. predicted token 下质量下降可控；
4. token corruption training 能显著改善错误传播；
5. few-step / MeanFlow 版本能显著降低 RTF。

### 应该调整方向的信号

1. ground-truth token 都无法稳定生成正确内容；
2. prompt waveform 无法控制说话人；
3. token-to-wave 比 CosyVoice2 原 decoder 质量明显差；
4. MeanFlow 质量损失过大，且无法通过训练稳定改善；
5. 主要提升只来自训练规模或数据，而不是方法设计。

## 11. 建议最终论文贡献表述

较弱表述：

```text
We replace the text condition in WavTTS with CosyVoice2 tokens.
```

不建议这样写，创新性不足。

较强表述：

```text
We propose a prompt-conditioned semantic token-to-waveform decoder that directly generates raw waveform from discrete speech tokens using flow matching.
```

更强表述：

```text
We study robust semantic token-to-waveform generation in raw waveform space, introduce token-to-patch alignment and token-noise robust training, and further explore few-step generation for efficient token-based TTS decoding.
```

## 12. 当前最推荐方案

短期最推荐做：

```text
CosyVoice2 ground-truth semantic token
  + prompt waveform
  + token-to-patch adaptor
  + WavTTS flow decoder
  + token corruption robust training
```

中期再做：

```text
predicted token inference
baseline comparison
MeanFlow / few-step acceleration
```

这样论文会从一个简单工程替换，变成一个围绕 semantic token-to-waveform decoding、鲁棒性和推理效率展开的完整研究问题。
