"""
WavTTS 核心模块: Conditional Flow Matching (CFM)
================================================
基于条件流匹配的原始波形生成模型，支持训练和推理。

参考论文:
  - Flow Matching for Generative Modeling (Lipman et al., 2023)
  - Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale (Le et al., 2023)

ein notation (张量维度命名约定):
  b  - batch          批次大小
  n  - sequence       序列长度 (waveform token 数量)
  nt - text sequence  文本序列长度 (拼音/字符 token 数量)
  nw - raw wave length 原始波形采样点数量 (= n × wav_frame_len)
  d  - dimension      特征维度

训练流程概览:
  1. 原始波形 inp=[B, N] 作为目标 x1
  2. 随机采样时间 t ~ logistic_normal
  3. 构造流: x0=高斯噪声, φ_t = (1-t)*x0 + t*x1  (线性插值)
  4. 构造 conditioning (self.cond_mode 二选一):
       - infill (默认, Voicebox): 随机 mask 70%-100% 区域, 未 mask 部分作为条件, loss 只算 mask 区
       - prefix (CosyVoice2):     50% 无 prompt, 否则取 [0,30%] 前缀当条件, loss 覆盖整段有效 wav
  5. DiT 预测 x_pred, 转为 velocity v_pred
  6. loss = MSE(v_pred, v_true) + 0.05 × 多分辨率 mel loss

推理流程概览:
  1. 纯噪声 y0 开始, Euler ODE 32步积分
  2. 每步 DiT 同时计算 cond + uncond, 用 CFG 组合
  3. 保留 condition 区域波形, 输出生成部分
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import random
import math
import inspect

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint  # 黑盒 ODE 求解器 (使用 Euler 方法)

from wavtts.model.modules import MelSpectrogramLoss
from wavtts.model.perceptual_loss import MultiResolutionSTFTLoss
from wavtts.model.repa import RepaAlignmentHead, repa_cosine_loss
from wavtts.model.utils import (
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM(nn.Module):
    """
    Conditional Flow Matching 模型.

    包装一个 Transformer (DiT/MMDiT) 骨干网络, 提供:
      - forward():  训练时的 flow matching loss 计算
      - sample():   推理时的 ODE 采样生成音频

    核心思想:
      学习一个从噪声到目标波形的直线概率路径,
      而不是扩散模型中弯曲的随机微分方程路径。
      这让采样可以用更少的步数 (32步即可) 达到高质量。
    """

    def __init__(
        self,
        transformer: nn.Module,
        # ---- 以下为 CFM 的各项配置参数 ----
        sigma=0.0,                       # 扩散噪声系数 (当前未使用, 保留兼容)
        odeint_kwargs: dict = dict(
            method="euler"               # ODE 求解方法: euler | midpoint | rk4
        ),
        audio_drop_prob=0.3,             # 训练时随机丢弃音频条件的概率 (独立丢弃模式)
        cond_drop_prob=0.2,              # 训练时同时丢弃音频+文本的概率 p_uncond (独立丢弃模式)
        joint_cond_drop_prob=0.0,        # 联合丢弃概率: >0 时切换为联合丢弃模式, 同时丢/不丢
        # -- 条件构造方式 --
        cond_mode: str = "infill",       # "infill": Voicebox 式随机段补全 (默认) | "prefix": CosyVoice2 式前缀 prompt
        prefix_drop_prob: float = 0.5,   # prefix 模式: 50% 概率无 prompt (纯无条件)
        prefix_frac: float = 0.3,        # prefix 模式: 否则随机取 [0, prefix_frac*len] 前缀当 prompt
        training_cfg_rate: float = 0.0,  # prefix 模式: >0 时每样本以该概率把 audio+content+spk 三条件一起清 (CFG)
        waveform_kwargs: dict = dict(),  # 波形参数 (wav_frame_len 等)
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),  # 随机 mask 的长度比例范围 (仅 infill 模式)
        vocab_char_map: dict[str:int] | None = None,  # 字符→ID 的映射表 (来自 vocab.txt)
        # -- 流匹配预测模式 --
        prediction: str = "flow",        # "flow": 预测 flow vector (x1-x0) | "x_pred": 直接预测 x1
        loss_space: str = "flow",        # "flow": flow空间算loss | "v": velocity空间 | "x": x1空间
        # -- 时间采样策略 --
        t_sampling: str = "uniform",     # "uniform": 均匀采样 | "logistic_normal": 逻辑正态采样
        P_mean: float = 0.0,             # logistic_normal 的均值 (当前配置: -0.8)
        P_std: float = 1.0,              # logistic_normal 的标准差 (当前配置: 0.8)
        time_shift: float = 1.0,         # 时间偏移系数, 1.0=不偏移 (推理时可用)
        t_eps: float = 1e-4,             # 时间最小裁剪值, 防止分母为0
        # -- 辅助 mel loss --
        use_aux_mel_loss: bool = False,  # 是否使用多分辨率 mel-spectrogram 辅助 loss
        aux_mel_loss_weight: float = 0.0,# mel loss 权重 (当前配置: 0.05)
        aux_mel_loss_masked: bool = True,# mel loss 是否只在被 mask 区域计算
        sample_rate: int = 16000,        # 目标采样率
        latents_scale: float = 1.0,      # 波形缩放系数 (当前配置: 9.0, 用于稳定训练)
        # -- training-only REPA branch --
        use_repa_loss: bool = False,
        repa_loss_weight: float = 0.0,
        # Python index 19 corresponds to the 20th DiT block.
        repa_layer: int = 19,
        repa_teacher_dim: int = 768,
        repa_alignment_hidden: int = 512,
        # -- training-only refined STFT/VAPA branch --
        use_stft_loss: bool = False,
        stft_loss_weight: float = 0.0,
        stft_gamma: float = 1.0,
        stft_eps: float = 0.01,
        stft_start_update: int = 0,
    ):
        super().__init__()

        # 随机 mask 的长度范围: 从每条音频中随机选取 70%-100% 的帧作为预测目标
        self.frac_lengths_mask = frac_lengths_mask

        # ========== 波形几何参数 ==========
        # wav_frame_len=160 意味着每 160 个采样点 (16kHz下=10ms) 作为一个 waveform token
        # 这是 WavTTS 的核心设计: 不转 mel-spectrogram, 直接在原始波形上建模
        waveform_kwargs = dict(waveform_kwargs)
        self.wav_frame_len = int(waveform_kwargs.pop("wav_frame_len", 160))
        self.num_channels = self.wav_frame_len  # 通道数 = 帧长, 用于 reshape 操作

        # ========== Classifier-Free Guidance (CFG) 的 dropout 配置 ==========
        # CFG 训练原理: 训练时随机丢弃条件, 推理时用 cond - uncond 的差值引导生成
        # 两种丢弃模式:
        #   模式1 (独立丢弃, joint_cond_drop_prob=0):
        #     30% 概率丢弃音频条件, 20% 概率全部丢弃 (音频+文本)
        #   模式2 (联合丢弃, joint_cond_drop_prob>0, 当前配置=0.1):
        #     10% 概率同时丢弃音频+文本, 90% 概率什么都不丢
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.joint_cond_drop_prob = joint_cond_drop_prob

        # prefix 模式 (CosyVoice2 式) 相关配置
        self.cond_mode = cond_mode
        self.prefix_drop_prob = prefix_drop_prob
        self.prefix_frac = prefix_frac
        self.training_cfg_rate = training_cfg_rate

        # ========== Transformer 骨干网络 ==========
        # 实际是 DiT 或 MMDiT, 负责接收 (noised波形, 条件波形, 文本) 并预测输出
        self.transformer = transformer
        self.dim = transformer.dim  # 特征维度 (WavTTS_Large: 1152)
        self.transformer_accepts_spk_embedding = (
            "spk_embedding" in inspect.signature(self.transformer.forward).parameters
        )
        self.transformer_accepts_hidden_state = (
            "return_hidden_state" in inspect.signature(self.transformer.forward).parameters
        )

        # 将 wav_frame_len 同步到 transformer (DiT 需要知道如何切分波形)
        if hasattr(self.transformer, "set_wav_frame_len"):
            self.transformer.set_wav_frame_len(self.wav_frame_len)

        # ========== 条件流相关 ==========
        self.sigma = sigma  # 噪声系数 (目前未使用, 保留兼容)

        # ========== 采样相关 ==========
        self.odeint_kwargs = odeint_kwargs  # ODE 求解器参数

        # ========== 文本 tokenization ==========
        # vocab_char_map: {"a": 0, "o": 1, "的拼音": 2, ...}
        # 来自预处理阶段生成的 vocab.txt
        self.vocab_char_map = vocab_char_map

        # ========== 增强流模型设置 ==========
        self.prediction = prediction          # 预测模式: DiT 直接输出 x_pred 还是 flow
        self.loss_space = loss_space          # loss 计算空间
        self.t_sampling = t_sampling          # 时间 t 的采样分布
        self.P_mean = P_mean                  # logistic_normal 均值
        self.P_std = P_std                    # logistic_normal 标准差
        self.time_shift = time_shift          # 时间偏移
        self.t_eps = t_eps                    # 数值稳定性的最小时间值
        self.latents_scale = latents_scale    # 波形缩放: x1_scaled = x1 × 9.0
        self.target_sample_rate = sample_rate # 采样率

        # REPA is enabled only when the trainer supplies frozen teacher
        # features.  The teacher itself is intentionally not registered here.
        self.use_repa_loss = bool(use_repa_loss)
        self.repa_loss_weight = float(repa_loss_weight)
        self.repa_layer = int(repa_layer)
        self.repa_alignment_head = (
            RepaAlignmentHead(self.dim, int(repa_teacher_dim), int(repa_alignment_hidden))
            if self.use_repa_loss and self.repa_loss_weight > 0
            else None
        )
        if self.repa_alignment_head is not None and not self.transformer_accepts_hidden_state:
            raise ValueError("REPA currently requires a DiT backbone with return_hidden_state support")

        self.use_stft_loss = bool(use_stft_loss)
        self.stft_loss_weight = float(stft_loss_weight)
        self.stft_gamma = float(stft_gamma)
        self.stft_eps = float(stft_eps)
        self.stft_start_update = int(stft_start_update)
        self.stft_loss = MultiResolutionSTFTLoss() if self.use_stft_loss and self.stft_loss_weight > 0 else None

        # ========== 辅助 Mel-Spectrogram Loss ==========
        # 为什么需要辅助 mel loss?
        #   纯 waveform 的 MSE loss 对所有频率一视同仁,
        #   但人耳对不同频率的敏感度不同 (低频更重要).
        #   多分辨率 mel loss 在感知相关的时频域做约束,
        #   能提升生成音频的听感质量, 权重仅 0.05, 不影响流匹配主训练.
        self.use_aux_mel_loss = use_aux_mel_loss
        self.aux_mel_loss_masked = aux_mel_loss_masked
        if self.use_aux_mel_loss:
            self.aux_mel_loss = MelSpectrogramLoss(
                sample_rate=sample_rate,
                # 7 个不同分辨率的 mel 频带, 从极粗 (5 mel bins) 到极细 (320 mel bins)
                n_mels=[5, 10, 20, 40, 80, 160, 320],
                # 对应的窗口长度, 覆盖从 2ms 到 128ms
                window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
                mel_fmin=[0, 0, 0, 0, 0, 0, 0],      # 所有频带从 0Hz 开始
                mel_fmax=[None] * 7,                    # 所有频带覆盖到 Nyquist 频率
                pow=1.0,                                # L1 loss (power=1), 比 L2 更鲁棒
                clamp_eps=1e-5,                         # 防止 log(0)
                weight=aux_mel_loss_weight,             # 最终 loss 中的权重 (0.05)
            )
        else:
            self.aux_mel_loss = None

        # ========== 计算 mask 对齐步长 ==========
        # 为了让随机 mask 的边界对齐到所有 mel 变换的 hop_length 的整数倍,
        # 计算所有 hop_length 的最小公倍数 (LCM).
        # 为什么需要对齐?
        #   如果 mask 边界落在 mel 帧中间, mel loss 会有边界效应.
        self.mask_align_to = 1
        alignments = []
        if self.aux_mel_loss is not None and hasattr(self.aux_mel_loss, "mel_transforms"):
            hop_lengths = [int(m.hop_length) for m in self.aux_mel_loss.mel_transforms]
            if len(hop_lengths) > 0:
                lcm_hop = hop_lengths[0]
                for h in hop_lengths[1:]:
                    lcm_hop = math.lcm(lcm_hop, h)
                alignments.append(lcm_hop)
        if len(alignments) > 0:
            lcm_align = alignments[0]
            for a in alignments[1:]:
                lcm_align = math.lcm(lcm_align, a)
            self.mask_align_to = max(1, lcm_align)

    @property
    def device(self):
        """获取模型所在设备 (CPU / CUDA:0 / ...)"""
        return next(self.parameters()).device

    def _sample_time(self, batch: int, dtype, device):
        """
        采样流匹配的时间步 t ∈ [0, 1].

        支持两种采样策略:
          - uniform:          t ~ U(0, 1)  简单均匀采样
          - logistic_normal:  z ~ N(P_mean, P_std²), t = sigmoid(z)
                              这是 Stable Diffusion 3 的做法,
                              让模型更关注中间时间步 (信息量最大的阶段).

        当前配置: P_mean=-0.8, P_std=0.8
          效果: t 更集中在 0.2~0.8 之间, 两端的极端噪声/几乎干净的情况采样较少.

        返回: t shape [B], 值在 (0, 1) 之间
        """
        if self.t_sampling == "uniform":
            # 均匀采样: 所有时间步等概率
            t = torch.rand((batch,), dtype=dtype, device=device)
        elif self.t_sampling == "logistic_normal":
            # 逻辑正态采样: z ~ N(P_mean, P_std), t = sigmoid(z)
            # sigmoid 将 (-∞, +∞) 映射到 (0, 1)
            # 当 P_mean=-0.8 时, sigmoid(-0.8)≈0.31, 分布中心在 0.3 左右
            z = torch.randn((batch,), device=device, dtype=dtype) * self.P_std + self.P_mean
            t = torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown t_sampling: {self.t_sampling}")

        # 可选的时间偏移 (推理时使用, 改变时间步分布)
        if self.time_shift != 1.0:
            t = t / (t + self.time_shift * (1 - t))

        return t

    def _x_to_v(self, x_pred, z, t):
        """
        将 x_prediction 转换为 velocity.

        公式: v_pred = (x_pred - z) / (1 - t)

        参数:
          x_pred: DiT 预测的 x1 (目标波形)
          z:      当前的 φ_t (加噪后的中间状态)
          t:      当前时间步

        返回: v_pred, 预测的 velocity (即 flow 方向)

        为什么需要这个转换?
          当 prediction="x_pred" 但 loss_space="v" 时,
          先让 DiT 输出 x_pred, 再转为 velocity,
          然后用 velocity 算 loss (v-loss 各时间步权重更均衡).
        """
        # 防止 t=1 时分母为 0, clamp 到 t_eps=0.02
        denom = (1.0 - t).clamp_min(self.t_eps)
        # 对齐维度: t 可能是 [B] 或 [B,1], z 是 [B,N] 或 [B,N,D]
        while denom.ndim < z.ndim:
            denom = denom.unsqueeze(-1)
        return (x_pred - z) / denom

    def _prepare_condition(self, text, speech_tokens, batch: int, device):
        condition = speech_tokens if exists(speech_tokens) else text
        if condition is None:
            raise ValueError("CFM requires either text or speech_tokens conditioning.")

        if exists(speech_tokens):
            if isinstance(condition, list):
                condition = pad_sequence(
                    [torch.as_tensor(t, dtype=torch.long) for t in condition],
                    batch_first=True,
                    padding_value=-1,
                )
            condition = condition.to(device=device, dtype=torch.long)
            assert condition.shape[0] == batch
            return condition

        if isinstance(condition, list):
            if exists(self.vocab_char_map):
                # 有 vocab map: 拼音字符串 → token IDs
                condition = list_str_to_idx(condition, self.vocab_char_map).to(device)
            else:
                # 无 vocab map: 直接用字符的 Unicode 编码
                condition = list_str_to_tensor(condition).to(device)
            assert condition.shape[0] == batch
        else:
            condition = condition.to(device=device, dtype=torch.long)

        return condition

    def _prepare_spk_embedding(self, spk_embedding, batch: int, device):
        if spk_embedding is None:
            return None
        spk_embedding = spk_embedding.to(device=device, dtype=next(self.parameters()).dtype)
        if spk_embedding.ndim != 2 or spk_embedding.shape[0] != batch:
            raise ValueError(f"spk_embedding expects [B, D] with B={batch}, got {tuple(spk_embedding.shape)}")
        return spk_embedding

    def _transformer_forward(self, *, spk_embedding=None, drop_spk: bool = False, **kwargs):
        if self.transformer_accepts_spk_embedding:
            kwargs["spk_embedding"] = spk_embedding
            kwargs["drop_spk"] = drop_spk
        return self.transformer(**kwargs)

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],   # 条件波形 [B, N_samples]
        text: int["b nt"] | list[str] | None,    # 目标文本
        duration: int | int["b"],                 # 目标生成时长 (采样点数量)
        *,
        speech_tokens: int["b nt"] | list[int] | None = None,  # CosyVoice2 semantic tokens
        spk_embedding: float["b d"] | None = None, # CAM++ speaker embedding
        lens: int["b"] | None = None,            # 条件波形的实际长度
        steps=32,                                 # ODE 采样步数
        cfg_strength=1.0,                         # CFG 引导强度
        sway_sampling_coef=None,                  # Sway Sampling 系数
        timestep_mapping="sway_sampling",         # 时间步映射策略
        timestep_power=None,                      # power 映射的指数
        shift=1.0,                                # 时间偏移
        seed: int | None = None,                  # 随机种子 (用于可复现)
        max_duration=4096,                        # 最大生成 token 数
        use_epss=True,                            # 是否使用经验剪枝步采样
        no_ref_audio=False,                       # 是否清零参考音频 (zero-shot TTS)
        duplicate_test=False,                     # 测试模式: 从中间时间步开始
        t_inter=0.1,                              # 测试模式下的起始时间
        edit_mask=None,                           # 编辑 mask (用于语音编辑)
    ):
        """
        ODE 采样: 从噪声生成目标语音波形.

        完整流程:
          1. 准备条件 (cond=参考音频, text=目标文本, cond_mask)
          2. 创建初始噪声 y0
          3. ODE 积分: y0 → y1, 每步 DiT 预测 velocity
          4. CFG: v = v_uncond + cfg_strength × (v_cond - v_uncond)
          5. 后处理: 还原缩放, 保留条件区域, 裁剪到目标长度

        这就是 WavTTS 的 "推理" — 给定参考音频和文本, 生成新语音.
        """
        self.eval()  # 切换到评估模式 (关闭 dropout 等)

        # ====== 输入检查与预处理 ======
        if cond.ndim != 2:
            raise ValueError(f"WavTTS expects raw waveform conditioning [B, N], got {tuple(cond.shape)}")

        # 统一 dtype 并缩放条件波形 (与训练一致)
        cond = cond.to(next(self.parameters()).dtype)
        cond = cond * self.latents_scale  # × 9.0

        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        # 如果没有提供 lens, 默认整段都是有效音频
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # ====== 条件处理 ======
        condition = self._prepare_condition(text, speech_tokens, batch=batch, device=device)
        spk_embedding = self._prepare_spk_embedding(spk_embedding, batch=batch, device=device)

        # ====== 时长处理 ======
        # cond_mask: 标记哪些采样点属于条件音频 (True=条件区域)
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            # 语音编辑模式: 额外的编辑区域 mask
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            # 单个整数 → 所有 batch 元素使用相同长度
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        # 计算最终的生成长度
        # max_duration=4096 是 waveform token 数量 (不是采样点!)
        # 转换为采样点: 4096 × 160 = 655,360 采样点 ≈ 41秒 @16kHz
        max_duration_limit = max_duration * self.wav_frame_len

        # 生成时长至少要比 (文本长度, 参考音频长度) 都长, 保证有内容生成
        duration = torch.maximum(
            torch.maximum((condition != -1).sum(dim=-1), lens) + 1, duration
        )
        duration = duration.clamp(max=max_duration_limit)  # 限制最大长度
        max_duration = duration.amax()  # batch 中最大长度

        # ====== Padding 到统一长度 ======
        # 条件音频右侧零填充到 max_duration
        if duplicate_test:
            # 测试模式: 保留测试用的条件
            test_cond = F.pad(cond, (cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, max_duration - cond_seq_len), value=0.0)

        if no_ref_audio:
            # Zero-shot TTS: 不要参考音频, 全部清零
            cond = torch.zeros_like(cond)

        # 扩展 cond_mask 到 max_duration (新增部分 = False, 即需要生成)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)

        # step_cond: 条件区域=原始波形, 非条件区域=0
        # 这就是给 DiT 的 conditioning input
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        # batch>1 时构造 mask 用于 DiT 内部的 padding 处理
        # batch=1 时省略 mask 以节省显存和加速
        if batch > 1:
            mask = lens_to_mask(duration, length=max_duration)
        else:
            mask = None

        # ============================================================
        # ODE 函数: 定义 dy/dt = v(y, t)
        # 这是 ODE 求解器在每一步需要调用的函数
        # ============================================================
        def fn(t, x):
            """
            ODE 右侧函数: 给定当前状态 x 和时间 t, 返回 velocity.

            当 cfg_strength > 0 时, 会同时计算 cond 和 uncond 两个 forward,
            然后用 CFG 公式组合: v = v_uncond + cfg_strength × (v_cond - v_uncond)

            为什么需要 CFG?
              条件生成 GAN/扩散模型对条件可能不够"忠实",
              CFG 通过放大 cond 和 uncond 之间的差异来增强条件控制.
              cfg_strength 越大 → 越遵从文本, 但可能引入失真.
            """
            def to_v(pred_x_or_v):
                """将 DiT 输出转为 velocity"""
                if self.prediction == "flow":
                    # DiT 直接输出 flow (velocity)
                    return pred_x_or_v
                else:  # x_pred 模式
                    # DiT 输出 x_pred, 需要转成 velocity
                    return self._x_to_v(pred_x_or_v, x, t)

            # ---- 不加 CFG 的情况 (cfg_strength ≈ 0) ----
            if cfg_strength < 1e-5:
                pred, *_ = self._transformer_forward(
                    x=x, cond=step_cond, text=condition, time=t, mask=mask,
                    # cfg=0 means pure unconditional sampling. Keep this
                    # branch single-pass, but apply the same condition drops
                    # as the uncond half of the packed CFG path.
                    drop_audio_cond=True, drop_text=True, cache=True,
                    spk_embedding=spk_embedding, drop_spk=True,
                )
                return to_v(pred)

            # ---- CFG 推理: 一次 forward 同时算 cond + uncond ----
            # cfg_infer=True 告诉 DiT 将 batch 翻倍处理 (输入不变, 内部分成两路)
            pred_cfg, *_ = self._transformer_forward(
                x=x, cond=step_cond, text=condition, time=t, mask=mask,
                cfg_infer=True, cache=True,  # cache=True 缓存 text embedding
                spk_embedding=spk_embedding, drop_spk=False,
            )
            # chunk 成两半: 前一半是 cond 预测, 后一半是 uncond 预测
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            v_cond = to_v(pred)
            v_uncond = to_v(null_pred)

            # CFG 组合公式: v = v_uncond + s × (v_cond - v_uncond)
            # s > 1: 增强条件控制, s = 1: 标准条件生成, s = 0: 无条件生成
            return v_uncond + (v_cond - v_uncond) * cfg_strength

        # ====== 初始化噪声 ======
        # 每个 batch 元素的噪声长度不同 (根据各自的 duration)
        # 为了支持 batch 推理, 用 pad_sequence 对齐到 max_duration
        y0 = []
        if exists(seed):
            # Seed once per sample batch. Resetting inside the loop makes
            # same-length batch items share identical initial noise.
            torch.manual_seed(seed)
        for dur in duration:
            y0.append(torch.randn(dur, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)  # [B, max_duration]

        t_start = 0

        # 测试模式: 从 t_inter (如 0.1) 开始, 观察中间时间步的行为
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        # ====== 时间步序列 ======
        use_epss = use_epss and timestep_mapping == "sway_sampling"

        if t_start == 0 and use_epss:
            # EPSS (Empirically Pruned Step Sampling):
            #   不等于均匀分布, 在 0 和 1 附近更密集采样
            #   因为 ODE 在这些区域变化剧烈, 需要更小的步长
            #   这让 32 步就能达到很好的效果
            t = get_epss_timesteps(steps, device=self.device, dtype=torch.float32)
        else:
            # 普通均匀分布: t ∈ [t_start, 1], steps+1 个点
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=torch.float32)

        # ====== 时间步映射 (可选) ======
        if timestep_mapping == "uniform":
            # 不做映射, 使用原始等距时间步
            pass
        elif timestep_mapping == "sway_sampling":
            # Sway Sampling: 余弦调制的非线性时间步映射
            # 在 0 和 1 附近更密集, 中间稀疏, 提升采样质量
            if sway_sampling_coef is not None:
                t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        elif timestep_mapping == "power":
            # Power 映射: t → t^p, p>1 时靠近 0 更密集
            if timestep_power is None:
                raise ValueError("timestep_power must be provided when timestep_mapping='power'")
            # p=0 collapses the time grid and torchdiffeq requires strictly
            # monotonic timesteps, so treat it as a very small valid power.
            t = t.pow(max(float(timestep_power), 5e-2))
        else:
            raise ValueError(f"Unknown timestep_mapping: {timestep_mapping}")

        # 时间偏移 (shift != 1 时调整时间步分布)
        # shift=0 similarly collapses all nonzero timesteps to 1.0, so clamp
        # it to a very small valid shift for ODE sampling.
        effective_shift = max(float(shift), 5e-2)
        if effective_shift != 1.0:
            denom = t + effective_shift * (1 - t)
            t = torch.where(denom > 0, t / denom.clamp_min(torch.finfo(t.dtype).eps), torch.zeros_like(t))

        # ====== ODE 求解 ======
        # 使用 Euler 方法 (强制覆盖, 因为实测 Euler 最快且效果不差)
        odeint_kwargs = dict(self.odeint_kwargs)
        odeint_kwargs["method"] = "euler"

        # odeint 自动调用 fn(t, y) 进行逐步积分
        # trajectory shape: [steps+1, B, max_duration]
        trajectory = odeint(fn, y0, t, **odeint_kwargs)

        # 清理 DiT 的文本缓存 (避免影响后续调用)
        self.transformer.clear_cache()

        # ====== 后处理 ======
        sampled = trajectory[-1]  # 最后一步的输出 (t=1)
        out = sampled

        # 还原缩放: /9.0
        out = out / self.latents_scale
        cond_unscaled = cond / self.latents_scale

        # 条件区域用原始音频替换 (保持输入不变), 非条件区域用生成结果
        out = torch.where(cond_mask, cond_unscaled, out)

        # 裁剪到各自的 target duration, 去掉 padding
        wav_list = []
        for b in range(batch):
            wav_list.append(out[b, : duration[b].item()])
        out = pad_sequence(wav_list, batch_first=True, padding_value=0.0)  # [B, N] (可能不同长度)

        return out, trajectory

    def forward(
        self,
        inp: float["b nw"],        # 原始波形 [B, N_samples], 即 flow matching 的 x1
        text: int["b nt"] | list[str] | None,  # 文本序列, 可以是 token IDs 或字符串列表
        *,
        speech_tokens: int["b nt"] | None = None,  # CosyVoice2 semantic tokens
        spk_embedding: float["b d"] | None = None, # CAM++ speaker embedding
        lens: int["b"] | None = None,   # 每条音频的实际长度 (不含 padding)
        noise_scheduler: str | None = None,  # 噪声调度器 (当前未使用)
        repa_teacher_features: torch.Tensor | None = None,
        repa_teacher_mask: torch.Tensor | None = None,
        global_update: int | None = None,
    ):
        """
        训练前向传播: 计算 Flow Matching loss.

        完整训练流程:
          ============================================
          原始波形 inp (x1)    随机噪声 x0
              │                    │
              ├─ scale ×9.0 ──┐    │
              │                │    │
              │    t ~ logistic_normal
              │                │    │
              │    φ_t = (1-t)·x0 + t·x1  ← 直线插值
              │                │
              ├─ rand_mask ──┐ │
              │   (70%-100%) │ │
              │              │ │
              cond = x1      │ │
              (mask区域=0)   │ │
              │              │ │
              └──┬───────────┘ │
                 │             │
               DiT(φ_t, cond, text, t)
                 │
                 ├→ x_pred → v_pred
                 │
                 ├→ flow_loss = MSE(v_pred, v_true) [只在mask区域]
                 └→ aux_mel_loss = 多分辨率MelLoss(x_pred, x1) × 0.05
          ============================================

        参数:
          inp:  原始波形 [B, N], N = 采样点数量
          text: 文本标注, 支持两种格式:
                - list[str]: ["拼音1 拼音2", ...] → 内部转为 token IDs
                - Tensor[int]: 已经是 token IDs [B, Nt]
          lens: 每条音频的实际采样点数量 [B], 用于创建 mask

        返回:
          total_loss:  总 loss = flow + mel + optional REPA + optional STFT/VAPA
          cond:        条件波形 (被部分 mask 的 x1)
          v_pred:      预测的 velocity
          loss_dict:   {"total_loss", "flow_loss", "aux_mel_loss", "repa_loss", "stft_loss"}
        """
        # ====== 输入检查 ======
        if inp.ndim != 2:
            raise ValueError(f"WavTTS expects raw waveform input [B, N], got {tuple(inp.shape)}")
        # import pdb; pdb.set_trace()
        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        # ====== 第1步: 条件处理 ======
        # 普通 WavTTS 走 text；token-to-wav 走 speech_tokens，但下游 DiT 接口保持 text 参数名。
        condition = self._prepare_condition(text, speech_tokens, batch=batch, device=device)
        spk_embedding = self._prepare_spk_embedding(spk_embedding, batch=batch, device=device)

        # ====== 第2步: 构造 mask (标记有效区域, 忽略 padding) ======
        if not exists(lens):
            # 如果没有提供 lens, 假设整段都是有效音频
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)  # [B, N], True=有效, False=padding

        # ====== 第3步: Flow Matching 核心 ======
        # x1 = 目标波形 (要生成的)
        x1 = inp

        # 缩放: 将波形值放大以稳定训练
        # 原始波形 ≈ [-1, 1], 乘 9.0 后 ≈ [-9, 9]
        # 这让 flow 的数值范围更大, 训练更稳定
        x1 = x1 * self.latents_scale

        # x0 = 纯高斯噪声 (起点)
        x0 = torch.randn_like(x1)

        # 采样时间步 t
        time = self._sample_time(batch, dtype=dtype, device=self.device)

        # 条件流: φ_t = (1-t)·x0 + t·x1
        # 这是从噪声到目标的直线路径
        #   t=0: φ_0 = x0 (纯噪声)
        #   t=1: φ_1 = x1 (目标波形)
        t = time.unsqueeze(-1)                   # [B] → [B, 1]
        φ = (1 - t) * x0 + t * x1                # [B, N], 加噪后的中间状态
        flow = x1 - x0                            # [B, N], 真实的 flow (velocity 的 ground truth)

        # ====== 第4步: 构造 conditioning + 定义 loss 覆盖区域 ======
        # 两种条件构造方式 (由 self.cond_mode 决定), 最终产物:
        #   cond:      给模型的波形条件 (未给出区域 = 0)
        #   loss_mask: loss 要覆盖的区域
        if self.cond_mode == "infill":
            # Voicebox 式随机段补全:
            #   随机选 70%-100% 的区域作为"要预测的目标" (loss 只算这里)
            #   剩余区域作为"已知条件" (给模型看到的)
            #   模型学会了从部分音频 + 完整文本 → 补全音频
            #
            # 这样训练出来的模型天然支持:
            #   - TTS:      cond=参考音频开头, pred=后续全部
            #   - 语音编辑:  cond=未编辑部分, pred=被替换的区域
            frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)

            if self.mask_align_to > 1:
                # 对齐到 mel hop_length 的倍数 (避免 mel loss 边界效应)
                rand_span_mask = MelSpectrogramLoss._aligned_random_span_mask(
                    lengths=lens,
                    frac_lengths=frac_lengths,
                    align_to=self.mask_align_to,
                    max_length=seq_len,
                )
            else:
                # 普通随机 mask (不强制对齐)
                rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

            # 与有效区域 mask 取交集: 只 mask 有效音频部分
            if exists(mask):
                rand_span_mask &= mask

            # 被 mask 区域 = 0 (模型需要预测这些区域)
            # 未 mask 区域 = x1 原始值 (模型可以看到, 作为条件)
            cond = torch.where(rand_span_mask, torch.zeros_like(x1), x1)
            loss_mask = rand_span_mask

        elif self.cond_mode == "prefix":
            # CosyVoice2 式前缀 prompt:
            #   50% 概率完全没有 prompt (纯无条件);
            #   否则随机取 [0, prefix_frac*len] 的波形前缀塞进 cond, 其余位置置 0.
            #   loss 覆盖整段有效 wav (含 prompt 前缀帧) — prompt 区的 cond 里写着答案,
            #   这部分 loss 自然很低, 模型真正学的是"给定前几秒音频, 预测后面发声",
            #   与推理时"给 prompt、生成目标"完全同构.
            cond = torch.zeros_like(x1)
            for i, j in enumerate(lens):
                if random.random() < self.prefix_drop_prob:
                    continue
                idx = random.randint(0, int(self.prefix_frac * j))
                idx -= idx % self.wav_frame_len  # 对齐到波形帧边界, 避免切断一个帧
                cond[i, :idx] = x1[i, :idx]
            loss_mask = mask if exists(mask) else torch.ones_like(x1, dtype=torch.bool)

        else:
            raise ValueError(f"Unknown cond_mode: {self.cond_mode}")

        # ====== 第5步: CFG Dropout (随机丢弃条件, 用于 CFG 训练) ======
        # 两种模式:
        #   infill: 沿用 joint / 独立 两种旧丢弃逻辑
        #   prefix: CosyVoice2 式 per-sample, 每样本以 training_cfg_rate (0.2) 概率
        #           把 audio prompt + content token + spk 三个条件一起清零,
        #           与上面的 50% prompt 丢弃相互独立.
        if self.cond_mode == "infill":
            if self.joint_cond_drop_prob > 0.0:
                # 联合丢弃模式:
                #   以 joint_cond_drop_prob 的概率同时丢弃音频和文本
                joint_drop = random.random() < self.joint_cond_drop_prob
                drop_audio_cond = joint_drop
                drop_text = joint_drop
                drop_spk = joint_drop
            else:
                # 独立丢弃模式:
                #   先决定是否丢弃音频 (30%)
                drop_audio_cond = random.random() < self.audio_drop_prob
                if random.random() < self.cond_drop_prob:
                    # 以 cond_drop_prob (20%) 的概率全丢弃
                    drop_audio_cond = True
                    drop_text = True
                    drop_spk = True
                else:
                    drop_text = False
                    drop_spk = False
        else:
            drop_audio_cond = False
            drop_text = False
            drop_spk = False
            if self.training_cfg_rate > 0.0:
                cfg_mask = torch.rand(batch, device=device) > self.training_cfg_rate  # [B]
                cond = cond * cfg_mask.view(-1, 1)
                if spk_embedding is not None:
                    # 零向量行经 InputEmbedding 内 F.normalize (clamp_min eps) 后仍为 0
                    spk_embedding = spk_embedding * cfg_mask.view(-1, 1)
                if condition is not None:
                    # 内容条件置为 padding (-1), 语义 token / 文本都会被 mask 成 0
                    condition = condition.clone()
                    condition[~cfg_mask] = -1

        # ====== 第7步: DiT 前向传播 ======
        # 输入: φ_t (加噪波形), cond (条件波形), text (文本), t (时间步)
        raw_pred, repa_hidden, *_ = self._transformer_forward(
            x=φ, cond=cond, text=condition, time=time,
            drop_audio_cond=drop_audio_cond, drop_text=drop_text,
            drop_spk=drop_spk,
            mask=mask,   # 有效区域 mask, DiT 内部会处理 padding
            lens=lens,   # 实际长度
            spk_embedding=spk_embedding,
            return_hidden_state=self.repa_alignment_head is not None,
            hidden_state_layer=self.repa_layer,
        )
        # raw_pred: DiT 的原始输出 [B, N]

        # ====== 第8步: 解读预测 ======
        # 根据 prediction 模式, 将 DiT 输出解读为 x_pred 或 v_pred
        if self.prediction == "flow":
            # DiT 直接预测 flow (velocity), 这是最简单的模式
            v_pred = raw_pred
            # 从 flow 反推 x_pred: x_pred = φ_t + (1-t) × v_pred
            x_pred = φ + (1.0 - t) * v_pred
        elif self.prediction == "x_pred":
            # DiT 预测 x1 (目标波形本身)
            x_pred = raw_pred
            # 从 x_pred 推算 velocity: v = (x_pred - φ_t) / (1-t)
            v_pred = self._x_to_v(x_pred, φ, time)
        else:
            raise ValueError(f"Unknown prediction: {self.prediction}")

        # ====== 第9步: 计算 Loss ======
        if self.loss_space == "flow":
            # 在 flow 空间算 MSE
            loss = F.mse_loss(v_pred, flow, reduction="none")

        elif self.loss_space == "v":
            # 在 velocity 空间算 MSE (当前配置使用此模式)
            # target = (x1 - φ_t) / (1-t), 即真实 velocity
            denom = (1.0 - time).clamp_min(self.t_eps)  # 防止 t=1 时分母为 0
            while denom.ndim < φ.ndim:
                denom = denom.unsqueeze(-1)
            target = (x1 - φ) / denom
            loss = F.mse_loss(v_pred, target, reduction="none")
            # 为什么用 v-loss?
            #   各时间步的 loss 量级更均衡:
            #     flow loss: t→1 时 flow≈0, loss 自然小 → 不均匀
            #     v-loss:   除以 (1-t) 做了归一化 → 各 t 权重接近

        elif self.loss_space == "x":
            # 在 x1 空间直接算 MSE
            loss = F.mse_loss(x_pred, x1, reduction="none")
        else:
            raise ValueError(f"Unknown loss_space: {self.loss_space}")

        # ====== 第10步: 只在 loss 覆盖区域计算 loss ======
        # infill: 只在被 mask 的预测区算 (未被 mask 的区域是 conditioning, 不需要预测)
        # prefix: 覆盖整段有效 wav (含 prompt 前缀帧)
        loss = loss[loss_mask]  # 取出 loss 覆盖区域内所有采样点的 loss 值
        flow_loss = loss.mean()      # 标量, 主 loss
        total_loss = flow_loss

        # ====== 第11步: REPA hidden-state alignment ======
        # The frozen WavLM branch is computed in Trainer.  Only the DiT-side
        # hidden state and the small alignment head receive gradients here.
        repa_loss = torch.tensor(0.0, device=device)
        if (
            self.repa_alignment_head is not None
            and repa_hidden is not None
            and repa_teacher_features is not None
            and repa_teacher_mask is not None
        ):
            repa_loss = repa_cosine_loss(
                repa_hidden,
                repa_teacher_features,
                repa_teacher_mask,
                self.repa_alignment_head,
                source_mask=loss_mask,
            ) * self.repa_loss_weight
            total_loss = total_loss + repa_loss

        # Perceptual branches operate on the original waveform range, not the
        # latents-scaled range used by flow matching.
        x1_flat_unscaled = x1 / self.latents_scale
        x1_pred_flat_unscaled = x_pred / self.latents_scale

        # ====== 第12步: 辅助 Mel-Spectrogram Loss ======
        # 多分辨率 mel loss, 帮助提升听感质量
        aux_mel_loss = torch.tensor(0.0, device=device)
        if self.use_aux_mel_loss and self.aux_mel_loss is not None:
            aux_mel_kwargs = {}
            if self.aux_mel_loss_masked:
                # 只在 loss 覆盖区域计算 mel loss (与主 loss 一致)
                aux_mel_kwargs.update(
                    frame_mask=loss_mask,
                    frame_lengths=lens,
                )

            # 7 个分辨率 (5~320 mel bins) 的加权和
            aux_mel_loss = self.aux_mel_loss(
                x1_pred_flat_unscaled,
                x1_flat_unscaled,
                **aux_mel_kwargs,
            )
            total_loss = total_loss + aux_mel_loss  # aux_mel_loss 内部已乘 0.05

        # ====== 第13步: refined STFT / VAPA loss ======
        stft_loss = torch.tensor(0.0, device=device)
        stft_active = (
            self.stft_loss is not None
            and (global_update is None or global_update >= self.stft_start_update)
        )
        if stft_active:
            stft_values = self.stft_loss(
                x1_pred_flat_unscaled,
                x1_flat_unscaled,
                frame_mask=loss_mask,
                frame_lengths=lens,
            )
            # Match the implicit data-to-velocity scaling of the v-loss.
            vapa_scale = (1.0 - time).clamp_min(self.stft_eps).pow(-self.stft_gamma)
            stft_loss = (stft_values * vapa_scale).mean() * self.stft_loss_weight
            total_loss = total_loss + stft_loss

        # ====== 返回 ======
        loss_dict = {
            "total_loss": total_loss,      # 总 loss, 用于 backward
            "flow_loss": flow_loss,        # 流匹配主 loss, 用于日志监控
            "aux_mel_loss": aux_mel_loss,  # 辅助 mel loss, 用于日志监控
            "repa_loss": repa_loss,
            "stft_loss": stft_loss,
        }

        return total_loss, cond, v_pred, loss_dict
