"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers.x_transformers import RotaryEmbedding

from wavtts.model.modules import (
    AdaLayerNorm_Final,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    TimestepEmbedding,
    precompute_freqs_cis,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self, text_num_embeds, text_dim, mask_padding=True, average_upsampling=False, conv_layers=0, conv_mult=2
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 waveform tokens is ~81.92s at 16k / wav_frame_len=160
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def average_upsample_text_by_mask(self, text, text_mask):
        batch, seq_len, text_dim = text.shape

        text_lens = text_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            valid_text_len = text_lens[i].item()

            if valid_text_len == 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [valid_text_len, text_dim]

            base_repeat = seq_len // valid_text_len
            remainder = seq_len % valid_text_len

            indices = []
            for j in range(valid_text_len):
                repeat_count = base_repeat + (1 if j >= valid_text_len - remainder else 0)
                indices.extend([j] * repeat_count)

            indices = torch.tensor(indices[:seq_len], device=text.device, dtype=torch.long)
            upsampled = valid_data[indices]  # [seq_len, text_dim]

            upsampled_text[i, :seq_len, :] = upsampled

        return upsampled_text

    def forward(self, text: int["b nt"], seq_len, drop_text=False):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than waveform tokens
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            text = text + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask)

        return text


class SemanticTokenEmbedding(nn.Module):
    def __init__(
        self,
        token_num_embeds,
        token_dim,
        mask_padding=True,
        conv_layers=0,
        conv_mult=2,
        upsampling="linear",
        token_offset=0,
    ):
        super().__init__()
        self.token_num_embeds = int(token_num_embeds)
        self.mask_padding = mask_padding
        self.upsampling = upsampling
        self.token_offset = int(token_offset)
        self.token_embed = nn.Embedding(self.token_num_embeds + 1, token_dim)  # 0 is padding / CFG-null.

        self.precompute_max_pos = 8192
        self.register_buffer("freqs_cis", precompute_freqs_cis(token_dim, self.precompute_max_pos), persistent=False)

        if conv_layers > 0:
            self.extra_modeling = True
            self.token_blocks = nn.Sequential(
                *[ConvNeXtV2Block(token_dim, token_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def _normalize_token_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.long()
        if self.token_offset > 0:
            non_pad = tokens >= 0
            tokens = torch.where(non_pad, tokens - self.token_offset, tokens)
        return tokens

    def _upsample_to_seq_len(self, token_embed: torch.Tensor, token_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        if token_embed.shape[1] == seq_len:
            return token_embed

        if self.upsampling != "linear":
            raise ValueError(f"Unsupported semantic token upsampling: {self.upsampling}")

        upsampled = F.interpolate(
            token_embed.transpose(1, 2),
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        if self.mask_padding:
            mask_f = token_mask.float().unsqueeze(1)
            up_mask = F.interpolate(mask_f, size=seq_len, mode="nearest").squeeze(1).bool()
            upsampled = upsampled.masked_fill(~up_mask.unsqueeze(-1), 0.0)

        return upsampled

    def forward(self, speech_tokens: int["b nt"], seq_len, drop_text=False):
        if speech_tokens.ndim != 2:
            raise ValueError(f"SemanticTokenEmbedding expects [B, T], got {tuple(speech_tokens.shape)}")

        speech_tokens = self._normalize_token_ids(speech_tokens)
        token_mask = speech_tokens >= 0
        speech_tokens = speech_tokens + 1

        if drop_text:
            speech_tokens = torch.zeros_like(speech_tokens)

        token_embed = self.token_embed(speech_tokens)
        pos_len = min(token_embed.shape[1], self.precompute_max_pos)
        pos_embed = F.pad(self.freqs_cis[:pos_len, :], (0, 0, 0, token_embed.shape[1] - pos_len), value=0.0)
        token_embed = token_embed + pos_embed

        if self.mask_padding:
            token_embed = token_embed.masked_fill(~token_mask.unsqueeze(-1), 0.0)

        if self.extra_modeling:
            if self.mask_padding:
                for block in self.token_blocks:
                    token_embed = block(token_embed)
                    token_embed = token_embed.masked_fill(~token_mask.unsqueeze(-1), 0.0)
            else:
                token_embed = self.token_blocks(token_embed)

        return self._upsample_to_seq_len(token_embed, token_mask, int(seq_len))


# noised waveform and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        wav_frame_len,
        text_dim,
        out_dim,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
        use_spk_embedding: bool = False,
        spk_embed_dim: int = 192,
        spk_proj_dim: int | None = None,
    ):
        super().__init__()

        self.use_audio_proj = use_audio_proj
        self.use_spk_embedding = bool(use_spk_embedding)
        self.spk_embed_dim = int(spk_embed_dim)
        self.spk_proj_dim = text_dim if spk_proj_dim is None else int(spk_proj_dim)
        if self.use_spk_embedding:
            self.spk_proj = nn.Linear(self.spk_embed_dim, self.spk_proj_dim)
        else:
            self.spk_proj = None

        spk_input_dim = self.spk_proj_dim if self.use_spk_embedding else 0

        if not use_audio_proj:
            self.proj = nn.Linear(wav_frame_len * 2 + text_dim + spk_input_dim, out_dim)
        else:
            audio_proj_dim = out_dim if audio_proj_dim is None else audio_proj_dim
            audio_proj_hidden = audio_proj_dim if audio_proj_hidden is None else audio_proj_hidden
            self.x_proj = nn.Sequential(
                nn.Linear(wav_frame_len, audio_proj_hidden, bias=False),
                nn.Linear(audio_proj_hidden, audio_proj_dim),
            )
            self.cond_proj = nn.Sequential(
                nn.Linear(wav_frame_len, audio_proj_hidden, bias=False),
                nn.Linear(audio_proj_hidden, audio_proj_dim),
            )

            self.fuse = nn.Linear(audio_proj_dim * 2 + text_dim + spk_input_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def _speaker_frame_embed(
        self,
        spk_embedding: torch.Tensor | None,
        batch: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        drop_spk: bool = False,
    ) -> torch.Tensor | None:
        if not self.use_spk_embedding:
            return None

        if spk_embedding is None or drop_spk:
            spk_embedding = torch.zeros(batch, self.spk_embed_dim, device=device, dtype=dtype)
        else:
            if spk_embedding.ndim != 2:
                raise ValueError(f"spk_embedding expects [B, D], got {tuple(spk_embedding.shape)}")
            if spk_embedding.shape[0] != batch or spk_embedding.shape[1] != self.spk_embed_dim:
                raise ValueError(
                    f"spk_embedding expects [{batch}, {self.spk_embed_dim}], got {tuple(spk_embedding.shape)}"
                )
            spk_embedding = spk_embedding.to(device=device, dtype=dtype)
            spk_embedding = F.normalize(spk_embedding, dim=1)

        spk_embed = self.spk_proj(spk_embedding)
        return spk_embed.unsqueeze(1).expand(-1, seq_len, -1)

    def forward(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        text_embed: float["b n d"],
        spk_embedding: float["b d"] | None = None,
        drop_audio_cond=False,
        drop_spk: bool = False,
        audio_mask: bool["b n"] | None = None,
    ):
        if drop_audio_cond:
            cond = torch.zeros_like(cond)

        spk_embed = self._speaker_frame_embed(
            spk_embedding,
            batch=x.shape[0],
            seq_len=x.shape[1],
            dtype=x.dtype,
            device=x.device,
            drop_spk=drop_spk,
        )

        if not self.use_audio_proj:
            inputs = (x, cond, text_embed) if spk_embed is None else (x, cond, text_embed, spk_embed)
            h = self.proj(torch.cat(inputs, dim=-1))
        else:
            x_h = self.x_proj(x)
            c_h = self.cond_proj(cond)
            inputs = (x_h, c_h, text_embed) if spk_embed is None else (x_h, c_h, text_embed, spk_embed)
            h = self.fuse(torch.cat(inputs, dim=-1))

        h = self.conv_pos_embed(h, mask=audio_mask) + h
        return h


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        wav_frame_len=160,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        condition_type="text",
        speech_token_num_embeds=6561,
        speech_token_dim=None,
        speech_token_mask_padding=True,
        speech_token_conv_layers=None,
        speech_token_upsampling="linear",
        speech_token_offset=0,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_audio_proj: bool = False,
        audio_proj_dim: int | None = None,
        audio_proj_hidden: int | None = None,
        use_spk_embedding: bool = False,
        spk_embed_dim: int = 192,
        spk_embed_hidden: int = 512,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = wav_frame_len
        self.condition_type = str(condition_type)
        if self.condition_type == "speech_token":
            text_dim = text_dim if speech_token_dim is None else speech_token_dim
            self.text_embed = SemanticTokenEmbedding(
                token_num_embeds=speech_token_num_embeds,
                token_dim=text_dim,
                mask_padding=speech_token_mask_padding,
                conv_layers=conv_layers if speech_token_conv_layers is None else speech_token_conv_layers,
                upsampling=speech_token_upsampling,
                token_offset=speech_token_offset,
            )
        elif self.condition_type == "text":
            self.text_embed = TextEmbedding(
                text_num_embeds,
                text_dim,
                mask_padding=text_mask_padding,
                average_upsampling=text_embedding_average_upsampling,
                conv_layers=conv_layers,
            )
        else:
            raise ValueError(f"Unsupported DiT condition_type: {condition_type}")
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(
            wav_frame_len, text_dim, dim,
            use_audio_proj=use_audio_proj,
            audio_proj_dim=audio_proj_dim,
            audio_proj_hidden=audio_proj_hidden,
            use_spk_embedding=use_spk_embedding,
            spk_embed_dim=spk_embed_dim,
            spk_proj_dim=text_dim,
        )
        self.use_spk_embedding = bool(use_spk_embedding)
        self.spk_embed_dim = int(spk_embed_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out_dim = wav_frame_len
        self.proj_out = nn.Linear(dim, wav_frame_len)
        self.proj_out_output_layer = self.proj_out

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

        # raw waveform tokenization.
        self.wav_frame_len = wav_frame_len

    def initialize_weights(self):
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # self.apply(_basic_init)

        # # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out_output_layer.weight, 0)
        if self.proj_out_output_layer.bias is not None:
            nn.init.constant_(self.proj_out_output_layer.bias, 0)

    def set_wav_frame_len(self, wav_frame_len: int):
        self.wav_frame_len = int(wav_frame_len)

        if self.wav_frame_len != self.proj_out_dim:
            raise ValueError(
                f"wav_frame_len ({self.wav_frame_len}) must equal proj_out_dim ({self.proj_out_dim}) "
                "for reshape wav front-end."
            )

    def _wav_to_tokens(
        self,
        wav: torch.Tensor,
        mask: torch.Tensor | None = None,
        lens: torch.Tensor | None = None,
    ):
        assert wav.ndim == 2, f"Expected [B, N] wav input, got {tuple(wav.shape)}"

        bsz, num_samples = wav.shape
        frame_len = self.wav_frame_len
        pad_len = (frame_len - (num_samples % frame_len)) % frame_len

        if pad_len > 0:
            wav = F.pad(wav, (0, pad_len), value=0.0)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)

        tokens = wav.view(bsz, -1, frame_len)

        token_mask = None
        if mask is not None:
            token_mask = mask.view(bsz, -1, frame_len).any(dim=-1)

        token_lens = None
        if lens is not None:
            token_lens = (lens.to(dtype=torch.long, device=wav.device) + frame_len - 1) // frame_len

        return tokens, token_mask, token_lens

    def _tokens_to_wav(self, tokens: torch.Tensor, target_num_samples: int):
        wav = tokens.reshape(tokens.shape[0], -1)
        return wav[:, :target_num_samples]

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        spk_embedding: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_spk: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(text, x.shape[1], drop_text=drop_text)
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)  # Calculate the actual sequence length for each sample
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(text_embed_list, batch_first=True, padding_value=0)
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x,
            cond,
            text_embed,
            spk_embedding=spk_embedding,
            drop_audio_cond=drop_audio_cond,
            drop_spk=drop_spk,
            audio_mask=audio_mask,
        )

        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b nw"],  # noised waveform
        cond: float["b nw"],  # masked conditioning waveform
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | bool["b nw"] | None = None,
        drop_audio_cond: bool = False,  # cfg for conditioning waveform
        drop_text: bool = False,  # cfg for text
        drop_spk: bool = False,  # cfg for speaker embedding
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        lens: int["b"] | None = None,
        spk_embedding: float["b d"] | None = None,
        return_hidden_state: bool = False,
        hidden_state_layer: int = -1,
    ):
        if x.ndim != 2 or cond.ndim != 2:
            raise ValueError(f"WavTTS DiT expects raw waveform x/cond [B, N], got {x.ndim}D and {cond.ndim}D.")

        target_num_samples = x.shape[1]
        x, token_mask, token_lens = self._wav_to_tokens(x, mask=mask, lens=lens)
        cond, _, _ = self._wav_to_tokens(cond, mask=None, lens=None)
        mask = token_mask
        lens = token_lens

        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised waveform + conditioning waveform + text
        t = self.time_embed(time)

        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x, cond, text, spk_embedding=spk_embedding,
                drop_audio_cond=False, drop_text=False, drop_spk=False, cache=cache, audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x, cond, text, spk_embedding=spk_embedding,
                drop_audio_cond=True, drop_text=True, drop_spk=True, cache=cache, audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x, cond, text, spk_embedding=spk_embedding,
                drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_spk=drop_spk,
                cache=cache, audio_mask=mask
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        selected_hidden = None
        if return_hidden_state:
            num_layers = len(self.transformer_blocks)
            resolved_layer = hidden_state_layer if hidden_state_layer >= 0 else num_layers + hidden_state_layer
            if not 0 <= resolved_layer < num_layers:
                raise ValueError(
                    f"hidden_state_layer={hidden_state_layer} is invalid for {num_layers} DiT blocks"
                )
        else:
            resolved_layer = None
            
        for i, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False)
            else:
                x = block(x, t, mask=mask, rope=rope)
            if resolved_layer is not None and i == resolved_layer:
                selected_hidden = x

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        output = self._tokens_to_wav(output, target_num_samples=target_num_samples)

        return output, selected_hidden, None
