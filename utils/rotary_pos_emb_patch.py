from typing import Optional

import torch
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.transformer import TransformerConfig
from torch import Tensor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def apply_patch():
    if hasattr(rope_utils, 'apply_rotary_pos_emb'):
        rope_utils.apply_rotary_pos_emb = apply_rotary_pos_emb_new
        print("WARNING: Monkey-patched 'mpu.rope_utils.apply_rotary_pos_emb' using 'apply_rotary_pos_emb_new'.")


def apply_rotary_pos_emb_new(
        t: Tensor,
        freqs: Tensor,
        config: TransformerConfig,
        cu_seqlens: Optional[Tensor] = None,
        mscale: float = 1.0,
        cp_group: torch.distributed.ProcessGroup = None,
):
    # 按最后一维拆分为dim长度的两部分，返回元组(cos_restored, sin_restored)
    print("freqs shape:", freqs.shape)  # 查看Megatron传入的freqs形状
    print("t (query) shape:", t.shape)  # 查看query的形状（通常是[batch, seq_len, num_heads, head_dim]）
    cos_restored, sin_restored = torch.split(freqs, freqs.shape[-1] // 2, dim=-1)
    # # 2. 扩展RoPE维度到匹配query的head_dim=128（每对复数维度共享相同的cos/sin）
    # # 方法：将64维的cos/sin重复一次，得到128维（对应128维head_dim的64对旋转）
    # cos_restored = cos_restored.repeat_interleave(2, dim=-1)  # [seq_len,1,128]
    # sin_restored = sin_restored.repeat_interleave(2, dim=-1)  # [seq_len,1,128]
    print("cos_restored shape:", cos_restored.shape)  # 查看Megatron传入的freqs形状
    print("sin_restored shape:", sin_restored.shape)  # 查看query的形状（通常是[batch, seq_len, num_heads, head_dim]）
    r1, r2 = apply_rotary_pos_emb(t, t, cos_restored, sin_restored, unsqueeze_dim=2)
    return r1
