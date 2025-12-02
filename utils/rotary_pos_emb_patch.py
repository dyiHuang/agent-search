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
        t: Tensor,  # [seq_len, batch, num_heads, head_dim]
        freqs: Tensor,  # [1, seq_len, 2*head_dim]
        config: TransformerConfig,
        cu_seqlens: Optional[Tensor] = None,
        mscale: float = 1.0,
        cp_group: torch.distributed.ProcessGroup = None,
):
    # 按最后一维拆分为dim长度的两部分，返回元组(cos_restored, sin_restored)
    # _freqs = torch.zeros_like(freqs, dtype=freqs.dtype,device=freqs.device)
    # _freqs = _freqs.copy_(freqs)
    cos_restored, sin_restored = torch.split(freqs, freqs.shape[-1] // 2, dim=-1)
    t = t.transpose(0, 1)
    print(f"apply_rotary_pos_emb_new cos - 形状: {cos_restored.shape}, mean: [{cos_restored.mean():.6f}, std: {cos_restored.std():.6f}]")
    print(f"apply_rotary_pos_emb_new sin - 形状: {sin_restored.shape}, mean: [{sin_restored.mean():.6f}, std: {sin_restored.std():.6f}]")
    r1, r2 = apply_rotary_pos_emb(t, t, cos_restored, sin_restored, unsqueeze_dim=2)
    r1 = r1.transpose(0, 1).contiguous()
    return r1
