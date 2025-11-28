from typing import Optional

import torch
from megatron.core.models.common.embeddings import rope_utils
from megatron.core.transformer import TransformerConfig
from torch import Tensor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def apply_patch():
    if hasattr(rope_utils, 'apply_rotary_pos_emb'):
        rope_utils.apply_rotary_pos_emb = apply_rotary_pos_emb_new


def apply_rotary_pos_emb_new(
        t: Tensor,
        freqs: Tensor,
        config: TransformerConfig,
        cu_seqlens: Optional[Tensor] = None,
        mscale: float = 1.0,
        cp_group: torch.distributed.ProcessGroup = None,
):
    # 按最后一维拆分为dim长度的两部分，返回元组(cos_restored, sin_restored)
    cos_restored, sin_restored = torch.split(freqs, freqs.shape[-1] // 2, dim=-1)
    r1, r2 = apply_rotary_pos_emb(t, t, cos_restored, sin_restored, unsqueeze_dim=2)
    return r1
