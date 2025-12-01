from functools import partial
from typing import Optional, Union, List, Dict, Callable

import torch
from torch import Tensor
import torch.nn as nn
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core import parallel_state, tensor_parallel, pipeline_parallel
from megatron.core.models.huggingface import qwen_model
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerConfig, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_utils import PreTrainedModel
from transformers import Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, eager_attention_forward
from transformers.activations import ACT2FN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from megatron.core.packed_seq_params import PackedSeqParams

from utils import utils, torch_functional
from tensor_parallel import vocab_parallel_log_probs_from_logits, vocab_parallel_compute_entropy_loss

import core_algos
import qwen_load
from tensordict import TensorDict


class Qwen2DotProductAttention(DotProductAttention):
    def __init__(self, config: TransformerConfig, layer_number: int, attn_mask_type: AttnMaskType, attention_type: str,
                 attention_dropout: float = None, softmax_scale: float = None, cp_comm_type: str = None,
                 pg_collection: ProcessGroupCollection = None):
        super().__init__(config, layer_number, attn_mask_type, attention_type, attention_dropout, softmax_scale,
                         cp_comm_type, pg_collection)

        self.num_key_value_groups = self.num_attention_heads_per_partition // self.num_query_groups_per_partition

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attention_mask: Tensor,
                attn_mask_type: AttnMaskType = None,
                attention_bias: Tensor = None,
                packed_seq_params: Optional[PackedSeqParams] = None, ):
        attention_interface: Callable = eager_attention_forward
        if self.qwen_config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.qwen_config._attn_implementation]
        # [sq, b, np, hn],[sq, b, gp, hn] -> [b, np, sq, hn],[b, gp, sq, hn]
        query = query.transpose(0, 1).transpose(1, 2).contiguous()
        key = key.transpose(0, 1).transpose(1, 2).contiguous()
        value = value.transpose(0, 1).transpose(1, 2).contiguous()

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=self.config.attention_dropout,
            scaling=self.softmax_scale,
            sliding_window=None,  # main diff with Llama
            # **kwargs,
        )

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value.size(0), query.size(1), query.size(2), value.size(3))

        # change view [b, np, sq, hn]
        context = attn_output.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context


class Qwen2MegatronAttention(SelfAttention):
    def __init__(self,
                 config: TransformerConfig,
                 layer_number: int,
                 attn_mask_type=AttnMaskType.padding,
                 cp_comm_type: str = None,
                 pg_collection: ProcessGroupCollection = None
                 ):
        config.add_qkv_bias = True
        super().__init__(
            config,
            submodules=SelfAttentionSubmodules(
                linear_qkv=tensor_parallel.ColumnParallelLinear,
                core_attention=Qwen2DotProductAttention,
                linear_proj=tensor_parallel.RowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )


class Qwen2MegatronMLP(MLP):
    def __init__(self, config: TransformerConfig):
        self.config = config
        mlp = MLPSubmodules(
            linear_fc1=tensor_parallel.ColumnParallelLinear,  # Qwen2.5 gate_proj, up_proj
            linear_fc2=tensor_parallel.RowParallelLinear,  # Qwen2.5 down_proj
        )
        super().__init__(config, submodules=mlp)
        # # Qwen2.5 用SwiGLU， 替换 Megatron 默认 GELU
        # self.activation_func = nn.SiLU  # SwiLU = SilU + 点积


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, config: TransformerConfig, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2MegatronTransformerLayer(TransformerLayer):
    def __init__(self, config: TransformerConfig, layer_number: int = 1, qwen_config: Qwen2Config = None):
        super().__init__(
            config,
            submodules=TransformerLayerSubmodules(
                input_layernorm=Qwen2RMSNorm,
                self_attention=ModuleSpec(
                    module=Qwen2MegatronAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                ),
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=Qwen2RMSNorm,
                mlp=Qwen2MegatronMLP,
                mlp_bda=get_bias_dropout_add,
                sharded_state_dict_keys_map={
                    #"input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                    #"pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_"
                },
            ),
            layer_number=layer_number)
        qwen_config._attn_implementation = 'sdpa'
        self.self_attention.core_attention.qwen_config = qwen_config
        self.attention_type = qwen_config.layer_types[layer_number-1]


class Qwen2PreTrainedModel(PreTrainedModel):
    config: Qwen2Config
    base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["Qwen2DecoderLayer"]
    # _skip_keys_device_placement = ["past_key_values"]
    # _supports_flash_attn = True
    # _supports_sdpa = True
    # _supports_flex_attn = True

    # _can_compile_fullgraph = True
    # _supports_attention_backend = True
    # _can_record_outputs = {
    #     "hidden_states": Qwen2DecoderLayer,
    #     "attentions": Qwen2Attention,
    # }


class Qwen2MegatronModel(MegatronModule):
    """Megatron 并行化的 Qwen2.5-3B 模型"""

    def __init__(self, g_config, qwen_config: Qwen2Config, megatron_config: TransformerConfig):
        super().__init__(megatron_config)
        self.ddp_config = DistributedDataParallelConfig(use_megatron_fsdp=False)
        self.vocab_size = qwen_config.vocab_size
        self.hidden_size = qwen_config.hidden_size
        self.num_layers = qwen_config.num_hidden_layers
        self.tp_size = megatron_config.tensor_model_parallel_size
        self.micro_batch_size = g_config.actor.ppo_micro_batch_size
        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()  # 获取 PP 总进程数
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()  # 当前 PP stage 编号（0~pp_size-1）

        # -------------------------- 1. PP 拆分：按 stage 分配层数 --------------------------
        # 计算当前 stage 负责的 Transformer 层数（均匀分配，支持余数处理）
        self.num_layers_per_stage = self.num_layers // self.pp_size
        self.start_layer = self.pp_rank * self.num_layers_per_stage
        self.end_layer = self.start_layer + self.num_layers_per_stage
        # 处理余数（最后一个 stage 多承担剩余层数）
        if self.pp_rank == self.pp_size - 1:
            self.end_layer = self.num_layers

        # -------------------------- 2. 仅当前 stage 初始化自己负责的层 --------------------------
        # 嵌入层：仅 PP stage 0 初始化（输入层必须在第一个 stage）
        # 嵌入层（Tensor Parallel）
        self.embedding = None
        if self.pp_rank == 0:
            self.embedding = tensor_parallel.VocabParallelEmbedding(
                self.vocab_size, self.hidden_size, init_method=torch.nn.init.xavier_uniform_, config=megatron_config
            )

        # Rotary Embedding：仅 PP stage 0 计算（后续 stage 复用或传递）
        # Rotary Embedding (Qwen2.5标准实现)
        # self.rotary_emb = RotaryEmbedding(
        #     megatron_config.kv_channels, rotary_percent=1.0, seq_len_interpolation_factor=1.0
        # )
        self.rotary_emb = Qwen2RotaryEmbedding(config=qwen_config)

        # Transformer 层：仅初始化当前 stage 负责的层（核心 PP 拆分）
        self.layers = nn.ModuleList([
            Qwen2MegatronTransformerLayer(megatron_config, i - self.start_layer + 1, qwen_config)
            for i in range(self.start_layer, self.end_layer)
        ])

        # 输出层（final_norm + lm_head）：仅 PP 最后一个 stage 初始化
        self.final_norm = None
        self.lm_head = None
        utils.print_rank_0(f"self.pp_rank: {self.pp_rank}, self.pp_size: {self.pp_size}")
        if self.pp_rank == self.pp_size - 1:
            self.final_norm = Qwen2RMSNorm(megatron_config, self.hidden_size, eps=qwen_config.rms_norm_eps)
            self.lm_head = tensor_parallel.ColumnParallelLinear(
                self.hidden_size, self.vocab_size, config=megatron_config, bias=False,
                init_method=megatron_config.init_method
            )
        self.model_type = ModelType.encoder_or_decoder

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, only_last_token: bool = False):
        """适配 PP+TP 并行的前向传播：处理跨 stage 数据流转"""

        # -------------------------- 1. 第一个 PP stage：执行嵌入层 + Rotary 编码 --------------------------
        # 嵌入 + Rotary 编码
        hidden_states = None
        rotary_pos_emb = None
        if self.pp_rank == 0:
            ori_input_ids = input_ids
            # [b, s, h] -> [s, b, h]
            input_ids = input_ids.transpose(1, 0).contiguous()
            # 嵌入层（仅 stage 0 有）
            hidden_states = self.embedding(input_ids)

            ori_input_ids.transpose(1, 0)

            # 提前处理 only_last_token（减少跨 stage 通信量）
            # if only_last_token:
            #     hidden_states = hidden_states[:, -1:]  # [batch, 1, hidden_size/tp_size]
            #     rotary_pos_emb = rotary_pos_emb[:, -1:] if isinstance(rotary_pos_emb, torch.Tensor) else None
        else:
            # self.hidden_states should be passed by Megatron
            hidden_states = self.input_tensor

        # -------------------------- 修正注意力掩码维度 --------------------------
        # if attention_mask is not None:
        #     # 自注意力需要的掩码形状：[batch_size, 1, seq_len, seq_len]
        #     # 1. 将 [batch_size, seq_len] 扩展为 [batch_size, 1, 1, seq_len]
        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 检查Rotary Embedding
        use_cache: Optional[bool] = None
        cache_position: Optional[torch.LongTensor] = None
        past_key_values: Optional[Cache] = None
        position_ids: Optional[torch.LongTensor] = None
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.model.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[0], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            hidden_states = hidden_states.transpose(1, 0)
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.model.config,
                "input_embeds": hidden_states,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            hidden_states = hidden_states.transpose(1, 0).contiguous()

        # seq_len = hidden_states.size(0)
        # position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        # 计算 Rotary 嵌入（仅 stage 0 计算，传递给后续 stage）
        cos, sin = self.rotary_emb(hidden_states, position_ids)  # [b, s, h]
        cos_sin = torch.cat([cos, sin], dim=-1).transpose(1, 0).contiguous()

        rotary_pos_emb = cos_sin, cos_sin

        # -------------------------- 2. 当前 stage 处理自己的 Transformer 层 --------------------------
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask_mapping[layer.attention_type], rotary_pos_emb=rotary_pos_emb
            )
            # 取元组的第一个元素作为新的 hidden_states（忽略 context 信息）
            hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
            # PP 下无需在层循环中处理 only_last_token（已在 stage 0 处理或后续 stage 保持）
            # if only_last_token:
            #     # 仅保留最后一个token的hidden states，减少后续计算量
            #     hidden_states = hidden_states[:, -1:]  # [batch, 1, hidden_size]
            #     # 对应的rotary_pos_emb也仅保留最后一个位置（若层内依赖）
            #     rotary_pos_emb = rotary_pos_emb[:, -1:] if isinstance(rotary_pos_emb, torch.Tensor) else None

        # 输出
        if self.pp_rank == self.pp_size - 1:
            hidden_states = self.final_norm(hidden_states)
            logits = self.lm_head(hidden_states)
            # [s, b, h] -> [b, s, h]
            logits = logits[0].transpose(1, 0).contiguous()

            # 若仅需最后一个token的logits，直接返回
            if only_last_token:
                logits = logits[:, -1:]  # [batch, 1, vocab_size/tp_size]

            logits = logits.float()
            return logits
        return hidden_states

    @torch.no_grad()  # 生成过程禁用梯度计算
    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int = 512,
            eos_token_id: Optional[int] = None,
            temperature: float = 1.0,
            top_k: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,
            pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成方法（适配 Megatron 并行逻辑）
        Args:
            input_ids: 输入prompt的token id，shape: [batch_size, seq_len]
            max_length: 生成的最大总长度（含prompt）
            eos_token_id: 终止符token id（Qwen2.5默认：151643）
            temperature: 采样温度（<1.0更确定，>1.0更多样）
            top_k: 仅从top-k个高概率token中采样（None表示不限制）
            attention_mask: 输入prompt的attention mask，shape: [batch_size, seq_len]
            pad_token_id: padding token id（用于填充batch中提前终止的样本）
        Returns:
            生成的完整token id，shape: [batch_size, max_length]
        """
        # 1. 初始化配置和设备
        self.eval()  # 生成时切换为评估模式（禁用dropout）
        batch_size, current_seq_len = input_ids.shape
        device = input_ids.device
        eos_token_id = eos_token_id or 151643  # Qwen2.5默认eos_token_id
        pad_token_id = pad_token_id or eos_token_id  # 默认为eos_token_id

        # 2. 初始化attention mask（若未提供）
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.bool()  # 确保是bool类型

        # 3. 检查最大长度合法性
        if max_length <= current_seq_len:
            return input_ids  # 若已达到最大长度，直接返回

        # 4. 初始化生成状态：记录每个样本是否已生成eos
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)  # [batch_size]

        # 5. 预分配生成结果的tensor（避免动态拼接）
        generated_ids = torch.full(
            (batch_size, max_length), pad_token_id, dtype=torch.long, device=device
        )
        generated_ids[:, :current_seq_len] = input_ids  # 填充prompt部分

        # 6. 自回归生成循环
        for step in range(current_seq_len, max_length):
            # a. 获取当前输入（prompt + 已生成的token）
            current_input_ids = generated_ids[:, :step]  # [batch_size, step]
            current_attention_mask = attention_mask[:, :step]  # [batch_size, step]

            batches = TensorDict(
                source={
                    "input_ids": current_input_ids,
                    "attention_mask": current_attention_mask
                },
                batch_size=batch_size)
            # b. 前向传播：仅计算最后一个token的logits（提升效率）
            logits = self.forward_backward_batch(
                batch=batches,
                only_last_token=True,  # 关键优化：仅返回最后一个token的logits
                forward_only=True,
            )  # PP+TP下：LIST[batch_size/pp_size, 1, vocab_size/tp_size]

            # logits = [
            #     # c. 张量并行聚合：收集所有TP进程的logits，得到完整vocab分布
            #     self.gather_logits_across_tp(l) for l in logits]  # [batch_size, 1, vocab_size]

            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # 确保正确聚合所有micro batch的logits
                if isinstance(logits, list):
                    logits = torch.cat(logits, dim=0)  # (batch_size, 1, vocab_size/tp_size)
                logits = logits.to(torch.float32)
            else:
                logits = torch.empty(size=(batch_size, 1, self.vocab_size),
                                     dtype=torch.float32,
                                     device=input_ids.device)

            # utils.print_rank_0(f"before broadcast, logits shape : {logits.shape}")
            # broadcast across pp ranks
            torch.distributed.broadcast(tensor=logits,
                                        src=parallel_state.get_pipeline_model_parallel_last_rank(),
                                        group=parallel_state.get_pipeline_model_parallel_group(),
                                        async_op=False)

            # utils.print_rank_0(f"after broadcast, logits shape : {logits.shape}")

            # d. 应用温度采样和Top-K过滤
            logits = logits / temperature  # 温度调整
            if top_k is not None and top_k > 0:
                logits = self._top_k_filter(logits, top_k)  # Top-K过滤

            # utils.print_rank_0(f"after top_k, logits shape : {logits.shape}")
            # e. 采样得到下一个token（greedy或随机采样）
            next_token_log_probs = torch.softmax(logits, dim=-1)  # [batch_size, 1, vocab_size]
            next_token = torch.multinomial(next_token_log_probs.squeeze(1), num_samples=1)  # [batch_size, 1]

            # f. 处理停止条件：标记已生成eos的样本
            finished_mask = finished_mask | (next_token.squeeze(1) == eos_token_id)
            # 对已完成的样本，填充pad_token_id
            next_token = torch.where(finished_mask.unsqueeze(1), torch.tensor(pad_token_id, device=device), next_token)

            # g. 保存生成的token
            generated_ids[:, step] = next_token.squeeze(1)

            # h. 更新attention_mask（新增token的mask为True）
            new_attention_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            attention_mask = torch.cat([attention_mask, new_attention_mask], dim=-1)  # [batch_size, step+1]

            # i. 若所有样本都已完成，提前退出
            if finished_mask.all():
                break

        return generated_ids

    def forward_backward_batch(self, batch: TensorDict, only_last_token=False,
                               forward_only=False, post_process_fn=None, meta_info: Dict = None):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (seq_len , micro_batch_size, hidden_size)
        """
        # broadcast from last pp rank to all other pp ranks
        # TODO: actually, we just need to control the sampling order.
        batch = batch.contiguous()
        torch_functional.broadcast_dict_tensor(
            batch,
            src=parallel_state.get_pipeline_model_parallel_last_rank(),
            group=parallel_state.get_pipeline_model_parallel_group())

        batches = self.split_dict_tensor_into_batches(batch, batch_size=self.micro_batch_size)
        # compute input shapes for pp stages
        # input_shapes = self.compute_transformers_input_shapes(
        #     batches,
        #     attention_mask,
        #     meta_info={
        #         'hidden_size': self.hidden_size
        #     })
        n_micro_batch = len(batches)
        seq_len = batches[0]["input_ids"].shape[1]

        forward_backward_func = pipeline_parallel.get_forward_backward_func()

        def loss_func(output, data):
            if forward_only:
                if post_process_fn is None:
                    output = self.gather_logits_across_tp(output)
                    return 1.0, output
                else:
                    return 1.0, post_process_fn(output, data)
            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            ref_log_probs = data['ref_log_probs']
            advantages = data['advantages']

            clip_ratio = 0.2
            entropy_coeff = 0.1
            if not meta_info:
                clip_ratio = meta_info['clip_ratio']
                entropy_coeff = meta_info['entropy_coeff']

            # compute policy loss
            logits = output
            logits = logits[:, -response_length - 1:-1].contiguous()
            log_prob = vocab_parallel_log_probs_from_logits(logits, responses)
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=ref_log_probs,
                                                                          log_prob=log_prob,
                                                                          advantages=advantages,
                                                                          eos_mask=response_mask,
                                                                          cliprange=clip_ratio)
            entropy_loss = vocab_parallel_compute_entropy_loss(logits, eos_mask=response_mask)
            policy_loss = pg_loss - entropy_loss * entropy_coeff
            # return loss and stats
            stats = {
                'actor/entropy_loss': entropy_loss.detach().item(),
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item()
            }
            return policy_loss, stats

        def forward_step(batch_iter, model):
            micro_batch = next(batch_iter)
            output = model(input_ids=micro_batch["input_ids"], attention_mask=None,
                           # attention_mask=micro_batch["attention_mask"],
                           only_last_token=only_last_token)
            return output, partial(loss_func, data=micro_batch)

        # batch should be a list of batches inside micro-batches
        batch_generator = self.make_batch_generator(batches, vpp_size=1)

        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self,
                num_microbatches=n_micro_batch,
                seq_length=seq_len,
                # hidden_size=self.model_config.hidden_size,
                micro_batch_size=self.micro_batch_size,
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self,
                num_microbatches=n_micro_batch,
                seq_length=seq_len,
                # hidden_size=self.model_config.hidden_size,
                micro_batch_size=self.micro_batch_size,
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        return losses_reduced

    @staticmethod
    def make_batch_generator(batches, vpp_size):
        if vpp_size > 1:
            # has vpp
            batch_generator = [batches] * vpp_size  # number of vpp chunks
            batch_generator = [iter(b) for b in batch_generator]
        else:
            # no vpp
            batch_generator = iter(batches)
        return batch_generator

    # @staticmethod
    # def compute_transformers_input_shapes(batches, attention_masks, meta_info):
    #     from flash_attn.bert_padding import unpad_input  # flash 2 is a must for Megatron
    #     # pre-compute input shapes for each micro-batch at each pp stage
    #     input_shapes = []
    #     for input_ids, attention_mask in zip(batches, attention_masks):
    #         input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]  # (total_nnz, 1)
    #         # compute shapes for model_inputs
    #         input_shapes.append(torch.Size([input_ids_rmpad.shape[0], 1, meta_info['hidden_size']]))
    #     return input_shapes

    @staticmethod
    def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
        assert tensors.batch_size[0] % batch_size == 0, \
            f'input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}'
        return tensors.split(batch_size)

    def gather_logits_across_tp(self, logits: torch.Tensor) -> torch.Tensor:
        """张量并行（TP）下聚合logits：将各进程的分片logits拼接为完整vocab分布
        Args:
            logits: TP下的分片logits，shape: [batch_size, 1, vocab_size/tp_size]
        Returns:
            完整logits，shape: [batch_size, 1, vocab_size]
        """
        if self.tp_size == 1:
            return logits  # 单卡无需聚合

        # 使用Megatron的tensor_model_parallel_all_gather聚合（跨TP进程）
        gathered_logits = tensor_parallel.gather_from_tensor_model_parallel_region(
            logits,
            group=parallel_state.get_tensor_model_parallel_group()
        )
        return gathered_logits

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-K过滤：将概率排名低于top-k的token的logits设为-无穷（不参与采样）"""
        if top_k >= logits.size(-1):
            return logits  # 若top-k大于vocab_size，无需过滤

        # 获取top-k的logits值和对应的索引
        top_k_values, _ = torch.topk(logits, k=top_k, dim=-1)
        min_top_k_value = top_k_values[:, :, -1:]  # [batch_size, 1, 1]
        # 过滤：低于min_top_k_value的logits设为-无穷
        logits = torch.where(logits >= min_top_k_value, logits, torch.tensor(-float("inf"), device=logits.device))
        return logits

    def detailed_forward_debug(self, input_ids, attention_mask=None):
        """详细的前向传播调试"""
        utils.print_rank_0("=== 详细前向传播调试开始 ===")

        # 原始输入
        utils.print_rank_0(f"输入形状: {input_ids.shape}")
        utils.print_rank_0(f"输入token: {input_ids[0].cpu().numpy()}")

        # 嵌入层
        hidden_states = self.embedding(input_ids.transpose(0, 1))
        utils.print_rank_0(
            f"嵌入层输出 - 形状: {hidden_states.shape}, 均值: {hidden_states.mean():.6f}, 标准差: {hidden_states.std():.6f}")

        # 检查Rotary Embedding
        seq_len = hidden_states.size(0)
        position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
        utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")
        cos_sin = torch.cat([cos, sin], dim=-1).transpose(1, 0).contiguous()

        rotary_pos_emb = cos_sin, cos_sin

        # 逐层检查Transformer
        for layer_idx, layer in enumerate(self.layers):
            utils.print_rank_0(f"\n--- 第{layer_idx}层 ---")

            # 输入统计
            input_mean = hidden_states.mean().item()
            input_std = hidden_states.std().item()
            utils.print_rank_0(f"输入 - 均值: {input_mean:.6f}, 标准差: {input_std:.6f}")

            # 层前向传播
            layer_output = layer(hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)

            # 处理输出（可能是tuple）
            if isinstance(layer_output, tuple):
                utils.print_rank_0(f"层输出是tuple，长度: {len(layer_output)}")
                hidden_states = layer_output[0]
                if len(layer_output) > 1:
                    utils.print_rank_0(
                        f"额外输出: {[x.shape if hasattr(x, 'shape') else type(x) for x in layer_output[1:]]}")
            else:
                hidden_states = layer_output

            # 输出统计
            output_mean = hidden_states.mean().item()
            output_std = hidden_states.std().item()
            utils.print_rank_0(f"输出 - 均值: {output_mean:.6f}, 标准差: {output_std:.6f}")

            # 检查变化
            change = abs(output_mean - input_mean)
            utils.print_rank_0(f"均值变化: {change:.6f}")

            # 检查NaN/Inf
            if torch.isnan(hidden_states).any():
                utils.print_rank_0(f"⚠️  第{layer_idx}层输出包含NaN!")
            if torch.isinf(hidden_states).any():
                utils.print_rank_0(f"⚠️  第{layer_idx}层输出包含Inf!")

            # 只检查前3层，避免输出过多
            if layer_idx >= 2:
                utils.print_rank_0("... (跳过后续层详细输出)")
                break

        # 最终层
        if self.final_norm:
            hidden_states = self.final_norm(hidden_states)
            utils.print_rank_0(f"最终归一化 - 均值: {hidden_states.mean():.6f}, 标准差: {hidden_states.std():.6f}")

        # LM Head
        hidden_states = self.lm_head(hidden_states)
        if isinstance(hidden_states, tuple):
            utils.print_rank_0(f"lm_head输出是tuple，长度: {len(hidden_states)}")
            hidden_states = hidden_states[0]
            if len(hidden_states) > 1:
                utils.print_rank_0(
                    f"lm_head额外输出: {[x.shape if hasattr(x, 'shape') else type(x) for x in hidden_states[1:]]}")
        else:
            hidden_states = hidden_states
        logits = hidden_states.transpose(0, 1)
        utils.print_rank_0(f"最终logits - 形状: {logits.shape}, 均值: {logits.mean():.6f}, 标准差: {logits.std():.6f}")

        # 检查logits的合理性
        top5_values, top5_indices = torch.topk(logits[0, -1], 5)
        utils.print_rank_0(f"最后一个token的top-5 logits:")
        for i, (value, idx) in enumerate(zip(top5_values, top5_indices)):
            utils.print_rank_0(f"  {i + 1}. token {idx.item()}: {value.item():.3f}")

        return logits

    def debug_generation_sampling(self, input_ids, num_steps=3):
        """调试生成过程中的采样"""
        utils.print_rank_0("=== 生成过程采样调试 ===")

        current_input = input_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        for step in range(num_steps):
            utils.print_rank_0(f"\n--- 生成步骤 {step + 1} ---")

            # 前向传播获取logits
            with torch.no_grad():
                logits = self.forward(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    only_last_token=True  # 只获取最后一个token的logits
                )

            utils.print_rank_0(f"Logits形状: {logits.shape}")
            utils.print_rank_0(f"Logits范围: [{logits.min():.3f}, {logits.max():.3f}]")

            # 检查logits的分布
            last_token_logits = logits[:, -1]  # [batch_size, vocab_size]
            probs = torch.softmax(last_token_logits, dim=-1)

            # 采样前的概率分布
            top_probs, top_indices = torch.topk(probs[0], 10)
            utils.print_rank_0("Top-10 概率分布:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token_str = self.tokenizer.decode([idx.item()]) if hasattr(self, 'tokenizer') else str(idx.item())
                utils.print_rank_0(f"  {i + 1}. [{idx.item():6d}] '{token_str}': {prob.item():.4f}")

            # 采样
            next_token = torch.multinomial(probs, num_samples=1)
            utils.print_rank_0(f"采样的下一个token: {next_token[0].item()}")

            # 更新输入
            current_input = torch.cat([current_input, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token, dtype=torch.bool)], dim=1)

            utils.print_rank_0(f"更新后输入长度: {current_input.shape[1]}")

        return current_input

    def run_comprehensive_debug(self, tokenizer):
        """运行全面的调试"""
        utils.print_rank_0("🚀 开始全面调试...")

        # 使用固定的简单输入
        test_prompt = "Hello"
        if tokenizer is not None:
            input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to('cuda')
        else:
            # 如果没有tokenizer，使用简单数字
            input_ids = torch.tensor([[1, 2, 3]], device='cuda')

        utils.print_rank_0(f"测试输入: '{test_prompt}' -> {input_ids.cpu().numpy()}")

        # 1. 详细前向传播
        utils.print_rank_0("\n" + "=" * 50)
        utils.print_rank_0("1. 详细前向传播检查")
        utils.print_rank_0("=" * 50)
        logits = self.detailed_forward_debug(input_ids)

        # 2. 生成过程采样检查
        utils.print_rank_0("\n" + "=" * 50)
        utils.print_rank_0("2. 生成过程采样检查")
        utils.print_rank_0("=" * 50)
        generated = self.debug_generation_sampling(input_ids)

        # 3. 验证最终输出
        if tokenizer is not None:
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            utils.print_rank_0(f"\n最终生成文本: '{generated_text}'")
        else:
            utils.print_rank_0(f"\n最终生成token: {generated[0].cpu().numpy()}")

        # 运行这些调试函数来定位具体问题
        self.debug_attention_mechanism(input_ids)
        self.debug_residual_connections(input_ids)
        self.debug_lm_head_output(input_ids)

        # 运行这些调试来确认问题
        self.debug_mlp_implementation(input_ids)
        self.check_mlp_weights(layer_idx=2)

        utils.print_rank_0("✅ 全面调试完成")

    def debug_attention_mechanism(self, input_ids):
        """调试注意力机制"""
        utils.print_rank_0("=== 注意力机制调试 ===")

        hidden_states = self.embedding(input_ids.transpose(0, 1))
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        # 检查Rotary Embedding
        seq_len = hidden_states.size(0)
        position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
        utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")
        cos_sin = torch.cat([cos, sin], dim=-1).transpose(1, 0).contiguous()

        rotary_pos_emb = cos_sin, cos_sin

        for layer_idx in range(3):  # 只检查前3层
            utils.print_rank_0(f"\n--- 第{layer_idx}层注意力调试 ---")
            layer = self.layers[layer_idx]

            # 检查输入
            input_stats = f"输入: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}"

            # 注意力层前向传播
            attention_output = layer.self_attention(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb
            )

            if isinstance(attention_output, tuple):
                attention_output = attention_output[0]

            utils.print_rank_0(
                f"{input_stats} -> 注意力输出: mean={attention_output.mean():.6f}, std={attention_output.std():.6f}")

            # 检查注意力权重
            if hasattr(layer.self_attention.core_attention, 'attention_scores'):
                scores = layer.self_attention.core_attention.attention_scores
                utils.print_rank_0(f"注意力分数范围: [{scores.min():.3f}, {scores.max():.3f}]")

            # 继续完整层的前向传播
            layer_output = layer(hidden_states)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

    def debug_residual_connections(self, input_ids):
        """调试残差连接和层归一化"""
        utils.print_rank_0("=== 残差连接调试 ===")

        hidden_states = self.embedding(input_ids.transpose(0, 1))
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        # 检查Rotary Embedding
        seq_len = hidden_states.size(0)
        position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
        utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")
        cos_sin = torch.cat([cos, sin], dim=-1).transpose(1, 0).contiguous()

        rotary_pos_emb = cos_sin, cos_sin

        for layer_idx in range(3):
            utils.print_rank_0(f"\n--- 第{layer_idx}层残差连接 ---")
            layer = self.layers[layer_idx]

            # 输入统计
            input_mean = hidden_states.mean().item()
            input_std = hidden_states.std().item()

            utils.print_rank_0(f"input_mean: mean={input_mean:.6f}, input_std={input_std:.6f}")

            # 输入层归一化
            norm_input = layer.input_layernorm(hidden_states)
            utils.print_rank_0(f"输入层归一化: mean={norm_input.mean():.6f}, std={norm_input.std():.6f}")

            # 注意力输出
            attention_output = layer.self_attention(norm_input, attention_mask, rotary_pos_emb)
            if isinstance(attention_output, tuple):
                attention_output = attention_output[0]

            # 残差连接1
            residual1 = hidden_states + attention_output
            utils.print_rank_0(f"残差连接1: mean={residual1.mean():.6f}, std={residual1.std():.6f}")

            # MLP层归一化
            norm_mlp = layer.pre_mlp_layernorm(residual1)
            utils.print_rank_0(f"MLP层归一化: mean={norm_mlp.mean():.6f}, std={norm_mlp.std():.6f}")

            # MLP输出
            mlp_output = layer.mlp(norm_mlp)

            # 残差连接2
            residual2 = residual1 + mlp_output[0]
            utils.print_rank_0(f"残差连接2: mean={residual2.mean():.6f}, std={residual2.std():.6f}")

            hidden_states = residual2

    def debug_lm_head_output(self, input_ids):
        """调试LM Head输出"""
        utils.print_rank_0("=== LM Head输出调试 ===")

        # 完整前向传播到最后一层
        hidden_states = self.embedding(input_ids.transpose(0, 1))
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        # 检查Rotary Embedding
        seq_len = hidden_states.size(0)
        position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
        utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")
        cos_sin = torch.cat([cos, sin], dim=-1).transpose(1, 0).contiguous()

        rotary_pos_emb = cos_sin, cos_sin

        for layer in self.layers:
            layer_output = layer(hidden_states, attention_mask, rotary_pos_emb)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        # 最终归一化
        if self.final_norm:
            hidden_states = self.final_norm(hidden_states)
            utils.print_rank_0(f"最终归一化输出: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

        # LM Head前向传播
        lm_output = self.lm_head(hidden_states)
        utils.print_rank_0(f"LM Head输出类型: {type(lm_output)}")

        if isinstance(lm_output, tuple):
            utils.print_rank_0(f"LM Head tuple长度: {len(lm_output)}")
            for i, _ in enumerate(lm_output):
                output = lm_output[i]
                if output is not None:
                    utils.print_rank_0(
                        f"  输出{i}: 形状={output.shape}, mean={output.mean():.6f}, std={output.std():.6f}")
            # 通常第一个元素是logits
            logits = lm_output[0]
        else:
            logits = lm_output

        logits = logits.transpose(0, 1)
        utils.print_rank_0(f"最终logits: 形状={logits.shape}, mean={logits.mean():.6f}, std={logits.std():.6f}")

        return logits

    def debug_mlp_implementation(self, input_ids):
        """详细调试MLP实现"""
        utils.print_rank_0("=== MLP实现详细调试 ===")

        hidden_states = self.embedding(input_ids.transpose(0, 1))
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        # 检查Rotary Embedding
        seq_len = hidden_states.size(0)
        position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
        utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")
        cos_sin = torch.cat([cos, sin], dim=-1).transpose(1, 0).contiguous()
        rotary_pos_emb = cos_sin, cos_sin

        # 只运行到第2层
        for layer_idx in range(3):
            layer = self.layers[layer_idx]

            # 运行到MLP输入
            attention_output = layer.self_attention(hidden_states, attention_mask, rotary_pos_emb)
            if isinstance(attention_output, tuple):
                attention_output = attention_output[0]
            residual1 = hidden_states + attention_output
            mlp_input = layer.pre_mlp_layernorm(residual1)

            if layer_idx <= 2:  # 重点调试第2层
                utils.print_rank_0(f"\n--- 第{layer_idx}层MLP详细调试 ---")
                utils.print_rank_0(f"MLP输入: mean={mlp_input.mean():.6f}, std={mlp_input.std():.6f}")

                mlp = layer.mlp
                # 检查MLP各组件
                utils.print_rank_0(f"MLP类型: {type(mlp)}")
                utils.print_rank_0(f"MLP配置: gated_linear_unit={mlp.config.gated_linear_unit}")

                # 逐步执行MLP前向传播
                # 1. linear_fc1
                fc1_output = mlp.linear_fc1(mlp_input)[0]
                utils.print_rank_0(
                    f"linear_fc1输出: shape={fc1_output.shape}, mean={fc1_output.mean():.6f}, std={fc1_output.std():.6f}")

                # 2. 检查是否分为gate和up
                if mlp.config.gated_linear_unit:
                    gate, up = torch.chunk(fc1_output, 2, dim=-1)
                    utils.print_rank_0(f"gate部分: shape={gate.shape}, mean={gate.mean():.6f}, std={gate.std():.6f}")
                    utils.print_rank_0(f"up部分: shape={up.shape}, mean={up.mean():.6f}, std={up.std():.6f}")

                    # 3. 激活函数
                    activated_gate = torch.nn.functional.silu(gate)
                    utils.print_rank_0(f"SiLU(gate): mean={activated_gate.mean():.6f}, std={activated_gate.std():.6f}")

                    # 4. 门控乘法
                    intermediate = activated_gate * up
                    utils.print_rank_0(f"gate * up: mean={intermediate.mean():.6f}, std={intermediate.std():.6f}")
                else:
                    intermediate = mlp.activation_func(fc1_output)
                    utils.print_rank_0(f"激活函数输出: mean={intermediate.mean():.6f}, std={intermediate.std():.6f}")

                # 5. linear_fc2
                fc2_output = mlp.linear_fc2(intermediate)[0]
                utils.print_rank_0(f"linear_fc2输出: mean={fc2_output.mean():.6f}, std={fc2_output.std():.6f}")

                # 6. dropout (如果有)
                if hasattr(mlp, 'dropout') and mlp.dropout.p > 0:
                    final_output = mlp.dropout(fc2_output)
                    utils.print_rank_0(f"Dropout后: mean={final_output.mean():.6f}, std={final_output.std():.6f}")
                else:
                    final_output = fc2_output

                utils.print_rank_0(f"MLP最终输出: mean={final_output.mean():.6f}, std={final_output.std():.6f}")

            # 完成这一层
            layer_output = layer(hidden_states)
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

    def check_mlp_weights(self, layer_idx=2):
        """检查MLP层权重"""
        utils.print_rank_0(f"=== 第{layer_idx}层MLP权重检查 ===")

        layer = self.layers[layer_idx]
        mlp = layer.mlp

        # 检查linear_fc1权重
        fc1_weight = mlp.linear_fc1.weight
        fc1_bias = mlp.linear_fc1.bias if hasattr(mlp.linear_fc1, 'bias') else None

        utils.print_rank_0(f"linear_fc1权重: shape={fc1_weight.shape}")
        utils.print_rank_0(f"linear_fc1权重 - mean: {fc1_weight.mean().item():.6f}, std: {fc1_weight.std().item():.6f}")
        utils.print_rank_0(f"linear_fc1权重 - 范围: [{fc1_weight.min().item():.6f}, {fc1_weight.max().item():.6f}]")

        if fc1_bias is not None:
            utils.print_rank_0(f"linear_fc1偏置 - mean: {fc1_bias.mean().item():.6f}, std: {fc1_bias.std().item():.6f}")

        # 检查linear_fc2权重
        fc2_weight = mlp.linear_fc2.weight
        fc2_bias = mlp.linear_fc2.bias if hasattr(mlp.linear_fc2, 'bias') else None

        utils.print_rank_0(f"linear_fc2权重: shape={fc2_weight.shape}")
        utils.print_rank_0(f"linear_fc2权重 - mean: {fc2_weight.mean().item():.6f}, std: {fc2_weight.std().item():.6f}")
        utils.print_rank_0(f"linear_fc2权重 - 范围: [{fc2_weight.min().item():.6f}, {fc2_weight.max().item():.6f}]")

        if fc2_bias is not None:
            utils.print_rank_0(f"linear_fc2偏置 - mean: {fc2_bias.mean().item():.6f}, std: {fc2_bias.std().item():.6f}")


class Qwen2MegatronCritic(Qwen2MegatronModel):
    """PPO Critic模型（价值网络），兼容Megatron TP并行"""

    def __init__(
            self,
            g_config,
            qwen_config: Qwen2Config,
            megatron_config: TransformerConfig,
            freeze_actor_backbone: bool = True,  # 是否冻结Actor底层参数
            use_bias: bool = True  # 价值头是否使用偏置
    ):
        # 调用父类初始化（复用嵌入层、Transformer层、LayerNorm等）
        super().__init__(g_config=g_config, qwen_config=qwen_config, megatron_config=megatron_config)
        self.micro_batch_size = g_config.critic.ppo_micro_batch_size
        self.freeze_actor_backbone = freeze_actor_backbone

        # 2. 价值输出头（兼容TP并行）
        # 输入：Actor的hidden_size（已按TP拆分，每个进程仅持有 hidden_size/TP_size 维度）
        # 输出：标量价值（维度=1，无需TP拆分）

        # 输出层（final_norm + lm_head）：仅 PP 最后一个 stage 初始化
        self.value_head = None
        if self.pp_rank == self.pp_size - 1:
            self.lm_head = None
            self.value_head = nn.Linear(
                in_features=self.hidden_size,
                out_features=1,  # 输出标量价值
                bias=False,
            )

            # 初始化分类头（可选：若从预训练模型加载，可跳过；若随机初始化，建议用Xavier）
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                nn.init.xavier_uniform_(self.value_head.weight)
                if hasattr(self.value_head, 'bias') and self.value_head.bias is not None:
                    nn.init.zeros_(self.value_head.bias)
            else:
                nn.init.zeros_(self.value_head.weight)
                if hasattr(self.value_head, 'bias') and self.value_head.bias is not None:
                    nn.init.zeros_(self.value_head.bias)

            self.value_head.weight.data = self.value_head.weight.data.to('cuda')
            # 广播rank0的参数到所有TP rank
            torch.distributed.broadcast(
                self.value_head.weight.data,
                src=0,  # TP group内的rank0
                group=parallel_state.get_tensor_model_parallel_group()
            )

        # 4. 冻结Actor底层参数（可选，根据训练策略调整）
        if self.freeze_actor_backbone:
            self._freeze_actor_components()

    def _freeze_actor_components(self):
        """冻结Actor的底层特征提取组件，仅训练价值头"""
        if not self.embedding:
            for param in self.embedding.parameters():
                param.requires_grad = False
        if not self.rotary_emb:
            for param in self.rotary_emb.parameters():
                param.requires_grad = False
        for param in self.layers.parameters():
            param.requires_grad = False
        if not self.final_norm:
            self.final_norm.requires_grad_(False)  # RMSNorm通常也冻结

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor = None,
            only_last_token: bool = True,  # PPO默认仅需最后一个token的价值
            normalize_value: bool = False  # 是否对价值预测归一化（稳定训练）
    ) -> torch.Tensor:
        """
        前向传播：输出序列的价值预测
        Args:
            input_ids: [batch_size, seq_len] 输入token ID
            attention_mask: [batch_size, seq_len] 注意力掩码（处理padding）
            only_last_token: 是否仅返回最后一个token的价值（代表整个序列的状态价值）
            normalize_value: 是否对价值预测做LayerNorm（稳定PPO训练）
        Returns:
            value_preds: 价值预测，维度为：
                - only_last_token=True: [batch_size, 1]（PPO标准输出）
                - only_last_token=False: [batch_size, seq_len, 1]（时序价值预测）
        """
        # -------------------------- 1. 第一个 PP stage：执行嵌入层 + Rotary 编码 --------------------------
        # 嵌入 + Rotary 编码
        rotary_pos_emb = None
        if self.pp_rank == 0:
            ori_input_ids = input_ids
            # [b, s, h] -> [s, b, h]
            input_ids = input_ids.transpose(1, 0).contiguous()
            # 1. 嵌入层 + Rotary编码（与Actor完全一致）
            hidden_states = self.embedding(input_ids)  # [batch, seq_len, hidden_size/TP_size]

            ori_input_ids.transpose(1, 0)
        else:
            # self.hidden_states should be passed by Megatron
            hidden_states = self.input_tensor

        # -------------------------- 修正注意力掩码维度 --------------------------
        if attention_mask is not None:
            # 自注意力需要的掩码形状：[batch_size, 1, seq_len, seq_len]
            # 1. 将 [batch_size, seq_len] 扩展为 [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        seq_len = hidden_states.size(0)
        position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
        # 计算 Rotary 嵌入（仅 stage 0 计算，传递给后续 stage）
        cos, sin = self.rotary_emb(hidden_states, position_ids)  # [b, s, h]
        cos_sin = torch.cat([cos, sin], dim=-1).transpose(1, 0).contiguous()

        rotary_pos_emb = cos_sin, cos_sin

        # -------------------------- 2. 当前 stage 处理自己的 Transformer 层 --------------------------
        # 2. Transformer层传播（完整序列，不提前截断，保证特征完整性）
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
            )
            # 取元组的第一个元素作为新的 hidden_states（忽略 context 信息）
            hidden_states = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states

        # 输出
        if self.pp_rank == self.pp_size - 1:
            # 3. 最终归一化
            hidden_states = self.final_norm(hidden_states)

            # 4. 价值预测（标量输出）
            value_preds = self.value_head(hidden_states)  # [batch, seq_len, 1]

            # [s, b, h] -> [b, s, h]
            value_preds = value_preds.transpose(1, 0).contiguous()

            # 5. 仅保留最后一个token的价值（PPO核心用法）
            if only_last_token:
                value_preds = value_preds[:, -1:, :].squeeze(-1)  # [batch, 1]

            # 6. 价值归一化（可选，缓解PPO训练不稳定性）
            if normalize_value:
                value_preds = nn.functional.layer_norm(
                    value_preds, normalized_shape=value_preds.shape[1:], eps=1e-5
                )
            value_preds = value_preds.float()
            value_preds = torch.squeeze(value_preds, dim=-1)
            return value_preds

        return hidden_states

    def forward_backward_batch(self, batch: TensorDict, only_last_token=False,
                               forward_only=False, post_process_fn=None, meta_info: Dict = None):
        # broadcast from last pp rank to all other pp ranks
        batch = batch.contiguous()
        torch_functional.broadcast_dict_tensor(batch,
                                               src=parallel_state.get_pipeline_model_parallel_last_rank(),
                                               group=parallel_state.get_pipeline_model_parallel_group())
        # split into micro-batches
        batch['attention_mask'] = batch['attention_mask'].to(bool)
        batches = self.split_dict_tensor_into_batches(batch, batch_size=self.micro_batch_size)
        n_micro_batch = len(batches)
        seq_len = batches[0]['input_ids'].shape[1]

        forward_backward_func = pipeline_parallel.get_forward_backward_func()

        def loss_func(output, data):
            if forward_only:
                return 1.0, output

            responses = data['responses']
            attention_mask = data['attention_mask']
            values = data['values']
            returns = data['returns']
            response_length = responses.size(1)

            eos_mask = attention_mask[:, -response_length:]

            cliprange_value = meta_info['cliprange_value']

            vpreds = output  # (bs, sequence_length)
            vpreds = vpreds[:, -response_length - 1:-1]

            vf_loss, vf_clipfrac = core_algos.compute_value_loss(vpreds=vpreds,
                                                                 values=values,
                                                                 returns=returns,
                                                                 eos_mask=eos_mask,
                                                                 cliprange_value=cliprange_value)
            stats = {
                'critic/vf_loss': vf_loss.detach().item(),
                'critic/vf_clipfrac': vf_clipfrac.detach().item(),
                'critic/vpred_mean': utils.masked_mean(vpreds, eos_mask).detach().item(),
            }

            return vf_loss, stats

        def forward_step(batch_iter, model):
            micro_batch = next(batch_iter)
            output = model(input_ids=micro_batch["input_ids"], attention_mask=None,
                           only_last_token=only_last_token)
            return output, partial(loss_func, data=batch)

        # batch should be a list of batches inside micro-batches
        batch_generator = self.make_batch_generator(batches, vpp_size=1)

        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self,
                num_microbatches=n_micro_batch,
                seq_length=seq_len,
                # hidden_size=self.model_config.hidden_size,
                micro_batch_size=self.micro_batch_size,
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self,
                num_microbatches=n_micro_batch,
                seq_length=seq_len,
                # hidden_size=self.model_config.hidden_size,
                micro_batch_size=self.micro_batch_size,
                forward_only=forward_only,
            )
        # loss_reduces contains the stats returned from loss_func
        return losses_reduced


def build_qwen2_megatron_model(config, tokenizer, qwen_model_path: str, lora_config: LoraConfig = None, is_critic=False) \
        -> Union[Qwen2MegatronModel, Qwen2MegatronCritic]:
    """构建 Megatron 并行化 Qwen2.5 模型，可集成 Lora"""
    qwen_config = Qwen2Config.from_pretrained(qwen_model_path)

    utils.print_rank_0(f"qwen2 config: {qwen_config}")

    params_dtype = get_params_dtype(config)

    # 配置 Megatron 并行参数（需与 DeepSpeed 对齐）
    megatron_config = TransformerConfig(
        hidden_size=qwen_config.hidden_size,  # 2048
        num_layers=qwen_config.num_hidden_layers,
        num_attention_heads=qwen_config.num_attention_heads,  # 16
        num_query_groups=qwen_config.num_key_value_heads,  # 2
        kv_channels=qwen_config.hidden_size // qwen_config.num_attention_heads,  # 128
        ffn_hidden_size=qwen_config.intermediate_size,
        layernorm_epsilon=qwen_config.rms_norm_eps,
        init_method=torch.nn.init.xavier_uniform_,
        output_layer_init_method=torch.nn.init.xavier_uniform_,
        # activation_func_clamp_value=10.0,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        bf16=config.deepspeed.bf16.enabled,
        fp16=config.deepspeed.fp16.enabled,
        activation_func=ACT2FN[qwen_config.hidden_act],
        gated_linear_unit=True,
        add_bias_linear=False,
        add_qkv_bias=True,
        attention_dropout=qwen_config.attention_dropout,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=True,
    )
    utils.print_rank_0(f"megatron TransformerConfig: {megatron_config}")
    # 加载预训练权重（Hugging Face -> Megatron 格式映射）
    # 注：需手动映射参数名（如 'embed_tokens.weight' -> 'embedding.weight'）
    hf_model = Qwen2ForCausalLM.from_pretrained(qwen_model_path, dtype=params_dtype).cuda()
    utils.print_rank_0(hf_model)
    utils.print_rank_0(f"hf_model qwen2 config: {hf_model.config}")

    # response = tokenizer.decode(hf_model.generate(inputs=tokenizer.encode("Hello World!", return_tensors="pt").to(hf_model.device),
    #                                               max_length=512,
    #                                               max_new_tokens=100,
    #                                               eos_token_id=151643,
    #                                               use_cache=True,
    #                                               pad_token_id=0,)[0],  # 若batch_size>1，需遍历所有样本
    #                         skip_special_tokens=True)
    # utils.print_rank_0(f"qwen2 response: {response}")

    if not is_critic:
        model = Qwen2MegatronModel(config, qwen_config, megatron_config)
        model.cuda()
        qwen_load.load_state_dict_to_megatron_qwen(hf_model.state_dict(), [model], qwen_config,
                                                   megatron_config.params_dtype)
    else:
        model = Qwen2MegatronCritic(config, hf_model.config, megatron_config)
        model.cuda()
        qwen_load.load_state_dict_to_megatron_qwen(hf_model.state_dict(), [model], hf_model.config,
                                                   megatron_config.params_dtype, is_value_model=is_critic)

    # 集成LoRa（仅对 Attention 层的 proj 层添加 LoRa）
    # if lora_config is not None:
    #     target_modules = ["query_proj", "key_proj", "value_proj", "output_proj"]
    #     lora_config.target_modules = target_modules
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters() # 验证 LoRa 训练参数占比（通常 <1%）

    diffs = utils.find_tensor_diff(hf_model.model.embed_tokens.weight, model.embedding.weight)
    utils.print_rank_0(f"model.embedding.weight差异位置：{diffs}")
    for i in range(len(hf_model.model.layers)):
        diffs = utils.find_tensor_diff(hf_model.model.layers[i].input_layernorm.weight,
                                       model.layers[i].input_layernorm.weight)
        utils.print_rank_0(f"model.layers[{i}].input_layernorm.weight：{diffs}")
        hf_qkv = torch.cat(
            [hf_model.model.layers[i].self_attn.q_proj.weight, hf_model.model.layers[i].self_attn.k_proj.weight,
             hf_model.model.layers[i].self_attn.v_proj.weight],
            dim=0)
        diffs = utils.find_tensor_diff(hf_qkv, model.layers[i].self_attention.linear_qkv.weight)
        utils.print_rank_0(f"model.layers[{i}].self_attention.linear_qkv.weight差异位置：{diffs}")
        hf_qkv_bias = torch.cat(
            [hf_model.model.layers[i].self_attn.q_proj.bias, hf_model.model.layers[i].self_attn.k_proj.bias,
             hf_model.model.layers[i].self_attn.v_proj.bias],
            dim=0)
        diffs = utils.find_tensor_diff(hf_qkv_bias, model.layers[i].self_attention.linear_qkv.bias)
        utils.print_rank_0(f"model.layers[{i}].self_attention.linear_qkv.bias差异位置：{diffs}")
        diffs = utils.find_tensor_diff(hf_model.model.layers[i].self_attn.o_proj.weight,
                                       model.layers[i].self_attention.linear_proj.weight)
        utils.print_rank_0(f"model.layers[{i}].self_attention.linear_proj.bias：{diffs}")
        linear_fc1 = torch.cat(
            [hf_model.model.layers[i].mlp.gate_proj.weight, hf_model.model.layers[i].mlp.up_proj.weight],
            dim=0)
        diffs = utils.find_tensor_diff(linear_fc1, model.layers[i].mlp.linear_fc1.weight)
        utils.print_rank_0(f"model.layers[{i}].mlp.linear_fc1.weight：{diffs}")
        diffs = utils.find_tensor_diff(hf_model.model.layers[i].mlp.down_proj.weight,
                                       model.layers[i].mlp.linear_fc2.weight)
        utils.print_rank_0(f"model.layers[{i}].mlp.linear_fc2.weight：{diffs}")
        diffs = utils.find_tensor_diff(hf_model.model.layers[i].post_attention_layernorm.weight,
                                       model.layers[i].pre_mlp_layernorm.weight)
        utils.print_rank_0(f"model.layers[{i}].pre_mlp_layernorm.weight：{diffs}")

    diffs = utils.find_tensor_diff(hf_model.model.norm.weight, model.final_norm.weight)
    utils.print_rank_0(f"model.final_norm.weight差异位置：{diffs}")
    if not is_critic:
        diffs = utils.find_tensor_diff(hf_model.lm_head.weight, model.lm_head.weight)
        utils.print_rank_0(f"model.lm_head.weight差异位置：{diffs}")

    run_comprehensive_debug(hf_model, tokenizer)
    return model


def get_params_dtype(config):
    params_dtype = torch.get_default_dtype()
    if config.deepspeed.bf16.enabled:
        params_dtype = torch.bfloat16
    if config.deepspeed.fp16.enabled:
        params_dtype = torch.float16
    return params_dtype


def detailed_forward_debug(self, input_ids, attention_mask=None):
    """详细的前向传播调试"""
    utils.print_rank_0("=== hf_model 详细前向传播调试开始 ===")

    # 原始输入
    utils.print_rank_0(f"输入形状: {input_ids.shape}")
    utils.print_rank_0(f"输入token: {input_ids[0].cpu().numpy()}")

    # 嵌入层
    hidden_states = self.model.embed_tokens(input_ids)
    utils.print_rank_0(
        f"嵌入层输出 - 形状: {hidden_states.shape}, 均值: {hidden_states.mean():.6f}, 标准差: {hidden_states.std():.6f}")

    # 检查Rotary Embedding
    use_cache: Optional[bool] = None
    cache_position: Optional[torch.LongTensor] = None
    past_key_values: Optional[Cache] = None
    position_ids: Optional[torch.LongTensor] = None
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.model.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    rotary_pos_emb = self.model.rotary_emb(hidden_states, position_ids)
    cos, sin = rotary_pos_emb
    utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
    utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.model.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }

    # 逐层检查Transformer
    for layer_idx, layer in enumerate(self.model.layers):
        utils.print_rank_0(f"\n--- 第{layer_idx}层 ---")

        # 输入统计
        input_mean = hidden_states.mean().item()
        input_std = hidden_states.std().item()
        utils.print_rank_0(f"输入 - 均值: {input_mean:.6f}, 标准差: {input_std:.6f}")

        # 层前向传播
        layer_output = layer(
            hidden_states,
            attention_mask=causal_mask_mapping[layer.attention_type],
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=rotary_pos_emb,
            # **kwargs,
        )

        # 处理输出（可能是tuple）
        if isinstance(layer_output, tuple):
            utils.print_rank_0(f"层输出是tuple，长度: {len(layer_output)}")
            hidden_states = layer_output[0]
            if len(layer_output) > 1:
                utils.print_rank_0(
                    f"额外输出: {[x.shape if hasattr(x, 'shape') else type(x) for x in layer_output[1:]]}")
        else:
            hidden_states = layer_output

        # 输出统计
        output_mean = hidden_states.mean().item()
        output_std = hidden_states.std().item()
        utils.print_rank_0(f"输出 - 均值: {output_mean:.6f}, 标准差: {output_std:.6f}")

        # 检查变化
        change = abs(output_mean - input_mean)
        utils.print_rank_0(f"均值变化: {change:.6f}")

        # 检查NaN/Inf
        if torch.isnan(hidden_states).any():
            utils.print_rank_0(f"⚠️  第{layer_idx}层输出包含NaN!")
        if torch.isinf(hidden_states).any():
            utils.print_rank_0(f"⚠️  第{layer_idx}层输出包含Inf!")

        # 只检查前3层，避免输出过多
        if layer_idx >= 2:
            utils.print_rank_0("... (跳过后续层详细输出)")
            break

    # 最终层
    if self.model.norm:
        hidden_states = self.model.norm(hidden_states)
        utils.print_rank_0(f"最终归一化 - 均值: {hidden_states.mean():.6f}, 标准差: {hidden_states.std():.6f}")

    # LM Head
    hidden_states = self.lm_head(hidden_states)
    if isinstance(hidden_states, tuple):
        utils.print_rank_0(f"lm_head输出是tuple，长度: {len(hidden_states)}")
        hidden_states = hidden_states[0]
        if len(hidden_states) > 1:
            utils.print_rank_0(
                f"lm_head额外输出: {[x.shape if hasattr(x, 'shape') else type(x) for x in hidden_states[1:]]}")
    else:
        hidden_states = hidden_states
    logits = hidden_states
    utils.print_rank_0(f"最终logits - 形状: {logits.shape}, 均值: {logits.mean():.6f}, 标准差: {logits.std():.6f}")

    # 检查logits的合理性
    top5_values, top5_indices = torch.topk(logits[0, -1], 5)
    utils.print_rank_0(f"最后一个token的top-5 logits:")
    for i, (value, idx) in enumerate(zip(top5_values, top5_indices)):
        utils.print_rank_0(f"  {i + 1}. token {idx.item()}: {value.item():.3f}")

    return logits


def run_comprehensive_debug(self, tokenizer):
    """运行全面的调试"""
    utils.print_rank_0("🚀 hf_model 开始全面调试...")

    # 使用固定的简单输入
    test_prompt = "Hello"
    if tokenizer is not None:
        input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to('cuda')
    else:
        # 如果没有tokenizer，使用简单数字
        input_ids = torch.tensor([[1, 2, 3]], device='cuda')

    utils.print_rank_0(f"测试输入: '{test_prompt}' -> {input_ids.cpu().numpy()}")

    # 1. 详细前向传播
    utils.print_rank_0("\n" + "=" * 50)
    utils.print_rank_0("1. 详细前向传播检查")
    utils.print_rank_0("=" * 50)
    logits = detailed_forward_debug(self, input_ids)

    # # 2. 生成过程采样检查
    # utils.print_rank_0("\n" + "=" * 50)
    # utils.print_rank_0("2. 生成过程采样检查")
    # utils.print_rank_0("=" * 50)
    # generated = self.debug_generation_sampling(input_ids)

    # 3. 验证最终输出
    # if tokenizer is not None:
    #     generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    #     utils.print_rank_0(f"\n最终生成文本: '{generated_text}'")
    # else:
    #     utils.print_rank_0(f"\n最终生成token: {generated[0].cpu().numpy()}")

    # # 运行这些调试函数来定位具体问题
    debug_attention_mechanism(self, input_ids)
    # self.debug_residual_connections(input_ids)
    # self.debug_lm_head_output(input_ids)

    # 运行这些调试来确认问题
    debug_mlp_implementation(self, input_ids)
    # self.check_mlp_weights(layer_idx=2)

    utils.print_rank_0("✅ hf_model 全面调试完成")


def debug_mlp_implementation(self, input_ids):
    """详细调试MLP实现"""
    utils.print_rank_0("=== MLP实现详细调试 ===")

    hidden_states = self.model.embed_tokens(input_ids)
    attention_mask: Optional[torch.Tensor] = None
    # 检查Rotary Embedding
    use_cache: Optional[bool] = None
    cache_position: Optional[torch.LongTensor] = None
    past_key_values: Optional[Cache] = None
    position_ids: Optional[torch.LongTensor] = None
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.model.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    rotary_pos_emb = self.model.rotary_emb(hidden_states, position_ids)
    cos, sin = rotary_pos_emb
    utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
    utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.model.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }

    # 只运行到第2层
    for layer_idx in range(3):
        layer = self.model.layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)

        # 运行到MLP输入
        attention_output = layer.self_attn(hidden_states=hidden_states,
                                           position_embeddings=rotary_pos_emb,
                                           attention_mask=causal_mask_mapping[layer.attention_type],
                                           past_key_values=past_key_values,
                                           cache_position=cache_position,
                                           use_cache=use_cache,
                                           position_ids=position_ids,
                                           )
        if isinstance(attention_output, tuple):
            attention_output = attention_output[0]
        residual1 = hidden_states + attention_output
        mlp_input = layer.post_attention_layernorm(residual1)

        if layer_idx <= 2:  # 重点调试第2层
            utils.print_rank_0(f"\n--- 第{layer_idx}层MLP详细调试 ---")
            utils.print_rank_0(f"MLP输入: mean={mlp_input.mean():.6f}, std={mlp_input.std():.6f}")

            mlp = layer.mlp
            # 检查MLP各组件
            utils.print_rank_0(f"MLP类型: {type(mlp)}")

            # 2. 分为gate和up
            gate = mlp.gate_proj(mlp_input)
            up = mlp.up_proj(mlp_input)
            utils.print_rank_0(f"gate部分: shape={gate.shape}, mean={gate.mean():.6f}, std={gate.std():.6f}")
            utils.print_rank_0(f"up部分: shape={up.shape}, mean={up.mean():.6f}, std={up.std():.6f}")

            # 3. 激活函数
            activated_gate = mlp.act_fn(gate)
            utils.print_rank_0(f"SiLU(gate): mean={activated_gate.mean():.6f}, std={activated_gate.std():.6f}")

            # 4. 门控乘法
            intermediate = activated_gate * up
            utils.print_rank_0(f"gate * up: mean={intermediate.mean():.6f}, std={intermediate.std():.6f}")

            # 5. linear_fc2
            fc2_output = mlp.down_proj(intermediate)
            utils.print_rank_0(f"linear_fc2输出: mean={fc2_output.mean():.6f}, std={fc2_output.std():.6f}")

            # 6. dropout (如果有)
            if hasattr(mlp, 'dropout') and mlp.dropout.p > 0:
                final_output = mlp.dropout(fc2_output)
                utils.print_rank_0(f"Dropout后: mean={final_output.mean():.6f}, std={final_output.std():.6f}")
            else:
                final_output = fc2_output

            utils.print_rank_0(f"MLP最终输出: mean={final_output.mean():.6f}, std={final_output.std():.6f}")

        # 完成这一层
        layer_output = layer(hidden_states,
                             attention_mask=causal_mask_mapping[layer.attention_type],
                             position_ids=position_ids,
                             past_key_values=past_key_values,
                             use_cache=use_cache,
                             cache_position=cache_position,
                             position_embeddings=rotary_pos_emb,
                             )
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output


def debug_attention_mechanism(self, input_ids):
    """调试注意力机制"""
    utils.print_rank_0("=== 注意力机制调试 ===")

    hidden_states = self.model.embed_tokens(input_ids)
    attention_mask: Optional[torch.Tensor] = None
    # 检查Rotary Embedding
    use_cache: Optional[bool] = None
    cache_position: Optional[torch.LongTensor] = None
    past_key_values: Optional[Cache] = None
    position_ids: Optional[torch.LongTensor] = None
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.model.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    rotary_pos_emb = self.model.rotary_emb(hidden_states, position_ids)
    cos, sin = rotary_pos_emb
    utils.print_rank_0(f"Rotary cos - 形状: {cos.shape}, 范围: [{cos.min():.3f}, {cos.max():.3f}]")
    utils.print_rank_0(f"Rotary sin - 形状: {sin.shape}, 范围: [{sin.min():.3f}, {sin.max():.3f}]")

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.model.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }

    for layer_idx in range(3):  # 只检查前3层
        utils.print_rank_0(f"\n--- 第{layer_idx}层注意力调试 ---")
        layer = self.model.layers[layer_idx]

        # 检查输入
        input_stats = f"输入: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}"

        # 注意力层前向传播
        attention_output = layer.self_attn(hidden_states=hidden_states,
                                           position_embeddings=rotary_pos_emb,
                                           attention_mask=causal_mask_mapping[layer.attention_type],
                                           past_key_values=past_key_values,
                                           cache_position=cache_position,
                                           use_cache=use_cache,
                                           position_ids=position_ids,
                                           )

        if isinstance(attention_output, tuple):
            attention_output = attention_output[0]

        utils.print_rank_0(
            f"{input_stats} -> 注意力输出: mean={attention_output.mean():.6f}, std={attention_output.std():.6f}")

        # 检查注意力权重
        if hasattr(layer.self_attn, 'attention_scores'):
            scores = layer.self_attn.attention_scores
            utils.print_rank_0(f"注意力分数范围: [{scores.min():.3f}, {scores.max():.3f}]")

        # 继续完整层的前向传播
        layer_output = layer(hidden_states,
                             attention_mask=causal_mask_mapping[layer.attention_type],
                             position_ids=position_ids,
                             past_key_values=past_key_values,
                             use_cache=use_cache,
                             cache_position=cache_position,
                             position_embeddings=rotary_pos_emb,
                             )
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
        else:
            hidden_states = layer_output
