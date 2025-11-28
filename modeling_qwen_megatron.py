from functools import partial
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core import parallel_state, tensor_parallel, pipeline_parallel
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
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

from utils import utils, torch_functional
from tensor_parallel import vocab_parallel_log_probs_from_logits, vocab_parallel_compute_entropy_loss

import core_algos
import qwen_load
from tensordict import TensorDict


class Qwen2MegatronAttention(SelfAttention):
    def __init__(self,
                 config: TransformerConfig,
                 layer_number: int,
                 attn_mask_type=AttnMaskType.padding,
                 cp_comm_type: str = None,
                 pg_collection: ProcessGroupCollection = None
                 ):
        super().__init__(
            config,
            submodules=SelfAttentionSubmodules(
                linear_qkv=tensor_parallel.ColumnParallelLinear,
                core_attention=DotProductAttention,
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
        self.config.gated_linear_unit = True
        mlp = MLPSubmodules(
            linear_fc1=tensor_parallel.ColumnParallelLinear,  # Qwen2.5 gate_proj, up_proj
            linear_fc2=tensor_parallel.RowParallelLinear,  # Qwen2.5 down_proj
        )
        super().__init__(config, submodules=mlp)
        # Qwen2.5 用SwiGLU， 替换 Megatron 默认 GELU
        self.activation_func = nn.SiLU  # SwiLU = SilU + 点积


@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, config, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
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
    def __init__(self, config: TransformerConfig, layer_number: int = 1):
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
            Qwen2MegatronTransformerLayer(megatron_config, i + 1)
            for i in range(self.start_layer, self.end_layer)
        ])

        # 输出层（final_norm + lm_head）：仅 PP 最后一个 stage 初始化
        self.final_norm = None
        self.lm_head = None
        if self.pp_rank == self.pp_size - 1:
            self.final_norm = Qwen2RMSNorm(g_config, self.hidden_size, eps=qwen_config.rms_norm_eps)
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
            # -------------------------- 修正注意力掩码维度 --------------------------
            if attention_mask is not None:
                # 自注意力需要的掩码形状：[batch_size, 1, seq_len, seq_len]
                # 1. 将 [batch_size, seq_len] 扩展为 [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            seq_len = hidden_states.size(0)
            position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
            # 计算 Rotary 嵌入（仅 stage 0 计算，传递给后续 stage）
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            cos_sin = torch.cat([cos, sin], dim=-1)
            rotary_pos_emb = cos_sin, cos_sin

            ori_input_ids.transpose(1, 0)

            # 提前处理 only_last_token（减少跨 stage 通信量）
            # if only_last_token:
            #     hidden_states = hidden_states[:, -1:]  # [batch, 1, hidden_size/tp_size]
            #     rotary_pos_emb = rotary_pos_emb[:, -1:] if isinstance(rotary_pos_emb, torch.Tensor) else None
        else:
            # self.hidden_states should be passed by Megatron
            hidden_states = self.input_tensor

        # -------------------------- 2. 当前 stage 处理自己的 Transformer 层 --------------------------
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb
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
                # only on last rank. It should be on every tp rank
                logits = torch.cat([o for o in logits], dim=0)  # (bs, seq_size)
                logits = logits.to(torch.float32)
            else:
                logits = torch.empty(size=(batch_size, step),
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
            output = model(input_ids=micro_batch["input_ids"], attention_mask=micro_batch["attention_mask"],
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
            # -------------------------- 修正注意力掩码维度 --------------------------
            if attention_mask is not None:
                # 自注意力需要的掩码形状：[batch_size, 1, seq_len, seq_len]
                # 1. 将 [batch_size, seq_len] 扩展为 [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            seq_len = hidden_states.size(0)
            position_ids = torch.arange(0, seq_len, device=hidden_states.device).unsqueeze(0)
            # 计算 Rotary 嵌入（仅 stage 0 计算，传递给后续 stage）
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            rotary_pos_emb = cos.transpose(1, 0).contiguous(), sin.transpose(1, 0).contiguous()

            ori_input_ids.transpose(1, 0)
        else:
            # self.hidden_states should be passed by Megatron
            hidden_states = self.input_tensor

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
            output = model(input_ids=micro_batch["input_ids"], attention_mask=micro_batch["attention_mask"],
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


def build_qwen2_megatron_model(config, qwen_model_path: str, lora_config: LoraConfig = None, is_critic=False) \
        -> Union[Qwen2MegatronModel, Qwen2MegatronCritic]:
    """构建 Megatron 并行化 Qwen2.5 模型，可集成 Lora"""
    qwen_config = Qwen2Config.from_pretrained(qwen_model_path)

    params_dtype = torch.get_default_dtype()
    if config.deepspeed.bf16.enabled:
        params_dtype = torch.bfloat16
    if config.deepspeed.fp16.enabled:
        params_dtype = torch.float16

    # 配置 Megatron 并行参数（需与 DeepSpeed 对齐）
    megatron_config = TransformerConfig(
        hidden_size=qwen_config.hidden_size,
        num_layers=qwen_config.num_hidden_layers,
        num_attention_heads=qwen_config.num_attention_heads,
        num_query_groups=qwen_config.num_key_value_heads,
        kv_channels=qwen_config.hidden_size // qwen_config.num_attention_heads,
        ffn_hidden_size=qwen_config.intermediate_size,
        layernorm_epsilon=qwen_config.rms_norm_eps,
        init_method=torch.nn.init.xavier_uniform_,
        tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        bf16=config.deepspeed.bf16.enabled,
        fp16=config.deepspeed.fp16.enabled,
    )
    # 加载预训练权重（Hugging Face -> Megatron 格式映射）
    # 注：需手动映射参数名（如 'embed_tokens.weight' -> 'embedding.weight'）
    hf_model = Qwen2ForCausalLM.from_pretrained(qwen_model_path)
    if not is_critic:
        model = Qwen2MegatronModel(config, qwen_config, megatron_config)
        model.cuda()
        qwen_load.load_state_dict_to_megatron_qwen(hf_model.state_dict(), [model], qwen_config,
                                                   megatron_config.params_dtype)
    else:
        model = Qwen2MegatronCritic(config, qwen_config, megatron_config)
        model.cuda()
        qwen_load.load_state_dict_to_megatron_qwen(hf_model.state_dict(), [model], qwen_config,
                                                   megatron_config.params_dtype, is_value_model=is_critic)

    # 集成LoRa（仅对 Attention 层的 proj 层添加 LoRa）
    # if lora_config is not None:
    #     target_modules = ["query_proj", "key_proj", "value_proj", "output_proj"]
    #     lora_config.target_modules = target_modules
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters() # 验证 LoRa 训练参数占比（通常 <1%）
    return model
