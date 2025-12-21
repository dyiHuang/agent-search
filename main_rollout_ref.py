import inspect
import time
from typing import Dict, Any

import hydra
import numpy as np
import ray
import torch
from transformers import PretrainedConfig
from vllm import LLM, SamplingParams


def init_ray_and_actor(qwen_model_path):
    """
    仅LOCAL_RANK=0的主进程：
    1. 初始化Ray并分配num_gpus张GPU
    2. 创建TP=num_gpus的VLLMActor
    其他进程：仅连接Ray
    """
    num_gpus = 1
    rank = 0

    ray.init(
        address=None,
        ignore_reinit_error=True,
        local_mode=False,  # 必须关闭local_mode，否则无法多卡TP
        num_cpus=16,
        num_gpus=num_gpus,  # 为Ray集群分配num_gpus张GPU
        # _temp_dir="/tmp/ray-tp4",
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0",  # 明确指定num_gpus张GPU给Actor
                "TRUST_REMOTE_CODE": "True",
            }
        },
        _node_ip_address="10.60.114.169",
        # 关键：允许Ray跨进程共享Actor
        # _node_ip_address="127.0.0.1"
    )

    @ray.remote(num_gpus=num_gpus)  # 必须设为num_gpus，匹配TP=num_gpus
    class VLLMActor:
        def __init__(self, model_name):
            # Actor内初始化vLLM，TP=num_gpus（独占num_gpus张GPU）
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=num_gpus,  # 张量并行卡数
                dtype=torch.bfloat16,
                # 新增显存优化参数
                # gpu_memory_utilization=0.4,  # 降低显存利用率，避免分配超时
                enforce_eager=True,  # 禁用CUDA图，减少初始化阻塞
                skip_tokenizer_init=False,
                # max_num_batched_tokens=4096  # 批处理优化
                trust_remote_code=True,  # 必设为True！原False是核心问题
                enable_chunked_prefill=False,  # 禁用chunked prefill（解决长序列兼容）
            )
            # 获取vllm底层的模型核心（TP分片后的模型）
            print(dir(self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model))
            print(f"{self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model}")
            self.vllm_model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
            self.hf_config: PretrainedConfig = self.llm.llm_engine.model_config.hf_config

        def generate_from_tensor(self, input_ids_cpu, sampling_params: SamplingParams):
            """接收cpu张量，返回输出token的cpu张量"""
            # input_ids = input_ids_cpu.to(f"cuda:{rank}")  # TP=num_gpus时，主卡为cuda:0
            # 修复1：张量转Python列表（先确保是CPU张量，再转列表）
            # 若input_ids_cpu是一维（单条数据），转为二维：[[token1, token2, ...]]
            if len(input_ids_cpu.shape) == 1:
                token_ids = input_ids_cpu.tolist()  # 一维转列表：[101, 200, 300]
                prompt_token_ids = [token_ids]  # 转为二维：[[101, 200, 300]]
            # 若已是二维（多条数据），直接转列表
            elif len(input_ids_cpu.shape) == 2:
                prompt_token_ids = input_ids_cpu.tolist()
            else:
                raise ValueError(f"input_ids_cpu维度错误，仅支持1D/2D，当前：{input_ids_cpu.shape}")
            outputs = self.llm.generate(
                prompts=None,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params
            )
            # 遍历所有批次的输出，拼接为二维结构（核心修复）
            output_token_list_batch = []
            batch_logits = []
            for output in outputs:  # 遍历每个批次的生成结果（共batch_size个）
                token_ids = output.outputs[0].token_ids
                sample_logits = output.outputs[0].logits
                # 核心修复：将元组转为列表
                token_ids = list(token_ids)  # 元组→列表，比如 (300,400) → [300,400]
                output_token_list_batch.append(token_ids)
                batch_logits.append(sample_logits)

            # 转为二维张量（shape=[batch_size, gen_len]）
            # 注意：若各批次生成长度不同，需padding到相同长度（可选，根据你的业务需求）
            max_gen_len = max(len(tokens) for tokens in output_token_list_batch)
            padded_outputs = []
            for tokens in output_token_list_batch:
                # 补0到最大长度（或其他padding token）
                padded = tokens + [151643] * (max_gen_len - len(tokens))
                padded_outputs.append(padded)

            pad_value = 0.0
            padded_logits = []
            vocab_size = self.llm.llm_engine.model_config.vocab_size  # 获取词表大小
            for logits in batch_logits:
                # 计算需要padding的长度
                pad_len = max_gen_len - len(logits)
                if pad_len > 0:
                    # 对vocab_size维度padding（shape=[pad_len, vocab_size]）
                    padding = np.full((pad_len, vocab_size), pad_value, dtype=np.float32)
                    padded = np.concatenate([logits, padding], axis=0)
                else:
                    padded = logits
                padded_logits.append(padded)

            # 转为2维CPU张量（匹配输入的批次维度）
            logits_tensor = torch.tensor(padded_logits, dtype=torch.bfloat16)
            output_token_ids = torch.tensor(padded_outputs, dtype=torch.long)  # shape=[batch_size, max_gen_len]

            return output_token_ids, logits_tensor

        def sync_model_params(self, state_dict: Dict[str, Any], tp_rank, tp_size):
            full_state_dict = self.vllm_model.named_parameters()
            print(f"rank:{rank} state_dict:{state_dict.keys()}")
            if tp_rank == 0:
                print(f"full_state_dict:{full_state_dict.keys}")
                with torch.no_grad():
                    for k, v in full_state_dict.items():
                        print(f"k:{k}, mean: {v.mean():.6f}, std: {v.std():.6f}")

            with torch.no_grad():
                for k, v in state_dict.items():
                    if k == "embedding.weight":
                        local_param = full_state_dict["model.embed_tokens.weight"]
                        self.param_copy(local_param, v, 0, tp_rank, tp_size)
                        continue

                    if k == "final_norm.weight":
                        local_param = full_state_dict["model.norm.weight"]
                        local_param.data.copy_(v.data)
                        continue

                    if k == "lm_head.weight":
                        local_param = full_state_dict["lm_head.weight"]
                        local_param.data.copy_(v.data)
                        continue

                    import re
                    match = re.match("layers.{}.self_attention.linear_qkv.weight".replace("{}", r"(\d+)"), k)
                    if match:
                        layer_idx = match.group(1)
                        layer_idx = int(layer_idx) - 1
                        hidden_size = self.hf_config.hidden_size
                        qkv_param = full_state_dict["model.layers.{}.self_attn.qkv_proj.weight".format(layer_idx)]
                        qk_size = (qkv_param.size(0) - hidden_size) // 2
                        q_param = qkv_param[0:hidden_size, :]
                        k_param = qkv_param[hidden_size:hidden_size + qk_size, :]
                        v_param = qkv_param[hidden_size + qk_size:hidden_size + 2*qk_size, :]
                        q_size = q_param.size(0) // tp_size
                        k_size = k_param.size(0) // tp_size
                        v_size = v_param.size(0) // tp_size
                        self.param_copy(q_param, v[0:q_size, :], 0, tp_rank, tp_size)
                        self.param_copy(k_param, v[q_size:q_size+k_size, :], 0, tp_rank, tp_size)
                        self.param_copy(v_param, v[q_size+k_size:q_size+k_size+v_size, :], 0, tp_rank, tp_size)
                        continue
                    match = re.match("layers.{}.self_attention.linear_qkv.bias".replace("{}", r"(\d+)"), k)
                    if match:
                        layer_idx = match.group(1)
                        layer_idx = int(layer_idx) - 1
                        hidden_size = self.hf_config.hidden_size
                        qkv_param = full_state_dict["model.layers.{}.self_attn.qkv_proj.bias".format(layer_idx)]
                        qk_size = (qkv_param.size(0) - hidden_size) // 2
                        q_param = qkv_param[0:hidden_size]
                        k_param = qkv_param[hidden_size:hidden_size + qk_size]
                        v_param = qkv_param[hidden_size + qk_size:hidden_size + 2 * qk_size]
                        q_size = q_param.size(0) // tp_size
                        k_size = k_param.size(0) // tp_size
                        v_size = v_param.size(0) // tp_size
                        self.param_copy(q_param, v[0:q_size], 0, tp_rank, tp_size)
                        self.param_copy(k_param, v[q_size:q_size+k_size], 0, tp_rank, tp_size)
                        self.param_copy(v_param, v[q_size+k_size:q_size+k_size+v_size], 0, tp_rank, tp_size)
                        continue

                    match = re.match("layers.{}.self_attention.linear_proj.weight".replace("{}", r"(\d+)"), k)
                    if match:
                        layer_idx = match.group(1)
                        layer_idx = int(layer_idx) - 1
                        o_param = full_state_dict["model.layers.{}.self_attn.o_proj.weight".format(layer_idx)]
                        self.param_copy(o_param, v, 1, tp_rank, tp_size)
                        continue

                    match = re.match("layers.{}.mlp.linear_fc1.weight".replace("{}", r"(\d+)"), k)
                    if match:
                        layer_idx = match.group(1)
                        layer_idx = int(layer_idx) - 1
                        intermediate_size = self.hf_config.intermediate_size
                        gate_up_proj_param = full_state_dict["model.layers.{}.mlp.gate_up_proj.weight".format(layer_idx)]
                        gate_param = gate_up_proj_param[0:intermediate_size, :]
                        up_param = gate_up_proj_param[intermediate_size:, :]
                        gate_size = gate_param.size(0) // tp_size
                        up_size = up_param.size(0) // tp_size
                        self.param_copy(gate_param, v[0:gate_size, :], 0, tp_rank, tp_size)
                        self.param_copy(up_param, v[gate_size:gate_size + up_size, :], 0, tp_rank, tp_size)
                        continue

                    match = re.match("layers.{}.mlp.linear_fc2.weight".replace("{}", r"(\d+)"), k)
                    if match:
                        layer_idx = match.group(1)
                        layer_idx = int(layer_idx) - 1
                        down_param = full_state_dict["model.layers.{}.mlp.down_proj.weight".format(layer_idx)]
                        self.param_copy(down_param, v, 1, tp_rank, tp_size)
                        continue

                    match = re.match("layers.{}.input_layernorm.weight".replace("{}", r"(\d+)"), k)
                    if match:
                        layer_idx = match.group(1)
                        layer_idx = int(layer_idx) - 1
                        local_param = full_state_dict["model.layers.{}.input_layernorm.weight".format(layer_idx)]
                        local_param.data.copy_(v.data)
                        continue

                    match = re.match("layers.{}.pre_mlp_layernorm.weight".replace("{}", r"(\d+)"), k)
                    if match:
                        layer_idx = match.group(1)
                        layer_idx = int(layer_idx) - 1
                        local_param = full_state_dict["model.layers.{}.post_attention_layernorm.weight".format(layer_idx)]
                        local_param.data.copy_(v.data)
                        continue

                    print(k)

            # 清空vllm推理缓存（关键：让新参数生效）
            # self.llm.llm_engine.cache_manager.clear_all()
            torch.cuda.empty_cache()

            return

        @staticmethod
        def param_copy(local_param, remote_pram, dim, tp_rank, tp_size):
            split_size = local_param.size(dim) // tp_size
            start = tp_rank * split_size
            end = (tp_rank + 1) * split_size
            if dim == 0:
                if len(local_param.shape) == 2:
                    local_param[start:end, :].data.copy_(remote_pram.data)
                if len(local_param.shape) == 1:
                    local_param[start:end].data.copy_(remote_pram.data)
            if dim == 1:
                local_param[:, start:end].data.copy_(remote_pram.data)

        def _vllm_to_hf_name(self, vllm_name):
            """vllm参数名 → HF参数名映射（适配Qwen2.5-7B）"""
            # mapping = {
            #     "embedding.weight": "model.embed_tokens.weight",
            #     "layers.{}.self_attention.linear_qkv.weight": "model.layers.{}.self_attn.q_proj.weight",
            #     "layers.{}.self_attention.linear_qkv.weight": "model.layers.{}.self_attn.q_proj.bias",
            #     "layers.{}.self_attention.linear_qkv.weight": "model.layers.{}.self_attn.k_prob.weight",
            #     "layers.{}.self_attention.linear_qkv.weight": "model.layers.{}.self_attn.k_prob.bias",
            #     "layers.{}.self_attention.linear_qkv.weight": "model.layers.{}.self_attn.v_proj.weight",
            #     "layers.{}.self_attention.linear_qkv.weight": "model.layers.{}.self_attn.v_proj.bias",
            #     "layers.{}.self_attention.linear_proj.weight": "model.layers.{}.self_attn.o_proj.weight",
            #     "layers.{}.mlp.linear_fc1.weight": "model.layers.{}.mlp.gate_proj.weight",
            #     "layers.{}.mlp.linear_fc1.weight": "model.layers.{}.mlp.up_proj.weight",
            #     "layers.{}.mlp.linear_fc2.weight": "model.layers.{}.mlp.down_proj.weight",
            #     "layers.{}.input_layernorm.weight": "model.layers.{}.input_layernorm.weight",
            #     "layers.{}.pre_mlp_layernorm.weight": "model.layers.{}.post_attention_layernorm.weight",
            #     "final_norm.weight": "model.norm.weight",
            #     "lm_head.weight": "lm_head.weight"
            # }
            #
            return vllm_name

    # 创建Actor实例，获取全局引用
    vllm_actor_ref = VLLMActor.options(
        name="VLLMActor",  # 命名Actor（适配所有2.0+版本）
        namespace="ppo_train",
        lifetime="detached",  # 持久化（Ray 2.0+支持）
        max_restarts=3  # 重启策略
    ).remote(qwen_model_path)
    # 等待Actor初始化完成（避免其他进程调用时未就绪）
    ray.get(vllm_actor_ref.__ray_ready__.remote())
    print(f"[Rank {rank}] VLLMActor (TP={num_gpus}) initialized")
    return vllm_actor_ref


@hydra.main(config_path='config', config_name='ppo_megatron_trainer', version_base=None)
def main(config):
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    vllm_actor_ref = init_ray_and_actor(config.qwen_model_path)

    print("Actor初始化完成，程序持续运行中（按Ctrl+C退出）...")
    try:
        while True:
            time.sleep(3600)  # 每小时醒一次，避免空循环占用CPU
    except KeyboardInterrupt:
        print("接收到退出信号，开始清理Ray资源...")
        ray.shutdown()  # 手动清理Ray资源
        print("程序正常退出")


if __name__ == '__main__':
    main()
