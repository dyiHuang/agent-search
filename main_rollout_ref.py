import hydra
import ray
import torch
from vllm import LLM, SamplingParams


def init_ray_and_actor(qwen_model_path):
    """
    仅LOCAL_RANK=0的主进程：
    1. 初始化Ray并分配num_gpus张GPU
    2. 创建TP=num_gpus的VLLMActor
    其他进程：仅连接Ray
    """
    vllm_actor_ref = None
    num_gpus = 1
    rank = 0

    ray.init(
        ignore_reinit_error=True,
        local_mode=False,  # 必须关闭local_mode，否则无法多卡TP
        num_gpus=num_gpus,  # 为Ray集群分配num_gpus张GPU
        # _temp_dir="/tmp/ray-tp4",
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": "0",  # 明确指定num_gpus张GPU给Actor
                "TRUST_REMOTE_CODE": "True",
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
            }
        },
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

        def generate_from_tensor(self, input_ids_cpu, sampling_params: SamplingParams):
            """接收cpu张量，返回输出token的cpu张量"""
            input_ids = input_ids_cpu.to(f"cuda:{rank}")  # TP=num_gpus时，主卡为cuda:0
            outputs = self.llm.generate(
                prompts=None,
                prompt_token_ids=input_ids,
                sampling_params=sampling_params
            )
            output_token_ids = torch.tensor(outputs[0].outputs[0].token_ids).long().to(f"cuda:{rank}")
            return output_token_ids.cpu()

    # 创建Actor实例，获取全局引用
    vllm_actor_ref = VLLMActor.remote(qwen_model_path)
    # 等待Actor初始化完成（避免其他进程调用时未就绪）
    ray.get(vllm_actor_ref.__ray_ready__.remote())
    print(f"[Rank {rank}] VLLMActor (TP={num_gpus}) initialized")


@hydra.main(config_path='config', config_name='ppo_megatron_trainer', version_base=None)
def main(config):
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    init_ray_and_actor(config.qwen_model_path)


if __name__ == '__main__':
    main()
