import os
from typing import Dict

import deepspeed
import torch
from megatron.core import parallel_state
from transformers import AutoTokenizer
from peft import LoraConfig

from tensordict import TensorDict
from tensor_parallel import vocab_parallel_log_probs_from_logits
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

import core_algos
from utils import utils
from modeling_qwen_megatron import build_qwen2_megatron_model
from omegaconf import OmegaConf, open_dict
import reward_score
from deepspeed.accelerator import get_accelerator

# if get_accelerator().device_name() == 'cuda':
#     from apex.optimizers import FusedAdam as Adam
# else:
#     from torch.optim import AdamW as Adam
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig, ChainedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.num_microbatches_calculator import get_num_microbatches
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig


class MegatronDeepSpeedPPOTrainer:
    def __init__(self, config):
        self.global_steps = None
        self.config = config
        self._init_logger()

        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载数据集（分布式采样）
        self._create_dataloader()

        # 1. 初始化分布式环境（Megatron + DeepSpeed 协同）
        self._init_distributed()

        # 2. 配置 LoRa
        self.lora_config = LoraConfig(
            # r=config.lora_r,
            # lora_alpha=config.lora_alpha,
            # target_modules=[],
            # lora_dropout=config.lora_dropout,
            # bias="none",
            # task_type="CAUSAL_LM"
        )

        # 3. 构建 PPO 三模型
        self.actor = build_qwen2_megatron_model(config=config, qwen_model_path=config.qwen_model_path,
                                                lora_config=self.lora_config)
        self.critic = build_qwen2_megatron_model(config=config, qwen_model_path=config.qwen_model_path,
                                                 lora_config=self.lora_config, is_critic=True)
        self.reference = build_qwen2_megatron_model(config=config, qwen_model_path=config.qwen_model_path)
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False

        # 4. 初始化 Deepspeed 引擎（ZeRO 优化）
        self._init_deepspeed()

    def _init_logger(self):
        from utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                               experiment_name=self.config.trainer.experiment_name,
                               default_backend=self.config.trainer.logger,
                               config=OmegaConf.to_container(self.config, resolve=True))

    def _init_distributed(self):
        """初始化 Megatron + Deepspeed 分布式环境"""
        # -------------------------- 步骤 1：解析配置 --------------------------
        # 加载配置
        os.environ["MASTER_ADDR"] = self.config.megatron.master_addr
        os.environ["MASTER_PORT"] = str(self.config.megatron.master_port)
        os.environ["WORLD_SIZE"] = str(self.config.megatron.tensor_model_parallel_size *
                                       self.config.megatron.pipeline_model_parallel_size)
        os.environ["RANK"] = str(self.config.megatron.rank)
        os.environ["LOCAL_RANK"] = str(self.config.megatron.local_rank)
        # if self.config.megatron.sequence_parallel:
        #     os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        # 计算数据并行度（DP_SIZE = 总进程数 / (TP_SIZE * PP_SIZE)）
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        tp_size = self.config.megatron.tensor_model_parallel_size
        pp_size = self.config.megatron.pipeline_model_parallel_size
        dp_size = world_size // (tp_size * pp_size)
        assert tp_size * pp_size * dp_size == world_size, "并行度不匹配：TP*PP*DP != WORLD_SIZE"

        # -------------------------- 步骤 2：DeepSpeed 分布式初始化 --------------------------
        # 替代原生 torch.distributed.init_process_group，创建全局分布式进程组
        deepspeed.init_distributed(
            dist_backend="nccl",  # GPU 训练必选，CPU 用 "gloo"
            init_method="env://"  # 从环境变量读取 master_addr/master_port（torchrun 传入）
        )

        # 获取 DeepSpeed 进程信息
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = torch.distributed.get_rank()
        is_main_process = (rank == 0)

        if is_main_process:
            print(f"DeepSpeed distributed init finished：WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}")
            print(f"parallel config：TP={tp_size}, PP={pp_size}, DP={dp_size}")

            # -------------------------- 步骤 3：megatron-core 模型并行组初始化 --------------------------
            # 基于 DeepSpeed 的全局进程组，创建 TP/PP/DP 组（DP 组复用 DeepSpeed 的数据并行组）
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=pp_size,
                virtual_pipeline_model_parallel_size=None  # RL/微调场景禁用虚拟流水线
            )

            # 验证并行组（确保与 DeepSpeed 进程组兼容）
            tp_group = parallel_state.get_tensor_model_parallel_group()
            dp_group = parallel_state.get_data_parallel_group()
            pp_group = parallel_state.get_pipeline_model_parallel_group()
            if is_main_process:
                print(
                    f"megatron-core parallel group：TP group size={torch.distributed.get_world_size(tp_group)}, "
                    f"PP group size={torch.distributed.get_world_size(pp_group)}, "
                    f"DP group size={torch.distributed.get_world_size(dp_group)}")

            # # -------------------------- 步骤 4：混合精度统一配置 --------------------------
            # # 与 DeepSpeed 配置保持一致（bf16/fp16/fp32）
            # if self.config.deepspeed.bf16.enabled:
            #     torch.set_default_dtype(torch.bfloat16)  # 原生 API，所有张量默认 BF16
            #     torch.backends.cuda.matmul.allow_tf32 = True
            #     torch.backends.cudnn.allow_tf32 = True
            # elif self.config.deepspeed.fp16.enabled:
            #     torch.set_default_dtype(torch.float16)
            #     # from megatron.core.optimizer.fp16_optimizer import initialize_fp16_optimizer_states
            #     # initialize_fp16_optimizer_states()
            # else:
            #     torch.set_default_dtype(torch.float32)

            # -------------------------- 步骤 5：设置随机种子（确保可复现） --------------------------
            # 每个进程的种子 = 全局种子 + 进程 rank（避免进程间随机不一致）
            seed = self.config.megatron.seed
            # torch.manual_seed(seed + rank)
            # torch.cuda.manual_seed(seed + rank)
            # torch.cuda.manual_seed_all(seed + rank)
            model_parallel_cuda_manual_seed(seed, te_rng_tracker=True, inference_rng_tracker=True)
            # rng_tracker = get_cuda_rng_tracker()
            # rng_tracker.add("model-parallel-rng", seed=seed)
            import numpy as np
            np.random.seed(seed + rank)
            import random
            random.seed(seed + rank)

    def _init_deepspeed(self):
        """初始化deepspeed引擎（仅优化 LoRa 参数）"""
        utils.print_rank_0("DeepSpeed is enabled.")
        # # 优化配置
        # optimizer = Adam(
        #     self.actor.parameters(),
        #     lr=self.config.actor.optimizer.lr,
        #     betas=(0.9, 0.95),
        #     weight_decay=0.01
        # )
        optimizer = get_megatron_optimizer(config=init_megatron_optim_config(self.config.actor.optimizer),
                                           model_chunks=[self.actor])
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer, config=self.config.actor.optimizer)
        assert isinstance(optimizer, ChainedOptimizer)

        # 将 config.deepspeed 转换为 dict
        # resolve=True 表示在转换前解析所有变量插值
        deepspeed_dict = OmegaConf.to_container(self.config.deepspeed, resolve=True)
        # DeepSpeed 配置（从 config dict 加载）
        ds_config = deepspeed.DeepSpeedConfig(deepspeed_dict)

        # 初始化 DeepSpeed 引擎
        self.actor, self.optimizer, _, _ = deepspeed.initialize(
            model=self.actor,
            optimizer=optimizer.optimizer(),
            config=ds_config,
            mpu=parallel_state,
            lr_scheduler=opt_param_scheduler,
            # model_parameters=self.actor.parameters()
        )

        # # 优化配置
        # critic_optimizer = Adam(
        #     self.critic.parameters(),
        #     lr=self.config.critic.optimizer.lr,
        # )
        critic_optimizer = get_megatron_optimizer(config=init_megatron_optim_config(self.config.critic.optimizer),
                                                  model_chunks=[self.critic])
        critic_opt_param_scheduler = get_optimizer_param_scheduler(critic_optimizer,
                                                                   config=self.config.critic.optimizer)
        assert isinstance(critic_optimizer, ChainedOptimizer)

        self.critic, self.critic_optimizer, _, _ = deepspeed.initialize(
            model=self.critic,
            optimizer=critic_optimizer.optimizer(),
            config=ds_config,
            lr_scheduler=critic_opt_param_scheduler,
            # model_parameters=self.critic.parameters()
        )

    # def _load_dataset(self):
    #     """加载分布式数据集（每个 rank 处理部分数据）"""
    #     dataset = load_dataset("json", data_files=self.config.data_file, split="train")
    #     # 分布式采样器
    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size, shuffle=False)
    #     return dataloader

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                utils.print_rank_0(
                    f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num,
                                                                                   random_state=42)
        utils.print_rank_0(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                utils.print_rank_0(
                    f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num,
                                                                               random_state=42)
        utils.print_rank_0(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        utils.print_rank_0(f'Size of train dataloader: {len(self.train_dataloader)}')
        utils.print_rank_0(f'Size of val dataloader: {len(self.val_dataloader)}')

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        utils.print_rank_0(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor.optimizer.total_training_steps = total_training_steps
            self.config.critic.optimizer.total_training_steps = total_training_steps

    def _rollout(self, batch):
        """Rollout阶段：生成 response 并计算 log prob（分布式采用）"""
        # prompts = batch["prompt"].to('cuda')
        # inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = batch["input_ids"].to('cuda')
        attention_mask = batch["attention_mask"].to('cuda')
        prompt_len = input_ids.shape[1]

        # 生成 response（actor模型）
        with torch.no_grad():
            if not self.config.do_search:
                outputs = self.actor.generate(
                    input_ids=input_ids,
                    max_length=self.config.rollout.max_new_token,
                    eos_token=self.tokenizer.eos_token,
                    pad_token_id=self.tokenizer.pad_token,
                    temperature=self.config.rollout.temperature,
                    attention_mask=attention_mask,
                    top_k=self.config.rollout.top_k,
                )

                responses = self.tokenizer.decode(outputs, skip_special_tokens=True)
            else:
                # Agent config preparation
                gen_config = GenerationConfig(
                    max_turns=self.config.max_turns,
                    max_start_length=self.config.data.max_start_length,
                    max_prompt_length=self.config.data.max_prompt_length,
                    max_response_length=self.config.data.max_response_length,
                    max_obs_length=self.config.data.max_obs_length,
                    num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                    no_think_rl=self.config.algorithm.no_think_rl,
                    search_url=self.config.retriever.url,
                    topk=self.config.retriever.topk,
                )
                generation_manager = LLMGenerationManager(
                    tokenizer=self.tokenizer,
                    actor_model=self.actor,
                    config=gen_config,
                    g_config=self.config
                )
                first_input_ids = batch['input_ids'][:, -gen_config.max_start_length:].clone().long()
                final_gen_batch_output = generation_manager.run_llm_loop(
                    gen_batch=batch,
                    initial_input_ids=first_input_ids,
                )

                # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                for key in final_gen_batch_output[0].keys():
                    if isinstance(final_gen_batch_output[0][key], torch.Tensor):
                        final_gen_batch_output[0][key] = final_gen_batch_output[0][key].long()

        response_mask = self._get_eos_mask(response_id=outputs[:, prompt_len:],
                                           eos_token=self.tokenizer.eos_token_id,
                                           dtype=attention_mask.dtype)
        mask = torch.cat((attention_mask, response_mask), dim=-1)

        # 计算 reference 的 log_prob
        ref_log_probs = self._compute_ref_log_probs(outputs, mask, responses[:, prompt_len:])

        return responses[:, prompt_len:], outputs, ref_log_probs, response_mask, mask

    @staticmethod
    def _get_eos_mask(response_id: torch.Tensor, eos_token: int = 2, dtype=torch.int64):
        '''
        e.g. end of sentence token=1
        response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
        eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
        '''
        eos_mask = response_id.eq(eos_token).long()
        eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
        eos_mask = torch.logical_not(eos_mask).to(dtype)
        return eos_mask

    def _compute_reward(self, batch, responses):
        """计算奖励分数（结合RM分数 + KL 惩罚）"""
        # rm_inputs = self.tokenizer([f"{p}\n{r}" for p, r in zip(prompts, responses)],
        #                            return_tensors="pt", padding=True, truncation=True)
        # rm_inputs = {k: v.to(self.actor.device) for k, v in rm_inputs.items()}
        #
        # with torch.no_grad():
        #     rm_scores = self.rm_model(**rm_inputs).logits.sequeeze(-1)
        #
        # # 2.计算KL惩罚
        # kl = self.actor_log_probs - self.ref_log_probs
        # kl_penalty = self.config.kl_coef * kl
        #
        # # 3.总奖励
        # rewards = rm_scores - kl_penalty
        batch["responses"] = responses
        rm = reward_score.RewardManager(tokenizer=self.tokenizer, num_examine=0)
        rewards = rm(batch)
        return rewards

    def _update_policy(self, ref_log_probs, outputs, responses, advantages, mask: torch.Tensor = None):
        """更新策略网络（PPO 核心逻辑）"""

        batches = TensorDict(
            source={
                "input_ids": outputs,
                "attention_mask": mask,
                "responses": responses,
                "ref_log_probs": ref_log_probs,
                "advantages": advantages,
            },
            batch_size=outputs.shape[0])

        meta_info = {
            'clip_ratio': self.config.actor.clip_ratio,
            'entropy_coeff': self.config.actor.entropy_coeff,
        }

        batches.to('cuda')

        dataloader = utils.make_iterator(batches, mini_batch_size=self.config.actor.ppo_mini_batch_size,
                                         epochs=self.config.actor.ppo_epochs,
                                         dataloader_kwargs={'shuffle': self.config.actor.shuffle})
        metrics = {}
        for data in dataloader:
            # 模型前向传播
            metric_micro_batch = self.actor.forward_backward_batch(
                batch=data,
                forward_only=False,
                post_process_fn=None,
                meta_info=meta_info
            )  # PP+TP下：LIST[batch_size/pp_size, 1, vocab_size/tp_size]

            increment = get_num_microbatches() * self.config.megtron.micro_batch_size * self.config.megtron.data_parallel_size

            self.actor.step(lr_kwargs={'increment': increment})

            update_successful = self.actor.was_step_applied()
            utils.print_rank_0(f"actor update_successful:{update_successful}, increment:{increment}")

            # self.optimizer.step()

            for metric in metric_micro_batch:
                utils.append_to_dict(metrics, metric)

            # # 计算 PPO 损失
            # ratio = torch.exp(actor_log_probs - self.ref_log_probs.detach())
            # clipped_ratio = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
            # policy_loss = -torch.min(advantages * ratio, clipped_ratio * advantages)
            # policy_loss = self.mask_mean(mask, policy_loss, -1)

            # 更新 actor 模型
            # self.actor.backward(policy_loss)

            # critic_loss = torch.nn.functional.mse_loss(critic_values, target_values)
            # critic_loss = self.mask_mean(mask, critic_loss, -1)
            # 更新 critic 模型
            # self.critic.backward(critic_loss)

        return metrics

    def train(self):
        """PPO 训练主循环"""
        self.actor.train()
        self.critic.train()
        self.global_steps = 0

        metrics = {}
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # 1. Rollout：生成相应并计算 log prob
                responses, dialogue_ids, ref_log_probs, response_mask, attention_mask = self._rollout(batch_dict)

                # 2. 计算奖励
                rewards = self._compute_reward(batch_dict, responses)

                # 3. Critic 预测价值
                critic_values = self.compute_values(dialogue_ids, responses, attention_mask)

                # 4. 计算优势函数
                advantages, returns = core_algos.compute_gae_advantage_return(rewards,
                                                                              critic_values,
                                                                              eos_mask=response_mask)

                # 5. 更新价值网络
                metrics_critic = self._update_critic(dialogue_ids, attention_mask, responses, critic_values, returns)
                metrics.update(metrics_critic)

                # 6. 更新策略
                metrics_actor = self._update_policy(ref_log_probs, responses, advantages, attention_mask)
                metrics.update(metrics_actor)

                self.logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps % 100 == 0:
                    print(f'train metrics: {metrics}, global_steps: {self.global_steps}')

                if self.global_steps % self.config.trainer.log_interval == 0:
                    # pprint(f'Final validation metrics: {val_metrics}')
                    # 保存 checkpoint 到自定义路径
                    checkpoint_path = f"./ds_checkpoints/actor/epoch_{epoch}/global_steps_{self.global_steps}"
                    self.actor.save_checkpoint(checkpoint_path)
                    checkpoint_path = f"./ds_checkpoints/critic/epoch_{epoch}/global_steps_{self.global_steps}"
                    self.critic.save_checkpoint(checkpoint_path)

                if self.global_steps >= self.total_training_steps:
                    # pprint(f'Final validation metrics: {val_metrics}')
                    # 保存 checkpoint 到自定义路径
                    checkpoint_path = f"./ds_checkpoints/actor/epoch_{epoch}"
                    self.actor.save_checkpoint(checkpoint_path)
                    checkpoint_path = f"./ds_checkpoints/critic/epoch_{epoch}"
                    self.critic.save_checkpoint(checkpoint_path)
                    return
            # 保存 checkpoint 到自定义路径
            checkpoint_path = f"./ds_checkpoints/actor/epoch_{epoch}"
            self.actor.save_checkpoint(checkpoint_path)
            checkpoint_path = f"./ds_checkpoints/critic/epoch_{epoch}"
            self.critic.save_checkpoint(checkpoint_path)

    @staticmethod
    def mask_mean(self, mask, loss, dim=-1):
        # 序列维度平均
        if mask is not None:
            loss = (loss * mask).sum(dim) / mask.sum(dim)
        else:
            loss = loss.mean(dim)
        # 批次维度平均
        loss = loss.mean()
        return loss

    def compute_values(self, outputs, responses, attention_mask):
        response_length = responses.size(1)
        batches = TensorDict(
            source={
                "input_ids": outputs,
                "attention_mask": attention_mask,
                "responses": responses,
            },
            batch_size=responses.shape[0])
        batches.to('cuda')
        with torch.no_grad():
            output = self.critic.forward_backward_batch(batch=batches, forward_only=True)
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                values = torch.cat([o for o in output], dim=0)  # (bs, seq_size, 1)
                values = values.to(torch.float32)
            else:
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            # each tp ranks should contain the same value
            values = values * attention_mask
            values = values[:, -response_length - 1:-1]
            values = values.contiguous()

            # sync among pp ranks
            torch.distributed.broadcast(tensor=values,
                                        src=parallel_state.get_pipeline_model_parallel_last_rank(),
                                        group=parallel_state.get_pipeline_model_parallel_group())

        values.to('cuda')
        # add empty cache after each compute
        torch.cuda.empty_cache()

        return values

    def _compute_ref_log_probs(self, input_ids, mask, responses):
        """计算生成序列的 log probability（适配 Megatron 并行）"""

        def compute_logprobs_fn(output, data):
            _response = data["responses"]
            _response_length = _response.size(1)
            # 张量并行聚合：收集所有TP进程的logits，得到完整vocab分布
            _logits = self.reference.gather_logits_across_tp(output)
            _logits = _logits[:, -_response_length - 1:-1]
            _log_probs = vocab_parallel_log_probs_from_logits(_logits, _response)
            return _log_probs

        batches = TensorDict(
            source={
                "input_ids": input_ids,
                "attention_mask": mask,
                "responses": responses,
            },
            batch_size=input_ids.shape[0])

        batches = batches.to('cuda')

        # 模型前向传播
        with torch.no_grad():
            log_probs = self.reference.forward_backward_batch(
                batch=batches,
                forward_only=True,
                post_process_fn=compute_logprobs_fn
            )  # PP+TP下：LIST[batch_size/pp_size, 1, vocab_size/tp_size]

            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                log_probs = torch.cat([o for o in log_probs], dim=0)  # (bs, seq_size)
                log_probs = log_probs.to(torch.float32)
            else:
                log_probs = torch.empty(size=(input_ids.shape[0], input_ids.shape[1]),
                                        dtype=torch.float32,
                                        device=input_ids.device)

            # broadcast across pp ranks
            torch.distributed.broadcast(tensor=log_probs,
                                        src=parallel_state.get_pipeline_model_parallel_last_rank(),
                                        group=parallel_state.get_pipeline_model_parallel_group(),
                                        async_op=False)

        # log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # # 提取生成部分（prompt 之后）的 log prob
        # gen_log_probs = log_probs[:, prompt_len - 1:-1, :]
        # gen_token_ids = outputs[:, prompt_len:]
        # gen_log_probs = gen_log_probs.gather(2, gen_token_ids.unsqueeze(-1)).squeeze(-1)

        log_probs.to('cpu')
        # add empty cache after each compute
        torch.cuda.empty_cache()

        return log_probs

    def _update_critic(self, input_ids, attention_mask, responses, values, returns):
        batches = TensorDict(
            source={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "responses": responses,
                "values": values,
                "returns": returns,
            },
            batch_size=input_ids.shape[0])
        batches.to('cuda')

        meta_info = {
            'cliprange_value': self.config.critic.cliprange_value,
        }

        dataloader = utils.make_iterator(batches, mini_batch_size=self.config.critic.ppo_mini_batch_size,
                                         epochs=self.config.critic.ppo_epochs,
                                         dataloader_kwargs={'shuffle': self.config.critic.shuffle})
        metrics = {}
        for data in dataloader:
            metric_micro_batch = self.critic.forward_backward_batch(batch=data, meta_info=meta_info)

            increment = (get_num_microbatches() *
                         self.config.megtron.micro_batch_size * self.config.megtron.data_parallel_size)

            self.critic.step(lr_kwargs={'increment': increment})

            update_successful = self.critic.was_step_applied()
            utils.print_rank_0(f"critic update_successful:{update_successful}, increment:{increment}")

            for metric in metric_micro_batch:
                utils.append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

        return metrics


def get_optimizer_param_scheduler(optimizer, config):
    """Build the learning rate scheduler."""
    # Iteration-based training.
    if config.total_training_steps:
        if config.lr_decay_steps is None:
            config.lr_decay_steps = config.total_training_steps
        lr_decay_steps = config.lr_decay_steps
        wd_incr_steps = config.total_training_steps
        if config.lr_warmup_fraction is not None:
            lr_warmup_steps = config.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = config.lr_warmup_steps
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=config.lr,
        min_lr=config.min_lr,
        init_lr=config.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.lr_decay_style,
        start_wd=config.start_weight_decay,
        end_wd=config.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=config.override_opt_param_scheduler)

    return opt_param_scheduler


def init_megatron_optim_config(optim_config: Dict) -> OptimizerConfig:
    config = OptimizerConfig(
        optimizer='adam',
        lr=optim_config.get('lr'),
        min_lr=optim_config.get('min_lr'),
        clip_grad=optim_config.get('clip_grad'),
        weight_decay=1e-2,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
    )
    return config
