import os
from typing import Dict
from utils import rotary_pos_emb_patch

rotary_pos_emb_patch.apply_patch()

import deepspeed
import numpy as np
import torch
from megatron.core import parallel_state
from transformers import AutoTokenizer
from peft import LoraConfig

from tensordict import TensorDict
from tensor_parallel import vocab_parallel_log_probs_from_logits
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

import core_algos
from utils import utils, parallel_state_patch
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
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

from torch.utils.tensorboard import SummaryWriter

# 初始化 Writer（指定日志目录）
writer = SummaryWriter(log_dir="./ds_tensorboard_logs/agent_search_tensorboard")


class MegatronDeepSpeedPPOTrainer:
    def __init__(self, config):
        self.global_steps = None
        self.config = config
        self._init_logger()

        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_path,
                                                       trust_remote_code=True,
                                                       )
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # self.tokenizer.pad_token = self.tokenizer.eos_token

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
        self.actor = build_qwen2_megatron_model(config=config, tokenizer=self.tokenizer,
                                                qwen_model_path=config.qwen_model_path,
                                                lora_config=self.lora_config, is_actor=True)
        utils.print_rank_0(self.actor)
        self.critic = build_qwen2_megatron_model(config=config, tokenizer=self.tokenizer,
                                                 qwen_model_path=config.qwen_model_path,
                                                 lora_config=self.lora_config, is_critic=True)
        utils.print_rank_0(self.critic)
        # 确保参数可训练
        for param in self.critic.value_head.parameters():
            param.requires_grad = True
        self.critic.config.enable_autocast = True
        self.critic.config.autocast_dtype = torch.bfloat16
        self.actor.config.enable_autocast = True
        self.actor.config.autocast_dtype = torch.bfloat16

        self.reference = build_qwen2_megatron_model(config=config, tokenizer=self.tokenizer,
                                                    qwen_model_path=config.qwen_model_path)
        self.reference.eval()
        utils.print_rank_0(self.reference)
        for name, param in self.reference.named_parameters():
            param.requires_grad = False
            # param.data = param.data.to(torch.float32)
        self.reference.config.enable_autocast = True
        self.reference.config.autocast_dtype = torch.bfloat16
        # 确保参数可训练
        for name, param in self.actor.named_parameters():
            param.requires_grad = True

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
        # os.environ["WORLD_SIZE"] = str(self.config.megatron.tensor_model_parallel_size *
        #                                self.config.megatron.pipeline_model_parallel_size)
        # os.environ["RANK"] = str(self.config.megatron.rank)
        # os.environ["LOCAL_RANK"] = str(self.config.megatron.local_rank)
        # if self.config.megatron.sequence_parallel:
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
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
        torch.cuda.set_device(local_rank)

        # if is_main_process:
        print(f"DeepSpeed distributed init finished：WORLD_SIZE={world_size}, RANK={rank}, LOCAL_RANK={local_rank}")
        print(f"parallel config：TP={tp_size}, PP={pp_size}, DP={dp_size}")

        # -------------------------- 步骤 3：megatron-core 模型并行组初始化 --------------------------
        # 基于 DeepSpeed 的全局进程组，创建 TP/PP/DP 组（DP 组复用 DeepSpeed 的数据并行组）
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=tp_size,
                pipeline_model_parallel_size=pp_size,
                virtual_pipeline_model_parallel_size=None,  # RL/微调场景禁用虚拟流水线
                order='tp-pp-dp'  # 2-tp 2-pp for megatron 2-dp for deepspeed
            )

        # 验证并行组（确保与 DeepSpeed 进程组兼容）
        tp_group = parallel_state.get_tensor_model_parallel_group()
        dp_group = parallel_state.get_data_parallel_group()
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if is_main_process:
            print(
                f"megatron-core parallel group：TP group={tp_group} size={torch.distributed.get_world_size(tp_group)}, "
                f"PP group={pp_group} size={torch.distributed.get_world_size(pp_group)}, "
                f"DP group={dp_group} size={torch.distributed.get_world_size(dp_group)}")

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
        torch.manual_seed(seed + rank)
        model_parallel_cuda_manual_seed(seed, te_rng_tracker=True, inference_rng_tracker=True)
        import numpy as np
        np.random.seed(seed + rank)
        import random
        random.seed(seed + rank)

    def _init_deepspeed(self):
        """初始化deepspeed引擎（仅优化 LoRa 参数）"""
        utils.print_rank_0("DeepSpeed is enabled.")
        # param_to_name = {param: name for name, param in self.actor.named_parameters()}
        #
        # trainable = [param_to_name.get(param) for param in self.actor.parameters() if param.requires_grad]
        # print(f"actor trainable params len:{len(trainable)}")
        # if len(trainable) > 5:
        #     print(f"first 5 actor trainable params: {trainable[0:5]}")

        parallel_state_patch.add_missing_mpu_methods()

        actor_optimizer = get_megatron_optimizer(config=init_megatron_optim_config(self.config.actor.optimizer),
                                                 model_chunks=[self.actor])
        opt_param_scheduler = get_optimizer_param_scheduler(actor_optimizer, config=self.config.actor.optimizer)
        assert isinstance(actor_optimizer, ChainedOptimizer)

        # # 核心修复：强制开启所有参数的 requires_grad
        # for group in actor_optimizer.optimizer.param_groups:
        #     for p in group["params"]:
        #         p.requires_grad = True  # 覆盖 Megatron 处理后的状态

        # 将 config.deepspeed 转换为 dict
        # resolve=True 表示在转换前解析所有变量插值
        deepspeed_dict = OmegaConf.to_container(self.config.deepspeed, resolve=True)

        # 1. 过滤掉空的 param_groups
        #    创建一个新的列表来存放可训练参数非空的 groups
        # filtered_param_groups = []
        # utils.print_rank_0(f"{actor_optimizer.optimizer.param_groups}")
        # for param_group in actor_optimizer.optimizer.param_groups:
        #     trainable = sum(1 for param in param_group['params'] if param.requires_grad)
        #     # 检查这个 group 的 'params' 列表是否为空
        #     if trainable > 0:
        #         filtered_param_groups.append(param_group)
        # # 2. 验证过滤后是否还有参数组
        # if not filtered_param_groups:
        #     print(
        #         f"[Rank {torch.distributed.get_rank()}] All actor param_groups are empty after filtering. No "
        #         f"parameters to"
        #         f"optimize.")
        #     self.actor.step = lambda *args, **kwargs: None
        # else:

        # actor_optimizer.optimizer.param_groups = filtered_param_groups
        # 初始化 DeepSpeed 引擎
        self.actor, self.optimizer, _, _ = deepspeed.initialize(
            model=self.actor,
            config=deepspeed_dict,
            mpu=parallel_state,
            lr_scheduler=opt_param_scheduler,
            model_parameters=self.actor.parameters()
        )
        print(
            f"当前进程 {torch.distributed.get_rank()}-self.optimizer的参数分区数：{len(self.optimizer.params_in_partition)}")

        critic_optimizer = get_megatron_optimizer(config=init_megatron_optim_config(self.config.critic.optimizer),
                                                  model_chunks=[self.critic])
        critic_opt_param_scheduler = get_optimizer_param_scheduler(critic_optimizer,
                                                                   config=self.config.critic.optimizer)
        assert isinstance(critic_optimizer, ChainedOptimizer)

        # 1. 过滤掉空的 param_groups
        #    创建一个新的列表来存放可训练参数非空的 groups
        filtered_param_groups_critic = []
        utils.print_rank_0(f"{critic_optimizer.optimizer.param_groups}")
        for param_group in critic_optimizer.optimizer.param_groups:
            trainable = sum(1 for param in param_group['params'] if param.requires_grad)
            # 检查这个 group 的 'params' 列表是否为空
            if trainable > 0:
                filtered_param_groups_critic.append(param_group)
        # 2. 验证过滤后是否还有参数组
        if not filtered_param_groups_critic:
            print(
                f"[Rank {torch.distributed.get_rank()}] All critic param_groups are empty after filtering. No "
                f"parameters to"
                f"optimize.")
            self.critic.step = lambda *args, **kwargs: None
        else:

            critic_optimizer.optimizer.param_groups = filtered_param_groups_critic
            self.critic, self.critic_optimizer, _, _ = deepspeed.initialize(
                model=self.critic,
                optimizer=critic_optimizer.optimizer,
                config=deepspeed_dict,
                lr_scheduler=critic_opt_param_scheduler,
                # model_parameters=self.critic.parameters()
            )
            print(
                f"当前进程 {torch.distributed.get_rank()}-self.critic_optimizer的参数分区数：{len(self.critic_optimizer.params_in_partition)}")

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
                                         truncation='left')
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
                                       truncation='left')
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
        # debug
        # input_ids = self.tokenizer.encode("Hello World!", return_tensors="pt").to('cuda')
        attention_mask = batch["attention_mask"].to('cuda')
        prompt_len = input_ids.shape[1]

        # 生成 response（actor模型）
        with torch.no_grad():
            if not self.config.do_search:
                outputs = self.actor.generate(
                    input_ids=input_ids,
                    max_length=self.config.rollout.max_new_token + prompt_len,
                    eos_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token),
                    pad_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
                    temperature=self.config.rollout.temperature,
                    attention_mask=attention_mask,
                    top_k=self.config.rollout.top_k,
                )
                batch['prompts'] = batch['input_ids'][:, -self.config.data.max_start_length:].clone().long()
                response_mask = self._get_eos_mask(response_id=outputs[:, prompt_len:],
                                                   eos_token=self.tokenizer.eos_token_id,
                                                   dtype=attention_mask.dtype)
                mask = torch.cat((attention_mask, response_mask), dim=-1).bool()
                batch["attention_mask"] = mask
                response = outputs[:, prompt_len:]
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
                mask = final_gen_batch_output[0]['attention_mask'].bool()
                batch["attention_mask"] = mask
                batch['prompts'] = final_gen_batch_output[0]['prompts']
                outputs = final_gen_batch_output[0]['input_ids']
                response = outputs[:, prompt_len:]

        print(f"outputs dtype: {outputs.dtype}, mask dtype: {mask.dtype}")
        response_mask = self._get_eos_mask(response_id=outputs[:, prompt_len:],
                                           eos_token=self.tokenizer.eos_token_id,
                                           dtype=attention_mask.dtype)

        # 计算 reference 的 log_prob
        # ref_log_probs = self._compute_ref_log_probs(outputs, mask, outputs[:, prompt_len:])
        ref_log_probs = None

        # 解码第一条输入，确认无乱码
        dialogue_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        utils.print_rank_0(f"输入文本：{dialogue_text}")  # 若输出乱码，需重新处理输入数据

        return response, outputs, ref_log_probs, response_mask, mask

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
        rewards = rewards.to('cpu')
        return rewards

    def _update_policy(self, ref_log_probs, outputs, responses, advantages, mask: torch.Tensor = None):
        """更新策略网络（PPO 核心逻辑）"""

        batches = TensorDict(
            source={
                "input_ids": outputs.to('cuda'),
                "attention_mask": mask.to('cuda'),
                "responses": responses.to('cuda'),
                "ref_log_probs": ref_log_probs.to('cuda'),
                "advantages": advantages.to('cuda'),
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
            metric_micro_batch = self.actor.module.forward_backward_batch(
                batch=data,
                forward_only=False,
                post_process_fn=None,
                meta_info=meta_info
            )  # PP+TP下：LIST[batch_size/pp_size, 1, vocab_size/tp_size]

            increment = data.batch_size[0]

            print(
                f"当前进程 {torch.distributed.get_rank()}-self.optimizer.averaged_gradients的keys：{list(self.optimizer.averaged_gradients.keys())}")
            # # 强制检查Actor参数梯度
            # has_grad = False
            # for name, param in self.actor.named_parameters():
            #     if param.grad is not None and param.grad.norm().item() > 0:
            #         print(f"Actor {name} 梯度范数：{param.grad.norm().item()}")
            #         has_grad = True
            # assert has_grad, "Actor无有效梯度！"

            self.actor.allreduce_gradients()

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

                utils.print_rank_0(f"batch_dict['input_ids'][0]:{self.tokenizer.decode(batch_dict["input_ids"][0])}")
                self.actor.run_comprehensive_debug(self.tokenizer, batch_dict)

                continue

                # 1. Rollout：生成相应并计算 log prob
                responses, dialogue_ids, ref_log_probs, response_mask, attention_mask = self._rollout(batch_dict)
                utils.print_rank_0(f"rollout successful:{self.global_steps}, "
                                   f"dialogue:{self.tokenizer.decode(dialogue_ids[0], skip_special_tokens=True)}")

                continue

                # 2. 计算奖励
                rewards = self._compute_reward(batch_dict, responses)
                utils.print_rank_0(f"_compute_reward successful:{self.global_steps}")

                # 3. Critic 预测价值
                critic_values = self.compute_values(dialogue_ids, responses, attention_mask)
                utils.print_rank_0(f"compute_values successful:{self.global_steps}")

                # 4. 计算优势函数
                advantages, returns = core_algos.compute_gae_advantage_return(rewards,
                                                                              critic_values,
                                                                              eos_mask=response_mask)
                utils.print_rank_0(f"compute_gae_advantage_return successful:{self.global_steps}")

                # 5. 更新价值网络
                metrics_critic = self._update_critic(dialogue_ids, attention_mask, responses, critic_values, returns)
                metrics.update(metrics_critic)

                # 6. 更新策略
                metrics_actor = self._update_policy(ref_log_probs, dialogue_ids, responses, advantages, attention_mask)
                metrics.update(metrics_actor)

                # ds tensorboard
                self.write_ds_scalars(metrics)

                self.logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps % 100 == 0:
                    utils.print_rank_0(f'train metrics: {metrics}, global_steps: {self.global_steps}')

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

    def write_ds_scalars(self, metrics):
        if torch.distributed.get_rank() == 0:
            writer.add_scalar("train/actor/entropy_loss", np.mean(metrics['actor/entropy_loss']),
                              global_step=self.global_steps)
            writer.add_scalar("train/actor/pg_loss", np.mean(metrics['actor/pg_loss']), global_step=self.global_steps)
            writer.add_scalar("train/actor/pg_clipfrac", np.mean(metrics['actor/pg_clipfrac']),
                              global_step=self.global_steps)
            writer.add_scalar("train/actor/ppo_kl", np.mean(metrics['actor/ppo_kl']), global_step=self.global_steps)
            writer.add_scalar("train/critic/vf_loss", np.mean(metrics['critic/vf_loss']), global_step=self.global_steps)
            writer.add_scalar("train/critic/vf_clipfrac", np.mean(metrics['critic/vf_clipfrac']),
                              global_step=self.global_steps)
            writer.add_scalar("train/critic/vpred_mean", np.mean(metrics['critic/vpred_mean']),
                              global_step=self.global_steps)

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
        attention_mask = attention_mask.to('cuda')
        batches = TensorDict(
            source={
                "input_ids": outputs.to('cuda'),
                "attention_mask": attention_mask,
                "responses": responses.to('cuda'),
            },
            batch_size=responses.shape[0])
        batches.to('cuda')
        with torch.no_grad():
            output = self.critic.module.forward_backward_batch(batch=batches, forward_only=True)
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # only on last rank. It should be on every tp rank
                values = torch.cat([o for o in output], dim=0)  # (bs, seq_size, 1)
                values = values.to(torch.float32)
            else:
                values = torch.empty_like(attention_mask, dtype=torch.float32)

            # utils.print_rank_0(f"values.shape: {values.shape}, attention_mask.shape: {attention_mask.shape}")
            # utils.print_rank_0(f"values.device: {values.device}, attention_mask.device: {attention_mask.device}")
            # each tp ranks should contain the same value
            values = values * attention_mask
            values = values[:, -response_length - 1:-1]
            values = values.contiguous()

            # sync among pp ranks
            torch.distributed.broadcast(tensor=values,
                                        src=parallel_state.get_pipeline_model_parallel_last_rank(),
                                        group=parallel_state.get_pipeline_model_parallel_group())

        values = values.to('cpu')
        # add empty cache after each compute
        torch.cuda.empty_cache()

        return values

    def _compute_ref_log_probs(self, input_ids, mask, responses):
        """计算生成序列的 log probability（适配 Megatron 并行）"""

        def compute_logprobs_fn(output, data):
            _response = data["responses"]
            _response_length = _response.size(1)
            # # 张量并行聚合：收集所有TP进程的logits，得到完整vocab分布
            # _logits = self.reference.gather_logits_across_tp(output)
            _logits = output[:, -_response_length - 1:-1].contiguous()
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
            log_probs = self.reference.module.forward_backward_batch(
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

        log_probs.to('cpu')
        # add empty cache after each compute
        torch.cuda.empty_cache()

        return log_probs

    def _update_critic(self, input_ids, attention_mask, responses, values, returns):
        batches = TensorDict(
            source={
                "input_ids": input_ids.to('cuda'),
                "attention_mask": attention_mask.to('cuda'),
                "responses": responses.to('cuda'),
                "values": values.to('cuda'),
                "returns": returns.to('cuda'),
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
            metric_micro_batch = self.critic.module.forward_backward_batch(batch=data, meta_info=meta_info)

            increment = data.batch_size[0]

            print(
                f"当前进程 {torch.distributed.get_rank()}-self.critic_optimizer.averaged_gradients的keys：{list(self.critic_optimizer.averaged_gradients.keys())}")
            # 强制打印value_head参数的梯度（bfloat16下需注意精度）
            for name, param in self.critic.value_head.named_parameters():
                if param.grad is None:
                    print(f"ERROR:当前进程 {torch.distributed.get_rank()}- {name} 无梯度！")
                else:
                    grad_norm = param.grad.norm().item()
                    print(
                        f"当前进程 {torch.distributed.get_rank()}- {name} 梯度范数：{grad_norm} 梯度数值：{param.grad}")  # 需>0才正常

            # self.critic_value_head.allreduce_gradients()
            # self.critic_value_head.step(lr_kwargs={'increment': increment})

            self.critic.allreduce_gradients()
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
