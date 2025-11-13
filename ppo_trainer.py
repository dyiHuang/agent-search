import deepspeed
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.pipeline_parallel import get_forward_backward_func
from transformers import AutoTokenizer
from peft import LoraConfig
from megatron.core.optimizer import DistributedOptimizer

import tensordict
from tensordict import TensorDict

import core_algos
import utils
from modeling_qwen_megatron import build_qwen2_megatron_model


class MegatronDeepSpeedPPOTrainer:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1. 初始化分布式环境（Megatron + DeepSpeed 协同）
        self._init_distributed()

        # 2. 配置 LoRa
        self.lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 3. 构建 PPO 三模型
        self.actor = build_qwen2_megatron_model(args.qwen_model_path, lora_config=self.lora_config)
        self.critic = build_qwen2_megatron_model(args.qwen_model_path, lora_config=self.lora_config, is_critic=True)
        self.reference = build_qwen2_megatron_model(args.qwen_model_path)
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False

        # 4. 初始化 Deepspeed 引擎（ZeRO 优化）
        self._init_deepspeed()

        # 5. 加载数据集（分布式采样）
        self.dataset = self._load_dataset()

    def _init_distributed(self):
        """初始化 Megatron + Deepspeed 分布式环境"""
        deepspeed.init_distributed(dist_backend="nccl")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.args.tp_size if self.args.tp_size > 1 else 1,
            pipeline_model_parallel_size=self.args.pp_size if self.args.pp_size > 1 else 1,
        )

    def _init_deepspeed(self):
        """初始化deepspeed引擎（仅优化 LoRa 参数）"""
        # 优化配置
        optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )

        # DeepSpeed 配置（从 JSON 加载）
        ds_config = deepspeed.DeepSpeedConfig(self.args.ds_config_file)

        # 初始化 DeepSpeed 引擎
        self.actor, self.optimizer, _, _ = deepspeed.initialize(
            model=self.actor,
            optimizer=optimizer,
            config=ds_config,
            model_parameters=self.actor.parameters()
        )

        # 优化配置
        critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.args.learning_rate,
        )
        self.critic, self.critic_optimizer, _, _ = deepspeed.initialize(
            model=self.critic,
            optimizer=critic_optimizer,
            config=ds_config,
            model_parameters=self.critic.parameters()
        )

    def _load_dataset(self):
        """加载分布式数据集（每个 rank 处理部分数据）"""
        dataset = load_dataset("json", data_files=self.args.data_file, split="train")
        # 分布式采样器
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size, shuffle=False)
        return dataloader

    def _rollout(self, batch):
        """Rollout阶段：生成 response 并计算 log prob（分布式采用）"""
        prompts = batch["prompt"]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].to(self.actor.device)

        # 生成 response（actor模型）
        with torch.no_grad():
            outputs = self.actor.generate(
                input_ids=input_ids,
                max_length=self.args.max_new_token,
                eos_token=self.tokenizer.eos_token,
                pad_token_id=self.tokenizer.pad_token,
                temperature=self.args.temperature,
                attention_mask=attention_mask,
                top_k=self.args.top_k,
            )
        prompt_len = input_ids.shape[1]
        responses = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_attention_mask = self._get_eos_mask(response_id=outputs[:, prompt_len:],
                                                     eos_token=self.tokenizer.eos_token_id, dtype=attention_mask.dtype)
        mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # 计算 actor/reference 的 log_prob
        # actor_log_probs = self._compute_log_probs(self.actor, outputs, input_ids, mask, responses[:, prompt_len:])
        ref_log_probs = self._compute_log_probs(self.reference, outputs, input_ids, mask, responses[:, prompt_len:])

        return prompts, responses[:, prompt_len:], outputs, ref_log_probs, response_attention_mask, mask

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

    def _compute_reward(self, prompts, responses):
        """计算奖励分数（结合RM分数 + KL 惩罚）"""
        rm_inputs = self.tokenizer([f"{p}\n{r}" for p, r in zip(prompts, responses)],
                                   return_tensors="pt", padding=True, truncation=True)
        rm_inputs = {k: v.to(self.actor.device) for k, v in rm_inputs.items()}

        with torch.no_grad():
            rm_scores = self.rm_model(**rm_inputs).logits.sequeeze(-1)

        # 2.计算KL惩罚
        kl = self.actor_log_probs - self.ref_log_probs
        kl_penalty = self.args.kl_coef * kl

        # 3.总奖励
        rewards = rm_scores - kl_penalty
        return rewards

    def _update_policy(self, ref_log_probs, outputs, responses, critic_values, rewards,
                       # 序列掩码：(timesteps, batch_size)，1=有效，0=无效
                       response_attention_mask, mask: torch.Tensor = None):
        """更新策略网络（PPO 核心逻辑）"""
        # 计算优势函数 （Advantage = Reward - Critic Value）
        target_values, advantages = self._compute_gae_with_mask(rewards.permute(1, 0), critic_values.permute(1, 0), mask=response_attention_mask.permute(1, 0))
        target_values = target_values.permute(1, 0)
        advantages = advantages.permute(1, 0)

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
            'clip_ratio': self.args.clip_ratio,
            'entropy_coeff': self.args.entropy_coeff,
        }

        dataloader = utils.make_iterator(batches, mini_batch_size=self.config.ppo_mini_batch_size,
                                  epochs=self.config.ppo_epochs,
                                  dataloader_kwargs={'shuffle': self.config.shuffle})
        metrics = {}
        for data in dataloader:

            # 模型前向传播
            metric_micro_batch = self.actor.forward_backward_batch(
                batch=data,
                forward_only=False,
                post_process_fn=None,
                meta_info=meta_info
            )  # PP+TP下：LIST[batch_size/pp_size, 1, vocab_size/tp_size]

            for metric in metric_micro_batch:
                utils.append_to_dict(metrics, metric)

            # # 计算 PPO 损失
            # ratio = torch.exp(actor_log_probs - self.ref_log_probs.detach())
            # clipped_ratio = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range)
            # policy_loss = -torch.min(advantages * ratio, clipped_ratio * advantages)
            # policy_loss = self.mask_mean(mask, policy_loss, -1)

            # 更新 actor 模型
            # self.actor.backward(policy_loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

            critic_loss = torch.nn.functional.mse_loss(critic_values, target_values)
            critic_loss = self.mask_mean(mask, critic_loss, -1)
            # 更新 critic 模型
            # self.critic.backward(critic_loss)
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

        return metrics

    def train(self):
        """PPO 训练主循环"""
        self.actor.train()
        self.critic.train()

        for epoch in range(self.args.epochs):
            for step, batch in enumerate(self.dataset):
                # 1. Rollout：生成相应并计算 log prob
                prompts, responses, outputs, ref_log_probs, response_attention_mask, mask = self._rollout(batch)

                # 2. 计算奖励
                rewards = self._compute_reward(prompts, responses)

                # 3. Critic 预测价值
                critic_inputs = self.tokenizer([f"{p}\n{r}" for p, r in zip(prompts, responses)], return_tensors="pt",
                                               padding=True)
                critic_inputs = {k: v.to(self.actor.device) for k, v in critic_inputs.items()}
                critic_values = self.critic(**critic_inputs)

                # 4. 更新策略和价值网络
                self._update_policy(ref_log_probs, critic_values, rewards, response_attention_mask, mask)

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

    @staticmethod
    def _compute_gae_with_mask(
            rewards: torch.Tensor,  # 奖励：(timesteps, batch_size)
            values: torch.Tensor,  # 状态价值：(timesteps, batch_size)
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            mask: torch.Tensor = None  # 序列掩码：(timesteps, batch_size)，1=有效，0=无效
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        支持序列掩码的 GAE 计算（内部自动推导 dones，适配多轨迹不等长序列）
        Returns:
            target_values: 目标价值
            advantages: 标准化后的优势函数
            dones: 内部生成的终止标记（方便外部验证或后续使用）
        """
        # 1. 校验输入维度
        if rewards.ndim != 2 or values.ndim != 2:
            raise ValueError(
                f"rewards/values 需为 2 维张量（timesteps, batch_size），当前维度：rewards={rewards.ndim}, values={values.ndim}")
        timesteps, batch_size = rewards.shape

        # 2. 处理掩码：默认所有步骤有效（兼容单轨迹/同步终止场景）
        if mask is None:
            mask = torch.ones_like(rewards, dtype=torch.float32, device=rewards.device)
        else:
            mask = mask.float()  # 转为 float 方便后续乘法运算

        # -------------------------- 核心：通过 mask 推导 dones --------------------------
        dones = torch.zeros_like(rewards, dtype=torch.bool, device=rewards.device)  # 初始化终止标记
        for t in range(timesteps):
            # 条件1：当前步骤是有效步（mask[t] = 1）
            is_valid = mask[t] == 1.0
            if not is_valid.any():
                continue  # 无有效步骤，跳过

            # 条件2：当前是最后一个时间步，或下一个步骤是无效步
            if t == timesteps - 1:
                # 最后一个时间步的有效步 → 终止步
                dones[t] = is_valid
            else:
                # 下一个步骤是无效步（mask[t+1] = 0）的有效步 → 终止步
                next_is_invalid = mask[t + 1] == 0.0
                dones[t] = is_valid & next_is_invalid

        # 3. GAE 核心计算（基于推导的 dones）
        advantages = torch.zeros_like(rewards, dtype=torch.float32, device=rewards.device)
        running_advantage = torch.zeros(batch_size, dtype=torch.float32, device=rewards.device)  # 每个轨迹独立累积
        dones_float = dones.float()
        next_value = torch.zeros((1, batch_size), dtype=torch.float32, device=rewards.device)

        # 反向遍历时间步
        for t in reversed(range(timesteps)):
            # 仅对有效步骤更新累积优势
            if mask[t].sum() > 0:
                # 下一个状态的价值（终止步下为 0）
                next_v = next_value * (1 - dones_float[t])  # (batch_size,)

                # 计算 TD 残差
                td_error = rewards[t] + gamma * next_v - values[t]  # (batch_size,)

                # 累积 GAE 优势（仅有效步骤参与更新）
                running_advantage = td_error + gamma * gae_lambda * (1 - dones_float[t]) * running_advantage
                advantages[t] = running_advantage * mask[t]  # 屏蔽无效步骤

            # 更新下一个状态的价值（用于 t-1 步计算）
            next_v_current = values[t] * mask[t]
            next_value = torch.where(mask[t].sum() > 0, next_v_current, next_value)

        # 4. 计算目标价值（仅有效步骤有意义）
        target_values = advantages + values * mask  # 无效步骤设为 0

        # 5. 优势函数标准化（仅基于有效步骤，避免无效 0 干扰）
        valid_advantages = advantages[mask == 1.0]
        if valid_advantages.numel() > 0:
            advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        else:
            raise ValueError("所有轨迹步骤均为无效，无法计算 GAE")

        return target_values, advantages

    @staticmethod
    def _compute_log_probs(model, outputs, input_ids, mask, responses):
        """计算生成序列的 log probability（适配 Megatron 并行）"""
        prompt_len = input_ids.shape[1]

        def compute_logprobs_fn(output, data):
            _response = data["responses"]
            _response_length = _response.size(1)
            # 张量并行聚合：收集所有TP进程的logits，得到完整vocab分布
            _logits = model.gather_logits_across_tp(output)
            _logits = _logits[:, -_response_length - 1:-1]
            _log_probs = core_algos.vocab_parallel_log_probs_from_logits(_logits, _response)
            return _log_probs

        batches = TensorDict(
            source={
                "input_ids": outputs,
                "attention_mask": mask,
                "responses": responses,
            },
            batch_size=outputs.shape[0])

        # 模型前向传播
        log_probs = model.forward_backward_batch(
            batch=batches,
            forward_only=True,
            post_process_fn=compute_logprobs_fn
        )  # PP+TP下：LIST[batch_size/pp_size, 1, vocab_size/tp_size]

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            # only on last rank. It should be on every tp rank
            log_probs = torch.cat([o for o in log_probs], dim=0)  # (bs, seq_size)
            log_probs = log_probs.to(torch.float32)
        else:
            log_probs = torch.empty(size=(outputs.shape[0], outputs.shape[1]),
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

        return log_probs

