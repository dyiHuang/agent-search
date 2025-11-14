import deepspeed
import torch
from megatron.core import parallel_state
from transformers import AutoTokenizer
from peft import LoraConfig

from tensordict import TensorDict
from tensor_parallel import vocab_parallel_log_probs_from_logits

import core_algos
from utils import utils
from modeling_qwen_megatron import build_qwen2_megatron_model
from omegaconf import OmegaConf, open_dict
import reward_score
from deepspeed.accelerator import get_accelerator

if get_accelerator().device_name() == 'cuda':
    from apex.optimizers import FusedAdam as Adam
else:
    from torch.optim import Adam

from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.num_microbatches_calculator import get_num_microbatches


class MegatronDeepSpeedPPOTrainer:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.qwen_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1. 初始化分布式环境（Megatron + DeepSpeed 协同）
        self._init_distributed()

        # 2. 配置 LoRa
        self.lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=[],
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 3. 构建 PPO 三模型
        self.actor = build_qwen2_megatron_model(config.qwen_model_path, lora_config=self.lora_config)
        self.critic = build_qwen2_megatron_model(config.qwen_model_path, lora_config=self.lora_config, is_critic=True)
        self.reference = build_qwen2_megatron_model(config.qwen_model_path)
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False

        # 4. 初始化 Deepspeed 引擎（ZeRO 优化）
        self._init_deepspeed()

        # 5. 加载数据集（分布式采样）
        self._create_dataloader()

    def _init_distributed(self):
        """初始化 Megatron + Deepspeed 分布式环境"""
        deepspeed.init_distributed(dist_backend="nccl")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.config.tp_size if self.config.tp_size > 1 else 1,
            pipeline_model_parallel_size=self.config.pp_size if self.config.pp_size > 1 else 1,
        )

    def _init_deepspeed(self):
        """初始化deepspeed引擎（仅优化 LoRa 参数）"""
        utils.print_rank_0("DeepSpeed is enabled.")
        # 优化配置
        optimizer = Adam(
            self.actor.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01
        )
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer, config=self.config.actor.optimizer)

        # DeepSpeed 配置（从 JSON 加载）
        ds_config = deepspeed.DeepSpeedConfig(self.config.ds_config_file)

        # 初始化 DeepSpeed 引擎
        self.actor, self.optimizer, _, _ = deepspeed.initialize(
            model=self.actor,
            optimizer=optimizer,
            config=ds_config,
            mpu=parallel_state,
            lr_scheduler=opt_param_scheduler,
            # model_parameters=self.actor.parameters()
        )

        # 优化配置
        critic_optimizer = Adam(
            self.critic.parameters(),
            lr=self.config.learning_rate,
        )

        critic_opt_param_scheduler = get_optimizer_param_scheduler(critic_optimizer,
                                                                   config=self.config.critic.optimizer)

        self.critic, self.critic_optimizer, _, _ = deepspeed.initialize(
            model=self.critic,
            optimizer=critic_optimizer,
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
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _rollout(self, batch):
        """Rollout阶段：生成 response 并计算 log prob（分布式采用）"""
        # prompts = batch["prompt"].to('cuda')
        # inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = batch["input_ids"].to('cuda')
        attention_mask = batch["attention_mask"].to('cuda')

        # 生成 response（actor模型）
        with torch.no_grad():
            outputs = self.actor.generate(
                input_ids=input_ids,
                max_length=self.config.max_new_token,
                eos_token=self.tokenizer.eos_token,
                pad_token_id=self.tokenizer.pad_token,
                temperature=self.config.temperature,
                attention_mask=attention_mask,
                top_k=self.config.top_k,
            )
            prompt_len = input_ids.shape[1]
            responses = self.tokenizer.decode(outputs, skip_special_tokens=True)
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
        rewards = reward_score.RewardManager(tokenizer=self.tokenizer, num_examine=0)(batch)
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
            'clip_ratio': self.config.clip_ratio,
            'entropy_coeff': self.config.entropy_coeff,
        }

        dataloader = utils.make_iterator(batches, mini_batch_size=self.config.ppo_mini_batch_size,
                                         epochs=self.config.ppo_epochs,
                                         dataloader_kwargs={'shuffle': self.config.shuffle})
        metrics = {}
        for data in dataloader:
            self.optimizer.zero_grad()
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
            utils.print_rank_0(f"update_successful:{update_successful}, increment:{increment}")

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

        for epoch in range(self.config.epochs):
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
                self._update_critic(dialogue_ids, attention_mask, responses, critic_values, returns)

                # 6. 更新策略
                self._update_policy(ref_log_probs, responses, advantages, attention_mask)

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
            'cliprange_value': self.config.cliprange_value,
        }

        dataloader = utils.make_iterator(batches, mini_batch_size=self.config.ppo_mini_batch_size,
                                         epochs=self.config.ppo_epochs,
                                         dataloader_kwargs={'shuffle': self.config.shuffle})
        metrics = {}
        for data in dataloader:
            self.critic_optimizer.zero_grad()
            metric_micro_batch = self.critic.forward_backward_batch(batch=data, meta_info=meta_info)

            for metric in metric_micro_batch:
                utils.append_to_dict(metrics, metric)  # append the metric from this micro-batch to global metrics.

            self.critic_optimizer.step()
        return metrics


def get_optimizer_param_scheduler(optimizer, config):
    """Build the learning rate scheduler."""
    # Iteration-based training.
    if config.train_iters:
        if config.lr_decay_iters is None:
            config.lr_decay_iters = config.train_iters
        lr_decay_steps = config.lr_decay_iters * config.global_batch_size
        wd_incr_steps = config.train_iters * config.global_batch_size
        if config.lr_warmup_fraction is not None:
            lr_warmup_steps = config.lr_warmup_fraction * lr_decay_steps
        else:
            lr_warmup_steps = config.lr_warmup_iters * config.global_batch_size
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        max_lr=config.lr,
        min_lr=config.min_lr,
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
