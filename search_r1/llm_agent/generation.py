import json
import os

import torch
import re
from typing import List, Dict, Any, Tuple

from megatron.core import parallel_state
from typing_extensions import LiteralString
from dataclasses import dataclass
from volcenginesdkarkruntime import Ark

from torch import Tensor

from utils import torch_functional
from .tensor_helper import TensorHelper, TensorConfig

import requests

client = Ark(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=os.environ.get("ARK_API_KEY"))


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = None
    topk: int = 3


class LLMGenerationManager:
    def __init__(
            self,
            tokenizer,
            actor_model,
            config: GenerationConfig,
            g_config,
            is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor = actor_model
        self.config = config
        self.g_config = g_config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> tuple[Tensor, list[str] | list[Any]]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                         if '</search>' in resp
                         else resp.split('</answer>')[0] + '</answer>'
        if '</answer>' in resp
        else resp
                         for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            # actions, _ = self.env.postprocess_predictions(responses_str)
            # responses_str = [f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in
            #                  enumerate(actions)]
            # print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""

        next_obs_ids = self.tokenizer(
            next_obs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(
                f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: Dict, cur_responses: torch.Tensor,
                              next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        resp = self.tensor_fn.concatenate_with_padding([
            cur_responses,
            next_obs_ids,
        ], pad_to_left=False)
        new_input_ids = self.tensor_fn.concatenate_with_no_padding([
            rollings['input_ids'],
            resp
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        # new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_response_length, effective_len)

        new_rollings = {
            'input_ids': new_input_ids[:, -max_len:],
            # 'position_ids': new_position_ids[:, -max_len:].to('cuda'),
            'attention_mask': new_attention_mask[:, -max_len:]
        }

        return new_rollings

    def _info_masked_concatenate_with_padding(self,
                                              prompt: torch.Tensor,
                                              prompt_with_mask: torch.Tensor,
                                              response: torch.Tensor,
                                              info: torch.Tensor = None,
                                              pad_to_left: bool = True
                                              ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)  # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict,
                           cur_responses: torch.Tensor,
                           next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                next_obs_ids,
                pad_to_left=False
            )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                right_side['responses'],
                right_side['responses_with_info_mask'],
                cur_responses,
                pad_to_left=False
            )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_response_length, effective_len)

        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: Dict) -> Dict:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        prompt_len = active_batch["input_ids"].shape[1]
        print(f"active_batch[attention_mask]={active_batch["attention_mask"]}")
        if num_gpus <= 1:
            output = self.actor.generate(
                input_ids=active_batch["input_ids"].to('cuda'),
                max_length=self.g_config.rollout.max_new_token + prompt_len,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token),
                pad_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
                temperature=self.g_config.rollout.temperature,
                attention_mask=active_batch["attention_mask"].to('cuda'),
                top_k=self.g_config.rollout.top_k,
            )
            output = output.to('cpu')
            return {
                "input_ids": output,
                "responses": output[:, prompt_len:],
            }

        batch_size = active_batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.keys():
            if isinstance(active_batch[key], torch.Tensor):
                active_batch[key] = active_batch[key].long()
        if remainder == 0:
            output = self.actor.generate(
                input_ids=active_batch["input_ids"].to('cuda'),
                max_length=self.g_config.rollout.max_new_token + prompt_len,
                eos_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token),
                pad_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
                temperature=self.g_config.rollout.temperature,
                attention_mask=active_batch["attention_mask"].to('cuda'),
                top_k=self.g_config.rollout.top_k,
            )
            output = output.to('cpu')
            return {
                "input_ids": output,
                "responses": output[:, prompt_len:],
            }

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_active_batch = {}

        for k, v in active_batch.items():
            if isinstance(v, torch.Tensor):
                # Use first sequence as padding template
                pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
                padded_active_batch[k] = torch.cat([v, pad_sequence], dim=0)

        for key in padded_active_batch.keys():
            if isinstance(padded_active_batch[key], torch.Tensor):
                padded_active_batch[key] = padded_active_batch[key].long()

        # Generate with padded batch
        padded_output = self.actor.generate(
            input_ids=padded_active_batch["input_ids"].to('cuda'),
            max_length=self.g_config.rollout.max_new_token + prompt_len,
            eos_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token),
            pad_token_id=self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token),
            temperature=self.g_config.rollout.temperature,
            attention_mask=padded_active_batch["attention_mask"].to('cuda'),
            top_k=self.g_config.rollout.top_k,
        )
        padded_output = padded_output.to('cpu')

        # Remove padding from output
        trimmed_batch = padded_output[:-padding_size]

        # Handle meta_info if present
        # if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
        #     trimmed_meta = {}
        #     for k, v in padded_output.meta_info.items():
        #         if isinstance(v, torch.Tensor):
        #             trimmed_meta[k] = v[:-padding_size]
        #         else:
        #             trimmed_meta[k] = v
        #     padded_output.meta_info = trimmed_meta

        padded_output = trimmed_batch
        return {
            "input_ids": padded_output,
            "responses": padded_output[:, prompt_len:],
        }

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []],
                               'responses_with_info_mask': initial_input_ids[:, []]}

        active_mask = torch.ones(gen_batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings = self.tensor_fn.cut_to_effective_len(
                rollings,
                keys=['input_ids', 'attention_mask'],
                cut_left=False
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = {
                k: v[active_mask] if isinstance(v, torch.Tensor) else v for k, v in rollings.items()
            }
            gen_output = self._generate_with_gpu_padding(rollings_active)
            if parallel_state.get_model_parallel_group().rank() == parallel_state.get_model_parallel_src_rank():
                responses_ids, responses_str = self._postprocess_responses(gen_output['responses'])
                responses_ids, responses_str = self.tensor_fn.example_level_pad(responses_ids, responses_str, active_mask)

                # Execute in environment and process observations
                next_obs, dones, valid_action, is_search = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask
                )

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())
                turns_stats[curr_active_mask] += 1
                valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
                valid_search_stats += torch.tensor(is_search, dtype=torch.int)

                next_obs_ids = self._process_next_obs(next_obs)

                # Update states
                rollings = self._update_rolling_state(
                    rollings,
                    responses_ids,
                    next_obs_ids
                )
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    next_obs_ids
                )

            local_rank = parallel_state.get_model_parallel_group().rank()  # 模型并行内的本地rank
            self.align_shape(local_rank, rollings)
            self.align_shape(local_rank, original_right_side)
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            rollings = {k: v.to(device) for k, v in rollings.items()}
            original_right_side = {k: v.to(device) for k, v in original_right_side.items()}
            torch_functional.broadcast_dict_tensor(
                rollings,
                src=parallel_state.get_model_parallel_src_rank(),
                group=parallel_state.get_model_parallel_group())
            torch_functional.broadcast_dict_tensor(
                original_right_side,
                src=parallel_state.get_model_parallel_src_rank(),
                group=parallel_state.get_model_parallel_group())
            rollings = {k: v.to('cpu') for k, v in rollings.items()}
            original_right_side = {k: v.to('cpu') for k, v in original_right_side.items()}

        meta_info = {}
        # final LLM rollout
        if active_mask.sum():
            rollings = self.tensor_fn.cut_to_effective_len(
                rollings,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = {
                k: v[active_mask] if isinstance(v, torch.Tensor) else v for k, v in rollings.items()
            }
            gen_output = self._generate_with_gpu_padding(rollings_active)

            responses_ids, responses_str = self._postprocess_responses(gen_output['responses'])
            responses_ids, responses_str = self.tensor_fn.example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )

        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def align_shape(self, local_rank, tensor_dict):
        for key in sorted(tensor_dict.keys()):
            t = tensor_dict[key]
            if local_rank == 0:
                # 源进程广播张量属性
                shape = t.shape
                torch.distributed.broadcast_object_list([shape], src=0, group=parallel_state.get_model_parallel_group())
            else:
                # 非源进程接收属性
                shape = None
                torch.distributed.broadcast_object_list([shape], src=0, group=parallel_state.get_model_parallel_group())
                # 对齐张量属性
                new_t = torch.zeros(shape, device=t.device, dtype=t.dtype)
                new_t[:, :t.shape[1]] = t
                tensor_dict[key] = new_t

    def _compose_final_output(self, left_side: Dict,
                              right_side: Dict,
                              meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        # final_output.update(meta_info)

        return final_output, meta_info

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> tuple[
        list[str], list[int], list[int], list[int]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []

        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search_by_doubao(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):

            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert len(search_results) == 0

        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> tuple[
        list[str | None | Any], list[LiteralString | str]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str):  # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                    if action == 'search' and len(content) <= 1:
                        content = ''
                        action = None
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)

        return actions, contents

    def batch_search(self, queries: List[str] = None) -> list[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']

        return [self._passages2string(result) for result in results]

    def batch_search_by_doubao(self, queries: List[str] = None) -> list[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        if not queries:
            print(f"[Warning] queries is empty")
            return []
        tools = [{
            "type": "web_search",
            "max_keyword": 2,
            "sources": [],
        }]
        results = []
        for q in queries:
            query_str = ""
            query_str = "".join([query_str, "<search>", q, "</search>\n"])
            _input = [{
                "content": f"检索<search>和</search>之间的内容一次，并用英文直接返回查询到的结果，最多150个token\n{query_str}",
                "role": "user"}]
            resp = client.responses.create(
                model="doubao-seed-1-6-251015",
                input=_input,
                max_output_tokens=150,
                thinking={"type": "disabled"},
                tools=tools,
            )
            print(f"client.responses.create, input={_input}, resp={resp}")

            max_content = ""
            for chunk in resp.output:  # 遍历每一个实时返回的片段（chunk）
                chunk_type = getattr(chunk, "type", "")
                print(chunk_type)
                if chunk_type == "message":
                    for content in chunk.content:
                        if content.type == "output_text" and len(content.text) > len(max_content):
                            max_content = content.text

            results.append(max_content)

        if not results:
            print(f"[Warning] results is empty")

        return results

    def _batch_search(self, queries):

        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }

        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            if isinstance(doc_item, dict):
                # 场景：document是字典，取contents
                content = doc_item['document']['contents']
                title = content.split("\n")[0]
                text = "\n".join(content.split("\n")[1:])
            elif isinstance(doc_item, str):
                # 尝试解析JSON字符串
                try:
                    doc_dict = json.loads(doc_item)
                    content = doc_dict.get('contents', doc_item).strip()
                except:
                    # 解析失败则直接用原字符串
                    content = doc_item.strip()
                text = content
                title = "_title_"
            else:
                text = "_content_"
                title = "_title_"
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"

        return format_reference
