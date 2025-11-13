from typing import Dict
from tensordict import TensorDict
from torch.utils.data import DataLoader, Dataset
import torch


def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis) / mask.sum(axis=axis)


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


class TensorDictDataset(Dataset):
    def __init__(self, tensor_dict: TensorDict):
        super().__init__()
        self.tensor_dict = tensor_dict
        # 自动获取 batch_size（优先用 TensorDict 的 batch_size 属性，没有则取第一个 key 的第 0 维）
        if hasattr(tensor_dict, "batch_size"):
            self.batch_size = tensor_dict.batch_size
        else:
            # 确保所有 key 的张量第 0 维（batch 维）一致
            first_key = next(iter(tensor_dict.keys()))
            self.batch_size = tensor_dict[first_key].shape[0]
            # 校验所有 key 的 batch 维一致（避免报错）
            for key, tensor in tensor_dict.items():
                assert tensor.shape[0] == self.batch_size, f"Key {key} 的 batch 维大小（{tensor.shape[0]}）与预期（{self.batch_size}）不一致"

    def __len__(self) -> int:
        """返回样本总数"""
        return self.batch_size

    def __getitem__(self, idx: int) -> TensorDict:
        """按索引提取单个样本（保留 TensorDict 结构）"""
        return self.tensor_dict[idx]  # 关键：TensorDict 支持索引切片，自动拆分单个样本


def collate_fn(x: list[TensorDict]):
    sample_keys = set(x[0].keys().__iter__())
    for data in x[1:]:
        if set(data.keys().__iter__()) != sample_keys:
            raise KeyError(
                f"样本key不一致：{set(data.keys().__iter__())} vs {sample_keys}"
            )
    sample_keys = list(sample_keys)
    batch = []
    for data in x:
        # 替换原堆叠逻辑（以处理变长序列为例）
        for key in sample_keys:
            tensors = data[key]
            if all(t.shape == tensors[0].shape for t in tensors):
                # 形状一致直接堆叠
                data[key] = torch.stack(tensors).contiguous()
            else:
                # 形状不一致：填充到最大长度（示例用0填充，可自定义）
                max_len = max(t.shape[0] for t in tensors)
                padded_tensors = []
                for t in tensors:
                    pad_len = max_len - t.shape[0]
                    padded = torch.nn.functional.pad(t, (0, pad_len))  # 右侧填充
                    padded_tensors.append(padded)
                data[key] = torch.stack(padded_tensors).contiguous()
        batch.append(data)
    return batch


def make_iterator(data: TensorDict, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
    """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
    dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

    Args:
        mini_batch_size (int): mini-batch size when iterating the dataset. We require that
            ``batch.batch_size[0] % mini_batch_size == 0``
        epochs (int): number of epochs when iterating the dataset.
        dataloader_kwargs: internally, it returns a DataLoader over the batch.
            The dataloader_kwargs is the kwargs passed to the DataLoader

    Returns:
        Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
        ``self.batch.batch_size * epochs // mini_batch_size``
        :param dataloader_kwargs:
        :param seed:
        :param epochs:
        :param mini_batch_size:
        :param data:
    """
    assert data.batch_size[0] % mini_batch_size == 0, f"{data.batch_size[0]} % {mini_batch_size} != 0"
    # we can directly create a dataloader from TensorDict
    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    else:
        generator = None

    assert isinstance(dataloader_kwargs, Dict)
    dataset = TensorDictDataset(data)
    train_dataloader = DataLoader(dataset=dataset,
                                  batch_size=mini_batch_size,
                                  collate_fn=collate_fn,
                                  generator=generator,
                                  **dataloader_kwargs)

    def get_data():
        for _ in range(epochs):
            for d in train_dataloader:
                yield d

    return iter(get_data())