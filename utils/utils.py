from datetime import datetime
from typing import Dict, Union
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
    def __init__(self, tensor_dict: TensorDict):  # 明确输入为字典（key是字符串，value是张量）
        super().__init__()
        self.tensor_dict = tensor_dict

        # 1. 优先从张量的第一维获取batch_size（最可靠）
        first_key = next(iter(tensor_dict.keys().__iter__()))  # 正确获取第一个key
        first_tensor = tensor_dict[first_key]
        self.batch_size = first_tensor.shape[0]  # 张量第一维是样本数（整数）

        # 2. 校验所有张量的batch维一致
        for key, tensor in tensor_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Key {key} 的值不是张量（类型：{type(tensor)}）")
            assert tensor.shape[0] == self.batch_size, \
                f"Key {key} 的batch维（{tensor.shape[0]}）与预期（{self.batch_size}）不一致"

    def __len__(self) -> int:
        """返回样本总数（确保是整数）"""
        return self.batch_size

    def __getitem__(self, idx: int) -> TensorDict:
        """按索引提取单个样本（保留字典结构）"""
        return self.tensor_dict[idx]


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def collate_tensordict(batch: list[TensorDict]) -> TensorDict:
    """
    基础版collate_fn：拼接batch维，支持嵌套TensorDict
    Args:
        batch: 单个样本TensorDict的列表
    Returns:
        batch级TensorDict（所有张量第0维为batch size）
    """
    if not batch:
        raise ValueError("Batch is empty")

    # 取第一个样本作为模板，获取所有键（假设所有样本结构一致）
    sample = batch[0]
    if not isinstance(sample, TensorDict):
        raise TypeError("Each sample must be a TensorDict")

    # 递归处理每个键：如果是TensorDict则递归，否则拼接张量
    def _collate(value_list: list):
        # 检查第一个元素类型：嵌套TensorDict则递归
        if isinstance(value_list[0], TensorDict):
            return collate_tensordict(value_list)
        # 否则拼接张量（确保所有元素都是张量）
        elif isinstance(value_list[0], torch.Tensor):
            # 统一设备（可选，按第一个张量的设备）
            device = value_list[0].device
            value_list = [v.to(device) for v in value_list]
            # 拼接batch维（dim=0）
            return torch.stack(value_list).contiguous()
        # 处理非张量类型（如标量、列表，可选）
        else:
            return torch.tensor(value_list)  # 转换为张量后拼接

    # 遍历所有键，对每个键的所有样本值执行_collate
    batch_data = {
        key: _collate([sample[key] for sample in batch])
        for key in sample.keys()
    }

    # 构造batch级TensorDict（保留原样本的device和其他属性）
    return TensorDict(
        batch_data,
        batch_size=len(batch),  # 显式指定batch size
        device=sample.device
    )


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
                                  collate_fn=collate_tensordict,
                                  generator=generator,
                                  **dataloader_kwargs)

    def get_data():
        for _ in range(epochs):
            for d in train_dataloader:
                yield d

    return iter(get_data())


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


import torch


def find_tensor_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, atol: float = 1e-7, rtol: float = 1e-6) -> list:
    """
    对比两个张量的元素差异，返回不相等元素的维度位置

    Args:
        tensor1: 第一个张量
        tensor2: 第二个张量
        atol: 绝对误差容限（用于浮点数比较）
        rtol: 相对误差容限（用于浮点数比较）

    Returns:
        不相等元素的位置列表，每个元素是一个元组表示维度索引
    """
    # 1. 检查形状是否匹配
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"张量形状不匹配: {tensor1.shape} vs {tensor2.shape}")

    # 2. 检查数据类型和设备是否一致
    if tensor1.dtype != tensor2.dtype:
        print_rank_0(f"警告：张量数据类型不同: {tensor1.dtype} vs {tensor2.dtype}")
    if tensor1.device != tensor2.device:
        raise ValueError(f"张量设备不匹配: {tensor1.device} vs {tensor2.device}")

    # 3. 逐元素比较（区分整数和浮点数）
    if torch.is_floating_point(tensor1):
        # 浮点数/复数用isclose处理精度
        diff_mask = ~torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    else:
        # 整数直接用相等判断
        equal_mask = torch.equal(tensor1, tensor2)
        if equal_mask:
            return []
        diff_mask = tensor1 != tensor2

    # 5. 提取差异位置
    if not diff_mask.any():
        return []
    diff_indices = torch.nonzero(diff_mask, as_tuple=False).tolist()
    return [tuple(idx) for idx in diff_indices]

