import megatron.core.parallel_state as mpu
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_data_parallel_rank,
    get_data_parallel_group,
    get_data_parallel_world_size,
)


def add_missing_mpu_methods():
    """
    为 Megatron-LM 最新版本的 mpu 模块动态添加缺失的旧版 API，
    以兼容依赖这些接口的旧代码（如 DeepSpeed）。
    """
    # 检查并添加 model parallel 相关方法
    if not hasattr(mpu, 'get_model_parallel_rank'):
        mpu.get_model_parallel_rank = get_tensor_model_parallel_rank
        print("WARNING: Monkey-patched 'mpu.get_model_parallel_rank' using 'get_tensor_model_parallel_rank'.")

    if not hasattr(mpu, 'get_model_parallel_group'):
        mpu.get_model_parallel_group = get_tensor_model_parallel_group
        print("WARNING: Monkey-patched 'mpu.get_model_parallel_group' using 'get_tensor_model_parallel_group'.")

    if not hasattr(mpu, 'get_model_parallel_world_size'):
        mpu.get_model_parallel_world_size = get_tensor_model_parallel_world_size
        print(
            "WARNING: Monkey-patched 'mpu.get_model_parallel_world_size' using 'get_tensor_model_parallel_world_size'.")
