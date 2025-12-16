import megatron.core.parallel_state as mpu


def get_model_parallel_group(check_initialized=True):
    """Get the model-parallel group the caller rank belongs to."""
    return mpu.get_tensor_model_parallel_group(check_initialized)


def get_model_parallel_rank():
    return mpu.get_tensor_model_parallel_group().rank()


def get_model_parallel_world_size():
    return mpu.get_tensor_model_parallel_world_size()


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    return mpu.get_tensor_model_parallel_src_rank()


def get_tensor_model_parallel_group(check_initialized=True):
    """Get the tensor-model-parallel group the caller rank belongs to."""
    return mpu.get_tensor_model_parallel_group(check_initialized)


def get_pipeline_model_parallel_group(check_initialized=True):
    """Get the pipeline-model-parallel group the caller rank belongs to."""
    return mpu.get_pipeline_model_parallel_group(check_initialized)


def get_data_parallel_group(with_context_parallel=False, partial_data_parallel=False):
    """Get the data-parallel group the caller rank belongs to."""
    return mpu.get_data_parallel_group(with_context_parallel, partial_data_parallel)


def get_data_parallel_group_ranks(with_context_parallel=False, partial_data_parallel=False):
    ws = mpu.get_data_parallel_world_size(with_context_parallel, partial_data_parallel)
    return [i for i in ws]


def get_data_parallel_group_gloo(with_context_parallel=False, partial_data_parallel=False):
    """Get the Gloo data-parallel group the caller rank belongs to."""
    return mpu.get_data_parallel_group_gloo(with_context_parallel, partial_data_parallel)


def get_context_parallel_group(check_initialized=True):
    """Get the context-parallel group the caller rank belongs to."""
    return mpu.get_context_parallel_group(check_initialized)


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context-parallel group that the caller rank belongs to."""
    return mpu.get_context_parallel_global_ranks(check_initialized)


def get_hierarchical_context_parallel_groups(check_initialized=True):
    """Get the inner ring of context parallel group the caller rank belongs to."""
    return mpu.get_hierarchical_context_parallel_groups(check_initialized)


def get_embedding_group(check_initialized=True):
    """Get the embedding group the caller rank belongs to."""
    return mpu.get_embedding_group(check_initialized)


def get_position_embedding_group(check_initialized=True):
    """Get the position embedding group the caller rank belongs to."""
    return mpu.get_position_embedding_group(check_initialized)


def get_amax_reduction_group(with_context_parallel=False, tp_only_amax_red=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    return mpu.get_amax_reduction_group(with_context_parallel, tp_only_amax_red)


def get_tensor_and_data_parallel_group(check_initialized=True, with_context_parallel=False):
    """Get the tensor- and data-parallel group the caller rank belongs to."""
    return mpu.get_tensor_and_data_parallel_group(check_initialized, with_context_parallel)


def get_tensor_and_context_parallel_group(check_initialized=True):
    """Get the tensor- and context-parallel group the caller rank belongs to."""
    return mpu.get_tensor_and_context_parallel_group(check_initialized)


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor-model-parallel group."""
    return mpu.get_tensor_model_parallel_world_size()


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline-model-parallel group."""
    return mpu.get_pipeline_model_parallel_world_size()


def get_tensor_model_parallel_rank():
    """Return caller's rank for the tensor-model-parallel group."""
    return mpu.get_tensor_model_parallel_rank()


def get_pipeline_model_parallel_rank():
    """Return caller's rank for the pipeline-model-parallel group."""
    return mpu.get_pipeline_model_parallel_rank()


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    return mpu.get_tensor_model_parallel_src_rank()


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    return mpu.get_data_parallel_src_rank(with_context_parallel)


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first stage in the current rank's pipeline."""
    return mpu.get_pipeline_model_parallel_first_rank()


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last stage in the current rank's pipeline."""
    return mpu.get_pipeline_model_parallel_last_rank()


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline."""
    return mpu.get_pipeline_model_parallel_next_rank()


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that precedes the caller in the pipeline."""
    return mpu.get_pipeline_model_parallel_prev_rank()


def get_data_parallel_world_size(with_context_parallel=False, partial_data_parallel=False):
    """Return world size for the data parallel group."""
    return mpu.get_data_parallel_world_size(with_context_parallel, partial_data_parallel)


def get_data_parallel_rank(with_context_parallel=False, partial_data_parallel=False):
    """Return caller's rank in the data-parallel group."""
    return mpu.get_data_parallel_rank(with_context_parallel, partial_data_parallel)


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    return mpu.get_context_parallel_world_size()


def get_context_parallel_rank():
    """Return caller's rank in the context-parallel group."""
    return mpu.get_context_parallel_rank()


def get_tensor_and_context_parallel_world_size():
    """Return world size for the tensor and context-parallel group."""
    return mpu.get_tensor_and_context_parallel_world_size()


def get_tensor_and_context_parallel_rank():
    """Return caller's rank in the joint tensor-model-parallel and context-parallel group."""
    return mpu.get_tensor_and_context_parallel_rank()


### Expert-related parallel states functions
def get_expert_model_parallel_group(check_initialized=True):
    """Get the expert-model-parallel group the caller rank belongs to."""
    return mpu.get_expert_model_parallel_group(check_initialized)


def get_expert_model_parallel_world_size():
    """Return world size for the expert-model-parallel group."""
    return mpu.get_expert_model_parallel_world_size()


def get_expert_model_parallel_rank():
    """Return caller's rank in the expert-model-parallel group."""
    return mpu.get_expert_model_parallel_rank()


def get_expert_tensor_parallel_group(check_initialized=True):
    """Get the expert-tensor-parallel group the caller rank belongs to."""
    return mpu.get_expert_tensor_parallel_group(check_initialized)


def get_expert_tensor_parallel_world_size():
    """Return world size for the expert tensor parallel group."""
    return mpu.get_expert_tensor_parallel_world_size()


def get_expert_tensor_parallel_rank():
    """Return my rank for the expert tensor parallel group."""
    return mpu.get_expert_tensor_parallel_rank()


def get_expert_tensor_and_model_parallel_group(check_initialized=True):
    """Get the expert-tensor and expert-model group the caller rank belongs to."""
    return mpu.get_expert_tensor_and_model_parallel_group(check_initialized)


def get_expert_tensor_and_model_parallel_world_size():
    """Return world size for the expert model parallel group times expert tensor parallel group."""
    return mpu.get_expert_tensor_and_model_parallel_world_size()


def get_expert_tensor_and_model_parallel_rank():
    """Return caller's rank in the joint tensor- and expert-model-parallel group."""
    return mpu.get_expert_tensor_and_model_parallel_rank()


def get_expert_tensor_model_pipeline_parallel_group(check_initialized=True):
    """Get expert tensor-model-pipeline parallel group."""
    return mpu.get_expert_tensor_model_pipeline_parallel_group(check_initialized)


def get_expert_data_parallel_group(check_initialized=True, partial_expert_data_parallel=False):
    """Get expert data parallel group."""
    return mpu.get_expert_data_parallel_group(check_initialized, partial_expert_data_parallel)


def get_data_modulo_expert_parallel_group(partial_expert_data_parallel=False):
    """[Deprecated] Get expert data parallel group."""
    return mpu.get_data_modulo_expert_parallel_group(partial_expert_data_parallel)


def get_expert_data_parallel_group_gloo(partial_expert_data_parallel=False):
    """Get expert data parallel group-gloo."""
    return mpu.get_expert_data_parallel_group_gloo(partial_expert_data_parallel)


def get_expert_data_parallel_rank(partial_expert_data_parallel=False):
    """Return caller's rank in the expert data parallel group."""
    return mpu.get_expert_data_parallel_rank(partial_expert_data_parallel)


def get_expert_data_parallel_world_size(partial_expert_data_parallel=False):
    """Return world size for the expert data parallel group."""
    return mpu.get_expert_data_parallel_world_size(partial_expert_data_parallel)


def get_intra_distributed_optimizer_instance_group(check_initialized=True):
    """Get the group of all GPUs in a distributed optimizer instance."""
    return mpu.get_intra_distributed_optimizer_instance_group(check_initialized)


def get_inter_distributed_optimizer_instance_group(check_initialized=True):
    """Get the group spanning the different distributed optimizer instances.
    Attention and MLP/Expert share same inter-instance group, so only built
    inter_partial_expert_data_parallel_group, and return it at here.
    """
    return mpu.get_inter_distributed_optimizer_instance_group(check_initialized)