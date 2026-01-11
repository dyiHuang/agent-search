#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export ARK_API_KEY=    # for doubao api
export RAY_ADDRESS=
export MASTER_ADDR=localhost  # or the IP of rank0 machine
export MASTER_PORT=6000  # ensure this port is free
export GLOO_SOCKET_IFNAME=eth0  # 或你的实际接口
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_NSOCKS_PERTHREAD=8

ray start --address=$RAY_ADDRESS

GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
       main_ppo.py