#!/bin/bash

# 分布式训练节点配置
WORKER_NODES=("job-58d3a4bd-2474-42ca-ae7b-0f7f59590507-worker-0" "job-58d3a4bd-2474-42ca-ae7b-0f7f59590507-worker-1")
NUM_NODES=3  # Master + 2 Workers
MASTER_GPUS=8
WORKER_GPUS=4
MASTER_ADDR="172.24.93.172"
MASTER_PORT="29500"
JOB_ID="100"

# 环境变量设置
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="DEBUG"

# world_size环境变量配置
MASTER_WORLD_SIZE=8
WORKER_WORLD_SIZE=4

# 训练后端配置
BACKEND="ptd"

# 数据集配置文件
TRAINING_DATASET_CONFIG="examples/training/sft/wan/pexel/training.json"
VALIDATION_DATASET_FILE="examples/training/sft/wan/pexel/validation.json"

# 并行策略配置
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_8="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_16="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 16 --dp_shards 1 --cp_degree 1 --tp_degree 1"

# 模型参数
model_cmd=(
  --model_name "wan_ibq_key_frame"
  --pretrained_model_name_or_path "/share/project/huangxu/model/Wan2.1-T2V-1.3B-diffusers"
)

# 数据集参数
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 10
)

# 数据加载器参数
dataloader_cmd=(
  --dataloader_num_workers 0
)

# 扩散模型参数
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
)

# 训练参数
training_cmd=(
  --training_type "full-finetune"
  --seed 42
  --batch_size 4
  --train_steps 15000
  --gradient_accumulation_steps 2
  --gradient_checkpointing
  --checkpointing_steps 2000
  --checkpointing_limit 2
  --transformer_id "/share/project/huangxu/model/Wan2.1-KeyFrame2V-1.3B/transformer"
  --enable_slicing
  --enable_tiling
)

# 优化器参数
optimizer_cmd=(
  --optimizer "adamw"
  --lr 5e-5
  --lr_scheduler "constant_with_warmup"
  --lr_warmup_steps 750
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# 验证参数
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 500
)

# 其他参数
miscellaneous_cmd=(
  --tracker_name "finetrainers-wan-ibq-key-frame-pexel-part2_0123"
  --output_dir "/share/project/huangxu/wan-ibq-key-frame-pexel-part2_0123"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Master节点命令（8卡，DDP8）
MASTER_CMD="torchrun \
    --nnodes=$NUM_NODES \
    --node_rank=0 \
    --nproc_per_node=$MASTER_GPUS \
    --rdzv_id=$JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    $DDP_16 \
    $(printf '%s ' "${model_cmd[@]}") \
    $(printf '%s ' "${dataset_cmd[@]}") \
    $(printf '%s ' "${dataloader_cmd[@]}") \
    $(printf '%s ' "${diffusion_cmd[@]}") \
    $(printf '%s ' "${training_cmd[@]}") \
    $(printf '%s ' "${optimizer_cmd[@]}") \
    $(printf '%s ' "${validation_cmd[@]}") \
    $(printf '%s ' "${miscellaneous_cmd[@]}")"

# 在Master节点上直接运行命令（后台运行）
echo "Starting torchrun on Master node (8 GPUs with DDP8)"
export WORLD_SIZE=$MASTER_WORLD_SIZE
$MASTER_CMD &

# 在Worker节点上通过ssh运行命令
for i in "${!WORKER_NODES[@]}"; do
    NODE="${WORKER_NODES[$i]}"
    WORKER_RANK=$((i + 1))  # Worker节点的rank从1开始
    
    # 为每个Worker节点创建对应的命令
    WORKER_CMD="torchrun \
        --nnodes=$NUM_NODES \
        --node_rank=$WORKER_RANK \
        --nproc_per_node=$WORKER_GPUS \
        --rdzv_id=$JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train.py \
        $DDP_16 \
        $(printf '%s ' "${model_cmd[@]}") \
        $(printf '%s ' "${dataset_cmd[@]}") \
        $(printf '%s ' "${dataloader_cmd[@]}") \
        $(printf '%s ' "${diffusion_cmd[@]}") \
        $(printf '%s ' "${training_cmd[@]}") \
        $(printf '%s ' "${optimizer_cmd[@]}") \
        $(printf '%s ' "${validation_cmd[@]}") \
        $(printf '%s ' "${miscellaneous_cmd[@]}")"
    
    echo "Starting torchrun on $NODE (4 GPUs with DDP4, node_rank=$WORKER_RANK)"
    ssh $NODE "source /opt/conda/etc/profile.d/conda.sh && cd /share/project/huangxu/workspace/KeyFrameToVideo && conda activate py310 && export WORLD_SIZE=$WORKER_WORLD_SIZE && $WORKER_CMD" &
done

# 等待所有节点完成
echo "Waiting for all nodes to complete..."
wait

echo "-------------------- Finished executing distributed training --------------------"