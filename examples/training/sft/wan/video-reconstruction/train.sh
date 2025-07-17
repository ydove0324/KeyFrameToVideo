#!/bin/bash

set -e -x

# Default distributed training settings
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
NUM_GPUS=${NUM_GPUS:-8}
JOB_ID=${JOB_ID:-"1"}

# Environment variables
export WANDB_MODE="offline"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
export TORCH_NCCL_BLOCKING_WAIT=1

# Training backend
BACKEND="ptd"

# Dataset configuration
TRAINING_DATASET_CONFIG="examples/training/sft/wan/video-reconstruction/training.json"
VALIDATION_DATASET_FILE="examples/training/sft/wan/video-reconstruction/validation.json"

# Parallel strategy configuration
DDP_1="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 1 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_2="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 2 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_4="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 4 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_8="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 8 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_16="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 16 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_32="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 32 --dp_shards 1 --cp_degree 1 --tp_degree 1"
DDP_24="--parallel_backend $BACKEND --pp_degree 1 --dp_degree 24 --dp_shards 1 --cp_degree 1 --tp_degree 1"

# Parallel arguments
parallel_cmd=(
  $DDP_32
)

# Model arguments
model_cmd=(
  --model_name "wan_ibq_key_frame"
  --pretrained_model_name_or_path "/share/project/huangxu/model/Wan2.1-T2V-1.3B-diffusers"
)

# Dataset arguments
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 10
)

# Dataloader arguments
dataloader_cmd=(
  --dataloader_num_workers 0
)

# Diffusion arguments
diffusion_cmd=(
  --flow_weighting_scheme "logit_normal"
  --flow_logit_mean 0.5
  --flow_logit_std 1
)
# Training arguments
training_cmd=(
  --training_type "full-finetune"
  --seed 42
  --batch_size 4
  --image_batch_size 16
  --train_steps 20000
  --gradient_accumulation_steps 4
  --gradient_checkpointing
  --resume_from_checkpoint "10000"
  --checkpointing_steps 250
  --checkpointing_limit 10
  --transformer_id "/share/project/huangxu/model/wan-ibq-key-frame-reconstruction-warmup/model_weights/005000/transformer"
  --enable_slicing
  --enable_tiling
)

# Optimizer arguments
optimizer_cmd=(
  --optimizer "adamw"
  --lr 1e-4
  --lr_scheduler "constant_with_warmup"
  --lr_warmup_steps 500
  --lr_num_cycles 1
  --beta1 0.9
  --beta2 0.95
  --weight_decay 0.02
  --epsilon 1e-8
  --max_grad_norm 0.05
)

# Validation arguments
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 500000
)

# Miscellaneous arguments
miscellaneous_cmd=(
  --tracker_name "finetrainers-wan-ibq-key-frame-video-reconstruction"
  --output_dir "/share/project/huangxu/model/wan-ibq-key-frame-video-reconstruction"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Execute training with torchrun
echo "torchrun \\
    --nnodes=$NUM_NODES \\
    --node_rank=$NODE_RANK \\
    --nproc_per_node=$NUM_GPUS \\
    --rdzv_id=$JOB_ID \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
    train.py \\
      ${parallel_cmd[@]} \\
      ${model_cmd[@]} \\
      ${dataset_cmd[@]} \\
      ${dataloader_cmd[@]} \\
      ${diffusion_cmd[@]} \\
      ${training_cmd[@]} \\
      ${optimizer_cmd[@]} \\
      ${validation_cmd[@]} \\
      ${miscellaneous_cmd[@]}"

torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=$NUM_GPUS \
  --rdzv_id=$JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py \
    "${parallel_cmd[@]}" \
    "${model_cmd[@]}" \
    "${dataset_cmd[@]}" \
    "${dataloader_cmd[@]}" \
    "${diffusion_cmd[@]}" \
    "${training_cmd[@]}" \
    "${optimizer_cmd[@]}" \
    "${validation_cmd[@]}" \
    "${miscellaneous_cmd[@]}"

echo -ne "-------------------- Finished executing script --------------------\n\n" 



