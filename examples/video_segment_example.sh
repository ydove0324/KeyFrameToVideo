#!/bin/bash

set -e -x

export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL="INFO"

# Finetrainers supports multiple backends for distributed training
BACKEND="ptd"

# GPU configuration
NUM_GPUS=1
CUDA_VISIBLE_DEVICES="0"

# Dataset configuration files
TRAINING_DATASET_CONFIG="path/to/your/training.json"
VALIDATION_DATASET_FILE="path/to/your/validation.json"

# Parallel configuration
parallel_cmd=(
  --parallel_backend $BACKEND 
  --pp_degree 1 
  --dp_degree 1 
  --dp_shards 1 
  --cp_degree 1 
  --tp_degree 1
)

# Model configuration
model_cmd=(
  --model_name "cogvideox"
  --pretrained_model_name_or_path "THUDM/CogVideoX-2B"
)

# Dataset configuration
dataset_cmd=(
  --dataset_config $TRAINING_DATASET_CONFIG
  --dataset_shuffle_buffer_size 32
)

# Video segmentation configuration
# Enable video segmentation with 17 frames per segment (you can also use 9)
video_segment_cmd=(
  --enable_video_segmentation
  --frames_per_segment 17
  --overlap_frames 0
)

# Training configuration
training_cmd=(
  --training_type "lora"
  --seed 42
  --batch_size 1
  --train_steps 1000
  --rank 32
  --lora_alpha 32
  --target_modules "(transformer_blocks|single_transformer_blocks).*(to_q|to_k|to_v|to_out.0)"
  --gradient_accumulation_steps 1
  --gradient_checkpointing
  --checkpointing_steps 200
  --checkpointing_limit 2
)

# Optimizer configuration
optimizer_cmd=(
  --optimizer "adamw"
  --lr 1e-4
  --lr_scheduler "constant_with_warmup"
  --lr_warmup_steps 100
  --beta1 0.9
  --beta2 0.99
  --weight_decay 1e-4
  --epsilon 1e-8
  --max_grad_norm 1.0
)

# Validation configuration
validation_cmd=(
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
)

# Miscellaneous configuration
miscellaneous_cmd=(
  --tracker_name "finetrainers-video-segment-example"
  --output_dir "/path/to/output"
  --init_timeout 600
  --nccl_timeout 600
  --report_to "wandb"
)

# Execute the training script
if [ "$BACKEND" == "accelerate" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    accelerate launch \
        --config_file "examples/accelerate_configs/multi_gpu.yaml" \
        --num_processes $NUM_GPUS \
        train.py \
        "${parallel_cmd[@]}" \
        "${model_cmd[@]}" \
        "${dataset_cmd[@]}" \
        "${video_segment_cmd[@]}" \
        "${training_cmd[@]}" \
        "${optimizer_cmd[@]}" \
        "${validation_cmd[@]}" \
        "${miscellaneous_cmd[@]}"
elif [ "$BACKEND" == "ptd" ]; then
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    torchrun \
        --nproc_per_node $NUM_GPUS \
        --nnodes 1 \
        --node_rank 0 \
        --master_addr localhost \
        --master_port 29500 \
        train.py \
        "${parallel_cmd[@]}" \
        "${model_cmd[@]}" \
        "${dataset_cmd[@]}" \
        "${video_segment_cmd[@]}" \
        "${training_cmd[@]}" \
        "${optimizer_cmd[@]}" \
        "${validation_cmd[@]}" \
        "${miscellaneous_cmd[@]}"
else
    echo "Unknown backend: $BACKEND"
    exit 1
fi 