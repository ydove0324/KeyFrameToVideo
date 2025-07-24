#!/usr/bin/env bash

# 分布式训练节点配置
WORKER_NODES=(job-4a5f9e84-308b-402f-970c-839a36444b0a-master-0 job-3288908a-08cb-42e5-bc04-05c31da928db-master-0)
NUM_NODES=3 # Master + Worker
MASTER_GPUS=8
WORKER_GPUS_LIST=(8 8)
MASTER_ADDR="job-0c49a1bd-fae4-406e-a805-14162fe03fbc-master-0"
MASTER_PORT="23456"
JOB_ID="100"

# 训练脚本路径
TRAIN_SCRIPT="examples/training/sft/wan/video-reconstruction_exp2/train.sh"

# Master节点启动
echo "Starting training on Master node ($MASTER_GPUS GPUs)"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NUM_NODES=$NUM_NODES
export NODE_RANK=0
export NUM_GPUS=$MASTER_GPUS
export JOB_ID=$JOB_ID

nohup bash $TRAIN_SCRIPT >> log/20250720_video_reconstruction_capsfusion.log 2>&1 < /dev/null &
# Worker节点启动
for i in "${!WORKER_NODES[@]}"; do
    NODE="${WORKER_NODES[$i]}"
    WORKER_RANK=$((i + 1))  # Worker节点的rank从1开始
    WORKER_GPUS=${WORKER_GPUS_LIST[$i]}
    
    echo "Local WORKER_RANK=$WORKER_RANK"
    echo "Starting training on $NODE ($WORKER_GPUS GPUs, node_rank=$WORKER_RANK)"
    ssh $NODE 'bash -c "
        source /opt/conda/etc/profile.d/conda.sh && \
        # Kill existing tmux session if it exists
        tmux kill-session -t video-train 2>/dev/null || true && \
        # First create tmux session with environment variables
        tmux new-session -A -d -s video-train \
            -e NODE_RANK='"$WORKER_RANK"' \
            -e MASTER_ADDR='"$MASTER_ADDR"' \
            -e MASTER_PORT='"$MASTER_PORT"' \
            -e NUM_NODES='"$NUM_NODES"' \
            -e NUM_GPUS='"$WORKER_GPUS"' \
            -e JOB_ID='"$JOB_ID"' \
            -e WANDB_MODE=offline && \
        # Then send commands to the session
        tmux send-keys -t video-train \"source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && cd /share/project/huangxu/workspace/KeyFrameToVideo && bash '"$TRAIN_SCRIPT"'\" C-m 
    " ' &
done

# 等待所有节点完成
echo "Waiting for all nodes to complete..."
wait

echo "-------------------- Finished executing distributed training --------------------"


