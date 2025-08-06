#!/usr/bin/env bash

# 分布式训练节点配置
WORKER_NODES=(job-9482f467-5d0f-41c3-aba6-8c76b1cba5da-master-0 job-548b4817-10df-4275-bbf0-c21e8239532d-master-0 job-66f41efe-d2d6-4065-8c06-c938b0aebc96-master-0 job-2ef01a80-b4b7-4546-822b-cc9ee7162ff2-master-0
job-0c49a1bd-fae4-406e-a805-14162fe03fbc-master-0 job-4a5f9e84-308b-402f-970c-839a36444b0a-master-0 job-3288908a-08cb-42e5-bc04-05c31da928db-master-0
job-e65ac8b9-2a99-45a2-94aa-30b42ff98bed-master-0 job-85c6310a-7e02-437c-a125-23074e2810fe-master-0)
NUM_NODES=10 # Master + Worker
MASTER_GPUS=8
WORKER_GPUS_LIST=(4 4 8 8 8 8 8 4 4)
MASTER_ADDR="job-12edf21f-dfb2-457a-8ae0-ca11421cf703-master-0"
MASTER_PORT="23456"
JOB_ID="100"

# 训练脚本路径
TRAIN_SCRIPT="examples/training/sft/wan/video-pretrain-first-frame-condition/train.sh"

# Master节点启动
echo "Starting training on Master node ($MASTER_GPUS GPUs)"
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NUM_NODES=$NUM_NODES
export NODE_RANK=0
export NUM_GPUS=$MASTER_GPUS
export JOB_ID=$JOB_ID

nohup bash $TRAIN_SCRIPT >> log/20250804_video_pretrain_koala_pexel_first_frame_condition_random_key_frame_interval.log 2>&1 < /dev/null &
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


