#!/usr/bin/env bash

# 分布式下载节点配置
WORKER_NODES=(job-9482f467-5d0f-41c3-aba6-8c76b1cba5da-master-0 job-548b4817-10df-4275-bbf0-c21e8239532d-master-0 job-2ef01a80-b4b7-4546-822b-cc9ee7162ff2-master-0 job-4a5f9e84-308b-402f-970c-839a36444b0a-master-0 job-3288908a-08cb-42e5-bc04-05c31da928db-master-0 job-0c49a1bd-fae4-406e-a805-14162fe03fbc-master-0 job-12edf21f-dfb2-457a-8ae0-ca11421cf703-master-0)
NUM_NODES=8
DOWNLOAD_SCRIPT="script/download_koala_data.py"

MASTER_ADDR="job-66f41efe-d2d6-4065-8c06-c938b0aebc96-master-0"

# 解析命令行参数
KILL_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --kill-only)
            KILL_ONLY=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--kill-only]"
            exit 1
            ;;
    esac
done

# 如果只是kill模式，则只kill所有tmux会话
if [ "$KILL_ONLY" = true ]; then
    echo "Killing all koala-download tmux sessions..."
    
    # Kill master node tmux session
    echo "Killing master node koala-download session..."
    tmux kill-session -t koala-download 2>/dev/null || echo "No koala-download session found on master"
    
    # Kill worker nodes tmux sessions
    for NODE in "${WORKER_NODES[@]}"; do
        echo "Killing koala-download session on $NODE..."
        ssh $NODE "tmux kill-session -t koala-download 2>/dev/null || echo 'No koala-download session found on $NODE'"
    done
    
    echo "All koala-download tmux sessions killed."
    exit 0
fi

# Master节点启动 (node_rank=0)
echo "Starting Koala data download on Master node (node_rank=0)"
export NODE_RANK=0
export WORLD_SIZE=$NUM_NODES

# 在Master节点上启动tmux会话
tmux kill-session -t koala-download 2>/dev/null || true
tmux new-session -A -d -s koala-download \
    -e NODE_RANK=$NODE_RANK \
    -e WORLD_SIZE=$WORLD_SIZE
tmux send-keys -t koala-download "source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && cd /share/project/huangxu/workspace/KeyFrameToVideo && python $DOWNLOAD_SCRIPT --world_size $WORLD_SIZE --node_rank $NODE_RANK" C-m

# Worker节点启动 (node_rank=1-7)
for i in "${!WORKER_NODES[@]}"; do
    NODE="${WORKER_NODES[$i]}"
    WORKER_RANK=$((i + 1))  # Worker节点的rank从1开始
    
    echo "Starting Koala data download on $NODE (node_rank=$WORKER_RANK)"
    ssh $NODE 'bash -c "
        source /opt/conda/etc/profile.d/conda.sh && \
        # Kill existing tmux session if it exists
        tmux kill-session -t koala-download 2>/dev/null || true && \
        # First create tmux session with environment variables
        tmux new-session -A -d -s koala-download \
            -e NODE_RANK='"$WORKER_RANK"' \
            -e WORLD_SIZE='"$NUM_NODES"' && \
        # Then send commands to the session
        tmux send-keys -t koala-download \"source /opt/conda/etc/profile.d/conda.sh && conda activate py310 && cd /share/project/huangxu/workspace/KeyFrameToVideo && python '"$DOWNLOAD_SCRIPT"' --world_size '"$NUM_NODES"' --node_rank '"$WORKER_RANK"'\" C-m 
    " ' &
done

# 等待所有节点完成
echo "Waiting for all nodes to complete..."
wait

echo "-------------------- Finished executing distributed Koala data download --------------------"

# 显示所有tmux会话状态
echo "Checking tmux sessions status:"
echo "Master node koala-download session:"
tmux list-sessions | grep koala-download || echo "No koala-download session found on master"

for NODE in "${WORKER_NODES[@]}"; do
    echo "Worker node $NODE koala-download session:"
    ssh $NODE "tmux list-sessions | grep koala-download || echo 'No koala-download session found on $NODE'"
done 