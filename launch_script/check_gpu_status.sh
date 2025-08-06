#!/usr/bin/env bash

# GPU状态检查脚本
# 从launch_3.sh中提取的节点信息
WORKER_NODES=(job-9482f467-5d0f-41c3-aba6-8c76b1cba5da-master-0 job-548b4817-10df-4275-bbf0-c21e8239532d-master-0 job-12edf21f-dfb2-457a-8ae0-ca11421cf703-master-0 job-2ef01a80-b4b7-4546-822b-cc9ee7162ff2-master-0
job-0c49a1bd-fae4-406e-a805-14162fe03fbc-master-0 job-4a5f9e84-308b-402f-970c-839a36444b0a-master-0 job-3288908a-08cb-42e5-bc04-05c31da928db-master-0
job-e65ac8b9-2a99-45a2-94aa-30b42ff98bed-master-0 job-85c6310a-7e02-437c-a125-23074e2810fe-master-0)
MASTER_ADDR="job-66f41efe-d2d6-4065-8c06-c938b0aebc96-master-0"

# 日志文件路径
LOG_FILE="log/gpu_status.log"

# 创建日志目录
mkdir -p log

# 清空日志文件
> $LOG_FILE

echo "=== GPU Status Check - $(date) ===" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 检查Master节点
echo "=== Checking Master Node: $MASTER_ADDR ===" | tee -a $LOG_FILE
echo "Machine: $MASTER_ADDR" | tee -a $LOG_FILE
echo "Timestamp: $(date)" | tee -a $LOG_FILE
echo "Command: nvidia-smi" | tee -a $LOG_FILE
echo "Output:" | tee -a $LOG_FILE

# 在本地执行nvidia-smi（假设当前机器是master）
nvidia-smi 2>&1 | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 检查所有Worker节点
for i in "${!WORKER_NODES[@]}"; do
    NODE="${WORKER_NODES[$i]}"
    echo "=== Checking Worker Node $((i+1)): $NODE ===" | tee -a $LOG_FILE
    echo "Machine: $NODE" | tee -a $LOG_FILE
    echo "Timestamp: $(date)" | tee -a $LOG_FILE
    echo "Command: nvidia-smi" | tee -a $LOG_FILE
    echo "Output:" | tee -a $LOG_FILE
    
    # SSH到远程节点执行nvidia-smi
    ssh $NODE 'nvidia-smi' 2>&1 | tee -a $LOG_FILE
    
    echo "" | tee -a $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
done

echo "=== GPU Status Check Completed ===" | tee -a $LOG_FILE
echo "Log saved to: $LOG_FILE" | tee -a $LOG_FILE 