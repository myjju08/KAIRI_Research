#!/bin/bash

# DeepCache Layer FID 계산을 백그라운드에서 실행하는 스크립트

# 로그 파일 설정
LOG_FILE="deepcache_layer_fid_calculation.log"
PID_FILE="deepcache_layer_fid.pid"

# 시작 시간 기록
echo "=== DeepCache Layer FID Calculation Started ===" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE

# Python 스크립트 실행 (백그라운드)
nohup python calculate_deepcache_layer_fid.py \
    --experiment_dir . \
    --output_file deepcache_layer_fid_results.csv \
    --progress_file deepcache_layer_fid_progress.csv \
    --save_interval 5 \
    > $LOG_FILE 2>&1 &

# 프로세스 ID 저장
echo $! > $PID_FILE
echo "Process started with PID: $(cat $PID_FILE)" | tee -a $LOG_FILE

# 프로세스 상태 확인
sleep 2
if ps -p $(cat $PID_FILE) > /dev/null; then
    echo "Process is running successfully" | tee -a $LOG_FILE
    echo "To check progress: tail -f $LOG_FILE" | tee -a $LOG_FILE
    echo "To stop process: kill $(cat $PID_FILE)" | tee -a $LOG_FILE
else
    echo "Process failed to start" | tee -a $LOG_FILE
    exit 1
fi 