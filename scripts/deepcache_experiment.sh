#!/bin/bash
set -euo pipefail

# 스크립트 기준 프로젝트 루트 경로 계산 및 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# DeepCache 실험 스크립트
echo "=== DeepCache 실험 시작 ==="
echo "프로젝트 루트: $PROJECT_ROOT"
echo "시작 시간: $(date)"
start_time=$(date +%s)

# 기본 설정
CUDA_VISIBLE_DEVICES=0
data_type=image
task=label_guidance
image_size=32
dataset="cifar10"
model_name_or_path='openai_cifar10.pt'
guide_network='resnet_cifar10.pt'
train_steps=1000
inference_steps=50
eta=1.0
target=2
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=128
num_samples=1024  # 1024개 이미지
logging_resolution=512
guidance_name='dps'
eval_batch_size=512
wandb=False
model_type='unet'

rho=1
mu=0.25
sigma=0.001
eps_bsz=1
iter_steps=4

use_deepcache=True
cache_block_id=1
skip_mode='uniform'

# 실험 결과 저장 디렉토리 (프로젝트 루트에 저장)
experiment_dir="$PROJECT_ROOT/deepcache_experiments_target=${target}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$experiment_dir"

echo "실험 결과 저장 디렉토리: $experiment_dir"

# 실험: Cache Interval별로 Clean Step 다르게
echo "=== Cache Interval별 Clean Step 실험 ==="

# Cache Interval과 Clean Step 조합
cache_intervals=(1 2 3 4 5)
clean_steps=(0 5 10 15 20 25 30 35 40 45 50)

total_experiments=$(( ${#cache_intervals[@]} * ${#clean_steps[@]} ))
current_experiment=0

for cache_interval in "${cache_intervals[@]}"; do
    for clean_step in "${clean_steps[@]}"; do
        current_experiment=$((current_experiment + 1))
        echo "실험 $current_experiment/$total_experiments: Cache Interval: $cache_interval, Clean Step: $clean_step"
        
        # 1024개 이미지 생성
        experiment_start_time=$(date +%s)
        
        # 실험 디렉토리 생성
        experiment_subdir="${experiment_dir}/cache_${cache_interval}_clean_${clean_step}"
        mkdir -p "$experiment_subdir"
        
        # 임시 로깅 디렉토리를 실험 디렉토리로 설정
        temp_logging_dir="$experiment_subdir"
        
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
            --data_type $data_type \
            --task $task \
            --image_size $image_size \
            --iter_steps $iter_steps \
            --dataset $dataset \
            --guide_network $guide_network \
            --logging_resolution $logging_resolution \
            --model_name_or_path $model_name_or_path \
            --train_steps $train_steps \
            --inference_steps $inference_steps \
            --target $target \
            --eta $eta \
            --clip_x0 $clip_x0 \
            --rho $rho \
            --mu $mu \
            --sigma $sigma \
            --eps_bsz $eps_bsz \
            --model_type $model_type \
            --wandb $wandb \
            --seed $seed \
            --logging_dir "$temp_logging_dir" \
            --per_sample_batch_size $per_sample_batch_size \
            --num_samples $num_samples \
            --guidance_name $guidance_name \
            --eval_batch_size $eval_batch_size \
            --use_deepcache $use_deepcache \
            --cache_interval $cache_interval \
            --cache_block_id $cache_block_id \
            --skip_mode $skip_mode \
            --clean_step $clean_step \
            --check_done False
        
        experiment_end_time=$(date +%s)
        experiment_duration=$((experiment_end_time - experiment_start_time))
        
        echo "실험 $current_experiment 완료: ${experiment_duration}초"
        echo "cache_${cache_interval}_clean_${clean_step}: ${experiment_duration}초" >> "${experiment_dir}/results.txt"
    done
done

# 종료 시간 기록 및 총 실행 시간 계산
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "=== 실험 완료 ==="
echo "종료 시간: $(date)"
echo "총 실행 시간: ${duration}초 ($((duration / 60))분 $((duration % 60))초)"

# 결과 요약
echo "=== 실험 결과 요약 ==="
echo "실험 결과 저장 위치: $experiment_dir"
echo "총 실험 수: $total_experiments"
echo "총 생성 이미지 수: $((total_experiments * num_samples))"
echo ""
echo "실험 결과:"
cat "${experiment_dir}/results.txt"

# 결과를 CSV 파일로도 저장
echo "cache_interval,clean_step,time_seconds" > "${experiment_dir}/results.csv"
cat "${experiment_dir}/results.txt" | sed 's/: /,/' | sed 's/cache_\([0-9]*\)_clean_\([0-9]*\): \([0-9]*\)초/\1,\2,\3/' >> "${experiment_dir}/results.csv"

# 실험별 평균 시간 계산
echo ""
echo "=== 실험별 평균 시간 분석 ==="
if [ -f "${experiment_dir}/results.csv" ]; then
    # CSV 파일에서 데이터 읽기 (헤더 제외)
    tail -n +2 "${experiment_dir}/results.csv" | while IFS=',' read -r cache_interval clean_step time_seconds; do
        echo "Cache Interval: $cache_interval, Clean Step: $clean_step, 시간: ${time_seconds}초"
    done
    
    # 전체 평균 시간 계산
    total_time=$(tail -n +2 "${experiment_dir}/results.csv" | awk -F',' '{sum+=$3} END {print sum}')
    avg_time=$(tail -n +2 "${experiment_dir}/results.csv" | awk -F',' '{sum+=$3; count++} END {print sum/count}')
    echo ""
    echo "총 실행 시간: ${total_time}초"
    echo "평균 실행 시간: ${avg_time}초"
fi

echo ""
echo "CSV 파일도 생성 완료:"
echo "- ${experiment_dir}/results.csv" 