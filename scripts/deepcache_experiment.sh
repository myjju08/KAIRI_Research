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
# Guidance strength 설정 (여러 값을 테스트하려면 배열로)
guidance_strengths=(1.0)  # 원하는 값들로 수정
# 대상 타깃들 설정
targets=(6)
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=128
num_samples=32  #n개 이미지
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

use_deepcache=False
skip_mode='uniform'

# Clean step 고정
clean_step=0

# Guidance strength 루프 시작
for guidance_strength in "${guidance_strengths[@]}"; do
	echo "=== Guidance Strength: $guidance_strength 실험 시작 ==="
	
	# Target 루프 시작
	for target in "${targets[@]}"; do
		# 실험 결과 저장 디렉토리 (guidance strength 포함)
		experiment_dir="$PROJECT_ROOT/deepcache_experiments_guidance=${guidance_strength}_target=${target}_$(date +%Y%m%d_%H%M%S)"
		mkdir -p "$experiment_dir"

		echo "실험 결과 저장 디렉토리: $experiment_dir"
		# 각 target별 타이밍 시작
		target_start_time=$(date +%s)

		# 실험: Cache Interval별로 Caching Depth 다르게
		echo "=== Cache Interval별 Caching Depth 실험 (Clean Step=0 고정) ==="

		# Cache Interval과 Caching Depth 조합
		cache_intervals=(1)
		cache_block_ids=(12)  # CIFAR10 UNet은 16개 output blocks를 가짐 (1~16)

		total_experiments=$(( ${#cache_intervals[@]} * ${#cache_block_ids[@]} ))
		current_experiment=0

		for cache_interval in "${cache_intervals[@]}"; do
		    for cache_block_id in "${cache_block_ids[@]}"; do
		        current_experiment=$((current_experiment + 1))
		        echo "실험 $current_experiment/$total_experiments: Cache Interval: $cache_interval, Caching Depth: $cache_block_id"
		        
		        # 1024개 이미지 생성
		        experiment_start_time=$(date +%s)
		        
		        # 실험 디렉토리 생성
		        experiment_subdir="${experiment_dir}/interval_${cache_interval}_depth_${cache_block_id}"
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
		            --guidance_strength $guidance_strength \
		            --check_done False
		        
		        experiment_end_time=$(date +%s)
		        experiment_duration=$((experiment_end_time - experiment_start_time))
		        
		        echo "실험 $current_experiment 완료: ${experiment_duration}초"
		        echo "interval_${cache_interval}_depth_${cache_block_id}: ${experiment_duration}초" >> "${experiment_dir}/results.txt"
		    done
		done

		# 종료 시간 기록 및 총 실행 시간 계산 (각 target별)
		end_time=$(date +%s)
		duration=$((end_time - target_start_time))

		echo "=== 실험 완료 (guidance=${guidance_strength}, target=${target}) ==="
		echo "종료 시간: $(date)"
		echo "총 실행 시간: ${duration}초 ($((duration / 60))분 $((duration % 60))초)"

		# 결과 요약
		echo "=== 실험 결과 요약 (guidance=${guidance_strength}, target=${target}) ==="
		echo "실험 결과 저장 위치: $experiment_dir"
		echo "총 실험 수: $total_experiments"
		echo "총 생성 이미지 수: $((total_experiments * num_samples))"
		echo ""
		echo "실험 결과:"
		cat "${experiment_dir}/results.txt"

		# 결과를 CSV 파일로도 저장
		echo "cache_interval,cache_block_id,time_seconds" > "${experiment_dir}/results.csv"
		cat "${experiment_dir}/results.txt" | sed 's/: /,/' | sed 's/interval_\([0-9]*\)_depth_\([0-9]*\): \([0-9]*\)초/\1,\2,\3/' >> "${experiment_dir}/results.csv"

		# 실험별 평균 시간 계산
		echo ""
		echo "=== 실험별 평균 시간 분석 (guidance=${guidance_strength}, target=${target}) ==="
		if [ -f "${experiment_dir}/results.csv" ]; then
			# CSV 파일에서 데이터 읽기 (헤더 제외)
			tail -n +2 "${experiment_dir}/results.csv" | while IFS=',' read -r cache_interval cache_block_id time_seconds; do
				echo "Cache Interval: $cache_interval, Caching Depth: $cache_block_id, 시간: ${time_seconds}초"
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

	done  # target 루프 종료
	
	echo ""
	echo "=== Guidance Strength $guidance_strength 실험 완료 ==="
	
done  # guidance strength 루프 종료