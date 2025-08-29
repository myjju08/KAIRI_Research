#!/bin/bash
set -euo pipefail

# 스크립트 기준 프로젝트 루트 경로 계산 및 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# DeepCache 실험 스크립트
echo "=== DeepCache Target 실험 시작 ==="
echo "프로젝트 루트: $PROJECT_ROOT"
echo "시작 시간: $(date)"
start_time=$(date +%s)

# 기본 설정
CUDA_VISIBLE_DEVICES=3
data_type=image
task=label_guidance
image_size=32
dataset="cifar10"
model_name_or_path='openai_cifar10.pt'
guide_network='resnet_cifar10.pt'
train_steps=1000
inference_steps=50
eta=1.0
# 대상 타깃들 설정 (0-10)
targets=(8 9)
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
skip_mode='uniform'

# 실험 설정 1: clean step 25, cache interval 5, cache block id 2
# 실험 설정 2: clean step 0, cache interval 1, cache block id 1 (기본값)

# Target 루프 시작
for target in "${targets[@]}"; do
	# 실험 결과 저장 디렉토리 (프로젝트 루트에 저장)
	experiment_dir="$PROJECT_ROOT/deepcache_target_experiments_target=${target}_$(date +%Y%m%d_%H%M%S)"
	mkdir -p "$experiment_dir"

	echo "실험 결과 저장 디렉토리: $experiment_dir"
	# 각 target별 타이밍 시작
	target_start_time=$(date +%s)

	echo "=== Target ${target} 실험 시작 ==="

	# 실험 1: clean step 15, cache interval 4, cache block id 7
	echo "실험 1: clean step 15, cache interval 4, cache block id 7"
	experiment_start_time=$(date +%s)
	
	experiment_subdir="${experiment_dir}/clean_step_15_cache_interval_4_cache_block_7"
	mkdir -p "$experiment_subdir"
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
		--cache_interval 4 \
		--cache_block_id 7 \
		--skip_mode $skip_mode \
		--clean_step 15 \
		--check_done False
	
	experiment_end_time=$(date +%s)
	experiment_duration=$((experiment_end_time - experiment_start_time))
	echo "실험 1 완료: ${experiment_duration}초"
	echo "clean_step_15_cache_interval_4_cache_block_7: ${experiment_duration}초" >> "${experiment_dir}/results.txt"

	# 실험 2: clean step 35, cache interval 5, cache block id 5
	echo "실험 2: clean step 35, cache interval 5, cache block id 5"
	experiment_start_time=$(date +%s)
	
	experiment_subdir="${experiment_dir}/clean_step_35_cache_interval_5_cache_block_5"
	mkdir -p "$experiment_subdir"
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
		--cache_interval 5 \
		--cache_block_id 5 \
		--skip_mode $skip_mode \
		--clean_step 35 \
		--check_done False
	
	experiment_end_time=$(date +%s)
	experiment_duration=$((experiment_end_time - experiment_start_time))
	echo "실험 2 완료: ${experiment_duration}초"
	echo "clean_step_35_cache_interval_5_cache_block_5: ${experiment_duration}초" >> "${experiment_dir}/results.txt"

	# 종료 시간 기록 및 총 실행 시간 계산 (각 target별)
	end_time=$(date +%s)
	duration=$((end_time - target_start_time))

	echo "=== 실험 완료 (target=${target}) ==="
	echo "종료 시간: $(date)"
	echo "총 실행 시간: ${duration}초 ($((duration / 60))분 $((duration % 60))초)"

	# 결과 요약
	echo "=== 실험 결과 요약 (target=${target}) ==="
	echo "실험 결과 저장 위치: $experiment_dir"
	echo "총 실험 수: 2"
	echo "총 생성 이미지 수: $((2 * num_samples))"
	echo ""
	echo "실험 결과:"
	cat "${experiment_dir}/results.txt"

	# 결과를 CSV 파일로도 저장
	echo "experiment_name,time_seconds" > "${experiment_dir}/results.csv"
	cat "${experiment_dir}/results.txt" | sed 's/: /,/' >> "${experiment_dir}/results.csv"

	echo ""
	echo "CSV 파일도 생성 완료:"
	echo "- ${experiment_dir}/results.csv"

done

# 전체 실험 완료
total_end_time=$(date +%s)
total_duration=$((total_end_time - start_time))

echo ""
echo "=== 전체 실험 완료 ==="
echo "종료 시간: $(date)"
echo "총 실행 시간: ${total_duration}초 ($((total_duration / 60))분 $((total_duration % 60))초)"
echo "총 target 수: ${#targets[@]}"
echo "총 실험 수: $((2 * ${#targets[@]}))"
echo "총 생성 이미지 수: $((2 * ${#targets[@]} * num_samples))" 