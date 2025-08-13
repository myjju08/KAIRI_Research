#!/bin/bash

# 시작 시간 기록
echo "=== CIFAR10 Label Guidance with DeepCache ==="
echo "시작 시간: $(date)"
start_time=$(date +%s)

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
num_samples=32
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
cache_interval=5
cache_block_id=1
skip_mode='uniform'

echo "=== 설정 정보 ==="
echo "DeepCache 사용: $use_deepcache"
echo "캐시 간격: $cache_interval"
echo "모델 타입: $model_type"
echo "추론 스텝: $inference_steps"
echo "샘플 수: $num_samples"
echo "배치 크기: $per_sample_batch_size"
echo "=================="

# 메인 실행
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
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --guidance_name $guidance_name \
    --eval_batch_size $eval_batch_size \
    --use_deepcache $use_deepcache \
    --cache_interval $cache_interval \
    --cache_block_id $cache_block_id \
    --skip_mode $skip_mode 

# 종료 시간 기록 및 총 실행 시간 계산
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "=== 실행 완료 ==="
echo "종료 시간: $(date)"
echo "총 실행 시간: ${duration}초 ($(($duration / 60))분 $(($duration % 60))초)"

# 시간을 파일로도 저장
echo "실행 시간: ${duration}초 ($(($duration / 60))분 $(($duration % 60))초)" >> timing_results.txt
echo "설정: DeepCache=$use_deepcache, Cache_Interval=$cache_interval, Steps=$inference_steps" >> timing_results.txt
echo "---" >> timing_results.txt 