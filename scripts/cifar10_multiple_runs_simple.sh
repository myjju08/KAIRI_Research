#!/bin/bash

# 여러 번 실행해서 이미지를 저장하는 간단한 스크립트

CUDA_VISIBLE_DEVICES=0
data_type=image
task=label_guidance
image_size=32
dataset="cifar10"
model_name_or_path='models/cifar10_uvit_small.pth'
guide_network='resnet_cifar10.pt'
train_steps=1000
inference_steps=50
eta=1.0
target=1
clip_x0=True
seed=41
per_sample_batch_size=128
num_samples=30
logging_resolution=512
guidance_name='dps'
eval_batch_size=512
wandb=False
log_traj=False

rho=1
mu=0.25
sigma=0.001
eps_bsz=1
iter_steps=4

# 실행할 횟수
num_runs=5

echo "=== CIFAR10 여러 번 실행 스크립트 ==="
echo "총 $num_runs번 실행합니다."

# 각 실행마다 다른 시드와 로그 디렉토리 사용
for i in $(seq 0 $((num_runs-1))); do
    echo ""
    echo "=== 실행 $((i+1))/$num_runs ==="
    
    # 각 실행마다 다른 시드 사용
    current_seed=$((seed + i))
    current_logging_dir="logs/run_${i}"
    
    echo "시드: $current_seed"
    echo "로그 디렉토리: $current_logging_dir"
    
    # check_done을 False로 설정하여 항상 실행되도록 함
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
        --wandb $wandb \
        --seed $current_seed \
        --logging_dir $current_logging_dir \
        --per_sample_batch_size $per_sample_batch_size \
        --num_samples $num_samples \
        --guidance_name $guidance_name \
        --eval_batch_size $eval_batch_size \
        --log_traj $log_traj \
        --check_done False
    
    if [ $? -eq 0 ]; then
        echo "실행 $((i+1)) 완료"
    else
        echo "실행 $((i+1)) 실패"
    fi
done

echo ""
echo "=== 모든 실행 완료 ==="
echo "생성된 이미지들:"
for i in $(seq 0 $((num_runs-1))); do
    if [ -d "logs/run_${i}" ]; then
        echo "  logs/run_${i}/images.npy"
    fi
done

echo ""
echo "이미지들을 합쳐서 FID score를 계산하려면 다음 명령어를 실행하세요:"
echo "python combine_and_evaluate_fid.py --num_runs $num_runs --num_samples $num_samples" 