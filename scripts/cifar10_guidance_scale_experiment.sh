#!/bin/bash

# 여러 guidance_scale 값에 대해 실험하는 스크립트

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
start_gradient=25

# start_gradient 값들
start_gradients=(10 15 20 25 30 35 40 45 50)

# guidance_scale 값들
guidance_scales=(0 0.1 1 2 5 10)

# 각 guidance_scale마다 실행할 횟수
num_runs_per_scale=5

echo "=== CIFAR10 Start Gradient 실험 ==="
echo "Start gradients: ${start_gradients[@]}"
echo "Guidance scales: ${guidance_scales[@]}"
echo "각 start_gradient마다 ${#guidance_scales[@]}개의 guidance_scale, 각 scale마다 ${num_runs_per_scale}번 실행"
echo "총 ${#start_gradients[@]} * ${#guidance_scales[@]} * ${num_runs_per_scale} = $(( ${#start_gradients[@]} * ${#guidance_scales[@]} * num_runs_per_scale )) 번 실행"
echo ""

# 각 start_gradient에 대해 실험
for start_gradient_idx in "${!start_gradients[@]}"; do
    start_gradient=${start_gradients[$start_gradient_idx]}
    echo "=== Start Gradient ${start_gradient} 실험 시작 ==="
    
    # 각 guidance_scale에 대해 실험
    for scale_idx in "${!guidance_scales[@]}"; do
        guidance_scale=${guidance_scales[$scale_idx]}
        echo "=== Start Gradient ${start_gradient}, Guidance Scale ${guidance_scale} 실험 시작 ==="
    
    # 각 guidance_scale마다 여러 번 실행
    for run_idx in $(seq 0 $((num_runs_per_scale-1))); do
        echo ""
        echo "--- Guidance Scale ${guidance_scale}, 실행 $((run_idx+1))/${num_runs_per_scale} ---"
        
        # 각 실행마다 다른 시드 사용
        current_seed=$((seed + run_idx))
        current_logging_dir="logs/start_gradient_${start_gradient}/guidance_scale_${guidance_scale}/run_${run_idx}"
        
        echo "시드: $current_seed"
        echo "로그 디렉토리: $current_logging_dir"
        
        # 실행
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
            --guidance_scale $guidance_scale \
            --start_gradient $start_gradient \
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
            echo "Guidance Scale ${guidance_scale}, 실행 $((run_idx+1)) 완료"
        else
            echo "Guidance Scale ${guidance_scale}, 실행 $((run_idx+1)) 실패"
        fi
    done
    
        echo ""
        echo "=== Start Gradient ${start_gradient}, Guidance Scale ${guidance_scale} 실험 완료 ==="
        echo "생성된 이미지들:"
        for run_idx in $(seq 0 $((num_runs_per_scale-1))); do
            if [ -d "logs/start_gradient_${start_gradient}/guidance_scale_${guidance_scale}/run_${run_idx}" ]; then
                echo "  logs/start_gradient_${start_gradient}/guidance_scale_${guidance_scale}/run_${run_idx}"
            fi
        done
        echo ""
    done
    
    echo ""
    echo "=== Start Gradient ${start_gradient} 실험 완료 ==="
    echo "생성된 디렉토리들:"
    for scale in "${guidance_scales[@]}"; do
        echo "  logs/start_gradient_${start_gradient}/guidance_scale_${scale}/"
    done
    
    echo ""
    echo "Start Gradient ${start_gradient}의 각 guidance_scale별로 이미지를 합쳐서 FID 계산하려면:"
    for scale in "${guidance_scales[@]}"; do
        echo "python calculate_fid_proper.py --guidance_scale ${scale} --num_runs ${num_runs_per_scale} --base_logging_dir logs/start_gradient_${start_gradient}"
    done
    echo ""
done

echo "=== 모든 실험 완료 ==="
echo "생성된 디렉토리들:"
for start_grad in "${start_gradients[@]}"; do
    echo "  logs/start_gradient_${start_grad}/"
done

echo ""
echo "각 start_gradient별로 모든 guidance_scale에 대해 FID 계산하려면:"
for start_grad in "${start_gradients[@]}"; do
    for scale in "${guidance_scales[@]}"; do
        echo "python calculate_fid_proper.py --guidance_scale ${scale} --num_runs ${num_runs_per_scale} --base_logging_dir logs/start_gradient_${start_grad}"
    done
done 