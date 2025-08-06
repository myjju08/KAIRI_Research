#!/bin/bash

# Early Exit Layer 실험 스크립트
# Fixed: start_gradient=35, guidance_scale=2
# Varying: early_exit_layer from 0 to 5
# 각 early_exit_layer마다 30개씩 5번 생성

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
start_gradient=50
guidance_scale=2.0

# early_exit_layer 값들
early_exit_layers=(0 1 2 3 4 5)

# 각 early_exit_layer마다 실행할 횟수
num_runs_per_layer=5

echo "=== CIFAR10 Early Exit Layer 실험 ==="
echo "Early exit layers: ${early_exit_layers[@]}"
echo "Start gradient: ${start_gradient}"
echo "Guidance scale: ${guidance_scale}"
echo "각 early_exit_layer마다 ${num_runs_per_layer}번 실행"
echo "총 ${#early_exit_layers[@]} * ${num_runs_per_layer} = $(( ${#early_exit_layers[@]} * num_runs_per_layer )) 번 실행"
echo ""
    
# 각 early_exit_layer에 대해 실험
for layer_idx in "${!early_exit_layers[@]}"; do
    early_exit_layer=${early_exit_layers[$layer_idx]}
    echo "=== Early Exit Layer ${early_exit_layer} 실험 시작 ==="
    
    # 각 early_exit_layer마다 여러 번 실행
    for run_idx in $(seq 0 $((num_runs_per_layer-1))); do
        echo ""
        echo "--- Early Exit Layer ${early_exit_layer}, 실행 $((run_idx+1))/${num_runs_per_layer} ---"
        
        # 각 실행마다 다른 시드 사용
        current_seed=$((seed + run_idx))
        current_logging_dir="logs/early_exit_layer_${early_exit_layer}/run_${run_idx}"
        
        echo "시드: $current_seed"
        echo "로그 디렉토리: $current_logging_dir"
        
        # 실행 시간 측정 시작
    start_time=$(date +%s.%N)
    
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
            --use_early_exit True \
            --early_exit_layer $early_exit_layer \
            --wandb $wandb \
            --seed $current_seed \
            --logging_dir $current_logging_dir \
            --per_sample_batch_size $per_sample_batch_size \
            --num_samples $num_samples \
            --guidance_name $guidance_name \
            --eval_batch_size $eval_batch_size \
            --log_traj $log_traj \
            --check_done False
        
        # 실행 시간 측정 종료
    end_time=$(date +%s.%N)
    execution_time=$(echo "$end_time - $start_time" | bc -l)
    
        # 속도 결과 저장
        timing_file="$current_logging_dir/timing.csv"
        echo "early_exit_layer,run_idx,execution_time,num_samples" > "$timing_file"
        echo "$early_exit_layer,$run_idx,$execution_time,$num_samples" >> "$timing_file"
        
        if [ $? -eq 0 ]; then
            echo "Early Exit Layer ${early_exit_layer}, 실행 $((run_idx+1)) 완료 (${execution_time}s)"
        else
            echo "Early Exit Layer ${early_exit_layer}, 실행 $((run_idx+1)) 실패"
        fi
    done
    
    echo ""
    echo "=== Early Exit Layer ${early_exit_layer} 실험 완료 ==="
    echo "생성된 이미지들:"
    for run_idx in $(seq 0 $((num_runs_per_layer-1))); do
        if [ -d "logs/early_exit_layer_${early_exit_layer}/run_${run_idx}" ]; then
            echo "  logs/early_exit_layer_${early_exit_layer}/run_${run_idx}"
        fi
    done
    echo ""
    
    echo "Early Exit Layer ${early_exit_layer}의 모든 run을 합쳐서 FID 계산하려면:"
    echo "python calculate_early_exit_fid.py --early_exit_layer ${early_exit_layer} --num_runs ${num_runs_per_layer}"
    echo ""
done

echo "=== 모든 Early Exit 실험 완료 ==="
echo "생성된 디렉토리들:"
for layer in "${early_exit_layers[@]}"; do
    echo "  logs/early_exit_layer_${layer}/"
done

echo ""
echo "각 early_exit_layer별로 FID 계산하려면:"
for layer in "${early_exit_layers[@]}"; do
    echo "python calculate_early_exit_fid.py --early_exit_layer ${layer} --num_runs ${num_runs_per_layer}"
done 