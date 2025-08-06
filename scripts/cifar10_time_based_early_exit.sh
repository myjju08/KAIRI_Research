#!/bin/bash

# Time-based Early Exit Layer 실험 스크립트
# Time step 0-50을 early exit 0-5로 매핑
# 마지막 9-0 step은 early exit 5로 처리

CUDA_VISIBLE_DEVICES=1
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

# Time-based early exit mapping 정의
# Python dict 형태로 전달: "{(0,8):5, (9,17):4, (18,26):3, (27,35):2, (36,44):1, (45,50):0}"
time_early_exit_mapping='{(0,8):5, (9,17):5, (18,26):0, (27,35):0, (36,44):0, (45,50):0}'

# 각 time-based early exit 설정마다 실행할 횟수
num_runs_per_config=5

echo "=== CIFAR10 Time-based Early Exit 실험 ==="
echo "Time-based early exit mapping: ${time_early_exit_mapping}"
echo "Start gradient: ${start_gradient}"
echo "Guidance scale: ${guidance_scale}"
echo "각 설정마다 ${num_runs_per_config}번 실행"
echo ""

# Time-based early exit 실험 실행
for run_idx in $(seq 0 $((num_runs_per_config-1))); do
    echo ""
    echo "--- Time-based Early Exit, 실행 $((run_idx+1))/${num_runs_per_config} ---"
    
    # 각 실행마다 다른 시드 사용
    current_seed=$((seed + run_idx))
    current_logging_dir="logs/550000/run_${run_idx}"
    
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
        --use_time_based_early_exit True \
        --time_early_exit_mapping "{(0,8):5, (9,17):5, (18,26):0, (27,35):0, (36,44):0, (45,50):0}" \
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
    echo "time_based_early_exit,run_idx,execution_time,num_samples" > "$timing_file"
    echo "time_based,$run_idx,$execution_time,$num_samples" >> "$timing_file"
    
    if [ $? -eq 0 ]; then
        echo "Time-based Early Exit, 실행 $((run_idx+1)) 완료 (${execution_time}s)"
    else
        echo "Time-based Early Exit, 실행 $((run_idx+1)) 실패"
    fi
done

echo ""
echo "=== Time-based Early Exit 실험 완료 ==="
echo "생성된 디렉토리들:"
for run_idx in $(seq 0 $((num_runs_per_config-1))); do
    if [ -d "logs/time_based_early_exit/run_${run_idx}" ]; then
        echo "  logs/time_based_early_exit/run_${run_idx}"
    fi
done

echo ""
echo "Time-based Early Exit의 모든 run을 합쳐서 FID 계산하려면:"
echo "python calculate_time_based_early_exit_fid.py --num_runs ${num_runs_per_config}" 