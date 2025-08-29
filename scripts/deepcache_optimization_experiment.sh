#!/bin/bash
set -euo pipefail

# 스크립트 기준 프로젝트 루트 경로 계산 및 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# 로그 디렉토리 생성
LOG_DIR="$PROJECT_ROOT/logs_optimization_experiment"
mkdir -p "$LOG_DIR"

# DeepCache 최적화 실험 스크립트
echo "=== DeepCache 최적화 실험 시작 ==="
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

# 실험 설정
# 1. Target classes: 1, 3, 5, 7
targets=(9)

# 2. Deep cache interval: 5 (0~clean_step은 full, clean_step~50은 deep cache)
cache_interval=5

# 3. Clean step: 15~30 범위에서 FID 감소 효과 확인
clean_steps=(15 20 25 30)

# 4. Cache depth: 2, 3 (최적값)
cache_block_ids=(2 3)

# 실험 결과 저장 디렉토리 (프로젝트 루트에 저장)
experiment_dir="$PROJECT_ROOT/deepcache_optimization_experiments_multi_target_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$experiment_dir"

echo "실험 결과 저장 디렉토리: $experiment_dir"
echo "Target Classes: ${targets[*]}"
echo "Deep Cache Interval: $cache_interval (0~clean_step: full, clean_step~50: deep cache)"
echo "테스트할 Clean Steps: ${clean_steps[*]}"
echo "테스트할 Cache Depths: ${cache_block_ids[*]}"

# 실험: Target Class, Clean Step, Cache Depth 조합별로 실험
echo "=== Target Class, Clean Step, Cache Depth 조합별 실험 (순차 실행) ==="

# 각 target별로 실험 수 계산
experiments_per_target=$(( ${#clean_steps[@]} * ${#cache_block_ids[@]} + 1 ))  # +1 for clean_step=0, cache_interval=1
total_experiments=$(( ${#targets[@]} * experiments_per_target ))
current_experiment=0

# 실험 시간 측정을 위한 배열 초기화
declare -a experiment_times
declare -a experiment_targets
declare -a experiment_clean_steps
declare -a experiment_cache_depths
declare -a experiment_cache_intervals

# 실험 시작 시간 기록
experiment_start_time=$(date +%s)

for target in "${targets[@]}"; do
    echo ""
    echo "=== Target Class $target 실험 시작 ==="
    
    # Target별 실험 디렉토리 생성
    target_experiment_dir="${experiment_dir}/target_${target}"
    mkdir -p "$target_experiment_dir"
    
    # 1. Clean Step과 Cache Depth 조합 실험
    for clean_step in "${clean_steps[@]}"; do
        for cache_block_id in "${cache_block_ids[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo ""
            echo "=== 실험 $current_experiment/$total_experiments ==="
            echo "Target: $target, Clean Step: $clean_step, Cache Depth (cache_block_id): $cache_block_id"
            echo "Deep Cache Interval: $cache_interval (0~$clean_step: full, $clean_step~50: deep cache)"
            echo "시작 시간: $(date)"
            
            # 실험 디렉토리 생성
            experiment_subdir="${target_experiment_dir}/clean_step_${clean_step}_cache_depth_${cache_block_id}"
            mkdir -p "$experiment_subdir"
            
            # 임시 로깅 디렉토리를 실험 디렉토리로 설정
            temp_logging_dir="$experiment_subdir"
            
            # 개별 실험 로그 파일
            experiment_log_file="${LOG_DIR}/target_${target}_clean_${clean_step}_depth_${cache_block_id}.log"
            
            echo "실행 중: Target $target, Clean Step $clean_step, Cache Depth $cache_block_id..."
            echo "로그 파일: $experiment_log_file"
            
            # 순차 실행
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
                --check_done False 2>&1 | tee "$experiment_log_file"
            
            # 실험 완료 시간 기록
            experiment_end_time=$(date +%s)
            experiment_duration=$((experiment_end_time - experiment_start_time))
            
            # 실험 시간 정보 저장
            experiment_times+=($experiment_duration)
            experiment_targets+=($target)
            experiment_clean_steps+=($clean_step)
            experiment_cache_depths+=($cache_block_id)
            experiment_cache_intervals+=($cache_interval)
            
            echo ""
            echo "=== 실험 $current_experiment 완료 ==="
            echo "완료 시간: $(date)"
            echo "소요 시간: ${experiment_duration}초 ($((experiment_duration / 60))분 $((experiment_duration % 60))초)"
            echo "진행률: $current_experiment/$total_experiments ($((current_experiment * 100 / total_experiments))%)"
            
            # 다음 실험 시작 시간 업데이트
            experiment_start_time=$(date +%s)
            
            # GPU 메모리 정리를 위한 잠시 대기
            echo "GPU 메모리 정리 중... (5초 대기)"
            sleep 5
        done
    done
    
    # 2. Clean Step 0, Cache Interval 1 실험 (각 target별로 추가)
    current_experiment=$((current_experiment + 1))
    echo ""
    echo "=== 실험 $current_experiment/$total_experiments ==="
    echo "Target: $target, Clean Step: 0, Cache Interval: 1"
    echo "시작 시간: $(date)"
    
    # 실험 디렉토리 생성
    experiment_subdir="${target_experiment_dir}/clean_step_0_cache_interval_1"
    mkdir -p "$experiment_subdir"
    
    # 임시 로깅 디렉토리를 실험 디렉토리로 설정
    temp_logging_dir="$experiment_subdir"
    
    # 개별 실험 로그 파일
    experiment_log_file="${LOG_DIR}/target_${target}_clean_0_interval_1.log"
    
    echo "실행 중: Target $target, Clean Step 0, Cache Interval 1..."
    echo "로그 파일: $experiment_log_file"
    
    # 순차 실행
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
        --cache_interval 1 \
        --cache_block_id 2 \
        --skip_mode $skip_mode \
        --clean_step 0 \
        --check_done False 2>&1 | tee "$experiment_log_file"
    
    # 실험 완료 시간 기록
    experiment_end_time=$(date +%s)
    experiment_duration=$((experiment_end_time - experiment_start_time))
    
    # 실험 시간 정보 저장
    experiment_times+=($experiment_duration)
    experiment_targets+=($target)
    experiment_clean_steps+=(0)
    experiment_cache_depths+=(2)
    experiment_cache_intervals+=(1)
    
    echo ""
    echo "=== 실험 $current_experiment 완료 ==="
    echo "완료 시간: $(date)"
    echo "소요 시간: ${experiment_duration}초 ($((experiment_duration / 60))분 $((experiment_duration % 60))초)"
    echo "진행률: $current_experiment/$total_experiments ($((current_experiment * 100 / total_experiments))%)"
    
    # 다음 실험 시작 시간 업데이트
    experiment_start_time=$(date +%s)
    
    # GPU 메모리 정리를 위한 잠시 대기
    echo "GPU 메모리 정리 중... (5초 대기)"
    sleep 5
    
    echo ""
    echo "=== Target Class $target 실험 완료 ==="
done

echo ""
echo "=== 모든 실험이 완료되었습니다 ==="
echo "실험 디렉토리: $experiment_dir"
echo "로그 디렉토리: $LOG_DIR"

# 실험 시간 결과를 CSV 파일로 저장
echo "=== 실험 시간 결과 저장 ==="
timing_csv_file="${experiment_dir}/experiment_timing_results.csv"

# CSV 헤더 작성
echo "target,clean_step,cache_depth,cache_interval,time_seconds,time_minutes" > "$timing_csv_file"

# 각 실험의 시간 정보를 CSV에 추가
for i in "${!experiment_times[@]}"; do
    target="${experiment_targets[$i]}"
    clean_step="${experiment_clean_steps[$i]}"
    cache_depth="${experiment_cache_depths[$i]}"
    cache_interval="${experiment_cache_intervals[$i]}"
    time_seconds="${experiment_times[$i]}"
    time_minutes=$(echo "scale=2; $time_seconds / 60" | bc -l)
    
    echo "$target,$clean_step,$cache_depth,$cache_interval,$time_seconds,$time_minutes" >> "$timing_csv_file"
done

echo "실험 시간 결과가 $timing_csv_file에 저장되었습니다."

# 시간 결과 요약 출력
echo ""
echo "=== 실험 시간 요약 ==="
echo "총 실험 수: ${#experiment_times[@]}"
echo "평균 시간: $(echo "scale=2; $(IFS=+; echo "$((${experiment_times[*]}))") / ${#experiment_times[@]}" | bc -l)초"
echo "최소 시간: $(printf '%s\n' "${experiment_times[@]}" | sort -n | head -1)초"
echo "최대 시간: $(printf '%s\n' "${experiment_times[@]}" | sort -n | tail -1)초"

# Target별 평균 시간
echo ""
echo "=== Target별 평균 시간 ==="
for target in "${targets[@]}"; do
    total_time=0
    count=0
    for i in "${!experiment_targets[@]}"; do
        if [[ "${experiment_targets[$i]}" == "$target" ]]; then
            total_time=$((total_time + experiment_times[i]))
            count=$((count + 1))
        fi
    done
    if [[ $count -gt 0 ]]; then
        avg_time=$(echo "scale=2; $total_time / $count" | bc -l)
        echo "Target $target: ${avg_time}초 (n=$count)"
    fi
done

# Clean Step별 평균 시간
echo ""
echo "=== Clean Step별 평균 시간 ==="
all_clean_steps=(0 15 20 25 30)
for step in "${all_clean_steps[@]}"; do
    total_time=0
    count=0
    for i in "${!experiment_clean_steps[@]}"; do
        if [[ "${experiment_clean_steps[$i]}" == "$step" ]]; then
            total_time=$((total_time + experiment_times[i]))
            count=$((count + 1))
        fi
    done
    if [[ $count -gt 0 ]]; then
        avg_time=$(echo "scale=2; $total_time / $count" | bc -l)
        echo "Clean Step $step: ${avg_time}초 (n=$count)"
    fi
done

# Cache Depth별 평균 시간
echo ""
echo "=== Cache Depth별 평균 시간 ==="
for depth in "${cache_block_ids[@]}"; do
    total_time=0
    count=0
    for i in "${!experiment_cache_depths[@]}"; do
        if [[ "${experiment_cache_depths[$i]}" == "$depth" ]]; then
            total_time=$((total_time + experiment_times[i]))
            count=$((count + 1))
        fi
    done
    if [[ $count -gt 0 ]]; then
        avg_time=$(echo "scale=2; $total_time / $count" | bc -l)
        echo "Cache Depth $depth: ${avg_time}초 (n=$count)"
    fi
done

# 결과 수집 스크립트 생성
cat > "${experiment_dir}/collect_results.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# 결과 수집 스크립트
echo "=== DeepCache 최적화 실험 결과 수집 ==="

# 현재 디렉토리에서 실행
cd "$(dirname "$0")"

# 결과 파일 생성
echo "결과 수집 중..."
> results.txt

# 각 target별로 결과 수집
for target in 1 3 5 7; do
    target_dir="target_${target}"
    if [ -d "$target_dir" ]; then
        echo "Target $target 결과 수집 중..."
        
        # Clean Step과 Cache Depth 조합 확인
        for clean_step in 15 20 25 30; do
            for cache_depth in 2 3; do
                experiment_dir="${target_dir}/clean_step_${clean_step}_cache_depth_${cache_depth}"
                if [ -d "$experiment_dir" ]; then
                    # images.npy 또는 .png 파일이 있는지 확인
                    if find "$experiment_dir" -name "images.npy" -type f | grep -q . || find "$experiment_dir" -name "*.png" -type f | grep -q .; then
                        echo "완료됨: Target $target, Clean Step $clean_step, Cache Depth $cache_depth"
                        echo "Target_${target}_Clean_${clean_step}_Depth_${cache_depth}: 완료" >> results.txt
                    else
                        echo "진행 중 또는 실패: Target $target, Clean Step $clean_step, Cache Depth $cache_depth"
                        echo "Target_${target}_Clean_${clean_step}_Depth_${cache_depth}: 실패" >> results.txt
                    fi
                fi
            done
        done
        
        # Clean Step 0, Cache Interval 1 확인
        experiment_dir="${target_dir}/clean_step_0_cache_interval_1"
        if [ -d "$experiment_dir" ]; then
            if find "$experiment_dir" -name "images.npy" -type f | grep -q . || find "$experiment_dir" -name "*.png" -type f | grep -q .; then
                echo "완료됨: Target $target, Clean Step 0, Cache Interval 1"
                echo "Target_${target}_Clean_0_Interval_1: 완료" >> results.txt
            else
                echo "진행 중 또는 실패: Target $target, Clean Step 0, Cache Interval 1"
                echo "Target_${target}_Clean_0_Interval_1: 실패" >> results.txt
            fi
        fi
    fi
done

# 결과를 CSV 파일로도 저장
echo "target,clean_step,cache_depth,cache_interval,status" > results.csv
cat results.txt | sed 's/: /,/' | sed 's/Target_\([0-9]*\)_Clean_\([0-9]*\)_Depth_\([0-9]*\): \(.*\)/\1,\2,\3,5,\4/' | sed 's/Target_\([0-9]*\)_Clean_\([0-9]*\)_Interval_\([0-9]*\): \(.*\)/\1,\2,2,\3,\4/' >> results.csv

echo ""
echo "=== 결과 수집 완료 ==="
echo "결과 파일:"
echo "- results.txt"
echo "- results.csv"

# 완료된 실험 수 계산
completed_experiments=$(grep -c "완료" results.txt 2>/dev/null || echo "0")
echo "완료된 실험 수: $completed_experiments"

echo ""
echo "결과 확인 방법:"
echo "1. 완료된 실험 목록: cat results.txt"
echo "2. CSV 형식 결과: cat results.csv"
echo "3. 특정 target 확인: ls -la target_*/clean_step_*/images.npy"
echo "4. PNG 파일 확인: ls -la target_*/*.png"
echo "5. 시간 측정 결과: cat experiment_timing_results.csv"
EOF

chmod +x "${experiment_dir}/collect_results.sh"

# 결과 시각화를 위한 Python 스크립트 생성
cat > "${experiment_dir}/plot_results.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 시간 측정 결과 파일 읽기
timing_file = "experiment_timing_results.csv"
if os.path.exists(timing_file):
    df = pd.read_csv(timing_file)
    
    # 그래프 설정
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Target별 평균 시간
    plt.subplot(3, 4, 1)
    target_means = df.groupby('target')['time_seconds'].mean()
    plt.bar(target_means.index, target_means.values)
    plt.xlabel('Target Class')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Time by Target Class')
    plt.xticks(target_means.index)
    
    # 2. Clean Step별 평균 시간
    plt.subplot(3, 4, 2)
    step_means = df.groupby('clean_step')['time_seconds'].mean()
    plt.bar(step_means.index, step_means.values)
    plt.xlabel('Clean Step')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Time by Clean Step')
    plt.xticks(step_means.index)
    
    # 3. Cache Depth별 평균 시간
    plt.subplot(3, 4, 3)
    depth_means = df.groupby('cache_depth')['time_seconds'].mean()
    plt.bar(depth_means.index, depth_means.values)
    plt.xlabel('Cache Depth')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Time by Cache Depth')
    plt.xticks(depth_means.index)
    
    # 4. Cache Interval별 평균 시간
    plt.subplot(3, 4, 4)
    interval_means = df.groupby('cache_interval')['time_seconds'].mean()
    plt.bar(interval_means.index, interval_means.values)
    plt.xlabel('Cache Interval')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Time by Cache Interval')
    plt.xticks(interval_means.index)
    
    # 5. Target별 Clean Step vs Time
    plt.subplot(3, 4, 5)
    for target in df['target'].unique():
        subset = df[df['target'] == target]
        step_means = subset.groupby('clean_step')['time_seconds'].mean()
        plt.plot(step_means.index, step_means.values, 'o-', label=f'Target {target}', linewidth=2, markersize=6)
    plt.xlabel('Clean Step')
    plt.ylabel('Average Time (seconds)')
    plt.title('Clean Step vs Time by Target')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Target별 Cache Depth vs Time
    plt.subplot(3, 4, 6)
    for target in df['target'].unique():
        subset = df[df['target'] == target]
        depth_means = subset.groupby('cache_depth')['time_seconds'].mean()
        plt.plot(depth_means.index, depth_means.values, 'o-', label=f'Target {target}', linewidth=2, markersize=6)
    plt.xlabel('Cache Depth')
    plt.ylabel('Average Time (seconds)')
    plt.title('Cache Depth vs Time by Target')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Heatmap: Target vs Clean Step
    plt.subplot(3, 4, 7)
    pivot_table = df.pivot_table(index='target', columns='clean_step', values='time_seconds', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Time (seconds)'})
    plt.title('Target vs Clean Step Heatmap')
    plt.xlabel('Clean Step')
    plt.ylabel('Target Class')
    
    # 8. Heatmap: Target vs Cache Depth
    plt.subplot(3, 4, 8)
    pivot_table = df.pivot_table(index='target', columns='cache_depth', values='time_seconds', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Time (seconds)'})
    plt.title('Target vs Cache Depth Heatmap')
    plt.xlabel('Cache Depth')
    plt.ylabel('Target Class')
    
    # 9. Box plot: Target별 시간 분포
    plt.subplot(3, 4, 9)
    df.boxplot(column='time_seconds', by='target', ax=plt.gca())
    plt.title('Time Distribution by Target')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Target Class')
    plt.ylabel('Time (seconds)')
    
    # 10. Box plot: Clean Step별 시간 분포
    plt.subplot(3, 4, 10)
    df.boxplot(column='time_seconds', by='clean_step', ax=plt.gca())
    plt.title('Time Distribution by Clean Step')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Clean Step')
    plt.ylabel('Time (seconds)')
    
    # 11. 3D Scatter plot: Target vs Clean Step vs Time
    ax = fig.add_subplot(3, 4, 11, projection='3d')
    scatter = ax.scatter(df['target'], df['clean_step'], df['time_seconds'], 
                        c=df['time_seconds'], cmap='viridis', s=30)
    ax.set_xlabel('Target Class')
    ax.set_ylabel('Clean Step')
    ax.set_zlabel('Time (seconds)')
    ax.set_title('3D: Target vs Clean Step vs Time')
    plt.colorbar(scatter, ax=ax, label='Time (seconds)')
    
    # 12. 통계 요약
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # 전체 통계
    total_stats = f"""
    전체 통계:
    
    총 실험 수: {len(df)}
    평균 시간: {df['time_seconds'].mean():.2f}초
    최소 시간: {df['time_seconds'].min():.2f}초
    최대 시간: {df['time_seconds'].max():.2f}초
    표준편차: {df['time_seconds'].std():.2f}초
    """
    
    # Target별 평균
    target_means = df.groupby('target')['time_seconds'].mean()
    target_stats = "\nTarget별 평균:\n"
    for target, mean_time in target_means.items():
        target_stats += f"  Target {target}: {mean_time:.2f}초\n"
    
    # Clean Step별 평균
    step_means = df.groupby('clean_step')['time_seconds'].mean()
    step_stats = "\nClean Step별 평균:\n"
    for step, mean_time in step_means.items():
        step_stats += f"  Step {step}: {mean_time:.2f}초\n"
    
    stats_text = total_stats + target_stats + step_stats
    plt.text(0.05, 0.95, stats_text, fontsize=9, verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('multi_target_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("그래프가 multi_target_optimization_results.png로 저장되었습니다.")
    
    # 추가 분석: 최적 조합 찾기
    print("\n=== 최적 조합 분석 ===")
    print("가장 빠른 조합:")
    fastest_combinations = df.nsmallest(5, 'time_seconds')[['target', 'clean_step', 'cache_depth', 'cache_interval', 'time_seconds']]
    for _, row in fastest_combinations.iterrows():
        print(f"  Target {row['target']}, Clean Step {row['clean_step']}, Cache Depth {row['cache_depth']}, Cache Interval {row['cache_interval']}: {row['time_seconds']:.2f}초")
    
    print("\nTarget별 최적 조합:")
    for target in df['target'].unique():
        subset = df[df['target'] == target]
        best_idx = subset['time_seconds'].idxmin()
        best_row = subset.loc[best_idx]
        print(f"  Target {target}: Clean Step {best_row['clean_step']}, Cache Depth {best_row['cache_depth']}, Cache Interval {best_row['cache_interval']} ({best_row['time_seconds']:.2f}초)")
        
else:
    print("experiment_timing_results.csv 파일을 찾을 수 없습니다.")
EOF

echo ""
echo "결과 시각화 스크립트 생성 완료:"
echo "- ${experiment_dir}/plot_results.py"
echo ""
echo "그래프 생성하려면:"
echo "cd ${experiment_dir} && python3 plot_results.py"

# 실험 정보 저장
echo "실험 정보:" > "${experiment_dir}/experiment_info.txt"
echo "시작 시간: $(date)" >> "${experiment_dir}/experiment_info.txt"
echo "총 실험 수: $total_experiments" >> "${experiment_dir}/experiment_info.txt"
echo "총 생성 이미지 수: $((total_experiments * num_samples))" >> "${experiment_dir}/experiment_info.txt"
echo "Target Classes: ${targets[*]}" >> "${experiment_dir}/experiment_info.txt"
echo "Deep Cache Interval: $cache_interval (0~clean_step: full, clean_step~50: deep cache)" >> "${experiment_dir}/experiment_info.txt"
echo "Clean Steps: 0 ${clean_steps[*]}" >> "${experiment_dir}/experiment_info.txt"
echo "Cache Depths: ${cache_block_ids[*]}" >> "${experiment_dir}/experiment_info.txt"
echo "실행 방식: 순차 실행" >> "${experiment_dir}/experiment_info.txt"
echo "시간 측정: 각 실험별 1024개 이미지 생성 시간 측정" >> "${experiment_dir}/experiment_info.txt"
echo "결과 파일: experiment_timing_results.csv" >> "${experiment_dir}/experiment_info.txt"

echo ""
echo "실험 정보가 ${experiment_dir}/experiment_info.txt에 저장되었습니다."

# 종료 시간 기록
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "=== 모든 실험 완료 ==="
echo "완료 시간: $(date)"
echo "총 소요 시간: ${duration}초 ($((duration / 60))분 $((duration % 60))초)"
echo ""
echo "실험 결과 확인:"
echo "1. 결과 수집: ${experiment_dir}/collect_results.sh"
echo "2. 그래프 생성: cd ${experiment_dir} && python3 plot_results.py"
echo "3. 실험 정보: cat ${experiment_dir}/experiment_info.txt"
echo "4. 시간 측정 결과: cat ${experiment_dir}/experiment_timing_results.csv" 