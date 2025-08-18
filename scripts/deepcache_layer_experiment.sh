#!/bin/bash
set -euo pipefail

# 스크립트 기준 프로젝트 루트 경로 계산 및 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# 로그 디렉토리 생성
LOG_DIR="$PROJECT_ROOT/logs_layer_experiment"
mkdir -p "$LOG_DIR"

# DeepCache Layer Depth 실험 스크립트
echo "=== DeepCache Layer Depth 실험 시작 ==="
echo "프로젝트 루트: $PROJECT_ROOT"
echo "시작 시간: $(date)"
start_time=$(date +%s)

# 기본 설정
CUDA_VISIBLE_DEVICES=1
data_type=image
task=label_guidance
image_size=32
dataset="cifar10"
model_name_or_path='openai_cifar10.pt'
guide_network='resnet_cifar10.pt'
train_steps=1000
inference_steps=50
eta=1.0
target=6
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
cache_interval=1
skip_mode='uniform'
clean_step=0

# Layer Depth 및 Cache Interval 실험 설정
cache_block_ids=(1)
cache_intervals=(4)

# 실험 결과 저장 디렉토리 (프로젝트 루트에 저장)
experiment_dir="$PROJECT_ROOT/deepcache_layer_experiments_target=${target}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$experiment_dir"

echo "실험 결과 저장 디렉토리: $experiment_dir"
echo "테스트할 Layer Depths: ${cache_block_ids[*]}"
echo "테스트할 Cache Intervals: ${cache_intervals[*]}"

# 실험: Layer Depth와 Cache Interval 조합별로 실험
echo "=== Layer Depth 및 Cache Interval 조합별 실험 (순차 실행) ==="

total_experiments=$(( ${#cache_block_ids[@]} * ${#cache_intervals[@]} ))
current_experiment=0

# 실험 시간 측정을 위한 배열 초기화
declare -a experiment_times
declare -a experiment_cache_intervals
declare -a experiment_layer_depths

# 실험 시작 시간 기록
experiment_start_time=$(date +%s)

for cache_interval in "${cache_intervals[@]}"; do
    for cache_block_id in "${cache_block_ids[@]}"; do
    current_experiment=$((current_experiment + 1))
    echo ""
    echo "=== 실험 $current_experiment/$total_experiments ==="
    echo "Cache Interval: $cache_interval, Layer Depth (cache_block_id): $cache_block_id"
    echo "시작 시간: $(date)"
    
    # 실험 디렉토리 생성
    experiment_subdir="${experiment_dir}/cache_interval_${cache_interval}_layer_depth_${cache_block_id}"
    mkdir -p "$experiment_subdir"
    
    # 임시 로깅 디렉토리를 실험 디렉토리로 설정
    temp_logging_dir="$experiment_subdir"
    
    # 개별 실험 로그 파일
    experiment_log_file="${LOG_DIR}/cache_${cache_interval}_layer_${cache_block_id}.log"
    
    echo "실행 중: Cache Interval $cache_interval, Layer Depth $cache_block_id..."
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
    experiment_cache_intervals+=($cache_interval)
    experiment_layer_depths+=($cache_block_id)
    
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

echo ""
echo "=== 모든 실험이 완료되었습니다 ==="
echo "실험 디렉토리: $experiment_dir"
echo "로그 디렉토리: $LOG_DIR"

# 실험 시간 결과를 CSV 파일로 저장
echo "=== 실험 시간 결과 저장 ==="
timing_csv_file="${experiment_dir}/experiment_timing_results.csv"

# CSV 헤더 작성
echo "cache_interval,layer_depth,time_seconds,time_minutes" > "$timing_csv_file"

# 각 실험의 시간 정보를 CSV에 추가
for i in "${!experiment_times[@]}"; do
    cache_interval="${experiment_cache_intervals[$i]}"
    layer_depth="${experiment_layer_depths[$i]}"
    time_seconds="${experiment_times[$i]}"
    time_minutes=$(echo "scale=2; $time_seconds / 60" | bc -l)
    
    echo "$cache_interval,$layer_depth,$time_seconds,$time_minutes" >> "$timing_csv_file"
done

echo "실험 시간 결과가 $timing_csv_file에 저장되었습니다."

# 시간 결과 요약 출력
echo ""
echo "=== 실험 시간 요약 ==="
echo "총 실험 수: ${#experiment_times[@]}"
echo "평균 시간: $(echo "scale=2; $(IFS=+; echo "$((${experiment_times[*]}))") / ${#experiment_times[@]}" | bc -l)초"
echo "최소 시간: $(printf '%s\n' "${experiment_times[@]}" | sort -n | head -1)초"
echo "최대 시간: $(printf '%s\n' "${experiment_times[@]}" | sort -n | tail -1)초"

# Cache Interval별 평균 시간
echo ""
echo "=== Cache Interval별 평균 시간 ==="
for interval in "${cache_intervals[@]}"; do
    total_time=0
    count=0
    for i in "${!experiment_cache_intervals[@]}"; do
        if [[ "${experiment_cache_intervals[$i]}" == "$interval" ]]; then
            total_time=$((total_time + experiment_times[i]))
            count=$((count + 1))
        fi
    done
    if [[ $count -gt 0 ]]; then
        avg_time=$(echo "scale=2; $total_time / $count" | bc -l)
        echo "Cache Interval $interval: ${avg_time}초 (n=$count)"
    fi
done

# Layer Depth별 평균 시간 (상위 5개)
echo ""
echo "=== Layer Depth별 평균 시간 (상위 5개) ==="
declare -A depth_times
declare -A depth_counts

for i in "${!experiment_layer_depths[@]}"; do
    depth="${experiment_layer_depths[$i]}"
    time="${experiment_times[$i]}"
    
    if [[ -z "${depth_times[$depth]}" ]]; then
        depth_times[$depth]=0
        depth_counts[$depth]=0
    fi
    
    depth_times[$depth]=$((depth_times[$depth] + time))
    depth_counts[$depth]=$((depth_counts[$depth] + 1))
done

# 평균 시간 계산 및 정렬
for depth in "${!depth_times[@]}"; do
    if [[ ${depth_counts[$depth]} -gt 0 ]]; then
        avg_time=$(echo "scale=2; ${depth_times[$depth]} / ${depth_counts[$depth]}" | bc -l)
        echo "$depth $avg_time" >> /tmp/depth_avg_times.txt
    fi
done

if [[ -f /tmp/depth_avg_times.txt ]]; then
    sort -k2,2n /tmp/depth_avg_times.txt | head -5 | while read depth avg_time; do
        echo "Layer Depth $depth: ${avg_time}초"
    done
    rm /tmp/depth_avg_times.txt
fi

# 결과 수집 스크립트 생성
cat > "${experiment_dir}/collect_results.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# 결과 수집 스크립트
echo "=== DeepCache Layer Experiment 결과 수집 ==="

# 현재 디렉토리에서 실행
cd "$(dirname "$0")"

# 결과 파일 생성
echo "결과 수집 중..."
> results.txt

# 각 실험 디렉토리에서 결과 수집
for cache_interval in 3; do
    for layer_depth in {1..15}; do
        experiment_dir="cache_interval_${cache_interval}_layer_depth_${layer_depth}"
        if [ -d "$experiment_dir" ]; then
            # images.npy 파일이 있는지 확인
            if find "$experiment_dir" -name "images.npy" -type f | grep -q .; then
                echo "완료됨: $experiment_dir"
                echo "${experiment_dir}: 완료" >> results.txt
            else
                echo "진행 중 또는 실패: $experiment_dir"
                echo "${experiment_dir}: 실패" >> results.txt
            fi
        fi
    done
done

# 결과를 CSV 파일로도 저장
echo "cache_interval,layer_depth,status" > results.csv
cat results.txt | sed 's/: /,/' | sed 's/cache_interval_\([0-9]*\)_layer_depth_\([0-9]*\): \(.*\)/\1,\2,\3/' >> results.csv

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
echo "3. 특정 실험 확인: ls -la cache_interval_*/images.npy"
echo "4. 시간 측정 결과: cat experiment_timing_results.csv"
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
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Heatmap: Cache Interval vs Layer Depth
    plt.subplot(2, 3, 1)
    pivot_table = df.pivot(index='layer_depth', columns='cache_interval', values='time_seconds')
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Time (seconds)'})
    plt.title('Cache Interval vs Layer Depth Heatmap')
    plt.xlabel('Cache Interval')
    plt.ylabel('Layer Depth')
    
    # 2. Line plot: Layer Depth별 시간 (Cache Interval별)
    plt.subplot(2, 3, 2)
    for interval in df['cache_interval'].unique():
        subset = df[df['cache_interval'] == interval]
        plt.plot(subset['layer_depth'], subset['time_seconds'], 'o-', label=f'Cache Interval {interval}', linewidth=2, markersize=6)
    plt.xlabel('Layer Depth')
    plt.ylabel('Time (seconds)')
    plt.title('Layer Depth vs Time by Cache Interval')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Box plot: Cache Interval별 시간 분포
    plt.subplot(2, 3, 3)
    df.boxplot(column='time_seconds', by='cache_interval', ax=plt.gca())
    plt.title('Time Distribution by Cache Interval')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Cache Interval')
    plt.ylabel('Time (seconds)')
    
    # 4. Box plot: Layer Depth별 시간 분포
    plt.subplot(2, 3, 4)
    df.boxplot(column='time_seconds', by='layer_depth', ax=plt.gca())
    plt.title('Time Distribution by Layer Depth')
    plt.suptitle('')  # Remove default title
    plt.xlabel('Layer Depth')
    plt.ylabel('Time (seconds)')
    
    # 5. 3D Scatter plot
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    scatter = ax.scatter(df['cache_interval'], df['layer_depth'], df['time_seconds'], 
                        c=df['time_seconds'], cmap='viridis', s=50)
    ax.set_xlabel('Cache Interval')
    ax.set_ylabel('Layer Depth')
    ax.set_zlabel('Time (seconds)')
    ax.set_title('3D View: Cache Interval vs Layer Depth vs Time')
    plt.colorbar(scatter, ax=ax, label='Time (seconds)')
    
    # 6. 통계 요약
    plt.subplot(2, 3, 6)
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
    
    # 가장 빠른/느린 조합
    fastest_idx = df['time_seconds'].idxmin()
    slowest_idx = df['time_seconds'].idxmax()
    fastest_combo = f"가장 빠른: Cache {df.loc[fastest_idx, 'cache_interval']}, Layer {df.loc[fastest_idx, 'layer_depth']} ({df.loc[fastest_idx, 'time_seconds']:.2f}초)"
    slowest_combo = f"가장 느린: Cache {df.loc[slowest_idx, 'cache_interval']}, Layer {df.loc[slowest_idx, 'layer_depth']} ({df.loc[slowest_idx, 'time_seconds']:.2f}초)"
    
    # Cache Interval별 평균
    interval_means = df.groupby('cache_interval')['time_seconds'].mean()
    interval_stats = "\nCache Interval별 평균:\n"
    for interval, mean_time in interval_means.items():
        interval_stats += f"  Cache {interval}: {mean_time:.2f}초\n"
    
    # Layer Depth별 평균
    depth_means = df.groupby('layer_depth')['time_seconds'].mean()
    depth_stats = "\nLayer Depth별 평균 (상위 5개):\n"
    top_depths = depth_means.nsmallest(5)
    for depth, mean_time in top_depths.items():
        depth_stats += f"  Layer {depth}: {mean_time:.2f}초\n"
    
    stats_text = total_stats + "\n" + fastest_combo + "\n" + slowest_combo + interval_stats + depth_stats
    plt.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('cache_interval_layer_depth_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("그래프가 cache_interval_layer_depth_results.png로 저장되었습니다.")
    
    # 추가 분석: 최적 조합 찾기
    print("\n=== 최적 조합 분석 ===")
    print("가장 빠른 조합:")
    fastest_combinations = df.nsmallest(5, 'time_seconds')[['cache_interval', 'layer_depth', 'time_seconds']]
    for _, row in fastest_combinations.iterrows():
        print(f"  Cache Interval {row['cache_interval']}, Layer Depth {row['layer_depth']}: {row['time_seconds']:.2f}초")
    
    print("\nCache Interval별 최적 Layer Depth:")
    for interval in df['cache_interval'].unique():
        subset = df[df['cache_interval'] == interval]
        best_depth = subset.loc[subset['time_seconds'].idxmin(), 'layer_depth']
        best_time = subset['time_seconds'].min()
        print(f"  Cache Interval {interval}: Layer Depth {best_depth} ({best_time:.2f}초)")
        
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
echo "Cache Intervals: ${cache_intervals[*]}" >> "${experiment_dir}/experiment_info.txt"
echo "Layer Depths: ${cache_block_ids[*]}" >> "${experiment_dir}/experiment_info.txt"
echo "Target: $target" >> "${experiment_dir}/experiment_info.txt"
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