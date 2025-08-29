#!/bin/bash
set -euo pipefail

# 스크립트 기준 프로젝트 루트 경로 계산 및 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"

# 로그 디렉토리 생성
LOG_DIR="$PROJECT_ROOT/logs_baseline_generation"
mkdir -p "$LOG_DIR"

# Baseline 이미지 생성 스크립트
echo "=== Baseline 이미지 생성 시작 (DeepCache 없음) ==="
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

# DeepCache 사용 안함
use_deepcache=False
skip_mode='uniform'

# Target classes
targets=(5)

# 실험 결과 저장 디렉토리
experiment_dir="$PROJECT_ROOT/baseline_images_no_deepcache_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$experiment_dir"

echo "실험 결과 저장 디렉토리: $experiment_dir"
echo "Target Classes: ${targets[*]}"
echo "DeepCache 사용: $use_deepcache"
echo "각 target별 1024개 이미지 생성"

# 실험: 각 target별로 이미지 생성
echo "=== 각 Target별 Baseline 이미지 생성 ==="

total_experiments=${#targets[@]}
current_experiment=0

# 실험 시간 측정을 위한 배열 초기화
declare -a experiment_times
declare -a experiment_targets

# 실험 시작 시간 기록
experiment_start_time=$(date +%s)

for target in "${targets[@]}"; do
    current_experiment=$((current_experiment + 1))
    echo ""
    echo "=== 실험 $current_experiment/$total_experiments ==="
    echo "Target: $target"
    echo "DeepCache 사용: $use_deepcache"
    echo "시작 시간: $(date)"
    
    # 실험 디렉토리 생성
    experiment_subdir="${experiment_dir}/target_${target}_baseline"
    mkdir -p "$experiment_subdir"
    
    # 임시 로깅 디렉토리를 실험 디렉토리로 설정
    temp_logging_dir="$experiment_subdir"
    
    # 개별 실험 로그 파일
    experiment_log_file="${LOG_DIR}/target_${target}_baseline.log"
    
    echo "실행 중: Target $target, Baseline (DeepCache 없음)..."
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
        --check_done False 2>&1 | tee "$experiment_log_file"
    
    # 실험 완료 시간 기록
    experiment_end_time=$(date +%s)
    experiment_duration=$((experiment_end_time - experiment_start_time))
    
    # 실험 시간 정보 저장
    experiment_times+=($experiment_duration)
    experiment_targets+=($target)
    
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

echo ""
echo "=== 모든 실험이 완료되었습니다 ==="
echo "실험 디렉토리: $experiment_dir"
echo "로그 디렉토리: $LOG_DIR"

# 실험 시간 결과를 CSV 파일로 저장
echo "=== 실험 시간 결과 저장 ==="
timing_csv_file="${experiment_dir}/baseline_timing_results.csv"

# CSV 헤더 작성
echo "target,time_seconds,time_minutes" > "$timing_csv_file"

# 각 실험의 시간 정보를 CSV에 추가
for i in "${!experiment_times[@]}"; do
    target="${experiment_targets[$i]}"
    time_seconds="${experiment_times[$i]}"
    time_minutes=$(echo "scale=2; $time_seconds / 60" | bc -l)
    
    echo "$target,$time_seconds,$time_minutes" >> "$timing_csv_file"
done

echo "실험 시간 결과가 $timing_csv_file에 저장되었습니다."

# 시간 결과 요약 출력
echo ""
echo "=== 실험 시간 요약 ==="
echo "총 실험 수: ${#experiment_times[@]}"
echo "평균 시간: $(echo "scale=2; $(IFS=+; echo "$((${experiment_times[*]}))") / ${#experiment_times[@]}" | bc -l)초"
echo "최소 시간: $(printf '%s\n' "${experiment_times[@]}" | sort -n | head -1)초"
echo "최대 시간: $(printf '%s\n' "${experiment_times[@]}" | sort -n | tail -1)초"

# Target별 시간 출력
echo ""
echo "=== Target별 시간 ==="
for i in "${!experiment_times[@]}"; do
    target="${experiment_targets[$i]}"
    time_seconds="${experiment_times[$i]}"
    time_minutes=$(echo "scale=2; $time_seconds / 60" | bc -l)
    echo "Target $target: ${time_seconds}초 (${time_minutes}분)"
done

# 결과 수집 스크립트 생성
cat > "${experiment_dir}/collect_results.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# 결과 수집 스크립트
echo "=== Baseline 이미지 생성 결과 수집 ==="

# 현재 디렉토리에서 실행
cd "$(dirname "$0")"

# 결과 파일 생성
echo "결과 수집 중..."
> results.txt

# 각 target별로 결과 수집
for target in 1 3 5 7 9; do
    experiment_dir="target_${target}_baseline"
    if [ -d "$experiment_dir" ]; then
        # images.npy 또는 .png 파일이 있는지 확인
        if find "$experiment_dir" -name "images.npy" -type f | grep -q . || find "$experiment_dir" -name "*.png" -type f | grep -q .; then
            echo "완료됨: Target $target (Baseline)"
            echo "Target_${target}_Baseline: 완료" >> results.txt
        else
            echo "진행 중 또는 실패: Target $target (Baseline)"
            echo "Target_${target}_Baseline: 실패" >> results.txt
        fi
    fi
done

# 결과를 CSV 파일로도 저장
echo "target,status" > results.csv
cat results.txt | sed 's/: /,/' | sed 's/Target_\([0-9]*\)_Baseline: \(.*\)/\1,\2/' >> results.csv

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
echo "3. 특정 target 확인: ls -la target_*_baseline/images.npy"
echo "4. PNG 파일 확인: ls -la target_*_baseline/*.png"
echo "5. 시간 측정 결과: cat baseline_timing_results.csv"
EOF

chmod +x "${experiment_dir}/collect_results.sh"

# 실험 정보 저장
echo "실험 정보:" > "${experiment_dir}/experiment_info.txt"
echo "시작 시간: $(date)" >> "${experiment_dir}/experiment_info.txt"
echo "총 실험 수: $total_experiments" >> "${experiment_dir}/experiment_info.txt"
echo "총 생성 이미지 수: $((total_experiments * num_samples))" >> "${experiment_dir}/experiment_info.txt"
echo "Target Classes: ${targets[*]}" >> "${experiment_dir}/experiment_info.txt"
echo "DeepCache 사용: $use_deepcache" >> "${experiment_dir}/experiment_info.txt"
echo "각 target별 이미지 수: $num_samples" >> "${experiment_dir}/experiment_info.txt"
echo "실행 방식: 순차 실행" >> "${experiment_dir}/experiment_info.txt"
echo "시간 측정: 각 실험별 1024개 이미지 생성 시간 측정" >> "${experiment_dir}/experiment_info.txt"
echo "결과 파일: baseline_timing_results.csv" >> "${experiment_dir}/experiment_info.txt"

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
echo "2. 실험 정보: cat ${experiment_dir}/experiment_info.txt"
echo "3. 시간 측정 결과: cat ${experiment_dir}/baseline_timing_results.csv" 