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
