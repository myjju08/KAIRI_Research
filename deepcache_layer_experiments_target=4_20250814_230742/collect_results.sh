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
    for layer_depth in {0..15}; do
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
