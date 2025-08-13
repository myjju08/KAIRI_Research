#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

def calculate_average_accuracy_proper():
    """Calculate average accuracy across all 4 targets with proper combination matching"""
    
    # 올바른 결과 파일들
    target_files = [
        'deepcache_target0_correct_accuracy_results.csv',
        'deepcache_target4_correct_accuracy_results.csv', 
        'deepcache_target6_correct_accuracy_results.csv',
        'deepcache_target8_correct_accuracy_results.csv'
    ]
    
    # 모든 결과를 저장할 리스트
    all_results = []
    
    # 각 target 파일 읽기
    for file in target_files:
        if os.path.exists(file):
            print(f"Reading {file}...")
            df = pd.read_csv(file)
            all_results.append(df)
        else:
            print(f"Warning: {file} not found")
    
    if not all_results:
        print("No result files found!")
        return
    
    # 모든 결과 합치기
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # cache_interval과 clean_step별로 그룹화하여 평균 계산
    # 이때 정확히 같은 조합끼리만 매칭
    avg_results = combined_df.groupby(['cache_interval', 'clean_step'])['condition_accuracy'].mean().reset_index()
    avg_results = avg_results.rename(columns={'condition_accuracy': 'average_condition_accuracy'})
    
    # 조합별 개수 확인 (모든 target에서 같은 조합이 있는지)
    combination_counts = combined_df.groupby(['cache_interval', 'clean_step']).size().reset_index(name='count')
    print(f"\n=== Combination Counts ===")
    print(f"Expected count per combination: 4 (one per target)")
    print(f"Min count: {combination_counts['count'].min()}")
    print(f"Max count: {combination_counts['count'].max()}")
    
    # 4개가 아닌 조합들 확인
    incomplete_combinations = combination_counts[combination_counts['count'] != 4]
    if len(incomplete_combinations) > 0:
        print(f"\n=== Incomplete Combinations (not 4 targets) ===")
        for _, row in incomplete_combinations.iterrows():
            print(f"Cache {row['cache_interval']}, Clean {row['clean_step']}: {row['count']} targets")
    
    # 결과 저장
    output_file = 'deepcache_average_accuracy_proper_results.csv'
    avg_results.to_csv(output_file, index=False)
    
    print(f"\n=== Average Accuracy Results (Proper Combination Matching) ===")
    print(f"Results saved to: {output_file}")
    print(f"Total combinations: {len(avg_results)}")
    
    # 전체 평균
    overall_avg = avg_results['average_condition_accuracy'].mean()
    print(f"Overall average accuracy: {overall_avg:.4f}")
    
    # Cache interval별 평균
    cache_avg = avg_results.groupby('cache_interval')['average_condition_accuracy'].mean()
    print(f"\n=== Cache Interval별 평균 ===")
    for cache_interval, avg_acc in cache_avg.items():
        print(f"Cache Interval {cache_interval}: {avg_acc:.4f}")
    
    # Clean step별 평균
    clean_avg = avg_results.groupby('clean_step')['average_condition_accuracy'].mean()
    print(f"\n=== Clean Step별 평균 ===")
    for clean_step, avg_acc in clean_avg.items():
        print(f"Clean Step {clean_step}: {avg_acc:.4f}")
    
    # 최고 성능 조합
    best_idx = avg_results['average_condition_accuracy'].idxmax()
    best_result = avg_results.loc[best_idx]
    print(f"\n=== Best Performance ===")
    print(f"Cache Interval: {best_result['cache_interval']}")
    print(f"Clean Step: {best_result['clean_step']}")
    print(f"Average Accuracy: {best_result['average_condition_accuracy']:.4f}")
    
    # 각 target별 개별 결과도 확인
    print(f"\n=== Individual Target Results ===")
    for i, file in enumerate(target_files):
        if os.path.exists(file):
            df = pd.read_csv(file)
            target_avg = df['condition_accuracy'].mean()
            print(f"Target {i*2}: {target_avg:.4f}")
    
    return avg_results

if __name__ == "__main__":
    calculate_average_accuracy_proper() 