#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

def calculate_average_accuracy():
    """Calculate average accuracy across all 4 targets"""
    
    # 결과 파일들
    target_files = [
        'deepcache_target0_accuracy_results.csv',
        'deepcache_target4_accuracy_results.csv', 
        'deepcache_target6_accuracy_results.csv',
        'deepcache_target8_accuracy_results.csv'
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
    avg_results = combined_df.groupby(['cache_interval', 'clean_step'])['condition_accuracy'].mean().reset_index()
    avg_results = avg_results.rename(columns={'condition_accuracy': 'average_condition_accuracy'})
    
    # 결과 저장
    output_file = 'deepcache_average_accuracy_results.csv'
    avg_results.to_csv(output_file, index=False)
    
    print(f"\n=== Average Accuracy Results ===")
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
    
    return avg_results

if __name__ == "__main__":
    calculate_average_accuracy() 