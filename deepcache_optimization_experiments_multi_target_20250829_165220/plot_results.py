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
