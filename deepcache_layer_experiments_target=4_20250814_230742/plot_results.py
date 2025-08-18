#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 결과 파일 읽기
results_file = "results.csv"
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    
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
    print("results.csv 파일을 찾을 수 없습니다.")
