import re
import numpy as np

# lr_on 로그에서 gradient norm 추출
with open('logs/lr_ablation_run.log', 'r') as f:
    log_content = f.read()

# [layer-routing] 패턴 찾기
lr_on_pattern = r'\[layer-routing\] t=(\d+) active=\[([^\]]+)\] grad_norm=([\d.e+-]+)'
lr_on_matches = re.findall(lr_on_pattern, log_content)

if lr_on_matches:
    print("=== Layer Routing ON Gradient Norms ===")
    print(f"Total entries: {len(lr_on_matches)}")
    
    # timestep별 grad_norm 분석
    grad_norms_by_phase = {
        'early (t=36-49, blocks 0-3)': [],
        'mid (t=16-35, blocks 4-7)': [],
        'late (t=0-15, blocks 8-11)': []
    }
    
    for t_str, blocks_str, grad_norm_str in lr_on_matches:
        t = int(t_str)
        grad_norm = float(grad_norm_str)
        
        if 36 <= t <= 49:
            grad_norms_by_phase['early (t=36-49, blocks 0-3)'].append(grad_norm)
        elif 16 <= t <= 35:
            grad_norms_by_phase['mid (t=16-35, blocks 4-7)'].append(grad_norm)
        elif 0 <= t <= 15:
            grad_norms_by_phase['late (t=0-15, blocks 8-11)'].append(grad_norm)
    
    print("\nGradient norm statistics by phase:")
    for phase, norms in grad_norms_by_phase.items():
        if norms:
            print(f"\n{phase}:")
            print(f"  Mean: {np.mean(norms):.4f}")
            print(f"  Std:  {np.std(norms):.4f}")
            print(f"  Min:  {np.min(norms):.4f}")
            print(f"  Max:  {np.max(norms):.4f}")
            print(f"  Count: {len(norms)}")
    
    # 전체 통계
    all_norms = [float(m[2]) for m in lr_on_matches]
    print(f"\nOverall lr_on:")
    print(f"  Mean grad_norm: {np.mean(all_norms):.4f}")
    print(f"  Std grad_norm:  {np.std(all_norms):.4f}")
else:
    print("No layer-routing gradient norms found in log")

print("\n" + "="*50)
print("Key insight: Gradient는 unit-norm으로 정규화됨!")
print("즉, 4개 블록 gradient나 12개 블록 gradient나")
print("최종 guidance 세기가 동일함 (guidance_scale * 1.0)")
print("="*50)
