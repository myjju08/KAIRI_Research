#!/usr/bin/env python3
"""
start_gradient 5부터 모든 guidance scale에 대해 FID score를 수집하는 스크립트
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

def get_all_start_gradients():
    """logs 디렉토리에서 start_gradient 0부터 모든 값들을 찾습니다."""
    logs_dir = Path("logs")
    start_gradients = []
    
    if not logs_dir.exists():
        print("logs 디렉토리가 존재하지 않습니다.")
        return []
    
    for item in logs_dir.iterdir():
        if item.is_dir() and item.name.startswith("start_gradient_"):
            try:
                gradient_num = int(item.name.replace("start_gradient_", ""))
                if gradient_num >= 0:  # 0부터 시작
                    start_gradients.append(gradient_num)
            except ValueError:
                continue
    
    return sorted(start_gradients)

def get_guidance_scales_for_start_gradient(start_gradient):
    """특정 start_gradient에 대한 모든 guidance scale을 찾습니다."""
    start_gradient_dir = Path(f"logs/start_gradient_{start_gradient}")
    guidance_scales = []
    
    if not start_gradient_dir.exists():
        print(f"start_gradient_{start_gradient} 디렉토리가 존재하지 않습니다.")
        return []
    
    for item in start_gradient_dir.iterdir():
        if item.is_dir() and item.name.startswith("guidance_scale_"):
            try:
                scale_str = item.name.replace("guidance_scale_", "")
                # 문자열을 숫자로 변환 (0.1, 0, 1, 2, 5, 10 등)
                if scale_str == "0":
                    scale = 0
                elif "." in scale_str:
                    scale = float(scale_str)
                else:
                    scale = int(scale_str)
                guidance_scales.append(scale)
            except ValueError:
                continue
    
    return sorted(guidance_scales)

def extract_metrics_from_json(metrics_path):
    """metrics.json 파일에서 FID score와 condition accuracy를 추출합니다."""
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        fid_score = metrics.get("fid", None)
        # resnet classifier의 condition accuracy (target=1에 대한 정확도)
        condition_accuracy = metrics.get("validity_resnet_cifar10.pt", None)
        
        return {
            "fid": fid_score,
            "condition_accuracy": condition_accuracy
        }
    except Exception as e:
        print(f"metrics.json 파일 읽기 오류: {e}")
        return {"fid": None, "condition_accuracy": None}

def find_metrics_json(start_gradient, guidance_scale, run_num=0):
    """특정 설정에 대한 metrics.json 파일을 찾습니다."""
    base_path = Path(f"logs/start_gradient_{start_gradient}/guidance_scale_{guidance_scale}/run_{run_num}")
    
    if not base_path.exists():
        return None
    
    # find 명령어로 metrics.json 파일 찾기
    try:
        result = subprocess.run(['find', str(base_path), '-name', 'metrics.json', '-type', 'f'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            metrics_path = result.stdout.strip().split('\n')[0]
            if os.path.exists(metrics_path):
                return metrics_path
    except Exception as e:
        print(f"metrics.json 파일 검색 오류: {e}")
    
    return None

def find_images_npy(start_gradient, guidance_scale, run_num=0):
    """특정 설정에 대한 images.npy 파일을 찾습니다."""
    base_path = Path(f"logs/start_gradient_{start_gradient}/guidance_scale_{guidance_scale}/run_{run_num}")
    
    if not base_path.exists():
        return None
    
    # find 명령어로 images.npy 파일 찾기
    try:
        result = subprocess.run(['find', str(base_path), '-name', 'images.npy', '-type', 'f'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            images_path = result.stdout.strip().split('\n')[0]
            if os.path.exists(images_path):
                return images_path
    except Exception as e:
        print(f"images.npy 파일 검색 오류: {e}")
    
    return None

def load_combined_images_for_setting(start_gradient, guidance_scale):
    """특정 설정의 모든 run에서 이미지를 합쳐서 로드합니다."""
    all_images = []
    
    for run_num in range(5):  # run_0부터 run_4까지
        images_path = find_images_npy(start_gradient, guidance_scale, run_num)
        
        if images_path:
            try:
                images_array = np.load(images_path)
                images = [Image.fromarray(img) for img in images_array]
                all_images.extend(images)
                print(f"    run_{run_num}: {len(images)}개 이미지 로드됨")
            except Exception as e:
                print(f"    run_{run_num}: 이미지 로드 오류 - {e}")
        else:
            print(f"    run_{run_num}: images.npy 파일을 찾을 수 없음")
    
    print(f"  총 {len(all_images)}개 이미지가 합쳐졌습니다.")
    return all_images

def calculate_fid_proper(images, dataset="cifar10"):
    """기존 calculate_fid_proper.py의 함수를 사용해서 FID를 계산합니다."""
    try:
        # calculate_fid_proper.py에서 필요한 함수들을 import
        from calculate_fid_proper import calculate_fid_proper as calc_fid
        
        # FID 계산
        fid_score = calc_fid(images, dataset)
        return fid_score
        
    except Exception as e:
        print(f"FID 계산 중 오류 발생: {e}")
        return None

def collect_metrics_scores():
    """모든 start_gradient와 guidance_scale에 대해 FID score와 condition accuracy를 수집합니다."""
    start_gradients = get_all_start_gradients()
    print(f"발견된 start_gradients: {start_gradients}")
    
    all_results = []
    
    for start_gradient in start_gradients:
        print(f"\n=== start_gradient_{start_gradient} 처리 중 ===")
        guidance_scales = get_guidance_scales_for_start_gradient(start_gradient)
        print(f"발견된 guidance_scales: {guidance_scales}")
        
        for guidance_scale in guidance_scales:
            print(f"  guidance_scale_{guidance_scale} 처리 중...")
            
            # 모든 run의 이미지를 합쳐서 FID 계산
            combined_images = load_combined_images_for_setting(start_gradient, guidance_scale)
            
            # 각 run에 대해 condition accuracy 수집
            run_accuracies = []
            for run_num in range(5):  # run_0부터 run_4까지
                metrics_path = find_metrics_json(start_gradient, guidance_scale, run_num)
                
                if metrics_path:
                    metrics = extract_metrics_from_json(metrics_path)
                    if metrics["condition_accuracy"] is not None:
                        run_accuracies.append(metrics["condition_accuracy"])
                        print(f"    run_{run_num}: Condition Accuracy = {metrics['condition_accuracy']:.4f}")
                    else:
                        print(f"    run_{run_num}: Condition Accuracy 추출 실패")
                else:
                    print(f"    run_{run_num}: metrics.json 파일을 찾을 수 없음")
            
            # 합쳐진 이미지로 FID 계산
            combined_fid = None
            if combined_images:
                print(f"  합쳐진 이미지로 FID 계산 중... (총 {len(combined_images)}개 이미지)")
                combined_fid = calculate_fid_proper(combined_images, "cifar10")
                if combined_fid:
                    combined_fid = combined_fid.get("fid")
                    print(f"  합쳐진 FID: {combined_fid:.4f}")
            
            # 평균 condition accuracy 계산
            avg_accuracy = sum(run_accuracies) / len(run_accuracies) if run_accuracies else None
            
            if combined_fid is not None or avg_accuracy is not None:
                result = {
                    "start_gradient": start_gradient,
                    "guidance_scale": guidance_scale,
                    "combined_fid": combined_fid,
                    "avg_condition_accuracy": avg_accuracy,
                    "individual_accuracies": run_accuracies,
                    "num_images": len(combined_images) if combined_images else 0,
                    "num_runs": len(run_accuracies)
                }
                all_results.append(result)
                
                if combined_fid is not None:
                    print(f"  합쳐진 FID: {combined_fid:.4f} ({len(combined_images)}개 이미지)")
                if avg_accuracy is not None:
                    print(f"  평균 Condition Accuracy: {avg_accuracy:.4f} (총 {len(run_accuracies)}개 실행)")
            else:
                print(f"  guidance_scale_{guidance_scale}: 유효한 metrics 없음")
    
    return all_results

def save_results_to_files(results):
    """결과를 다양한 형식으로 저장합니다."""
    if not results:
        print("저장할 결과가 없습니다.")
        return
    
    # JSON 형식으로 저장
    with open("metrics_collected.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("결과가 metrics_collected.json에 저장되었습니다.")
    
    # CSV 형식으로 저장 (간단한 버전)
    csv_data = []
    for result in results:
        csv_data.append({
            "start_gradient": result["start_gradient"],
            "guidance_scale": result["guidance_scale"],
            "combined_fid": result["combined_fid"],
            "avg_condition_accuracy": result["avg_condition_accuracy"],
            "num_images": result["num_images"],
            "num_runs": result["num_runs"]
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv("metrics_summary.csv", index=False)
    print("요약 결과가 metrics_summary.csv에 저장되었습니다.")
    
    # 상세 CSV (개별 실행 결과 포함)
    detailed_csv_data = []
    for result in results:
        for i, accuracy in enumerate(result["individual_accuracies"]):
            row = {
                "start_gradient": result["start_gradient"],
                "guidance_scale": result["guidance_scale"],
                "run_num": i,
                "condition_accuracy": accuracy
            }
            detailed_csv_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_csv_data)
    detailed_df.to_csv("metrics_detailed.csv", index=False)
    print("상세 결과가 metrics_detailed.csv에 저장되었습니다.")

def print_summary(results):
    """결과 요약을 출력합니다."""
    if not results:
        print("표시할 결과가 없습니다.")
        return
    
    print("\n" + "=" * 80)
    print("METRICS 수집 결과 요약")
    print("=" * 80)
    
    # start_gradient별로 그룹화
    for start_gradient in sorted(set(r["start_gradient"] for r in results)):
        print(f"\nstart_gradient_{start_gradient}:")
        print("-" * 60)
        
        gradient_results = [r for r in results if r["start_gradient"] == start_gradient]
        gradient_results.sort(key=lambda x: x["guidance_scale"])
        
        for result in gradient_results:
            fid_str = f"FID = {result['combined_fid']:>8.4f}" if result['combined_fid'] is not None else "FID = N/A"
            acc_str = f"Acc = {result['avg_condition_accuracy']:>6.4f}" if result['avg_condition_accuracy'] is not None else "Acc = N/A"
            img_str = f"({result['num_images']} images)" if result['num_images'] > 0 else "(no images)"
            print(f"  guidance_scale_{result['guidance_scale']:>5}: {fid_str} | {acc_str} {img_str}")
    
    # 전체 통계
    valid_fids = [r["combined_fid"] for r in results if r["combined_fid"] is not None]
    valid_accs = [r["avg_condition_accuracy"] for r in results if r["avg_condition_accuracy"] is not None]
    
    if valid_fids:
        print(f"\nFID 통계:")
        print(f"  총 설정 수: {len(valid_fids)}")
        print(f"  평균 FID: {sum(valid_fids)/len(valid_fids):.4f}")
        print(f"  최소 FID: {min(valid_fids):.4f}")
        print(f"  최대 FID: {max(valid_fids):.4f}")
    
    if valid_accs:
        print(f"\nCondition Accuracy 통계:")
        print(f"  총 설정 수: {len(valid_accs)}")
        print(f"  평균 Accuracy: {sum(valid_accs)/len(valid_accs):.4f}")
        print(f"  최소 Accuracy: {min(valid_accs):.4f}")
        print(f"  최대 Accuracy: {max(valid_accs):.4f}")

def main():
    print("start_gradient 5부터 모든 guidance scale에 대해 FID score와 condition accuracy 수집을 시작합니다...")
    
    # metrics 수집
    results = collect_metrics_scores()
    
    if results:
        # 결과 저장
        save_results_to_files(results)
        
        # 요약 출력
        print_summary(results)
        
        print(f"\n총 {len(results)}개의 설정에서 metrics를 수집했습니다.")
    else:
        print("수집된 metrics가 없습니다.")

if __name__ == "__main__":
    main() 