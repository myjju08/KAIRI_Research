#!/usr/bin/env python3
"""
start_gradient가 10 이상인 경우에만 condition accuracy를 계산하여 기존 JSON 파일을 업데이트하는 스크립트
"""

import os
import json
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image
from evaluations.image import ImageEvaluator
from utils.configs import Arguments
from logger import setup_logger

def load_existing_metrics():
    """기존 metrics_collected.json 파일을 로드합니다."""
    try:
        with open("metrics_collected.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("metrics_collected.json 파일을 찾을 수 없습니다.")
        return []

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

def load_images_from_npy(images_path):
    """images.npy 파일에서 이미지들을 로드합니다."""
    try:
        images_array = np.load(images_path)
        images = [Image.fromarray(img) for img in images_array]
        return images
    except Exception as e:
        print(f"이미지 로드 오류: {e}")
        return []

def create_config_for_evaluation():
    """평가를 위한 설정을 생성합니다."""
    args = Arguments()
    args.data_type = 'image'
    args.dataset = 'cifar10'
    args.task = 'label_guidance'
    args.target = '1'  # target=1에 대한 정확도
    args.guide_network = 'resnet_cifar10.pt'
    args.device = 'cuda'
    args.eval_batch_size = 512
    args.image_size = 32
    args.classifiers_path = './pretrained_models/evaluate_mu/best_checkpoint.npy'
    args.args_classifiers_path = './pretrained_models/evaluate_mu/args.pickle'
    args.logging_dir = 'logs/temp_evaluation'
    args.log_suffix = ''
    
    # targets 설정
    args.targets = ['1']
    args.tasks = ['label_guidance']
    args.guide_networks = ['resnet_cifar10.pt']
    
    # logger 설정
    setup_logger(args)
    
    return args

def calculate_condition_accuracy_for_setting(start_gradient, guidance_scale):
    """특정 설정에 대해 condition accuracy를 계산합니다."""
    print(f"  guidance_scale_{guidance_scale} 처리 중...")
    
    # 각 run에 대해 이미지 수집
    all_images = []
    
    for run_num in range(5):  # run_0부터 run_4까지
        images_path = find_images_npy(start_gradient, guidance_scale, run_num)
        
        if images_path:
            images = load_images_from_npy(images_path)
            if images:
                all_images.extend(images)
                print(f"    run_{run_num}: {len(images)}개 이미지 로드됨")
            else:
                print(f"    run_{run_num}: 이미지 로드 실패")
        else:
            print(f"    run_{run_num}: images.npy 파일을 찾을 수 없음")
    
    # 모든 이미지로 condition accuracy 계산
    if all_images:
        print(f"  총 {len(all_images)}개 이미지로 condition accuracy 계산 중...")
        
        # ImageEvaluator 설정
        config_args = create_config_for_evaluation()
        
        try:
            evaluator = ImageEvaluator(config_args)
            metrics = evaluator.evaluate(all_images)
            
            # validity_resnet_cifar10.pt 키에서 condition accuracy 추출
            condition_accuracy = metrics.get("validity_resnet_cifar10.pt", None)
            
            if condition_accuracy is not None:
                print(f"  Condition Accuracy: {condition_accuracy:.4f}")
                return condition_accuracy, [condition_accuracy]  # 모든 run이 동일한 accuracy
            else:
                print(f"  Condition Accuracy 계산 실패")
                return None, []
                
        except Exception as e:
            print(f"  평가 중 오류 발생: {e}")
            return None, []
    else:
        print(f"  guidance_scale_{guidance_scale}: 유효한 이미지 없음")
        return None, []

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

def fill_missing_condition_accuracy():
    """start_gradient가 10 이상인 경우에만 condition accuracy를 계산하여 JSON을 업데이트합니다."""
    # 기존 데이터 로드
    existing_data = load_existing_metrics()
    if not existing_data:
        print("기존 데이터를 로드할 수 없습니다.")
        return
    
    print(f"기존 데이터 {len(existing_data)}개 항목을 로드했습니다.")
    
    # start_gradient가 10 이상인 항목들 찾기
    target_start_gradients = [10, 15, 20, 25, 30, 35, 40, 45]
    updated_count = 0
    
    for start_gradient in target_start_gradients:
        print(f"\n=== start_gradient_{start_gradient} 처리 중 ===")
        guidance_scales = get_guidance_scales_for_start_gradient(start_gradient)
        print(f"발견된 guidance_scales: {guidance_scales}")
        
        for guidance_scale in guidance_scales:
            # 기존 데이터에서 해당 항목 찾기
            existing_item = None
            for item in existing_data:
                if item["start_gradient"] == start_gradient and item["guidance_scale"] == guidance_scale:
                    existing_item = item
                    break
            
            if existing_item is None:
                print(f"  guidance_scale_{guidance_scale}: 기존 데이터에서 찾을 수 없음")
                continue
            
            # condition accuracy가 이미 있으면 건너뛰기
            if existing_item["avg_condition_accuracy"] is not None:
                print(f"  guidance_scale_{guidance_scale}: 이미 condition accuracy가 있음 ({existing_item['avg_condition_accuracy']:.4f})")
                continue
            
            # condition accuracy 계산
            avg_accuracy, individual_accuracies = calculate_condition_accuracy_for_setting(start_gradient, guidance_scale)
            
            # 기존 데이터 업데이트
            if avg_accuracy is not None:
                existing_item["avg_condition_accuracy"] = avg_accuracy
                existing_item["individual_accuracies"] = individual_accuracies
                existing_item["num_runs"] = len(individual_accuracies)
                updated_count += 1
                print(f"  guidance_scale_{guidance_scale}: condition accuracy 업데이트 완료")
            else:
                print(f"  guidance_scale_{guidance_scale}: condition accuracy 계산 실패")
    
    # 업데이트된 데이터 저장
    if updated_count > 0:
        with open("metrics_collected.json", 'w') as f:
            json.dump(existing_data, f, indent=2)
        print(f"\n총 {updated_count}개 항목의 condition accuracy를 업데이트했습니다.")
        print("결과가 metrics_collected.json에 저장되었습니다.")
    else:
        print("\n업데이트할 항목이 없습니다.")

def print_summary():
    """업데이트된 결과 요약을 출력합니다."""
    try:
        with open("metrics_collected.json", 'r') as f:
            data = json.load(f)
        
        print("\n" + "=" * 80)
        print("업데이트된 METRICS 요약")
        print("=" * 80)
        
        # start_gradient별로 그룹화
        for start_gradient in sorted(set(item["start_gradient"] for item in data)):
            print(f"\nstart_gradient_{start_gradient}:")
            print("-" * 60)
            
            gradient_items = [item for item in data if item["start_gradient"] == start_gradient]
            gradient_items.sort(key=lambda x: x["guidance_scale"])
            
            for item in gradient_items:
                fid_str = f"FID = {item['combined_fid']:>8.4f}" if item['combined_fid'] is not None else "FID = N/A"
                acc_str = f"Acc = {item['avg_condition_accuracy']:>6.4f}" if item['avg_condition_accuracy'] is not None else "Acc = N/A"
                runs_str = f"({item['num_runs']} runs)" if item['num_runs'] > 0 else "(0 runs)"
                print(f"  guidance_scale_{item['guidance_scale']:>5}: {fid_str} | {acc_str} {runs_str}")
        
        # 전체 통계
        valid_accs = [item["avg_condition_accuracy"] for item in data if item["avg_condition_accuracy"] is not None]
        
        if valid_accs:
            print(f"\nCondition Accuracy 통계:")
            print(f"  총 설정 수: {len(valid_accs)}")
            print(f"  평균 Accuracy: {sum(valid_accs)/len(valid_accs):.4f}")
            print(f"  최소 Accuracy: {min(valid_accs):.4f}")
            print(f"  최대 Accuracy: {max(valid_accs):.4f}")
        
    except Exception as e:
        print(f"요약 출력 중 오류 발생: {e}")

def main():
    print("start_gradient가 10 이상인 경우에만 condition accuracy를 계산하여 JSON을 업데이트합니다...")
    
    # 누락된 condition accuracy 채우기
    fill_missing_condition_accuracy()
    
    # 요약 출력
    print_summary()

if __name__ == "__main__":
    main() 