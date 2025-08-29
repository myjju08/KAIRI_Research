#!/usr/bin/env python3
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg
import os
from tqdm import tqdm

def load_inception_model():
    """Inception v3 모델을 로드합니다."""
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # 마지막 분류 레이어 제거
    model.eval()
    return model

def preprocess_images(images):
    """이미지를 Inception v3 입력 형식으로 전처리합니다."""
    # [0, 255] -> [0, 1] -> [-1, 1]
    images = images.astype(np.float32) / 255.0
    images = images * 2 - 1
    
    # PyTorch 텐서로 변환
    images = torch.from_numpy(images).permute(0, 3, 1, 2)  # NHWC -> NCHW
    
    # Inception v3 입력 크기로 리사이즈 (299x299)
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
    ])
    
    processed_images = []
    for i in range(images.shape[0]):
        img = transform(images[i])
        processed_images.append(img)
    
    return torch.stack(processed_images)

def extract_features(model, images, batch_size=32):
    """이미지에서 Inception 특징을 추출합니다."""
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size), desc="특징 추출"):
            batch = images[i:i+batch_size]
            if torch.cuda.is_available():
                batch = batch.cuda()
                model = model.cuda()
            
            # Inception v3의 중간 특징 추출
            batch_features = model(batch)
            features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def calculate_fid(real_features, fake_features):
    """FID score를 계산합니다."""
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # 평균 차이의 제곱
    diff = mu_real - mu_fake
    diff_squared = diff.dot(diff)
    
    # 공분산 행렬의 제곱근
    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
    
    # 수치적 안정성을 위한 처리
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # FID 계산
    fid = diff_squared + np.trace(sigma_real + sigma_fake - 2 * covmean)
    
    return fid

def main():
    print("=== FID Score 계산 시작 ===")
    
    # Inception 모델 로드
    print("Inception v3 모델 로딩 중...")
    inception_model = load_inception_model()
    
    # 두 실험 결과 로드
    path1 = 'deepcache_experiments_target=3_20250820_023940/cache_1_clean_0/guidance_name=dps+recur_steps=1+iter_steps=4/model=openai_cifar10.pt/guide_net=resnet_cifar10.pt/target=3/bon=1/guidance_strength=1.0/images.npy'
    path2 = 'deepcache_layer_experiments_target=3_20250820_021906/cache_interval_1_layer_depth_1/guidance_name=dps+recur_steps=1+iter_steps=4/model=openai_cifar10.pt/guide_net=resnet_cifar10.pt/target=3/bon=1/guidance_strength=1.0/images.npy'
    
    print(f"실험 1 데이터 로딩: {path1}")
    data1 = np.load(path1)
    print(f"실험 2 데이터 로딩: {path2}")
    data2 = np.load(path2)
    
    print(f"데이터 형태: {data1.shape}")
    
    # 이미지 전처리
    print("이미지 전처리 중...")
    processed_data1 = preprocess_images(data1)
    processed_data2 = preprocess_images(data2)
    
    # 특징 추출
    print("실험 1 특징 추출 중...")
    features1 = extract_features(inception_model, processed_data1)
    print("실험 2 특징 추출 중...")
    features2 = extract_features(inception_model, processed_data2)
    
    # FID 계산
    print("FID score 계산 중...")
    fid_score = calculate_fid(features1, features2)
    
    print(f"\n=== 결과 ===")
    print(f"FID Score: {fid_score:.4f}")
    
    # 추가 통계
    print(f"\n=== 추가 통계 ===")
    print(f"실험 1 특징 평균: {np.mean(features1, axis=0)[:5]}...")
    print(f"실험 2 특징 평균: {np.mean(features2, axis=0)[:5]}...")
    print(f"특징 차이의 L2 norm: {np.linalg.norm(np.mean(features1, axis=0) - np.mean(features2, axis=0)):.4f}")
    
    # 결과 저장
    with open('fid_comparison_results.txt', 'w') as f:
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"실험 1: {path1}\n")
        f.write(f"실험 2: {path2}\n")
        f.write(f"데이터 형태: {data1.shape}\n")
        f.write(f"특징 차이의 L2 norm: {np.linalg.norm(np.mean(features1, axis=0) - np.mean(features2, axis=0)):.4f}\n")
    
    print(f"\n결과가 'fid_comparison_results.txt'에 저장되었습니다.")

if __name__ == "__main__":
    main() 