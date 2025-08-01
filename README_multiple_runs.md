# 여러 번 실행하여 FID Score 계산하기

이 문서는 Training-Free-Guidance에서 여러 번 실행하여 생성된 이미지들을 합쳐서 FID score를 계산하는 방법을 설명합니다.

## 문제 상황

기존 `cifar10_label.sh` 스크립트는 이미 `logs` 폴더에 이미지가 있으면 실행을 중단합니다. 이를 해결하기 위해 여러 번 실행할 수 있는 스크립트를 만들었습니다.

## 해결 방법

### 1. 여러 번 실행 스크립트

`scripts/cifar10_multiple_runs_simple.sh` 스크립트를 사용하여 여러 번 실행할 수 있습니다:

```bash
cd Training-Free-Guidance
bash scripts/cifar10_multiple_runs_simple.sh
```

이 스크립트는:
- 5번 실행합니다 (기본값)
- 각 실행마다 다른 시드를 사용합니다 (41, 42, 43, 44, 45)
- 각 실행마다 다른 로그 디렉토리를 사용합니다 (`logs/run_0`, `logs/run_1`, ...)
- `--check_done False` 옵션을 사용하여 항상 실행되도록 합니다

### 2. 정확한 FID 계산

모든 실행이 완료된 후, 기존 Training-Free-Guidance의 평가 시스템을 사용해서 정확한 FID score를 계산할 수 있습니다:

```bash
python calculate_fid_proper.py
```

이 스크립트는:
- 기존 시스템의 `calculate_fid` 함수를 사용
- 정확한 CIFAR-10 참조 데이터셋 사용
- InceptionV3 모델로 특징 추출
- 검증된 FID 계산 방법 사용

### 3. 수동으로 실행하기

만약 스크립트를 사용하지 않고 수동으로 실행하고 싶다면:

```bash
# 첫 번째 실행
python main.py \
    --data_type image \
    --task label_guidance \
    --dataset cifar10 \
    --logging_dir logs/run_0 \
    --seed 41 \
    --check_done False \
    --num_samples 30 \
    # ... 기타 옵션들

# 두 번째 실행
python main.py \
    --data_type image \
    --task label_guidance \
    --dataset cifar10 \
    --logging_dir logs/run_1 \
    --seed 42 \
    --check_done False \
    --num_samples 30 \
    # ... 기타 옵션들

# ... 더 많은 실행들
```

## 생성되는 파일들

### 실행 후 생성되는 디렉토리 구조:
```
logs/
├── run_0/
│   └── guidance_name=dps+recur_steps=1+iter_steps=4/model=models_cifar10_uvit_small.pth/guide_net=resnet_cifar10.pt/target=1/bon=1/guidance_strength=1.0/
│       ├── images.npy
│       ├── images.png
│       └── finished_sampling
├── run_1/
│   └── guidance_name=dps+recur_steps=1+iter_steps=4/model=models_cifar10_uvit_small.pth/guide_net=resnet_cifar10.pt/target=1/bon=1/guidance_strength=1.0/
│       ├── images.npy
│       ├── images.png
│       └── finished_sampling
├── ...
└── combined/
    ├── images.npy (모든 이미지 합친 파일)
    └── fid_results_proper.json (FID 결과)
```

### 파일 설명:
- `images.npy`: 생성된 이미지들의 numpy 배열
- `images.png`: 이미지 그리드 (시각화용)
- `finished_sampling`: 실행 완료 표시 파일
- `fid_results_proper.json`: FID score 결과

## 옵션 설명

### cifar10_multiple_runs_simple.sh 옵션:
- `num_runs`: 실행할 횟수 (기본값: 5)
- 각 실행마다 시드가 1씩 증가합니다

### calculate_fid_proper.py 옵션:
- 자동으로 `logs/combined/images.npy`에서 이미지를 로드
- 기존 Training-Free-Guidance의 정확한 FID 계산 시스템 사용
- CIFAR-10 테스트셋과 비교하여 FID 계산

## 예시 사용법

1. 3번 실행하여 각각 50개씩 이미지 생성:
```bash
# 스크립트 수정 (num_runs=3, num_samples=50)
bash scripts/cifar10_multiple_runs_simple.sh
```

2. 생성된 이미지들을 합쳐서 FID 계산:
```bash
python calculate_fid_proper.py
```

## 주의사항

1. 각 실행마다 충분한 GPU 메모리가 필요합니다
2. 실행 횟수와 샘플 수에 따라 시간이 오래 걸릴 수 있습니다
3. `logs` 디렉토리에 충분한 저장 공간이 필요합니다
4. 기존 `logs` 폴더의 내용이 덮어쓸 수 있으니 주의하세요

## 문제 해결

### 이미지 로드 실패
- 각 `logs/run_X` 디렉토리에 `images.npy` 파일이 있는지 확인
- 파일 권한 문제가 있는지 확인

### FID 계산 실패
- 필요한 의존성 패키지들이 설치되어 있는지 확인
- GPU 메모리가 충분한지 확인

### 메모리 부족
- `num_samples`를 줄여서 실행
- `eval_batch_size`를 줄여서 실행

## FID Score 해석

- **좋은 FID Score**: 1.0 이하
- **보통 FID Score**: 1.0 ~ 5.0
- **높은 FID Score**: 5.0 이상 (이미지 품질이 낮음)

기존 Training-Free-Guidance 시스템과 동일한 방법으로 계산되므로 신뢰할 수 있는 결과입니다. 