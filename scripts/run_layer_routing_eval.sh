#!/usr/bin/env bash
# Run two sampling sweeps (layer_routing on/off) over CIFAR-10 targets 0..9, 1024 imgs each,
# then optionally evaluate FID/cond-acc with scripts/eval_condacc_fid.py.

set -e

# ----- user configs -----
# GPUs to use (space-separated). You can override by: GPU_LIST="0 1 2 3" bash ...
GPU_LIST=${GPU_LIST:-"0"}
MODEL_PATH="models/dit-cifar10-s4.pt"
GUIDE_NET="resnet_cifar10.pt"
VAE="mse"
NUM_SAMPLES=256
PER_SAMPLE_BSZ=32
INFERENCE_STEPS=50
LOGDIR_BASE="logs/lr_ablation"
# Set COMPUTE_FID=true to also compute FID (requires torch_fidelity)
COMPUTE_FID=${COMPUTE_FID:-"true"}

# layer-routing schedule (DiT-S/4 depth=12, t in [0,49])
# t가 클수록(초반, noisy) early blocks, 작을수록(후반, detail) late blocks
# t 36~49 (초반) → blocks [0..3] (shallow, coarse semantic)
# t 16~35 (중반) → blocks [4..7] (mid)
# t 0~15 (후반) → blocks [8..11] (deep, detail)
LR_SCHEDULE='{"enabled": true, "mode": "hard_detach", "schedule": [ {"t_min": 0,  "t_max": 15, "blocks": [8,9,10,11],    "strength": 1.0}, {"t_min": 16, "t_max": 35, "blocks": [0,1,2,3,4,5,6,7,8,9,10,11],    "strength": 1.0}, {"t_min": 36, "t_max": 49, "blocks": [0,1,2,3],  "strength": 1.0} ]}'

run_case() {
  local label=$1    # on / off
  local enabled=$2  # true / false
  local strength=$3 # guidance_strength
  local lr_flag=""
  if [ "$enabled" = "true" ]; then
    lr_flag="$LR_SCHEDULE"
  else
    lr_flag='{"enabled": false}'
  fi

  local gpu_arr=($GPU_LIST)
  local n_gpu=${#gpu_arr[@]}

  # Launch all classes in parallel (one per GPU)
  for cls in {0..9}; do
    gpu=${gpu_arr[$((cls % n_gpu))]}
    echo "[run] label=$label target=$cls on GPU $gpu (background)"
    CUDA_VISIBLE_DEVICES=$gpu python main.py \
      --data_type image \
      --task label_guidance \
      --dataset cifar10 \
      --image_size 32 \
      --model_type transformer \
      --model_name_or_path "$MODEL_PATH" \
      --guide_network "$GUIDE_NET" \
      --vae "$VAE" \
      --train_steps 1000 \
      --inference_steps $INFERENCE_STEPS \
      --eta 1.0 \
      --clip_x0 True \
      --seed 0 \
      --target $cls \
      --logging_dir "${LOGDIR_BASE}/${label}/target=${cls}" \
      --per_sample_batch_size $PER_SAMPLE_BSZ \
      --num_samples $NUM_SAMPLES \
      --logging_resolution 32 \
      --guidance_name dps \
      --eval_batch_size 256 \
      --log_traj False \
      --rho 0 --mu 0 --sigma 0 --eps_bsz 0 --iter_steps 0 \
      --guidance_strength $strength \
      --layer_routing "$lr_flag" \
      --ablate_layer_routing False &
  done
  
  # Wait for all background jobs to complete
  echo "[wait] Waiting for all $label sampling jobs to complete..."
  wait
  echo "[done] All $label sampling jobs completed"
}

echo "[run] layer_routing=ON"
run_case "lr_on" "true" "1.0"

echo "[run] layer_routing=OFF"
run_case "lr_off" "false" "1.0"

echo "Sampling finished. Logs at ${LOGDIR_BASE}/lr_on and lr_off"
echo "[eval] Running cond-acc${COMPUTE_FID:+ and FID}..."
eval_args="--root ${LOGDIR_BASE} --classifier_ckpt ${GUIDE_NET}"
if [ "$COMPUTE_FID" = "true" ]; then
  eval_args="$eval_args --compute_fid"
fi
python scripts/eval_condacc_fid.py $eval_args

