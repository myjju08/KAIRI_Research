#!/bin/bash

CUDA_VISIBLE_DEVICES="2"
data_type=image
image_size=256
per_sample_batch_size=4
dataset="cat"

model_name_or_path='google/ddpm-ema-cat-256'

task=super_resolution
guide_network='no'
target=no

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=False
seed=42
logging_resolution=512
wandb=True
wandb_project=sweep_$task
metrics='lpips'

# These are just placeholders to be re-written by the sweep
# wandb_project='trails'
guidance_strength=0.0
rho=0.0
mu=0.0
sigma=0.0

# These are fixed hyperparameters for large scale running
num_samples=256
logging_dir='logs'
filter_type='no'
filter_rate=1.0
eps_bsz=1

# Fixed hyperparameters during the sweep
rho_schedule='increase'
mu_schedule='increase'
sigma_schedule='decrease'

sweep_dir='sweep_audio_diffusion_guidance'
cuda_ids=$CUDA_VISIBLE_DEVICES
topk=3
max_sweep=8
init_rho=0.25
max_rho=8
init_mu=0.25
max_mu=8
init_sigma=0.01
max_sigma=10
init_guidance_strength=0.25
max_guidance_strength=16
beam_sample_size=32
num_large_scale=1

echo "CUDA_VISIBLE_DEVICES=$cuda_ids"
echo "target=$target"
echo "per_sample_batch_size=$per_sample_batch_size"
echo "beam_sample_size=$beam_sample_size"

iter_steps=4
for guidance_name in 'tfg';
do
    for recur_steps in 1;
    do
        cmd="python searching.py \
            --task $task \
            --sweep_dir $sweep_dir \
            --cuda_ids $cuda_ids \
            --guide_network $guide_network \
            --topk $topk \
            --max_sweep $max_sweep \
            --beam_sample_size $beam_sample_size \
            --init_rho $init_rho \
            --max_rho $max_rho \
            --init_mu $init_mu \
            --max_mu $max_mu \
            --init_sigma $init_sigma \
            --max_sigma $max_sigma \
            --init_guidance_strength $init_guidance_strength \
            --max_guidance_strength $max_guidance_strength \
            --data_type $data_type \
            --image_size $image_size \
            --wandb $wandb \
            --filter_type $filter_type \
            --filter_rate $filter_rate \
            --metrics $metrics \
            --iter_steps $iter_steps \
            --wandb_project $wandb_project \
            --dataset $dataset \
            --logging_resolution $logging_resolution \
            --model_name_or_path $model_name_or_path \
            --train_steps $train_steps \
            --inference_steps $inference_steps \
            --target $target \
            --eta $eta \
            --clip_x0 $clip_x0 \
            --guidance_strength $guidance_strength \
            --recur_steps $recur_steps \
            --seed $seed \
            --logging_dir $logging_dir \
            --per_sample_batch_size $per_sample_batch_size \
            --num_samples $num_samples \
            --guidance_name $guidance_name \
            --eval_batch_size $eval_batch_size \
            --rho $rho \
            --mu $mu \
            --sigma $sigma \
            --rho_schedule $rho_schedule \
            --mu_schedule $mu_schedule \
            --sigma_schedule $sigma_schedule \
            --num_large_scale $num_large_scale \
            --eps_bsz $eps_bsz \
        "
        echo $cmd
        eval "$cmd"
        wait
    done
done