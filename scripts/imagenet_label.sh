CUDA_VISIBLE_DEVICES=0
data_type=image
image_size=256
dataset="imagenet"
model_name_or_path='transformer_model.pt'

# DiT-XL/2 specific settings
task=label_guidance
guide_network='google/vit-base-patch16-224'
target=0

# VAE settings (same as DiT)
vae_type='mse'  # 'mse' or 'ema'

# DiT training settings (1000 steps, linear schedule)
train_steps=1000
inference_steps=100  # Reduced for faster inference
eta=1.0
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=1  # Reduced for DiT-XL/2 memory usage
num_samples=2
logging_resolution=256  # Match image_size
guidance_name='tfg'
eval_batch_size=1  # Reduced for DiT-XL/2 memory usage
wandb=False
log_traj=False

# Training-Free-Guidance parameters
#rho=2
#mu=0.5
#guidance 없이
rho=0
mu=0
sigma=0.1
eps_bsz=1
iter_steps=4

# DiT specific parameters
learn_sigma=True  # DiT-XL/2 uses learn_sigma=True
in_channels=4     # DiT uses 4 channels in latent space

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --dataset $dataset \
    --guide_network $guide_network \
    --logging_resolution $logging_resolution \
    --model_name_or_path $model_name_or_path \
    --vae $vae_type \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --target $target \
    --iter_steps $iter_steps \
    --eta $eta \
    --clip_x0 $clip_x0 \
    --rho $rho \
    --mu $mu \
    --sigma $sigma \
    --eps_bsz $eps_bsz \
    --wandb $wandb \
    --seed $seed \
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --guidance_name $guidance_name \
    --eval_batch_size $eval_batch_size \
    --log_traj $log_traj


