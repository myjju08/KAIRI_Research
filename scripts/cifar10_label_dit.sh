CUDA_VISIBLE_DEVICES=0
data_type=image
task=label_guidance
dataset="cifar10"
image_size=32
model_type=transformer
model_name_or_path='models/dit-cifar10-s4.pt'
guide_network='resnet_cifar10.pt'
vae='mse'  # sd-vae-ft-mse

train_steps=1000
inference_steps=50
eta=1.0
clip_x0=True
seed=0
target=0                # 필요에 맞게 클래스 인덱스 수정 (0~9, CIFAR-10)

logging_dir='logs'
per_sample_batch_size=32
num_samples=32
logging_resolution=32
guidance_name='dps'
eval_batch_size=256
wandb=False
log_traj=False

# DPS related
rho=0
mu=0
sigma=0
eps_bsz=0
iter_steps=0
guidance_strength=1.0
# Layer-routing guidance (set enabled to true to activate)
# DiT-S/4 has depth=12 blocks (indices 0..11); inference_steps=50 (t=0..49)
layer_routing='{"enabled": false, "mode": "hard_detach", "schedule": [ {"t_min": 0,  "t_max": 15, "blocks": [8,9,10,11],    "strength": 1.0}, {"t_min": 16, "t_max": 35, "blocks": [4,5,6,7],    "strength": 1.0}, {"t_min": 36, "t_max": 49, "blocks": [1,2,3,4],  "strength": 1.0} ]}'
ablate_layer_routing=False

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --dataset $dataset \
    --image_size $image_size \
    --model_type $model_type \
    --model_name_or_path $model_name_or_path \
    --guide_network $guide_network \
    --vae $vae \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --eta $eta \
    --clip_x0 $clip_x0 \
    --seed $seed \
    --target $target \
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --logging_resolution $logging_resolution \
    --guidance_name $guidance_name \
    --eval_batch_size $eval_batch_size \
    --log_traj $log_traj \
    --rho $rho \
    --mu $mu \
    --sigma $sigma \
    --eps_bsz $eps_bsz \
    --iter_steps $iter_steps \
    --guidance_strength $guidance_strength \
    --layer_routing "$layer_routing" \
    --ablate_layer_routing $ablate_layer_routing


