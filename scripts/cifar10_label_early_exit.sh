CUDA_VISIBLE_DEVICES=0
data_type=image
task=label_guidance
image_size=32
dataset="cifar10"
model_name_or_path='models/cifar10_uvit_small.pth'
guide_network='resnet_cifar10.pt'
train_steps=1000
inference_steps=50
eta=1.0
target=6
clip_x0=True
seed=40
logging_dir='logs'
per_sample_batch_size=128
num_samples=30
logging_resolution=512
guidance_name='dps'
eval_batch_size=512
wandb=False
log_traj=False

rho=1
mu=0.25
sigma=0.001
eps_bsz=1
iter_steps=4
guidance_scale=2.0

# Early exit 설정
use_early_exit=True
early_exit_layer=7  # decode layer의 5번째 layer에서 early exit (총 6개 decode layer 중 5번째)

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --iter_steps $iter_steps \
    --dataset $dataset \
    --guide_network $guide_network \
    --logging_resolution $logging_resolution \
    --model_name_or_path $model_name_or_path \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --target $target \
    --eta $eta \
    --clip_x0 $clip_x0 \
    --rho $rho \
    --mu $mu \
    --sigma $sigma \
    --eps_bsz $eps_bsz \
    --guidance_scale $guidance_scale \
    --wandb $wandb \
    --seed $seed \
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --guidance_name $guidance_name \
    --eval_batch_size $eval_batch_size \
    --log_traj $log_traj \
    --use_early_exit $use_early_exit \
    --early_exit_layer $early_exit_layer 