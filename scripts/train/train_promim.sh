nproc_per_node=4
world_size=4
master_port=20888
config_path=./configs/train/promim.yml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port=$master_port train_promim.py \
    --config $config_path \
    --world_size $world_size
