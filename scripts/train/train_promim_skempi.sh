config_path=./configs/train/promim_ddg_skempi.yml
idx_cvfolds=0
device=cuda:0

python train_promim_skempi.py \
    --config $config_path\
    --idx_cvfolds $idx_cvfolds \
    --device $device
    