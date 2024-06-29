config=./configs/inference/7fae.yml
ckpt_path=./trained_models/promim_skempi_cvfold_0.pt
device=cuda:0
idx_cvfolds=0

python test_promim_7fae.py \
    --config $config \
    --ckpt_path $ckpt_path \
    --device $device \
    --idx_cvfolds $idx_cvfolds