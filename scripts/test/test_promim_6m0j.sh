config=./configs/inference/6m0j.yml
ckpt_path=./trained_models/promim_skempi_cvfold_1.pt
device=cuda:0
idx_cvfolds=1

python test_promim_6m0j.py \
    --config $config \
    --ckpt_path $ckpt_path \
    --device $device \
    --idx_cvfolds $idx_cvfolds