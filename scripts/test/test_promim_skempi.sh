ckpt=./trained_models/promim_skempi_cvfold_2.pt
device=cuda:0
idx_cvfolds=2

python test_promim_skempi.py \
    --ckpt $ckpt \
    --device $device \
    --idx_cvfolds $idx_cvfolds