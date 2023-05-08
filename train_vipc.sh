
# CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train.sh 1 \
#     --config ./cfgs/PCN_models/PoinTr.yaml \
#     --exp_name example --resume

# Train with single GPU
# bash ./scripts/train.sh 2 \
#     --config ./cfgs/ShapeNetViPC_models/PoinTr.yaml \
#     --exp_name shapenetvipc --num_workers 16 --val_freq 10

set -x
bash ./scripts/train.sh 0 \
    --config ./cfgs/ShapeNetViPC_models/PoinTr_Original_Split.yaml \
    --exp_name lamp --num_workers 16 --val_freq 10