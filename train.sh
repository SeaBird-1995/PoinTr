set -x

# CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train.sh 1 \
#     --config ./cfgs/PCN_models/PoinTr.yaml \
#     --exp_name example --resume

# Train with single GPU
bash ./scripts/train.sh 2 \
    --config ./cfgs/ShapeNet34_models/PoinTr.yaml \
    --exp_name shapenet34 --num_workers 16