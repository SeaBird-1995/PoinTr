

bash ./scripts/test.sh 2 \
    --ckpts ./experiments/PoinTr/ShapeNetViPC_models/shapenetvipc/ckpt-best.pth \
    --config ./cfgs/ShapeNetViPC_models/PoinTr.yaml \
    --exp_name shapenetvipc_watercraft --num_workers 16 --val_freq 10