#!/bin/bash
python evaluate.py \
    --checkpoint checkpoints/swin3dnet_500epoch.pth\
    --data_dir data/processed_oasis/test\
    --batch_size 4\
    --model swinunet