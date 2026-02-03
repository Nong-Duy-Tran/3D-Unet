#!/bin/bash
python train.py \
    --train_dir data/processed_oasis/train/\
    --val_dir data/processed_oasis/val/\
    --batch_size 4\
    --epochs 500\
    --model swinunet