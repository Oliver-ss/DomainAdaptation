#!/bin/bash
#SBATCH -o log_out
#SBATCH -e log_err
#SBATCH --gres=gpu:1
python3 train.py --resume='/usr/xtmp/satellite/train_models/resnet50.baseline.vegas.aug/models/best.pth'
