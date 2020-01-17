#!/bin/bash
#SBATCH -o train_out
#SBATCH -e train_err
#SBATCH --gres=gpu:1
make test MODEL=train_log/models/best.pth BN=False SAVE=0
make test MODEL=train_log/models/best.pth BN=True SAVE=0
