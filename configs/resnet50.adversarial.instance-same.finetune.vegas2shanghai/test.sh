#!/bin/bash
#SBATCH -o test_out
#SBATCH -e test_err
#SBATCH --gres=gpu:1
make test MODEL=train_log/models/best.pth BN=False SAVE=0
make test MODEL=train_log/models/best.pth BN=True SAVE=0
