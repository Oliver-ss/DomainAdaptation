#!/bin/bash
#SBATCH -o log_out
#SBATCH -e log_err
#SBATCH --gres=gpu:1
python3 train.py
