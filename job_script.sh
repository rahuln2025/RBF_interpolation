#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=1443

srun python3 dash_app.py