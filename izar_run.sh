#!/bin/bash
#SBATCH --time 5:59:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu_free
#SBATCH --gres=gpu:1  # Request 1 GPU

source ~/venvs/course_py-3.10/bin/activate

python main.py

deactivate