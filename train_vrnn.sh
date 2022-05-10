#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/mc821/VideoPrediction/out/%j.out

export PATH=/vol/bitbucket/mc821/videopred_venv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/mc821/VideoPrediction

vsion=v1
num_epochs=150

# python train.py
python main_vrnn.py --epochs $num_epochs --beta=5 --version $vsion
python main_vrnn.py --epochs $num_epochs --beta=10 --version $vsion
python main_vrnn.py --epochs $num_epochs --beta=15 --version $vsion
python main_vrnn.py --epochs $num_epochs --beta=20 --version $vsion
python main_vrnn.py --epochs $num_epochs --beta=25 --version $vsion
