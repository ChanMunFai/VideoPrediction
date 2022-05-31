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
num_epochs=3
warmup_steps=3000

# python train.py
# python main_vrnn.py --epochs $num_epochs --beta=1 --version $vsion
# python main_vrnn.py --epochs $num_epochs --beta=4.0 --version $vsion
# python main_vrnn.py --epochs $num_epochs --beta=2.0 --version $vsion
# python main_vrnn.py --epochs $num_epochs --beta=5.0 --version $vsion
python main_vrnn.py --epochs $num_epochs --beta=0 --version $vsion

# python main_vrnn.py --epochs $num_epochs --beta=1 --version $vsion --step_size=100
# python main_vrnn.py --epochs $num_epochs --beta=1 --version $vsion --step_size=250
# python main_vrnn.py --epochs $num_epochs --beta=1 --version $vsion --step_size=500
# python main_vrnn.py --epochs $num_epochs --beta=1 --version $vsion --step_size=1000 --learning_rate=0.1 --warmup $warmup_steps
# python main_vrnn.py --epochs $num_epochs --beta=1 --version $vsion --step_size=1500 --learning_rate=0.1 --warmup $warmup_steps
# python main_vrnn.py --epochs $num_epochs --beta=1 --version $vsion --step_size=2000 --learning_rate=0.1 --warmup $warmup_steps

# Finetune beta > 0 (v1)
# python main_vrnn_custom.py --epochs=200 --beta=0.0001 --version=v1 --step_size=1000000 --learning_rate=0.001
# python main_vrnn_custom.py --epochs=200 --beta=0.001 --version=v1 --step_size=1000000 --learning_rate=0.001
# python main_vrnn_custom.py --epochs=200 --beta=0.01 --version=v1 --step_size=1000000 --learning_rate=0.001
# python main_vrnn_custom.py --epochs=150 --beta=0.4 --version=v1 --step_size=1000000 --learning_rate=0.0001
# python main_vrnn_custom.py --epochs=150 --beta=0.1 --version=v1 --step_size=1000000 --learning_rate=0.0001

