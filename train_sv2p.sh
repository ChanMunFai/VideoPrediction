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

# num_epochs=1

# python train.py
# python main_sv2p.py --epochs $num_epochs --stage2_epochs=1

# python main_sv2p_custom.py --stage2_epochs=1 --stage=2
python main_sv2p_new.py --stage=3 --epochs=200 --learning_rate=1e-4
