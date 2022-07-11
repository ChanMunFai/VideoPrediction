#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/mc821/VideoPrediction/out/%j.out

export PATH=/vol/bitbucket/mc821/VideoPrediction/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/mc821/VideoPrediction

dset=BouncingBall_20

python main_kvae2.py --subdirectory=v3 --scale=0.3 --epoch=80  --batch_size=128 --learning_rate=0.007 --dataset $dset --wandb_on=True

