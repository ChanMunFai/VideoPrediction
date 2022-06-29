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

dset=BouncingBall 
# dset=MovingMNIST 

# python main_kvae.py --scale=0.3 --epoch=100 --dataset $dset --learning_rate=1e-5
# python main_kvae.py --scale=0.2 --epoch=100 --dataset $dset --learning_rate=1e-6
python main_kvae.py --scale=0.25 --epoch=100 --dataset $dset --learning_rate=1e-6
# python main_kvae.py --scale=0.5 --epoch=100 --dataset $dset --learning_rate=1e-5
# python main_kvae.py --scale=0.1 --epochs=100 --dataset $dset

