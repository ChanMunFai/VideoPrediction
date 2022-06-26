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

# python main_kvae.py --scale=0.3 --epoch=200
python main_kvae.py --scale=0.1 --epochs=150
python main_kvae.py --scale=0.05 --epochs=150
# python main_kvae.py --scale=0.05 --epochs=200
# python main_kvae.py --scale=0.5 --epochs=200
# python main_kvae.py --scale=0.8 --epochs=200
# python main_kvae.py --scale=1 --epochs=200

# python main_kvae.py --scale=50 --epochs=200
# python main_kvae.py --scale=100 --epochs=200
# python main_kvae.py --scale=1000 --epochs=200