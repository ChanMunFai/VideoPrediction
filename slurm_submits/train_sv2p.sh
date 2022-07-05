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

python main_sv2p.py --stage=0 --dataset $dset

# python main_sv2p.py --stage=3 --epochs=100 --learning_rate=1e-4 --beta_end=0.0001
# python main_sv2p.py --stage=3 --epochs=100 --learning_rate=1e-4 --beta_end=0.01
# python main_sv2p.py --stage=3 --epochs=100 --learning_rate=1e-4 --beta_end=0.1
# python main_sv2p.py --stage=3 --epochs=1000 --learning_rate=1e-6 --beta_end=0.001 --batch_size=52

### Train posterior networks 
# python -m sv2p.train_posterior_sv2p --epochs=50 --beta=10

### Updated training