import os 
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import torchvision

from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple 
from kvae.elbo_loss import ELBO
from kvae.model_kvae import KalmanVAE
from data.MovingMNIST import MovingMNIST
from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader


if __name__ == "__main__": 
    state_dict_path = "saves/BouncingBall/kvae/v2/finetuned2/scale=0.3/kvae_state_dict_scale=0.3_79.pth" 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = "BouncingBall", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall]")
    parser.add_argument('--x_dim', default=1, type=int)
    parser.add_argument('--a_dim', default=2, type=int)
    parser.add_argument('--z_dim', default=4, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--scale', default=0.3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--alpha', default="mlp", type=str, 
                    help = "choose between [mlp, rnn]")

    args = parser.parse_args()

    kvae = KalmanVAE(args = args).to(args.device)
    state_dict = torch.load(state_dict_path, map_location = args.device)
    kvae.load_state_dict(state_dict)

    if args.dataset == "MovingMNIST": 
        train_set = MovingMNIST(root='dataset/mnist', train=True, download=True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=args.batch_size,
                    shuffle=False)

    elif args.dataset == "BouncingBall": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/v2/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)
    
    data, target = next(iter(train_loader))
    data = data.to(args.device)
    data = (data - data.min()) / (data.max() - data.min())
    data = torch.where(data > 0.5, 1.0, 0.0)

    target = target.to(args.device)
    target = (target - target.min()) / (target.max() - target.min())
    target = torch.where(target > 0.5, 1.0, 0.0)

    B, T, C, H, W = data.size()

    with torch.no_grad():
        kvae(data)
    #     for name, param in kvae.named_parameters():
    #         print(name) 
 
    # print(kvae.A.size()) # K X z_dim X z_dim 
    # print(kvae.A) # different 
    # print(kvae.C) # a bit similar but still different 

    # Model does not learn to use different weights to get different dynamics 

    # pred_seq, *_ = kvae.predict(data, pred_len = 50)
    # print(pred_seq[0,0])

    # print(data[0,0,0])
    # reconstructed = kvae.reconstruct(data)
    # print(reconstructed[0,0,0])

    ### Test parameter net 
    with torch.no_grad(): 
        a_sample, _, _ = kvae._encode(data) 
        # a1 = kvae.a1.reshape(1,1,-1).expand(args.batch_size,-1,-1)
        # joint_obs = torch.cat([a1,a_sample[:,:-1,:]],dim=1)
        # print("a1", a1)
        joint_obs = a_sample
        # print("===> Joint Observations", joint_obs) # Missing last value of a1 

        # Print a_next and check how far they are away from a_sample
        

        dyn_emb, state_dyn_net = kvae.parameter_net(joint_obs)
        # dyn_emb = (dyn_emb - dyn_emb.min()) / (dyn_emb.max() - dyn_emb.min())
        dyn_emb = dyn_emb.reshape(B*T,50) 
        print("===> Dynamic Embeddings", dyn_emb[:,0])
        print(dyn_emb.size()) # BS * T X 50 

        # random_dynamics = torch.rand(B*T, 50) 

        # random_linear = nn.Linear(50, 3)
        # random_embeddings = random_linear(random_dynamics)
        # print("===> Random Dynamic Embeddings", random_embeddings)

        dyn_emb = kvae.alpha_out(dyn_emb.reshape(B*T,50))
        print("===> Dynamic Embeddings", dyn_emb)

        weights = dyn_emb.softmax(-1)

        # A, C, weights = kvae._interpolate_matrices(a_sample)

    print("===> a_sample", a_sample) # this gives different a_t 
    print(weights)
    



    
        


        