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

def plot_predictions(x, target, pred_len, plot_len = None):
    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/KVAE/attempt6/predictions/"
        output_dir_gt = f"results/{args.dataset}/KVAE/attempt6/ground_truth/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)
        if not os.path.exists(output_dir_gt):
            os.makedirs(output_dir_gt)

        if plot_len == None: 
            plot_len = pred_len

        i = i[:plot_len,:,:,:] 
        predicted_frames = torchvision.utils.make_grid(
                                        i,
                                        i.size(0)
                                        )

        ground_truth = target[batch_item,:plot_len,:,:,:]
        ground_truth_frames = torchvision.utils.make_grid(
                                        ground_truth,
                                        ground_truth.size(0)
                                        )

        stitched_frames = torchvision.utils.make_grid(
                                        [ground_truth_frames, predicted_frames],
                                        1
                                        )

        plt.imsave(
                output_dir_pred + f"predictions_{batch_item}.jpeg",
                predicted_frames.cpu().permute(1, 2, 0).numpy()
                )

        plt.imsave(
                output_dir_gt + f"ground_truth_{batch_item}.jpeg",
                ground_truth_frames.cpu().permute(1, 2, 0).numpy()
                )


def plot_reconstructions(x, plot_len, reconstruct_kalman = True):
    if reconstruct_kalman == True: 
        x_reconstructed = kvae.reconstruct_kalman(x)
    else: 
        x_reconstructed = kvae.reconstruct(x)
    
    for batch_item, i  in enumerate(x_reconstructed):
        if reconstruct_kalman == False: 
            output_dir = f"results/{args.dataset}/KVAE/attempt6/reconstructions/"
        else: 
            output_dir = f"results/{args.dataset}/KVAE/attempt6/reconstructions_kf/"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        i = i[:plot_len,:,:,:] 

        reconstructed_frames = torchvision.utils.make_grid(
                                        i,
                                        i.size(0)
                                        )

        ground_truth = x[batch_item,:plot_len,:,:,:]
        ground_truth_frames = torchvision.utils.make_grid(
                                        ground_truth,
                                        ground_truth.size(0)
                                        )

        stitched_frames = torchvision.utils.make_grid(
                                        [ground_truth_frames, reconstructed_frames],
                                        1
                                        )

        plt.imsave(
                output_dir + f"reconstructions_{batch_item}.jpeg",
                stitched_frames.cpu().permute(1, 2, 0).numpy()
                )

def plot_loss_over_time():
    pass


if __name__ == "__main__": 
    state_dict_path = "saves/BouncingBall/kvae/v4/scale=0.3/kvae_state_dict_scale=0.3_40.pth" 

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = "BouncingBall", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall]")
    parser.add_argument('--x_dim', default=1, type=int)
    parser.add_argument('--a_dim', default=2, type=int)
    parser.add_argument('--z_dim', default=4, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--scale', default=0.3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--alpha', default="rnn", type=str, 
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

    # with torch.no_grad():
    #     kvae(data)
    #     for name, param in kvae.named_parameters():
    #         print(name) 
 
    # print(kvae.A.size()) # K X z_dim X z_dim 
    # print(kvae.A) # different 
    # print(kvae.C) # a bit similar but still different 

    # Model does not learn to use different weights to get different dynamics 

    plot_predictions(data, target, pred_len = 20)  
    # plot_reconstructions(data, 20, reconstruct_kalman = False)

    # pred_seq, *_ = kvae.predict(data, pred_len = 50)
    # print(pred_seq[0,0])

    # print(data[0,0,0])
    # reconstructed = kvae.reconstruct(data)
    # print(reconstructed[0,0,0])
    



    
        


        