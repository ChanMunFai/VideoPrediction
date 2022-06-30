import os
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_msssim import ssim
import numpy as np
from utils import checkdir

from sv2p.cdna import CDNA 
from sv2p.model_sv2p import PosteriorInferenceNet, LatentVariableSampler
from vrnn.model_vrnn import VRNN

from data.MovingMNIST import MovingMNIST

seed = 128
torch.manual_seed(seed)

batch_size = 32
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = "MovingMNIST"

state_dict_cdna_path = f"saves/{dataset}/sv2p/stage3/final_beta=0.001/finetuned2/sv2p_cdna_state_dict_199.pth"
state_dict_posterior_path = f"saves/{dataset}/sv2p/stage3/final_beta=0.001/finetuned2/sv2p_posterior_state_dict_199.pth"

state_dict_cdna = torch.load(state_dict_cdna_path, map_location = device)
state_dict_posterior = torch.load(state_dict_posterior_path, map_location = device)

model =  CDNA(in_channels = 1, cond_channels = 1,n_masks = 10).to(device) 
model.load_state_dict(state_dict_cdna)

q_net = PosteriorInferenceNet(tbatch = 10).to(device)
q_net.load_state_dict(state_dict_posterior) 

sampler = LatentVariableSampler()

# Load in dataset
test_set = MovingMNIST(root='dataset/mnist', train=False, download=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

train_set = MovingMNIST(root='dataset/mnist', train=True, download=True)
train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=False)
    
def print_reconstructions():
    # Just use training set for now 
    
    output_dir = f"results/images/sv2p/reconstructions/train/"
    checkdir(output_dir)

    data, _ = next(iter(train_loader))
    data = data.to(device)
    data = torch.unsqueeze(data, 2)
    data = (data - data.min()) / (data.max() - data.min())

    inputs, targets = split_data(data) # define later
    inputs = inputs.to(device)
    targets = targets.to(device)

    predictions = []

    # Sample latent variable z from posterior - same z for all time steps
    with torch.no_grad():    
        mu, sigma = q_net(data) 
    z = sampler.sample(mu, sigma).to(device) 

    # Sample latent variables from prior 
    # z = sampler.sample_prior((batch_size, 1, 8, 8)).to(device)

    # Use latent variables that are very different 
    # z = torch.full((batch_size, 1, 8, 8), 1000).to(device)

    hidden = None
    with torch.no_grad():
        for t in range(inputs.size(1)):
            x_t = inputs[:, t, :, :, :]
            targets_t = targets[:, t, :, :, :] # x_t+1

            predictions_t, hidden, _, _ = model(inputs = x_t, conditions = z,
                                                hidden_states=hidden)

            predictions.append(predictions_t)

    predictions = torch.stack(predictions)
    predictions = predictions.permute(1,0,2,3,4) 

    i = 0
    for item in predictions: # iterate over batch items 
        item = torchvision.utils.make_grid(item, item.size(0))
        target = targets[i]
        target = torchvision.utils.make_grid(target, target.size(0))

        stitched_image = torchvision.utils.make_grid([target, item],1)
        
        plt.imsave(
            output_dir + f"reconstructions{i+1}.jpeg",
            stitched_image.cpu().permute(1, 2, 0).numpy()
            )
        i+= 1
    
def print_predictions(num_samples = 1, true_posterior = False):
    # Just use training set for now 
    
    output_dir = f"results/{dataset}/sv2p/predictions/"
    checkdir(output_dir)

    data, targets = next(iter(train_loader))
    data = data.to(device)
    data = (data - data.min()) / (data.max() - data.min())
    targets = targets.to(device)
    targets = (targets - targets.min()) / (targets.max() - targets.min())
    
    total_len = data.size(1) + targets.size(1)

    for n in range(num_samples): 
        z = sampler.sample_prior((batch_size, 1, 8, 8)).to(device)     # Sample latent variables from prior 

        if true_posterior == True: 
            mu, sigma = q_net(data)
            z = sampler.sample(mu, sigma).to(device)

            output_dir = f"results/{dataset}/sv2p/predictions/true_posterior/"
        
        hidden = None
        predicted_frames = torch.zeros(batch_size, targets.size(1), 1, 64, 64, device=device)

        with torch.no_grad():
            for t in range(total_len):
                if t < data.size(1): # seen data
                    x_t = data[:, t, :, :, :]

                    predictions_t, hidden, _, _ = model(inputs = x_t, conditions = z,
                                                    hidden_states=hidden)

                else: 
                    x_t = predictions_t # use predicted x_t instead of actual x_t

                    predictions_t, hidden, _, _ = model(inputs = x_t, conditions = z,
                                                    hidden_states=hidden)

                    predicted_frames[:, t-data.size(1)] = predictions_t

        i = 0
        for item in predicted_frames: # iterate over batch items 
            item = torchvision.utils.make_grid(item, item.size(0))

            target = targets[i]
            target = torchvision.utils.make_grid(target, target.size(0))

            seen_frames = data[i]
            seen_frames = torchvision.utils.make_grid(seen_frames, seen_frames.size(0))

            stitched_image = torchvision.utils.make_grid([seen_frames, target, item],1)
            new_output_dir = output_dir
            checkdir(new_output_dir)

            plt.imsave(
                new_output_dir + f"predictions{i+1}_{n}.jpeg",
                stitched_image.cpu().permute(1, 2, 0).numpy()
                )
            i+= 1

        print(z[0][0][0][0])
        
def split_data(data):
    """ Splits sequence of video frames into inputs and targets

    Both have shapes (Batch_Size X Seq_Len–1 X
                        Num_Channels X Height X Width)

    data: Batch Size X Seq Length X Channels X Height X Width

    Inputs: x_0 to x_T-1
    Targets: x_1 to x_T
    """

    inputs = data[:, :-1, :, :, :]
    targets = data[:, 1:, :, :, :]

    assert inputs.shape == targets.shape
    return inputs, targets

def calc_sv2p_losses_over_time(): 
    """Calculate MSE for each predicted frame
    over time"""

    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    mse_loss = []
    mse_loss_black = []
    mse_loss_last_seen = []

    data, targets = next(iter(train_loader))
    data = data.to(device)
    data = (data - data.min()) / (data.max() - data.min())
    targets = targets.to(device)
    targets = (targets - targets.min()) / (targets.max() - targets.min())
    
    total_len = data.size(1) + targets.size(1)
    z = sampler.sample_prior((batch_size, 1, 8, 8)).to(device)     # Sample latent variables from prior 

    hidden = None
    black_img = torch.full_like(data[:, 0, :, :, :], 0) # batch of black image

    with torch.no_grad():
        for t in range(total_len): 
            if t < data.size(1): # seen data
                x_t = data[:, t, :, :, :] # t = 0, 1, 2, 3, 4
                predictions_t, hidden, _, _ = model(inputs = x_t, conditions = z,
                                                hidden_states=hidden) # t = 1, 2, 3, 4, 5
        
                last_seen_image = x_t # t = 0, 1, 2, 3, 4

            else: 
                ground_truth_t = targets[:, t - data.size(1), :, :, :] # t = 6, 7, 8, 9
                x_t = predictions_t # use predicted x_t instead of actual x_t
                predictions_t, hidden, _, _ = model(inputs = x_t, conditions = z,
                                                hidden_states=hidden) 

                print("Size of Predictions", predictions_t.size()) # BS X 1 X 64 X 64
                print("Size of input", x_t.size()) # BS X 1 X 64 X 64

                mse_loss.append(mse(predictions_t, ground_truth_t).item()) # mse averaged across all batch sizes and pixels 
                mse_loss_black.append(mse(black_img, ground_truth_t).item()) 
                mse_loss_last_seen.append(mse(last_seen_image, ground_truth_t).item())
    
    print(mse_loss, mse_loss_black, mse_loss_last_seen)

    ### Plot VRNN losses here 
    return mse_loss, mse_loss_black, mse_loss_last_seen

def plot_losses_over_time():
    mse_loss, mse_loss_black, mse_loss_last_seen = calc_sv2p_losses_over_time()
    vrnn_losses = [0.02385283, 0.05241073, 0.05279471, 0.05820563, 0.06583029, 0.06497522,
                0.07184452, 0.06949529, 0.07262978, 0.07187897]
    plt.plot(mse_loss, label="SV2P")
    plt.plot(mse_loss_black, label="Black Frame")
    plt.plot(mse_loss_last_seen, label="Previous Frame")
    plt.plot(vrnn_losses, label = "VRNN")
    
    plt.title("MSE between ground truth and predicted frame over time")
    plt.ylabel('MSE')
    plt.xlabel('Time')
    plt.xticks(np.arange(0, 10, 1.0))
    plt.legend(loc="upper left")

    output_dir = "plots/SV2P/"
    plt.savefig(output_dir + f"SV2P_loss_over_time.jpeg")
    plt.close('all')


if __name__ == "__main__":
    # check_data()
    # print_reconstructions()
    # print_predictions(num_samples=1, true_posterior=True)
    print_predictions(num_samples=1, true_posterior=False)
    # calc_sv2p_losses_over_time()
    plot_losses_over_time()