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

from data.MovingMNIST import MovingMNIST

seed = 128
torch.manual_seed(seed)

batch_size = 16
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state_dict_path = 'saves/sv2p/stage3/final_beta=0.0001/sv2p_state_dict_99.pth'
state_dict = torch.load(state_dict_path, map_location = device)

model =  CDNA(in_channels = 1, cond_channels = 1,n_masks = 10).to(device) # stochastic
model.load_state_dict(state_dict)
model.to(device)

q_net = PosteriorInferenceNet(tbatch = 10).to(device)
sampler = LatentVariableSampler()

# Load in dataset
test_set = MovingMNIST(root='.dataset/mnist', train=False, download=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=False)


def check_data():
    """ This shows that we need more work to align train and test samples. 
    """

    data_train, _ = next(iter(train_loader)) 
    data_test, _ = next(iter(test_loader)) 

    data_train = data_train.to(device)
    data_train = torch.unsqueeze(data_train, 2)
    data_train = (data_train - data_train.min()) / (data_train.max() - data_train.min())

    data_test = data_test.to(device)
    data_test = torch.unsqueeze(data_test, 2)
    data_test = (data_test - data_test.min()) / (data_test.max() - data_test.min())

    output_dir = f"results/images/sv2p/sample_images/"
    checkdir(output_dir)

    for i in range(len(data_train)):
        sequence_train = data_train[i]
        sequence_train = torchvision.utils.make_grid(sequence_train, sequence_train.size(0))
        
        sequence_test = data_test[i]
        sequence_test = torchvision.utils.make_grid(sequence_test, sequence_test.size(0))

        combined_sequence = torchvision.utils.make_grid(
                                [sequence_train, sequence_test],
                                1
                                )

        plt.imsave(
            output_dir + f"sample{i}.jpeg",
            combined_sequence.cpu().permute(1, 2, 0).numpy()
            )
    
    print(data_train.shape)
    print(data_test.shape)

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


if __name__ == "__main__":
    # check_data()
    print_reconstructions()