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

from vrnn.model_vrnn import VRNN
from data.MovingMNIST import MovingMNIST

model_version = "v1"
dataset = "MovingMNIST"
state_dict_path = f'saves/{dataset}/vrnn/{model_version}/important/vrnn_state_dict_v1_beta=0.4_step=1000000_149.pth'

det_or_stoch = "stochastic"
stage = "stage_c"

# Deterministic model 
# /vol/bitbucket/mc821/VideoPrediction/saves/v1/important/vrnn_state_dict_v1_beta=0.0_step=1000000_299.pth

# Stochastic model 
# /vol/bitbucket/mc821/VideoPrediction/saves/v1/important/vrnn_state_dict_v1_beta=0.1_step=1000000_99.pth
# /vol/bitbucket/mc821/VideoPrediction/saves/v1/important/vrnn_state_dict_v1_beta=0.1_step=1000000_249.pth
# /vol/bitbucket/mc821/VideoPrediction/saves/v1/important/vrnn_state_dict_v1_beta=0.4_step=1000000_149.pth

if model_version == "v0":
    x_dim = 64
    h_dim = 1024
    z_dim = 32
    n_layers =  3
elif model_version == "v1":
    x_dim = 64
    h_dim = 1024
    z_dim = 32
    n_layers =  1
    
batch_size = 32

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state_dict = torch.load(state_dict_path, map_location = device)
model = VRNN(x_dim, h_dim, z_dim, n_layers)
model.load_state_dict(state_dict)
model.to(device)

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

def calc_test_loss():
    """uses test data to evaluate losses"""

    mean_kld_loss, mean_nll_loss = 0, 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data = torch.unsqueeze(data, 2)
            data = (data - data.min()) / (data.max() - data.min())

            kld_loss, nll_loss, _ = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()

    mean_kld_loss /= len(test_loader)
    mean_nll_loss /= len(test_loader)

    print('====> Test set loss: KLD Loss = {:.8f}, NLL Loss = {:.8f} '.format(
        mean_kld_loss, mean_nll_loss))

def plot_images(
    train = True, 
    print_current_frames = False,
    print_reconstructions = True,
    prediction = False,
    batch_item = 0):
    """ Plot images for a single sequence. 

    print_current_frames: print all current frames
    print_all_predictions: print all predicted frames of current frames
    prediction mode: separate data into seen and unseen data.
    batch_item: example number"""

    if train == True: 
        data, targets = next(iter(train_loader))
    else: 
        data, targets = next(iter(test_loader))

    data = data.to(device)
    data = (data - data.min()) / (data.max() - data.min())

    targets = targets.to(device)
    targets = (targets - targets.min()) / (targets.max() - targets.min())

    if train == True: 
        if det_or_stoch == "determistic": 
            output_dir = f"results/{dataset}/{model_version}/{det_or_stoch}/train/"
        elif det_or_stoch == "stochastic": 
            output_dir = f"results/{dataset}/VRNN/{model_version}/" # we just care about this
    else: 
        if det_or_stoch == "determistic": 
            output_dir = f"results/{dataset}/{model_version}/{det_or_stoch}/test/"
        elif det_or_stoch == "stochastic": 
            output_dir = f"results/{dataset}/{model_version}/{det_or_stoch}/{stage}/test/"

    checkdir(output_dir)
    
    # Current Frames
    if print_current_frames == True:
        current_frames = data[batch_item] # Seq Length X Num_channels X Width X Height
        current_frames = torchvision.utils.make_grid(current_frames, current_frames.size(0))
        plt.imsave(
            output_dir + f"current_frames{batch_item}.jpeg",
            current_frames.cpu().permute(1, 2, 0).numpy()
            )

    # Generate reconstructed frames for all of current frames
    if print_reconstructions == True:
        with torch.no_grad():
            current_frames = data[batch_item].unsqueeze(0)
            reconstructed_frames = model.reconstruct(current_frames)

            # Plot MSE 
            mse = nn.MSELoss(reduction = 'sum') # MSE over all pixels and whole sequence 
            avg_mse = 0
            for i, j in zip(current_frames, reconstructed_frames.unsqueeze(0)): 
                avg_mse += mse(i, j)
            
            print("Avg image MSE for each time step: ", avg_mse)
            print("Avg pixel-wise MSE for each time step : ", avg_mse/(64*64))

            reconstructed_frames = torchvision.utils.make_grid(
                                        reconstructed_frames,
                                        reconstructed_frames.size(0)
                                        )
            
            reconstructed_frames = reconstructed_frames.cpu().permute(1, 2, 0).numpy()

            # Plot actual frames at top, reconstructed frames at bottom 
            current_frames = current_frames.view(10, 1, 64, 64)
            current_frames = torchvision.utils.make_grid(current_frames, current_frames.size(0))
            current_frames = current_frames.cpu().permute(1, 2, 0).numpy()

            combined = np.concatenate((current_frames, reconstructed_frames), axis = 0)
            plt.imshow(combined)
            plt.axis('off')
            
            plt.text(120, 350, f"Pixel-wise MSE {avg_mse/(64*64)}", fontsize = 10)

            plt.savefig(output_dir + f"reconstructions_{batch_item}.jpeg")
            plt.close('all')

            
    # Predicted frames - unseen future
    if prediction == True:
        with torch.no_grad():
            seen_frames = data[batch_item].unsqueeze(0)
            ground_truth_frames = targets[batch_item].unsqueeze(0)

            predicted_frames = model.predict_new(seen_frames, ground_truth_frames)
            
            seen_frames = seen_frames.squeeze(0)
            ground_truth_frames = ground_truth_frames.squeeze(0)
            predicted_frames = predicted_frames.squeeze(0)
            
            seen_frames_grid = torchvision.utils.make_grid(
                                        seen_frames,
                                        seen_frames.size(0)
                                        )

            ground_truth_frames_grid = torchvision.utils.make_grid(
                                    ground_truth_frames,
                                    ground_truth_frames.size(0)
                                    )

            predicted_frames_grid = torchvision.utils.make_grid(
                                    predicted_frames,
                                    predicted_frames.size(0)
                                    )

            stitched_image = torchvision.utils.make_grid(
                                        [seen_frames_grid, ground_truth_frames_grid, predicted_frames_grid],
                                        1
                                        )

            plt.imsave(
                output_dir + f"predictions_{batch_item}.jpeg",
                stitched_image.cpu().permute(1, 2, 0).numpy()
                )


def calc_losses_over_time(train = True, batch_item = 0): 
    """Calculate MSE for each predicted frame
    over time"""

    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    
    if train == True: 
        data, targets = next(iter(train_loader))
    else: 
        data, targets = next(iter(test_loader))

    data = data.to(device)
    data = (data - data.min()) / (data.max() - data.min())

    targets = targets.to(device)
    targets = (targets - targets.min()) / (targets.max() - targets.min())

    mse_loss = []

    with torch.no_grad():
        seen_frames = data[batch_item].unsqueeze(0)
        ground_truth_frames = targets[batch_item].unsqueeze(0)
        predicted_frames = model.predict_new(seen_frames, ground_truth_frames)

        seen_frames = seen_frames.squeeze(0)
        ground_truth_frames = ground_truth_frames.squeeze(0)
        predicted_frames = predicted_frames.squeeze(0)

        for i, j in zip(ground_truth_frames, predicted_frames): 
            mse_loss.append(mse(i, j).item())

    return mse_loss

def plot_losses_over_time(train = True):
    if train == True: 
        if det_or_stoch == "determistic": 
            output_dir = f"results/{dataset}/{model_version}/{det_or_stoch}/train/"
        elif det_or_stoch == "stochastic": 
            output_dir = f"results/{dataset}/VRNN/{model_version}/"
    else: 
        if det_or_stoch == "determistic": 
            output_dir = f"results/{dataset}/{model_version}/{det_or_stoch}/test/"
        elif det_or_stoch == "stochastic": 
            output_dir = f"results/{dataset}/{model_version}/{det_or_stoch}/{stage}/test/"

    combined_mse_over_time = []

    for i in range(batch_size):
        combined_mse_over_time.append(calc_losses_over_time(train = train, batch_item = i)) 

    combined_mse_over_time = np.array(combined_mse_over_time)
    print(combined_mse_over_time.shape)

    avg_mse_over_time = np.mean(combined_mse_over_time, axis = 0)

    plt.plot(avg_mse_over_time)
    print("VRNN MSE", avg_mse_over_time)

    # for j in combined_mse_over_time: 
    #     plt.plot(j)

    plt.title("MSE between ground truth and predicted frame over time")
    plt.xticks(np.arange(0, 10, 1.0))
    plt.ylabel('MSE')
    plt.xlabel('Time')

    plt.savefig(output_dir + f"loss_over_time.jpeg")
    plt.close('all')


def calc_SSIM(true_seq_dir, predicted_seq_dir):
    true_seq = Image.open(true_seq_dir)
    predicted_seq = Image.open(predicted_seq_dir)
    convert_tensor = transforms.ToTensor()

    true_seq = convert_tensor(true_seq).unsqueeze(0)
    predicted_seq = convert_tensor(predicted_seq).unsqueeze(0)

    ssim_val = ssim(true_seq, predicted_seq, data_range=1, size_average=False) # return (N,)
    print(ssim_val.item())

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Make dir: %s'%directory)

if __name__ == "__main__":
    # test()

    # for i in range(batch_size):
    #     plot_images(batch_item = i, 
    #                 print_reconstructions = True, 
    #                 prediction = True)

    # calc_SSIM(
    #     true_seq_dir = "results/images/0/current_frames0.jpeg",
    #     predicted_seq_dir = "results/images/0/current_frames_predicted0.jpeg")

    # calc_losses_over_time(train = True, batch_item = 0)

    plot_losses_over_time(train = True)

    pass


