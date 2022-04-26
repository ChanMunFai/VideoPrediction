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

from model import VRNN
from data.MovingMNIST import MovingMNIST


# Load in model
x_dim = 64
h_dim = 1024
z_dim = 32
n_layers =  3
batch_size = 16

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state_dict = torch.load('saves/vrnn_state_dict_291.pth', map_location = device)
model = VRNN(x_dim, h_dim, z_dim, n_layers)
model.load_state_dict(state_dict)
model.to(device)

# Load in dataset
test_set = MovingMNIST(root='.dataset/mnist', train=False, download=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

def test(plot = True):
    """uses test data to evaluate
    likelihood of the model"""

    mean_kld_loss, mean_nll_loss = 0, 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data = torch.unsqueeze(data, 2)
            data = (data - data.min()) / (data.max() - data.min())

            break

            kld_loss, nll_loss, _ = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)

    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))


def plot_images(
    print_current_frames = False,
    print_all_predictions = False,
    prediction_mode = False,
    new_prediction_mode = True,
    batch_item = 0):
    """ Plot images for a single sequence given 3 modes.

    print_current_frames: print all current frames
    print_all_predictions: print all predicted frames of current frames
    prediction mode: separate data into seen and unseen data.
    batch_item: example number"""

    data, _ = next(iter(test_loader))
    data = data.to(device)
    data = torch.unsqueeze(data, 2)
    data = (data - data.min()) / (data.max() - data.min())

    output_dir = f"results/images/{batch_item}/"
    checkdir(output_dir)

    # Current Frames
    if print_current_frames == True:
        current_frames = data[batch_item] # Seq Length X Num_channels X Width X Height
        current_frames = torchvision.utils.make_grid(current_frames, current_frames.size(0))
        plt.imsave(
            output_dir + f"current_frames{batch_item}.jpeg",
            current_frames.cpu().permute(1, 2, 0).numpy()
            )

    # Generate predicted frames for all of current frames
    if print_all_predictions == True:
        with torch.no_grad():
            current_frames = data[batch_item].unsqueeze(0)
            current_predicted_frames = model.generate_frames(current_frames)
            current_predicted_frames = torchvision.utils.make_grid(
                                        current_predicted_frames,
                                        current_predicted_frames.size(0)
                                        )
            # print(current_predicted_frames.shape)
            plt.imsave(
                output_dir + f"current_frames_predicted{batch_item}.jpeg",
                current_predicted_frames.cpu().permute(1, 2, 0).numpy()
                )

    # Predicted frames - unseen future
    if prediction_mode == True:
        with torch.no_grad():
            current_frames = data[batch_item].unsqueeze(0)
            seen_frames = current_frames[0][0:5]
            true_future_frames = current_frames[0][5:]

            # print(seen_frames.shape)
            seen_frames = torchvision.utils.make_grid(
                                        seen_frames,
                                        seen_frames.size(0)
                                        )
            # print(seen_frames.shape)

            future_predicted_frames = model.predict(current_frames)
            future_predicted_frames = torchvision.utils.make_grid(
                                        future_predicted_frames,
                                        future_predicted_frames.size(0)
                                        )
            # print(future_predicted_frames.shape)

            true_future_frames = torchvision.utils.make_grid(
                                        true_future_frames,
                                        true_future_frames.size(0)
                                        )
            # print(true_future_frames.shape)

            stitched_image = torchvision.utils.make_grid(
                                        [seen_frames, true_future_frames,future_predicted_frames],
                                        1
                                        )
            # print(stitched_image.shape)

            plt.imsave(
                output_dir + f"stitched_image{batch_item}.jpeg",
                stitched_image.cpu().permute(1, 2, 0).numpy()
                )
    # Predicted frames - unseen future
    if new_prediction_mode == True:
        with torch.no_grad():
            current_frames = data[batch_item].unsqueeze(0)
            seen_frames = current_frames[0][0:5]
            true_future_frames = current_frames[0][5:]

            # print(seen_frames.shape)
            seen_frames = torchvision.utils.make_grid(
                                        seen_frames,
                                        seen_frames.size(0)
                                        )
            # print(seen_frames.shape)

            future_predicted_frames = model.predict_new(current_frames)
            future_predicted_frames = torchvision.utils.make_grid(
                                        future_predicted_frames,
                                        future_predicted_frames.size(0)
                                        )
            # print(future_predicted_frames.shape)

            true_future_frames = torchvision.utils.make_grid(
                                        true_future_frames,
                                        true_future_frames.size(0)
                                        )
            # print(true_future_frames.shape)

            stitched_image = torchvision.utils.make_grid(
                                        [seen_frames, true_future_frames,future_predicted_frames],
                                        1
                                        )
            # print(stitched_image.shape)

            plt.imsave(
                output_dir + f"new_stitched_image{batch_item}.jpeg",
                stitched_image.cpu().permute(1, 2, 0).numpy()
                )


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

    for i in range(batch_size):
        plot_images(batch_item = i)

    # calc_SSIM(
    #     true_seq_dir = "results/images/0/current_frames0.jpeg",
    #     predicted_seq_dir = "results/images/0/current_frames_predicted0.jpeg")

    pass


