"""Architecture modified from https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Conv(nn.Module):
    """
    Convolutional layers to embed x_t (shape: Number of channels X Width X Height) to
    x_t_tilde (shape: h_dim)

    h_dim = 1024
    """
    def __init__(self, image_channels = 1):
        super(Conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, bias = False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Flatten()
            # shape: batch_size X 1024 (input size of 64 X 64)
        )

    def forward(self, input):
        return self.main(input)

class Conv_64(nn.Module):
    """
    Convolutional layers to embed x_t (shape: Number of channels X Width X Height) to
    x_t_tilde (shape: h_dim)

    h_dim = 64
    """
    def __init__(self, image_channels = 1):
        super(Conv_64, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, bias = False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Flatten()
            # shape: batch_size X 64 (input size of 64 X 64)
        )

    def forward(self, input):
        return self.main(input)


class UnFlatten(nn.Module):
    def forward(self, input):
        output = input.view(input.size(0), input.size(1), 1, 1)
        return output

class Deconv(nn.Module):
    """
    Deconvolutional (tranposed convolutional) layers to embed h_t-1 and z_t_tilde (combined dimension: 2 * h_dim)
    into x_t_hat (shape: Number of channels X Width X Height)
    """
    def __init__(self, image_channels = 1, h_dim = 1024):
        super(Deconv, self).__init__()
        self.main = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim * 2, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
            # shape: batch_size X number_channels X width X height
        )

    def forward(self, input):
        return self.main(input)

def test_conv():
    img = torch.zeros(10, 1, 64, 64) # batch size X number of channels X height X width
    print(img.shape)
    conv_encoder = Conv()
    output = conv_encoder(img)
    print(output.shape)

def test_conv_v2():
    img = torch.zeros(10, 1, 64, 64) # batch size X number of channels X height X width
    print(img.shape)
    conv_encoder = Conv_64()
    output = conv_encoder(img)
    print(output.shape)

def test_deconv():
    h_t = torch.zeros(10, 1024) # batch size X h_dim
    z_t_tilde = torch.zeros(10, 1024)
    input = torch.cat([h_t, z_t_tilde], 1)
    print(input.shape)
    decoder = Deconv()
    output = decoder(input)
    print(output.shape)

if __name__ == "__main__":
    test_conv_v2()

