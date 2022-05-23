import torch

from model_vrnn import VRNN
from data.MovingMNIST import MovingMNIST
import torch.nn as nn

# Test for min and max values of input - DONE
# Ensure that data is normalised between 0 and 1 - DONE
# Calculate MSE for network that always produces black image

train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=64,
                shuffle=True)

model = VRNN(64, 1024, 32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
mse_loss = nn.MSELoss() # by default, mean --- average over all pixels and time steps

for data, _ in train_loader:
    data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
    data = (data - data.min()) / (data.max() - data.min())
    # print(torch.max(data)) # 1
    # print(torch.min(data)) # 0

    optimizer.zero_grad()
    kld_loss, recon_loss, _ = model(data)

    # Normalise reconstruction loss by number of pixels and sequence length
    print("MSE before normalisation", recon_loss)
    recon_loss = recon_loss / (64 * 64 * 10)
    print("Pixel MSE for 1 time step for random model", recon_loss) # 6.2398e-05

    # print out black image at all times
    black_seq = torch.full_like(data, 0)
    black_mse = mse_loss(black_seq, data)

    print("Pixel MSE of black image for 1 time step", black_mse)

    # print("MSE of black image before normalisation", black_mse * 10)
    # black_mse = black_mse/(64 * 64 * 10)
    # print("MSE of black image after normalisation", black_mse) # 1.0636e-06 - low because most of MNIST is black

    # print out white image at all times
    white_seq = torch.full_like(data, 1)
    white_mse = mse_loss(white_seq, data)
    print(white_mse) # 2.3067e-05

    rand_seq = torch.rand(data.size(0), data.size(1), data.size(2), data.size(3), data.size(4))
    # print(rand_seq.shape)
    # print(torch.max(rand_seq))
    # print(torch.min(rand_seq))
    rand_mse = mse_loss(rand_seq, data)
    print(rand_mse) # 7.9787e-06

    break





