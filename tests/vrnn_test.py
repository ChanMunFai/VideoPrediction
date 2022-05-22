import torch

from model import VRNN
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
mse_loss = nn.MSELoss()

for data, _ in train_loader:
    data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
    data = (data - data.min()) / (data.max() - data.min())
    # print(torch.max(data)) # 1
    # print(torch.min(data)) # 0

    optimizer.zero_grad()
    kld_loss, nll_loss, _ = model(data)

    # Normalise reconstruction loss by number of pixels and sequence length
    print("MSE before normalisation", nll_loss)
    nll_loss = nll_loss / (64 * 64 * 10)
    print("MSE after normalisation", nll_loss) # 6.2398e-05

    # print out black image at all times
    black_seq = torch.full_like(data, 0)
    black_mse = mse_loss(black_seq, data)

    print("MSE of black image before normalisation", black_mse * 10)
    black_mse = black_mse/(64 * 64 * 10)
    print("MSE of black image after normalisation", black_mse) # 1.0636e-06 - low because most of MNIST is black

    # print out white image at all times
    white_seq = torch.full_like(data, 1)
    white_mse = mse_loss(white_seq, data)
    white_mse = white_mse/(64 * 64 * 10)
    print(white_mse) # 2.3067e-05

    rand_seq = torch.rand(data.size(0), data.size(1), data.size(2), data.size(3), data.size(4))
    # print(rand_seq.shape)
    # print(torch.max(rand_seq))
    # print(torch.min(rand_seq))
    rand_mse = mse_loss(rand_seq, data)/(64 * 64 * 10)
    print(rand_mse) # 7.9787e-06

    break





