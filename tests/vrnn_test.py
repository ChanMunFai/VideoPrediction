import torch

from model_vrnn import VRNN
from data.MovingMNIST import MovingMNIST
import torch.nn as nn

EPS = torch.finfo(torch.float).eps # numerical logs

def kld_gauss(mean_1, std_1, mean_2, std_2):
    """Using std to compute KLD
    
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """

    kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
    return	0.5 * torch.sum(kld_element)

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

# KLD 
def benchmark_kld(): 
    mean0 = torch.zeros(1024)
    mean0point01 = torch.full_like(mean0, 0.001)
    ones = torch.full_like(mean0, 1)

    kld = kld_gauss(mean0, ones, mean0point01, ones)
    print(kld)

def test_recon_loss_benchmarks(): 
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
        print("Pixel MSE for 1 time step for random model", recon_loss) # 0.2470

        # print out black image at all times

        black_seq = torch.full_like(data, 0)
        black_mse = mse_loss(black_seq, data)
        print("Pixel MSE of black image for 1 time step", black_mse) # 0.0423

        # print out white image at all times

        # white_seq = torch.full_like(data, 1)
        # white_mse = mse_loss(white_seq, data) 
        # print(white_mse) # 0.9441

        # rand_seq = torch.rand(data.size(0), data.size(1), data.size(2), data.size(3), data.size(4))
        # rand_mse = mse_loss(rand_seq, data) 
        # print(rand_mse) #0.3269

        break





