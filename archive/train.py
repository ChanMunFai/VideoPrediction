import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import VRNN
from data.MovingMNIST import MovingMNIST

def train(epoch):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        #transforming data
        data = data.to(device)
        data = torch.unsqueeze(data, 2)
        # print("data shape", data.shape) # Batch Size X Seq Length X Channels X Height X Width

        data = (data - data.min()) / (data.max() - data.min())

        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, _ = model(data)
        loss = kld_loss + nll_loss
        loss.backward()
        optimizer.step()

        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        #printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * batch_size, batch_size * (len(train_loader.dataset)//batch_size),
                100. * batch_idx / len(train_loader),
                kld_loss / batch_size,
                nll_loss / batch_size))

            # sample = model.sample(torch.tensor(2, device=device))
            # sample = sample.squeeze() # remove channel dimension
            # plt.imshow(sample[0].to(torch.device('cpu')).numpy())
            # plt.pause(1e-6)

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    """uses test data to evaluate
    likelihood of the model"""

    mean_kld_loss, mean_nll_loss = 0, 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):

            data = data.to(device)
            data = torch.unsqueeze(data, 2)
            data = (data - data.min()) / (data.max() - data.min())

            kld_loss, nll_loss, _ = model(data)
            mean_kld_loss += kld_loss.item()
            mean_nll_loss += nll_loss.item()

    mean_kld_loss /= len(test_loader.dataset)
    mean_nll_loss /= len(test_loader.dataset)

    print('====> Test set loss: KLD Loss = {:.4f}, NLL Loss = {:.4f} '.format(
        mean_kld_loss, mean_nll_loss))


# changing device
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

#hyperparameters
x_dim = 64
h_dim = 1024
z_dim = 32
n_layers =  3
n_epochs = 300
clip = 10
learning_rate = 1e-3
batch_size = 64 #128
seed = 128
print_every = 20 # batches
save_every = 10 # epochs

#manual seed
torch.manual_seed(seed)
# plt.ion()

#init model + optimizer + datasets

train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
test_set = MovingMNIST(root='.dataset/mnist', train=False, download=True)

train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

model = VRNN(x_dim, h_dim, z_dim, n_layers)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, n_epochs + 1):

    #training + testing
    train(epoch)
    # test(epoch) # need to change

    #saving model
    if epoch % save_every == 1:
        fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), fn)
        print('Saved model to '+fn)
