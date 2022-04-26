import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from model import VRNN

#hyperparameters
x_dim = 64
h_dim = 1024
z_dim = 32
n_layers =  3

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

state_dict = torch.load('saves/vrnn_state_dict_291.pth', map_location = device)
model = VRNN(x_dim, h_dim, z_dim, n_layers)
model.load_state_dict(state_dict)
model.to(device)

sample = model.sample(100)
# sample = torch.squeeze(sample)
frame1 = sample[0]
frame10 = sample[-1]

sample = torchvision.utils.make_grid(sample, 10)
plt.imsave("sample.jpeg", sample.cpu().permute(1, 2, 0).numpy())

# plt.imshow(frame1.cpu().numpy(), cmap='gray')
# plt.savefig('frame1.jpeg')

# plt.imshow(frame10.cpu().numpy(), cmap='gray')
# plt.savefig('frame10.jpeg')
# plt.show()
