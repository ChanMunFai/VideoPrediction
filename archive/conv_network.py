import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

h_dim = 100

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 160) # 3380 = 27040/batch size
        self.fc2 = nn.Linear(160, h_dim) # change output size to dimension of h

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 3380)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tconv1 = nn.Conv2DTranspose(1,  10, kernel_size=5)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 160)
        self.fc2 = nn.Linear(160, h_dim) # change output size to dimension of h

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class DeconvNet(nn.Module): # convert 2h into h and t
    def __init__(self):
        super(DeconvNet, self).__init__()
        nz = 32
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nz is the length of the z input vector
            # ngf is the size of feature maps for the output generated image
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 8, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 8, 2, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, 6, 2, 0, bias=False),
            nn.Tanh()
        )


    def forward(self, input):
        return self.main(input)

if __name__ == "__main__":
    z = torch.randn(8, 32, 1, 1) # batch of latent variables
    print(z.shape)
    deconv_net = DeconvNet()
    output = deconv_net(z)
    print(output.shape)



