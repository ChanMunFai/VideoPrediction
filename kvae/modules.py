import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFastDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNFastDecoder, self).__init__()
        self.in_dec = nn.Linear(input_dim, 32*8*8)
        self.hidden_convs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ConvTranspose2d(in_channels=64,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1)])
        self.out_conv = nn.Conv2d(in_channels=32,
                      out_channels=output_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1)

    def forward(self, x):
        b, *_ = x.size()
        x = self.in_dec(x).reshape((b, -1, 8, 8))
        for hidden_conv in self.hidden_convs:
            x = F.relu(hidden_conv(x))
            x = F.pad(x, (0,1,0,1))

        x = torch.sigmoid(self.out_conv(x))
        return x

class CNNFastEncoder(nn.Module):
    def __init__(self, input_channels, output_dim, log_var=True):
        super(CNNFastEncoder, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=32,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.hidden_conv = nn.ModuleList([
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1)
        for _ in range(1)])

        self.out_mean = nn.Linear(32*8*8, output_dim)
        if log_var:
            self.out_log_var = nn.Linear(32*8*8, output_dim)
        else:
            self.out_log_var = None

    def forward(self, x):
        x = F.relu(self.in_conv(x))
        for hidden_layer in self.hidden_conv:
            x = F.relu(hidden_layer(x))
        x = x.flatten(-3, -1)
        if self.out_log_var is None:
            return self.out_mean(x)
        mean, log_var = self.out_mean(x), self.out_log_var(x)
        return mean, log_var

class SubPixelDecoder(nn.Module): 
    """ Decodes a_t to x_t

    Uses the sub-pixel network 

    Code adapted from https://github.com/yjn870/ESPCN-pytorch/blob/master/models.py
    """

    def __init__(self, scale_factor, num_channels=1):
        super(SubPixelDecoder, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=5//2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=3//2),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x

if __name__ == "__main__": 
    # decoder = SubPixelDecoder(scale_factor=1)
    decoder = CNNFastDecoder(input_dim=5, output_dim=64)
    sample_a = torch.zeros(6, 10, 5) 
    x_hat = decoder(sample_a.reshape(6*10, -1))
    print(x_hat.shape)

