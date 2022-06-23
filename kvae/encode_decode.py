"""Code adapted from Tensorflow implementation at https://github.com/simonkamronn/kvae/blob/849d631dbf2faf2c293d56a0d7a2e8564e294a51/kvae/utils/nn.py#L161"""

import torch 
import torch.nn as nn 
import numpy as np


class Decoder(nn.Module):
    """ Convolutional variational decoder to decode latent code to image reconstruction
        
    :param a_seq: latent code
    :return: x_mu
    """

    def __init__(self, a_dim, enc_shape, device):
        super(Decoder, self).__init__()
        self.device = device 
        self.a_dim = a_dim
        self.enc_shape = enc_shape 
        self.dec_upscale = nn.Linear(self.a_dim, np.prod(enc_shape)).to(device)
        
        self.num_filters = [32,32,32]
        self.filter_size = 3

        self.mu_out = nn.Sequential(
            nn.Conv2d(in_channels = 116, out_channels = 1, kernel_size = 1), 
        )
        self.mu_out2 = nn.Linear(60 * 32, 64 * 64) 


    def forward(self, a_seq):
        B, T, a_dim = a_seq.size()
        a_seq = torch.reshape(a_seq, (-1, a_dim))
        
        dec_hidden = self.dec_upscale(a_seq)
        dec_hidden = torch.reshape(dec_hidden, (B*T, 32, 7, 7)) # change this later

        for filters in reversed(self.num_filters):
            dec_hidden_layer = nn.Conv2d(
                in_channels = dec_hidden.size(1),
                out_channels = filters * 4,
                kernel_size =  self.filter_size).to(self.device)
            dec_hidden = dec_hidden_layer(dec_hidden)
            dec_hidden = self.subpixel_reshape(dec_hidden, 2)   
            # print(dec_hidden.size())   

        x_mu = self.mu_out(dec_hidden)
        x_mu = torch.reshape(x_mu, (x_mu.size(0), -1))
        x_mu = self.mu_out2(x_mu)
        x_mu = torch.reshape(x_mu, (B, T, 64, 64))
        # print(x_mu.size()) 
        
        return x_mu 

    def subpixel_reshape(self, x, factor):
        """ Reshape function for subpixel upsampling
        x: tensor, shape = (bs,c h,w)
        factor: interger, upsample factor
        Return: tensorflow tensor, shape = (bs,h*factor,w*factor,c//factor**2)
        """

        # input and output shapes
        bs, ic, ih, iw = x.size()
        oh, ow, oc = ih * factor, iw * factor, ic // factor ** 2

        assert ic % factor == 0, "Number of input channels must be divisible by factor"

        x = torch.reshape(x, (-1, oh, ow, oc))
        # print(x.size())
        return x

class Encoder(nn.Module):
    """
    Returns
        a_mu: 
        a_log_var: 
        encoder_shape: Not returned yet 
    """

    def __init__(self, input_channels = 1, a_dim = 2):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.a_dim = 2
        self.encode = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride = 2), 
                nn.ReLU(), 
                nn.Conv2d(32, 32, 3, stride = 2), 
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride = 2), 
                nn.ReLU(),
            )

        if input_channels == 1: 
            self.mu_out = nn.Linear(1568, self.a_dim) 
            self.log_var_out = nn.Linear(1568, self.a_dim)
        else: 
            raise NotImplementedError 
        
    def forward(self, x):
        B, T, NC, H, W = x.size()
        x = torch.reshape(x, (B * T, NC, H, W))

        x = self.encode(x)
        encoder_shape = x.size() # [32, 7, 7]
        x = torch.flatten(x, start_dim = 1)

        a_mu = self.mu_out(x)
        a_log_var = 0.1 * self.log_var_out(x)

        a_mu = torch.reshape(a_mu, (B, T, self.a_dim))
        a_log_var = torch.reshape(a_log_var, (B, T, self.a_dim))

        return a_mu, a_log_var, encoder_shape
        
if __name__ == "__main__":
    encoder = Encoder(input_channels=1, a_dim = 2)
    x_seq_sample = torch.rand(32, 10, 1, 64, 64)
    a_mu, a_log_var, encoder_shape = encoder(x_seq_sample)

    print(a_mu.size())
    print(a_log_var.size())
    print(encoder_shape) # wrong - remove first dimension 

    decoder = Decoder(a_dim = 2, enc_shape = [32, 7, 7], device = x_seq_sample.device)
    a_seq_sample = torch.rand(32, 10, 2)
    decoded = decoder(a_seq_sample)
    print(decoded.size())

    


