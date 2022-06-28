"""Code adapted from Tensorflow implementation at https://github.com/simonkamronn/kvae/blob/849d631dbf2faf2c293d56a0d7a2e8564e294a51/kvae/utils/nn.py#L161"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class KvaeEncoder(nn.Module):
    """ Encodes a B X T X 1 X 32 X 32 or B X T X 1 X 64 X 64 video sequence. 

    KvaeEncoder because architecture is modified from KVAE paper. 

    Arguments: 
        input_channels: 1 # not yet implemented for 3 channels 
        input_size: 32 or 64 
        a_dim: int 

    Returns
        a_mu: BS X T X 2 
        a_log_var: BS X T X 2 
        encoder_shape: (tuple)
    """

    def __init__(self, input_channels = 1, input_size = 32, a_dim = 2):
        super(KvaeEncoder, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        if self.input_size != 32 and self.input_size != 64: 
            raise NotImplementedError

        self.a_dim = 2
        self.encode = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, stride = 2), 
                nn.ReLU(), 
                nn.Conv2d(32, 32, 3, stride = 2), 
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride = 2), 
                nn.ReLU(),
            )

        if self.input_channels == 1: 
            if self.input_size == 64: 
                self.mu_out = nn.Linear(1568, self.a_dim) 
                self.log_var_out = nn.Linear(1568, self.a_dim)
            elif self.input_size == 32: 
                self.mu_out = nn.Linear(288, self.a_dim) 
                self.log_var_out = nn.Linear(288, self.a_dim)
        else: 
            raise NotImplementedError 
        
    def forward(self, x):
        B, T, NC, H, W = x.size()
        x = torch.reshape(x, (B * T, NC, H, W))

        x = self.encode(x)
        encoder_shape = list(x.size()) # [32, 7, 7]
        x = torch.flatten(x, start_dim = 1)

        a_mu = self.mu_out(x)
        a_log_var = 0.1 * self.log_var_out(x)

        a_mu = torch.reshape(a_mu, (B, T, self.a_dim))
        a_log_var = torch.reshape(a_log_var, (B, T, self.a_dim))

        return a_mu, a_log_var, encoder_shape[1:]


class Decoder64(nn.Module):
    """ Decodes a latent code of a_dim to a 1 X 64 X 64 image. 
        
    :param a_seq: latent code
    :return: x_mu
    """

    def __init__(self, a_dim, enc_shape, device):
        super(Decoder64, self).__init__()
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

class Encoder64_simple(nn.Module):
    """ Encodes a 1 X 64 X 64 image 

    Modified from encoder network I used in VRNN. 

    IGNORE this encoder - it kinda sucks. 
    """
    def __init__(self, input_channels = 1, a_dim = 2):
        super(Encoder64_simple, self).__init__()
        self.a_dim = a_dim 
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, bias = False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Flatten()
        )
        if input_channels == 1: 
            self.mu_out = nn.Linear(1024, self.a_dim) 
            self.log_var_out = nn.Linear(1024, self.a_dim)
        else: 
            raise NotImplementedError 

    def forward(self, x):
        B, T, NC, H, W = x.size()
        x = torch.reshape(x, (B * T, NC, H, W))

        x = self.encode(x)
        
        a_mu = self.mu_out(x)
        a_log_var = 0.1 * self.log_var_out(x)
        a_mu = torch.reshape(a_mu, (B, T, self.a_dim))
        a_log_var = torch.reshape(a_log_var, (B, T, self.a_dim))

        return a_mu, a_log_var

class UnFlatten(nn.Module):
    def forward(self, input):
        output = input.view(input.size(0), input.size(1), 1, 1)
        return output

class DecoderSimple(nn.Module):
    """ Decodes latent sequence to video sequence. 

    Code modified from https://github.com/charlio23/bouncing-ball/blob/main/models/modules.py

    Arguments: 
        input_dim: dimension of latent variable 
        output_channels: typically 1 or 3 
        output_size: 32 or 64
    """
    def __init__(self, input_dim, output_channels, output_size):
        super(DecoderSimple, self).__init__()
        self.output_size = output_size 
        
        if self.output_size == 64: 
            self.latent_size = 16 
        elif self.output_size == 32: 
            self.latent_size = 8 
        else: 
            raise NotImplementedError

        self.in_dec = nn.Linear(input_dim, 32*self.latent_size*self.latent_size)
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
                        out_channels=output_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1)

    def forward(self, a_seq):
        B, T, *_ = a_seq.size()
        a_seq = torch.reshape(a_seq, (B * T, -1))
        a_seq = self.in_dec(a_seq).reshape((B * T, -1, self.latent_size, self.latent_size))
        for hidden_conv in self.hidden_convs:
            a_seq = F.relu(hidden_conv(a_seq))
            a_seq = F.pad(a_seq, (0,1,0,1))

        x_mu = torch.sigmoid(self.out_conv(a_seq))
        x_mu = torch.reshape(x_mu, (B, T, self.output_size, self.output_size))
        return x_mu

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    encoder = KvaeEncoder(input_channels=1, input_size = 32, a_dim =2 )
    x_seq_sample = torch.rand(32, 10, 1, 32, 32)
    print("Number of parameters in encoder", count_parameters(encoder))
    a_mu, a_log_var, encoder_shape = encoder(x_seq_sample)
    # print(a_mu.size())
    # print(a_log_var.size())
    # print(encoder_shape) 

    encoder = KvaeEncoder(input_channels=1, input_size = 64, a_dim = 2)
    print("Number of parameters in encoder", count_parameters(encoder))
    x_seq_sample = torch.rand(32, 10, 1, 64, 64)
    a_mu, a_log_var, encoder_shape = encoder(x_seq_sample)
    # print(a_mu.size())
    # print(a_log_var.size())
    # print(encoder_shape) 

    decoder = Decoder64(a_dim = 2, enc_shape = encoder_shape, device = "cpu")
    print("Number of parameters in decoder", count_parameters(decoder))
    # a_seq_sample = torch.rand(32, 10, 2)
    # decoded = decoder(a_seq_sample)
    # print(decoded.size())

    # encoder = Encoder64_simple(input_channels=1, a_dim = 2)
    # x_seq_sample = torch.rand(32, 10, 1, 64, 64)
    # a_mu, a_log_var = encoder(x_seq_sample)
    # print(a_mu.size())
    # print(a_log_var.size())
    # print("Number of parameters in encoder", count_parameters(encoder))

    decoder = DecoderSimple(input_dim = 2, output_channels = 1, output_size = 64)
    a_seq_sample = torch.rand(32, 10, 2)
    decoded = decoder(a_seq_sample)
    print("Size of xhat", decoded.size())
    print("Number of parameters in Simple Decoder (64)", count_parameters(decoder))

    decoder = DecoderSimple(input_dim = 2, output_channels = 1, output_size = 32)
    a_seq_sample = torch.rand(32, 10, 2)
    decoded = decoder(a_seq_sample)
    print("Size of xhat", decoded.size())
    print("Number of parameters in Simple Decoder (32)", count_parameters(decoder))
    


    


