""" Adapted from Tensorflow implementation from https://github.com/simonkamronn/kvae/blob/master/kvae/KalmanVariationalAutoencoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module): 
    """ Decodes a_t to x_t

    Uses the sub-pixel network 

    Code adapted from https://github.com/yjn870/ESPCN-pytorch/blob/master/models.py
    """

    def __init__(self, scale_factor, num_channels=1):
        super(ESPCN, self).__init__()
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


class KalmanVariationalAutoencoder(object):
    def __init__(self):
        pass

        # Define initialisiers for LGSSM variables
        # A, B, C, Q, R 
        # A is initialised from identity matrices 
        # B and C randomly from Gaussians 

        # p(z-1)
        # Initial variable a_0

    def encoder(self, in_channels, x_t):
        """ Encodes image frame x_t into low-dimension space a_t

        Parameters:
            x_t: 
                frame at time t of dim (Batch_Size X Channels X Height X Width)

        Returns: 
            a_t: 
                latent variable at time t of dim (Batch_Size X 2)
        """
        encode = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride = 1, padding = 'same'), # need to change this to different stride
            nn.ReLU(), 
            nn.Conv2d(32, 32, 3, stride = 1, padding = 'same'), 
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride = 1, padding = 'same'), 
            nn.ReLU(),
            nn.Flatten()
        )
        a_t = encode(x_t)
        linear = nn.Linear(a_t.size(1), 2)
        a_t = linear(a_t)
        return a_t

    def alpha(self, input, state = None, K = 3): 
        """ Dynamics Parameter Network for mixing transitions in state space model

        Only implements RNN for now

        Input is encoded states a. 

        RNN takes in the input sequentially, and produces an output. 
        Output at each time step is then used (along with the next a_t) to predict the next output. 

        Returns: 
            alpha: mixing vector of dimension (Batch Size X Seq Length X K)

        """
        self.rnn = nn.LSTM(2, K, batch_first = True)
        output, state = self.rnn(input, state) 
        print(output.shape)

        # ouput is of dim (L, N, K) where L is seq length and N is batch size and K is the number of modes 
        # Alternative implementation will be to have a separate dim for hidden state dim and embed that to K 

        alpha = F.softmax(output, dim = 2)
        return alpha, state  

    def foward(self, x):
        pass
        # Encoder q(a|x)

        # alpha RNN 

        # Initialise Kalman filter 

        # Get smoothed posterior over z 

        # Get filtered posterior over z, over for imputation 

        # Decoder 
        
        # Losses 
        # VAE loss 
        # LGSSM []

def test_encoder():
    kvae = KalmanVariationalAutoencoder()
    x_t = torch.zeros(52, 1, 64, 64)
    a_t = kvae.encoder(1, x_t)
    print(a_t.shape) # 2 dim 

def test_alpha(): 
    kvae = KalmanVariationalAutoencoder()
    a = torch.randn(20, 10, 2) # batch size X seq len X 2
    alpha, state = kvae.alpha(a)
    print(alpha.shape)

# test_encoder()
test_alpha()