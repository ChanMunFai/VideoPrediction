import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

from vrnn.conv_layers import Conv, Conv_64, Deconv

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps # numerical logs

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        self.mse_loss = nn.MSELoss(reduction = 'sum') # MSE over all pixels 

        # embedding - embed xt to xt_tilde (dim h_dim)
        if self.h_dim == 1024: 
            self.embed = Conv()
        elif self.h_dim == 64: 
            self.embed = Conv_64()
        else: 
            print("Current parameters not supported.")
    
        #encoder - encode xt_tilde and h_t-1 into ht
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        # reparameterisation 1 - get mean and variance of zt
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoding - generate xt_hat from h_t-1 and zt_tilde
        self.phi_z = nn.Sequential( # convert zt to zt_tilde (shape: h_dim)
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        self.dec = Deconv(h_dim = h_dim)

        #prior - sample zt from h_t-1
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #recurrence - inputs are itself, xt_tilde and h_t-1_tilde
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        # all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        reconstruction_loss = 0

        h = torch.zeros(self.n_layers, x.size(0), self.h_dim, device=device)

        for t in range(x.size(1)): # sequence length

            xt = x[:,t,:,:,:]
            xt_tilde = self.embed(xt)

            #encoder and reparameterisation
            enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1)) # final layer of h
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

            #decoding
            zt_tilde = self.phi_z(zt) # convert dim from z_dim to h_dim
            input_latent = torch.cat([zt_tilde, h[-1]], 1)
            xt_hat = self.dec(input_latent)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            reconstruction_loss += self.mse_loss(xt_hat, xt)
            # print("MSE Loss at time t:", self.mse_loss(xt_hat, xt))

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)

            # print("Prior mean", prior_mean_t.shape) # 50 by 32
            # print("Prior std", prior_std_t.shape)
            # print("Posterior mean", enc_mean_t)
            # print("Posterior std", enc_std_t)

        reconstruction_loss = reconstruction_loss/x.size(0) # divide by batch size
        # print("MSE loss after 1 forward pass:", reconstruction_loss)

        return kld_loss, reconstruction_loss, \
            (all_enc_mean, all_enc_std)


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, 1, self.x_dim, self.x_dim, device=device)

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            zt_tilde = self.phi_z(z_t)

            #decoder
            xt_hat = self.dec(torch.cat([zt_tilde, h[-1]], 1))
            # print(dec_mean_t.shape) # batch X 1 X H x W
            #dec_std_t = self.dec_std(dec_t)

            xt_tilde = self.embed(xt_hat) # Batch X h dim

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

            sample[t] = xt_hat.data

        return sample

    def reconstruct(self, x):
        """ Generate reconstructed frames x_t_hat. 
        """
        h = torch.zeros(self.n_layers, x.size(0), self.h_dim, device=device)

        reconstructed_frames = torch.zeros(x.size(1), 1, self.x_dim, self.x_dim, device=device)

        for t in range(x.size(1)):
            xt = x[:,t,:,:,:] # assume x has channel dimension

            xt_tilde = self.embed(xt)

            #encoder and reparameterisation
            enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1)) # final layer of h
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

            #decoding
            zt_tilde = self.phi_z(zt) # convert dim from z_dim to h_dim
            input_latent = torch.cat([zt_tilde, h[-1]], 1)
            xt_hat = self.dec(input_latent)

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

            reconstructed_frames[t] = xt_hat

        return reconstructed_frames

    def predict(self, x):
        """Predicts for video frames given that the model has no 
        ground truth access to future.

        Splits x into 2 sub-sequences. Model does not see the 2nd sub sequence, and predicts
        purely using 1st sub-sequence and predicted frames.
        """
        seq_length = x.size(1)
        seq_length_train = int(seq_length/2)
        seq_length_test = seq_length - seq_length_train

        h = torch.zeros(self.n_layers, x.size(0), self.h_dim, device=device)

        predicted_frames = torch.zeros(seq_length_test, 1, self.x_dim, self.x_dim, device=device)
        true_future_frames = torch.zeros(seq_length_train, 1, self.x_dim, self.x_dim, device=device)

        for t in range(x.size(1)):
            if t < seq_length_train: # seen data
                xt = x[:,t,:,:,:]
                xt_tilde = self.embed(xt)

                #encoder and reparameterisation
                enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

                #decoding
                zt_tilde = self.phi_z(zt)
                input_latent = torch.cat([zt_tilde, h[-1]], 1)
                xt_hat = self.dec(input_latent)

            else: # unseen data
                xt = xt_hat # use predicted xt instead of actual xt
                xt_tilde = self.embed(xt)

                #encoder and reparameterisation
                enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

                #decoding
                zt_tilde = self.phi_z(zt)
                input_latent = torch.cat([zt_tilde, h[-1]], 1)
                xt_hat = self.dec(input_latent)

                predicted_frames[t-seq_length_train] = xt_hat

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

        return predicted_frames

    def predict_new(self, input, target): 
        total_len = input.size(1) + target.size(1)
        train_len = input.size(1)
        predicted_frames = torch.full_like(target, 0) 
        h = torch.zeros(self.n_layers, input.size(0), self.h_dim, device=device)

        for t in range(total_len):
            if t < input.size(1): # seen data
                xt = input[:,t,:,:,:]
                xt_tilde = self.embed(xt)

                #encoder and reparameterisation
                enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

                #decoding
                zt_tilde = self.phi_z(zt)
                input_latent = torch.cat([zt_tilde, h[-1]], 1)
                xt_hat = self.dec(input_latent)

            else: 
                xt = xt_hat # use predicted xt instead of actual xt
                xt_tilde = self.embed(xt)

                #encoder and reparameterisation
                enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

                #decoding
                zt_tilde = self.phi_z(zt)
                input_latent = torch.cat([zt_tilde, h[-1]], 1)
                xt_hat = self.dec(input_latent)

                predicted_frames[:,t-train_len] = xt_hat

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

        return predicted_frames


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD
        
        https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        """

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)

    # Unused functions 

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))
