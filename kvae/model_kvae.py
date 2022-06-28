"""Modified from: https://github.com/charlio23/bouncing-ball/blob/main/models/KalmanVAE.py """

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli

from kvae.encode_decode import KvaeEncoder, Decoder64, DecoderSimple 
from kvae.elbo_loss import ELBO

class KalmanVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(KalmanVAE, self).__init__()
        self.args = kwargs['args']
        self.x_dim = self.args.x_dim
        self.a_dim = self.args.a_dim
        self.z_dim = self.args.z_dim
        self.K = self.args.K
        self.scale = self.args.scale
        self.device = self.args.device 

        if self.args.dataset == "MovingMNIST": 
            self.encoder = KvaeEncoder(input_channels=1, input_size = 64, a_dim = 2).to(self.device)
            self.decoder = DecoderSimple(input_dim = 2, output_channels = 1, output_size = 64).to(self.device)
            # self.decoder = Decoder64(a_dim = 2, enc_shape = [32, 7, 7], device = self.device).to(self.device) # change this to encoder shape
        elif self.args.dataset == "BouncingBall": 
            self.encoder = KvaeEncoder(input_channels=1, input_size = 32, a_dim = 2).to(self.device)
            self.decoder = DecoderSimple(input_dim = 2, output_channels = 1, output_size = 32).to(self.device)

        self.parameter_net = nn.LSTM(self.a_dim, 50, 1, batch_first=True).to(self.device)
        self.alpha_out = nn.Linear(50, self.K).to(self.device) 

        # Initialise a_1 (optional)
        self.a1 = nn.Parameter(torch.zeros(self.a_dim).to(self.device))
        self.state_dyn_net = None

        # Initialise p(z_1) 
        self.mu_0 = (torch.zeros(self.z_dim)).to(torch.float64) 
        self.sigma_0 = (20*torch.eye(self.z_dim)).to(torch.torch.float64) 

        # A initialised with identity matrices. B initialised from Gaussian 
        self.A = nn.Parameter(torch.eye(self.z_dim).unsqueeze(0).repeat(self.K,1,1).to(self.device))
        self.C = nn.Parameter(torch.randn(self.K, self.a_dim, self.z_dim).to(self.device)*0.05)

        # Covariance matrices - fixed. Noise values obtained from paper. 
        self.Q = 0.08*torch.eye(self.z_dim).to(torch.torch.float64).to(self.device) 
        self.R = 0.03*torch.eye(self.a_dim).to(torch.torch.float64).to(self.device) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _encode(self, x):
        """ Encodes observations x into a. 

        Arguments: 
            x: input data of shape [BS X T X NC X H X W]
        
        Returns: 
            a_sample: shape [BS X T X a_dim]
            a_mu: shape [BS X T X a_dim]
            a_log_var: shape [BS X T X a_dim]
        """
        (a_mu, a_log_var, _) = self.encoder(x)
        eps = torch.normal(mean=torch.zeros_like(a_mu)).to(x.device)
        a_std = (a_log_var*0.5).exp()
        a_sample = a_mu + a_std*eps
        a_sample = a_sample.to(x.device)
        
        return a_sample, a_mu, a_log_var

    def _interpolate_matrices(self, obs, learn_a1 = True):
        """ Generate weights to choose and interpolate between K operating modes. 

        Dynamic parameters network generates weights that sums to 1 for each K operating mode. 
        The weights are of dimension [BS * T X K]. For each item in a time step and in a batch, 
        we have K different weights. 

        Weights are multiplied with A and C matrices to produce At and Ct respectively.
        
        A and C matrices are different state transition and emission matrices for K different modes. 
        In contrast, At and Ct are the interpolated (weighted) versions of them. 
        
        Parameters:
            obs: vector a of dimension [BS X T X a_dim]
            learn_a1: bool. If learn_a1 is True, we replace a_1 with a learned embedding. 
                Otherwise, we generate a_1 from encoding x_1. 

        Returns: 
            A_t: interpolated state transition matrix of dimension [BS X T X z_dim X z_dim]
            C_t: interpolated state transition matrix of dimension [BS X T X a_dim X z_dim]

        """
        (B, T, _) = obs.size()
        
        if learn_a1: 
            a1 = self.a1.reshape(1,1,-1).expand(B,-1,-1)
            joint_obs = torch.cat([a1,obs[:,:-1,:]],dim=1)
            # print("Initialised a1 shape", a1.shape) # BS X 1 X a_dim
            # print("joint_obs shape", joint_obs.shape) # BS X T X a_dim

        else: 
            joint_obs = obs
            # print("joint_obs shape", joint_obs.shape) # BS X T X a_dim
        
        dyn_emb, self.state_dyn_net = self.parameter_net(joint_obs)
        # print("dyn_emb shape", dyn_emb.size()) # BS X T X 50
        dyn_emb = self.alpha_out(dyn_emb.reshape(B*T,50))
        inter_weight = dyn_emb.softmax(-1)
        
        # print("Weights shape", inter_weight.shape) # B*T X K 
        # print("A shape", self.A.shape) # K X z_dim X z_dim  
        # print("C shape", self.C.shape) # K X a_dim X z_dim  
        
        A_t = torch.matmul(inter_weight, self.A.reshape(self.K,-1)).reshape(B,T,self.z_dim,self.z_dim).to(torch.torch.float64)
        C_t = torch.matmul(inter_weight, self.C.reshape(self.K,-1)).reshape(B,T,self.a_dim,self.z_dim).to(torch.torch.float64)
        
        return A_t, C_t

    def filter_posterior(self, obs, A, C): 
        """ Generates filtered posterior p(z_t|a_0:t) using Kalman filtering. 

        Parameters: 
            obs: vector a of dim [B X T X a_dim] 
            A: (interpolated) state transition matrix of dim [B X T X z_dim X z_dim]*
            C: (interpolated) emission matrix of dim [B X T X a_dim X z_dim]*

        Returns: 
            mu_filt: mean of p(z_t|a_0:t). Dim [T X B X z_dim]
            sigma_filt: sigma of p(z_t|a_0:t). Dim [T X B X z_dim X z_dim]
            mu_pred: vector containing Amu_t. Dim [T X B X z_dim]
            mu_pred: vector containing P_t-1. Dim [T X B X z_dim X z_dim]

        * We have individual matrices for each time step.  
        """

        (B, T, *_) = obs.size()
        obs = obs.reshape(T, B, -1) # place T in first dimension for easier calculations 
        obs = obs.unsqueeze(-1)

        mu_filt = torch.zeros(T, B, self.z_dim, 1).to(obs.device).to(torch.float64)
        sigma_filt = torch.zeros(T, B, self.z_dim, self.z_dim).to(obs.device).to(torch.float64)

        mu_t = self.mu_0.expand(B,-1).unsqueeze(-1).to(torch.float64).to(self.device) 
        sigma_t = self.sigma_0.expand(B,-1,-1).to(torch.float64).to(self.device)
        mu_predicted = mu_t
        sigma_predicted = sigma_t 

        mu_pred = torch.zeros_like(mu_filt).to(self.device) # A u_t
        sigma_pred = torch.zeros_like(sigma_filt).to(self.device) # P_t

        A = A.to(torch.float64).to(self.device) 
        C = C.to(torch.float64).to(self.device)

        for t in range(T): 
            mu_pred[t] = mu_predicted
            sigma_pred[t] = sigma_predicted

            ### Define Kalman Gain Matrix 
            kalman_gain = torch.matmul(sigma_predicted, torch.transpose(C[:,t,:, :], 1, 2)) 
            s = torch.matmul(torch.matmul(C[:,t,:, :], sigma_predicted), torch.transpose(C[:,t,:, :],1, 2)) + self.R.unsqueeze(0) 
            kalman_gain = torch.matmul(kalman_gain, torch.inverse(s))
        
            ### Update mean 
            error = obs[t] - torch.matmul(C[:,t,:, :], torch.matmul(A[:,t,:, :], mu_t)) # extra A compared to old code
            mu_t = torch.matmul(A[:,t,:, :], mu_t) + torch.matmul(kalman_gain, error)

            ### Update Variance 
            I_ = torch.eye(self.z_dim).to(self.device)
            sigma_t = torch.matmul((I_ - torch.matmul(kalman_gain, C[:,t,:, :])), sigma_predicted)

            mu_filt[t] = mu_t 
            sigma_filt[t] = sigma_t 

            if t != T-1 : # no need to update predicted for last time step 
                sigma_predicted = torch.matmul(torch.matmul(A[:,t+1,:, :], sigma_t), torch.transpose(A[:,t+1,:, :], 1, 2)) + self.Q 
                mu_predicted = torch.matmul(A[:,t+1,:, :], mu_t)

        return (mu_filt.to(torch.float64), sigma_filt.to(torch.float64)), (mu_pred.to(torch.float64), sigma_pred.to(torch.float64))

        
    def smooth_posterior(self, A, filtered, predicted): 
        """ Generates smoothed posterior p(z_t|a_T). 

        Requires outputs of `filter_posterior()` as prerequisite.  

        Parameters: 
            A: (interpolated) state transition matrix of dim [B X T X z_dim X z_dim]*
            filtered: tuple output from `filter_posterior()`. Contains 
                        mu_filt, sigma_filt, mu_pred, sigma_pred
                        
        Returns: 
            mu_smoothed: mean of p(z_t|a_T). Dim [B X T X z_dim]
            sigma_smoothed: sigma of p(z_t|a_T). Dim [B X T X z_dim X z_dim]
            
        * We have individual matrices for each time step.  
        """
        mu_filt, sigma_filt = filtered
        mu_pred, sigma_pred = predicted

        mu_filt = mu_filt.to(torch.float64)
        sigma_filt= sigma_filt.to(torch.float64)
        mu_pred = mu_pred.to(torch.float64)
        sigma_pred = sigma_pred.to(torch.float64)

        mu_z_smooth = torch.zeros_like(mu_filt).to(torch.float64).to(self.device)
        sigma_z_smooth = torch.zeros_like(sigma_filt).to(torch.float64).to(self.device)
        mu_z_smooth[-1] = mu_filt[-1]
        sigma_z_smooth[-1] = sigma_filt[-1]

        (T, *_) = mu_filt.size()
        A = A.to(torch.float64).to(self.device)

        for t in reversed(range(T-1)):
            J = torch.matmul(sigma_filt[t], torch.matmul(torch.transpose(A[:,t+1,:,:], 1,2), torch.inverse(sigma_pred[t+1])))

            mu_diff = mu_z_smooth[t+1] - mu_pred[t+1] 
            mu_z_smooth[t] = mu_filt[t] + torch.matmul(J, mu_diff)

            cov_diff = sigma_z_smooth[t+1] - sigma_pred[t+1]
            sigma_z_smooth[t] = sigma_filt[t] + torch.matmul(torch.matmul(J, cov_diff), torch.transpose(J, 1, 2))
        
        mu_z_smooth = torch.transpose(mu_z_smooth, 1, 0).to(self.device)
        sigma_z_smooth = torch.transpose(sigma_z_smooth, 1, 0).to(self.device)

        return mu_z_smooth, sigma_z_smooth

    def _kalman_posterior(self, obs, mask=None, filter_only=False):
        
        A, C = self._interpolate_matrices(obs)
        filtered, pred = self.filter_posterior(obs, A, C)
        smoothed = self.smooth_posterior(A, filtered, pred)
        
        return smoothed, A, C

    def _decode(self, a):
        """
        Arguments:
            a: Dim [B X T X a_dim]
        
        Returns: 
            x_mu: [B X T X 64 X 64]
        """
        B, T, *_ = a.size()
        # a = a.reshape(B*T, -1)

        x_mu = self.decoder(a)
        # print("Decoded x:", x_mu.size())

        return x_mu 
        
    def _sample(self, size):
        eps = torch.normal(mean=torch.zeros(size))
        return self._decode(eps)

    def forward(self, x):
        (B,T,C,H,W) = x.size()
        # q(a_t|x_t)
        a_sample, a_mu, a_log_var = self._encode(x) 
        
        # q(z|a)
        smoothed, A_t, C_t = self._kalman_posterior(a_sample)
        # p(x_t|a_t)
        x_hat = self._decode(a_sample).reshape(B,T,C,H,W)
        x_mu = x_hat # assume they are the same for now
        # ELBO

        elbo_calculator = ELBO(x, x_mu, x_hat, 
                        a_sample, a_mu, a_log_var, 
                        smoothed, A_t, C_t, self.scale)
        loss, recon_loss, latent_ll, elbo_kf, mse_loss = elbo_calculator.compute_loss()

        return loss, recon_loss, latent_ll, elbo_kf, mse_loss 

    def reconstruct(self, input): 
        """ Reconstruct x_hat based on input x. 
        """
        (B, T, C, H, W) = input.size()
        a_sample, _, _ = self._encode(x) 
        x_hat = self._decode(a_sample).reshape(B,T,C,H,W)

        return x_hat 

    def predict(self, input, pred_len)
        """ Predicts a sequence of length pred_len given input. 
        """
        (B, T, C, H, W) = input.size()
        a_sample, _, _ = self._encode(x) 
        smoothed, A_t, C_t = self._kalman_posterior(a_sample) # may not have access to smoothed anymore
        mu_z_smooth, sigma_z_smooth = smoothed # may need to constrain sigma_z_smooth 
        z_dist = MultivariateNormal(mu_z_smooth, scale_tril=torch.linalg.cholesky(sigma_z_smooth))
        z_sample = z_dist.sample()

        ### Predict forward 
        # Need to predict what a_hat is 
        # Need to predict what 



    def predict_sequence(self, input, seq_len=None):
        (B,T,C,H,W) = input.size()
        if seq_len is None:
            seq_len = T
        a_sample, _, _ = self._encode(input.reshape(B*T,C,H,W))
        a_sample = a_sample.reshape(B,T,-1)
        filt, A_t, C_t = self._kalman_posterior(a_sample, filter_only=True)
        filt_mean, filt_cov = filt
        eps = 1e-6*torch.eye(self.z_dim).to(input.device).reshape(1,self.z_dim,self.z_dim).repeat(B, 1, 1)
        filt_z = MultivariateNormal(filt_mean[-1].squeeze(-1), scale_tril=torch.linalg.cholesky(filt_cov[-1] + eps))
        z_sample = filt_z.sample()
        _shape = [a_sample.size(i) if i!=1 else seq_len for i in range(len(a_sample.size()))]
        obs_seq = torch.zeros(_shape).to(input.device)
        _shape = [z_sample.unsqueeze(1).size(i) if i!=1 else seq_len for i in range(len(a_sample.size()))]
        latent_seq = torch.zeros(_shape).to(input.device)
        latent_prev = z_sample
        obs_prev = a_sample[:,-1]
        for t in range(seq_len):
            # Compute alpha from a_0:t-1
            alpha_, cell_state = self.state_dyn_net
            dyn_emb, self.state_dyn_net = self.parameter_net(obs_prev.unsqueeze(1), (alpha_, cell_state))
            dyn_emb = self.alpha_out(dyn_emb)
            inter_weight = dyn_emb.softmax(-1).squeeze(1)
            ## Compute A_t, C_t
            A_t = torch.matmul(inter_weight, self.A.reshape(self.K,-1)).reshape(B,self.z_dim,self.z_dim)
            C_t = torch.matmul(inter_weight, self.C.reshape(self.K,-1)).reshape(B,self.a_dim,self.z_dim)

            # Calculate new z_t
            ## Update z_t
            latent_prev = torch.matmul(A_t, latent_prev.unsqueeze(-1)).squeeze(-1)
            latent_seq[:,t] = latent_prev
            # Calculate new a_t
            obs_prev = torch.matmul(C_t, latent_prev.unsqueeze(-1)).squeeze(-1)
            obs_seq[:,t] = obs_prev

        image_seq = self._decode(obs_seq.reshape(B*seq_len,-1)).reshape(B,seq_len,C,H,W)

        return image_seq, obs_seq, latent_seq


    