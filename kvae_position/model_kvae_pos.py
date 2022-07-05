import argparse 
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import matplotlib.pyplot as plt 

from kvae_position.elbo_loss_pos import ELBO

class KalmanVAE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(KalmanVAE, self).__init__()
        self.args = kwargs['args']
        self.a_dim = self.args.a_dim
        self.z_dim = self.args.z_dim
        self.K = self.args.K
        self.scale = self.args.scale
        self.device = self.args.device 

        self.parameter_net = nn.LSTM(self.a_dim, 50, 1, batch_first=True).to(self.device) 
        self.alpha_out = nn.Linear(50, self.K).to(self.device)

        # Initialise a_1 (optional)
        self.a1 = nn.Parameter(torch.zeros(self.a_dim, requires_grad=True, device = self.device))
        self.state_dyn_net = None

        # Initialise p(z_1) 
        self.mu_0 = (torch.zeros(self.z_dim)).double()
        self.sigma_0 = (20*torch.eye(self.z_dim)).double()

        # A initialised with identity matrices. B initialised from Gaussian 
        self.A = nn.Parameter(torch.eye(self.z_dim).unsqueeze(0).repeat(self.K,1,1).to(self.device))
        self.C = nn.Parameter(torch.randn(self.K, self.a_dim, self.z_dim).to(self.device)*0.05)

        # Covariance matrices - fixed. Noise values obtained from paper. 
        self.Q = 0.08*torch.eye(self.z_dim).double().to(self.device) 
        self.R = 0.03*torch.eye(self.a_dim).double().to(self.device) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _interpolate_matrices(self, obs, learn_a1 = True):
        """ Generate weights to choose and interpolate between K operating modes. 

        Parameters:
            obs: vector a of dimension [BS X T X a_dim]
            learn_a1: bool. If learn_a1 is True, we replace a_1 with a learned embedding. 
                Otherwise, we generate a_1 from encoding x_1. 

        Returns: 
            A_t: interpolated state transition matrix of dimension [BS X T X z_dim X z_dim]
            C_t: interpolated state transition matrix of dimension [BS X T X a_dim X z_dim]
            weights: 
        """
        (B, T, _) = obs.size()
        
        if learn_a1: 
            a1 = self.a1.reshape(1,1,-1).expand(B,-1,-1)
            joint_obs = torch.cat([a1,obs[:,:-1,:]],dim=1).double()
            # print("Initialised a1 shape", a1.shape) # BS X 1 X a_dim
            # print("joint_obs shape", joint_obs.shape) # BS X T X a_dim

        else: 
            joint_obs = obs
            # print("joint_obs shape", joint_obs.shape) # BS X T X a_dim

        # print(joint_obs.dtype)
        
        dyn_emb, self.state_dyn_net = self.parameter_net(joint_obs.to(torch.float32))
        # print("dyn_emb shape", dyn_emb.size()) # BS X T X 50
        dyn_emb = self.alpha_out(dyn_emb.reshape(B*T,50))
        weights = dyn_emb.softmax(-1)
        
        # print("Weights shape", inter_weight.shape) # B*T X K 
        # print("A shape", self.A.shape) # K X z_dim X z_dim  
        # print("C shape", self.C.shape) # K X a_dim X z_dim  
        
        A_t = torch.matmul(weights, self.A.reshape(self.K,-1)).reshape(B,T,self.z_dim,self.z_dim).double()
        C_t = torch.matmul(weights, self.C.reshape(self.K,-1)).reshape(B,T,self.a_dim,self.z_dim).double()
        
        return A_t, C_t, weights 

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

        mu_filt = torch.zeros(T, B, self.z_dim, 1).to(obs.device).double()
        sigma_filt = torch.zeros(T, B, self.z_dim, self.z_dim).to(obs.device).double()

        mu_t = self.mu_0.expand(B,-1).unsqueeze(-1).double().to(self.device) 
        sigma_t = self.sigma_0.expand(B,-1,-1).double().to(self.device)
        mu_predicted = mu_t
        sigma_predicted = sigma_t 

        mu_pred = torch.zeros_like(mu_filt).to(self.device) # A u_t
        sigma_pred = torch.zeros_like(sigma_filt).to(self.device) # P_t

        A = A.double().to(self.device) 
        C = C.double().to(self.device)

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

        return (mu_filt, sigma_filt), (mu_pred, sigma_pred)

        
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

        mu_filt = mu_filt
        sigma_filt= sigma_filt
        mu_pred = mu_pred
        sigma_pred = sigma_pred

        mu_z_smooth = torch.zeros_like(mu_filt).to(self.device)
        sigma_z_smooth = torch.zeros_like(sigma_filt).to(self.device)
        mu_z_smooth[-1] = mu_filt[-1]
        sigma_z_smooth[-1] = sigma_filt[-1]

        (T, *_) = mu_filt.size()
        A = A.double().to(self.device)

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
        
        A, C, weights = self._interpolate_matrices(obs)
        filtered, pred = self.filter_posterior(obs, A, C)
        smoothed = self.smooth_posterior(A, filtered, pred)
        
        return smoothed, A, C, weights 

    def forward(self, a):
        (B,T,a_dim) = a.size()
        
        # q(z|a)
        smoothed, A_t, C_t, weights = self._kalman_posterior(a)

        elbo_calculator = ELBO(a, smoothed, A_t, C_t)
        loss = elbo_calculator.compute_loss()

        return loss


    