"""Source: https://github.com/charlio23/bouncing-ball/blob/main/models/KalmanVAE.py """

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli

from kvae.encode_decode import Encoder, Decoder 
from kvae.elbo_loss import ELBO

class KalmanVAE(nn.Module):
    def __init__(self, x_dim, a_dim, z_dim, K, device, scale=0.3):
        super(KalmanVAE, self).__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim
        self.K = K
        self.scale = scale

        self.device = device 

        self.encoder = Encoder(input_channels=1, a_dim = 2).to(self.device)
        self.decoder = Decoder(a_dim = 2, enc_shape = [32, 7, 7], device = self.device).to(self.device) # change this to encoder shape
        
        self.parameter_net = nn.LSTM(self.a_dim, 50, 2, batch_first=True).to(self.device)
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
        # print(self.state_dyn_net)
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


def trial_forward(): 
    net = KalmanVAE(x_dim=1, a_dim=2, z_dim=4, K=3)
    
    from torch.autograd import Variable
    sample = Variable(torch.rand((6,10,1,32,32)), requires_grad=True) 
    (B,T,C,H,W) = sample.size()

    net.forward(sample)


def trial_run(): 
    # Trial run
    net = KalmanVAE(x_dim=1, a_dim=2, z_dim=4, K=3)
    
    from torch.autograd import Variable
    sample = Variable(torch.rand((6,10,1,64,64)), requires_grad=True) 
    (B,T,C,H,W) = sample.size()

    torch.autograd.set_detect_anomaly(True)

    # print(net)
    # print(net.parameter_net) 
    # print(net.alpha_out)

    ##############################
    ######## Forward Pass ########
    ##############################

    ### Encode - p(a), sample a

    a_sample, a_mu, a_log_var = net._encode(sample.reshape(B*T,C,H,W))
    a_sample = a_sample.reshape(B,T,-1)
    a_mu = a_mu.reshape(B,T,-1)
    a_log_var = a_log_var.reshape(B,T,-1)

    # print(a_sample.shape) # BS X Time X a_dim 
    # print(a_mu.shape) # BS X Time X a_dim 
    # print(a_log_var.shape) # BS X Time X a_dim 

    ### Posterior - p(z|a), sample z

    smoothed, A_t, C_t = net._kalman_posterior(a_sample, None)
    mu_z_smoothed, sigma_z_smoothed = smoothed
    # print(mu_z_smoothed.shape) # T X BS X z_dim X 1 
    # print(sigma_z_smoothed.shape) # T X BS X z_dim X z_dim 
    
    # print(A_t.shape) # BS X T X z_dim X z_dim 
    # print(C_t.shape) # BS X T X a_dim X z_dim 

    smoothed_z = MultivariateNormal(mu_z_smoothed.squeeze(-1), 
                                        scale_tril=torch.linalg.cholesky(sigma_z_smoothed))
    z_sample = smoothed_z.sample()
    z_sample = z_sample.reshape(B, T, -1)
    
    ### LGSSM - p(zt|zt-1) and p(at|zt)
    a_pred, z_next = net._decode_latent(z_sample, A_t, C_t) 
    a_pred = a_pred.reshape(B, T, -1)
    z_next = z_next.reshape(B, T, -1)

    # print(z_sample.shape) # BS X T X z_dim # z_t 
    # print(a_pred.shape) # BS X T X a_dim # a_t
    # print(z_next.shape) # BS X T X z_dim # z_t+1 hat 

    ### Decode 

    x_hat = net._decode(a_sample.reshape(B*T,-1)).reshape(B,T,C,H,W)

    # Losses: max ELBO = log p(x_t|a_t) - [log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)]
    
    ##############################
    ####### log p(x_t|a_t) #######
    ##############################
    
    # NLL loss - can modify that to MSE 

    ##############################
    #########  log q(a)  #########
    ##############################
    
    q_a = MultivariateNormal(a_mu, torch.diag_embed(torch.exp(a_log_var))) # BS X T X a_dim
    # pdf of a given q_a 
    loss_qa = q_a.log_prob(a_sample).mean(dim=0).sum() # summed across all time steps, averaged in batch 
    
    ##############################
    #########  log q(z)  #########
    ##############################

    loss_qz = smoothed_z.log_prob(z_sample.reshape(T, B, -1)).mean(dim=1).sum()

    ##############################
    ##### log p(z_t|z_{t-1}) #####
    ##############################

    # Covariance matrices - are they learnable parameters???
    Q = 0.08*torch.eye(4).to(torch.float64) 

    mu_z1 = (torch.zeros(4)).to(torch.float64) 
    sigma_z1 = (20*torch.eye(4)).to(torch.float64) 
    decoder_z1 = MultivariateNormal(mu_z1, scale_tril=torch.linalg.cholesky(sigma_z1))
    decoder_z = MultivariateNormal(torch.zeros(4), scale_tril=torch.linalg.cholesky(Q))
    
    loss_z1 = decoder_z1.log_prob(z_sample[0]).mean(dim=0)
    loss_zt_ztminus1 = decoder_z.log_prob((z_sample[:, 1:] - z_next[:,:-1])).mean(dim=0).sum() # averaged across batches, summed over time

    ##############################
    ####### log p(a_t|z_t) #######
    ##############################

    R = 0.03*torch.eye(2).to(torch.float64) 
    decoder_a = MultivariateNormal(torch.zeros(2), scale_tril=torch.linalg.cholesky(R))
    
    # print(a_sample.shape) # BS X T X a_dim
    # print(a_pred.shape) # BS X T X a_dim

    loss_a = decoder_a.log_prob((a_sample - a_pred)).mean(dim=1).sum()
    # print(loss_a.shape)

    # print(loss_a)
    
if __name__=="__main__":
    trial_run()
    # trial_forward()

    