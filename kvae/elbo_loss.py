import sys
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ELBO():
    def __init__(self, x, x_mu, x_hat, a_sample, a_mu, a_log_var, smoothed, \
            A_t, C_t, scale = 0.3):
        """ Object to compute ELBO for Kalman VAE. 

        Here, instead of maximising ELBO, we min -ELBO, whereby 
        
        ELBO = recon_ll - latent_ll + elbo_kf
             = log p(x) - log q(a) + [log p(z_t|z_t-1)} + log p(a_t|z_t)} - log p(z_t|a_{1:T}}]

        To min loss, we min: 
        Loss = recon_loss + latent_ll - elbo_kf
             = -log p(x) + log q(a) - [log p(z_t|z_t-1)} + log p(a_t|z_t)} - log p(z_t|a_{1:T}}]

        whereby recon_loss = - recon_ll. 

        Arguments: 
            x: Dim [B X T X NC X H X W]
            x_mu: Dim [B X T X NC X H X W]
            x_var: Dim [B X T X NC X H X W]
            x_hat: Dim [B X T X NC X H X W]
            a_sample: Dim [BS X Time X a_dim]
            a_mu: Dim [BS X Time X a_dim]
            a_log_var: Dim [BS X Time X a_dim]
            smoothed: tuple containing mu_z_smoothed and sigma_z_smoothed
            A_t: 
            C_t: 
        """
        self.device = x.device

        self.x = x
        self.x_mu = x_mu # Not yet supported in decoder # no need for x_var because it is a constant
        self.x_hat = x_hat
        self.a_sample = a_sample 
        self.a_mu = a_mu
        self.a_log_var = a_log_var

        self.mu_z_smoothed, self.sigma_z_smoothed = smoothed
        self.A_t = A_t
        self.C_t = C_t 

        self.scale = scale 
        self.z_dim = self.mu_z_smoothed.size(2)
        self.a_dim = self.a_mu.size(2)

        # Fixed covariance matrices 
        self.Q = 0.08*torch.eye(self.z_dim).to(torch.float64).to(self.device) 
        self.R = 0.03*torch.eye(self.a_dim).to(torch.float64).to(self.device) 

        # Initialise p(z_1) 
        self.mu_z0 = (torch.zeros(self.z_dim)).double().to(self.device)
        self.sigma_z0 = (20*torch.eye(self.z_dim)).double().to(self.device)

    def compute_loss(self): 
        """
        Instead of using beta as in Beta-VAE, we use a scale parameter on the 
        recon_loss. 

        Returns: 
            loss: self.scale * recon_loss + latent_ll - elbo_kf
            recon_loss: -log p(x_t|a_t)
                * Can either be MSE, or NLL of a Bernoulli or Gaussian distribution 
            latent_ll: log q(a)
            elbo_kf: [log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)]

        During training, we want loss to go down, recon_loss to go down, latent_ll to go down, 
        elbo_kf to go up. 
        """

        (B, T, *_) = self.x.size()

        recon_loss = self.compute_reconstruction_loss(mode = "bernoulli")
        kld = 0 
        latent_ll = self.compute_a_marginal_loglikelihood() # log q(a)

        ### LGSSM 
        self.smoothed_z = MultivariateNormal(self.mu_z_smoothed.squeeze(-1), 
                                        scale_tril=torch.linalg.cholesky(self.sigma_z_smoothed))
        self.z_sample = self.smoothed_z.sample()
        self.z_sample = self.z_sample.reshape(B, T, -1).to(self.device)
        z_marginal_ll = self.compute_z_marginal_loglikelihood() # log q(z)

        a_pred, z_next = self._decode_latent(self.z_sample, self.A_t, self.C_t) 
        self.a_pred = a_pred.reshape(B, T, -1).to(self.device)
        self.z_next = z_next.reshape(B, T, -1).to(self.device)

        z_cond_ll = self.compute_z_conditional_loglikelihood() # log p(z_t| z_t-1)
        a_cond_ll = self.compute_a_conditional_loglikelihood() # log p(a_t| z_t)

        # elbo_kf =  (z_marginal_ll - a_cond_ll - z_cond_ll) # wait this seems to be wrong 
        elbo_kf = a_cond_ll + z_cond_ll - z_marginal_ll

        print("q(z) is ", z_marginal_ll.item())
        print("q(a|z) is ", a_cond_ll.item())
        print("q(z|zt is", z_cond_ll.item())

        loss = self.scale * recon_loss + latent_ll - elbo_kf

        # Calculate MSE for tracking 
        mse_loss = self.compute_reconstruction_loss(mode = "mse")

        return loss, recon_loss, latent_ll, elbo_kf, mse_loss  

    def _decode_latent(self, z_sample, A, C):
        """ Returns z_t+1 and a_t given z_t and matrices A and C. 

        Used to calculate p(z_t|z_t-1) and p(a_t|z_t) for ELBO loss.

        p(z_t|z_t-1) = N(z_t; A_t * z_t-1, Q)
        p(a_t|z_t) = N(a_t; C_t * z_t, R)

        Parameters: 
            z_sample: vector z_t of dim [B X T X z_dim] 
            A: (interpolated) state transition matrix of dim [B X T X z_dim X z_dim]
            C: (interpolated) emission matrix of dim [B X T X a_dim X z_dim]

        Returns: 
            z_next: vector z_t+1 of dim [B X T X z_dim]
            a_pred: vector a_t of dim [B X T X a_dim]
        """
        # Do this just in case I got matrix multiplication wrong
        B, T, z_dim = z_sample.size() 
        z_sample = torch.reshape(z_sample, (B*T, -1)) # [B*T, z_dim]
        A = torch.reshape(A, (B*T, z_dim, z_dim)) # [B*T, z_dim, z_dim]
        B, T, a_dim, z_dim = C.size()
        C = torch.reshape(C, (B*T, a_dim, z_dim))

        z_next = torch.matmul(A, z_sample.unsqueeze(-1)).squeeze(-1)
        a_pred = torch.matmul(C, z_sample.unsqueeze(-1)).squeeze(-1)

        z_next = torch.reshape(z_next, (B, T, z_dim))
        a_pred = torch.reshape(a_pred, (B, T, a_dim))
        
        return a_pred, z_next

    def compute_reconstruction_loss(self, mode = "bernoulli"): 
        """ Compute reconstruction loss of x_hat against x. 
        
        Arguments: 
            mode: 'bernoulli', 'gaussian' or 'mse'

        Returns: 
            recon_loss: Reconstruction Loss summed across all pixels and all time steps, 
                 averaged over batch size. When using 'bernoulli' or 'gaussian', this is 
                 the Negative Log-Likelihood. 
        """
        if mode == "mse": 
            calc_mse_loss = nn.MSELoss(reduction = 'sum').to(self.device) # MSE over all pixels
            mse = calc_mse_loss(self.x, self.x_hat)
            mse = mse / self.x.size(0)
            return mse.to(self.device) 
        
        elif mode == "bernoulli": 
            eps = 1e-5
            prob = torch.clamp(self.x_mu, eps, 1 - eps) # prob = x_mu 
            ll = self.x * torch.log(prob) + (1 - self.x) * torch.log(1-prob)
            ll = ll.mean(dim = 0).sum() 
            return - ll 

        elif mode == "gaussian": 
            x_var = torch.full_like(self.x_mu, 0.01)
            x_dist = MultivariateNormal(self.x_mu, torch.diag_embed(x_var))
            ll = x_dist.log_prob(x).mean(dim = 0).sum() # verify this 
            return - ll
            

    def compute_a_marginal_loglikelihood(self):
        """ Compute q(a). 

        We define the distribution q(.) given its mean and variance. We then 
        find its pdf given a_sample. 

        Arguments: 
            a_sample: Dim [BS X Time X a_dim ]
            a_mu: Dim [BS X Time X a_dim ]
            a_log_var: Dim [BS X Time X a_dim ]

        Returns: 
            latent_ll: q(a_sample)
        """
        a_var = torch.exp(self.a_log_var)
        a_var = torch.clamp(a_var, min = 1e-8) # force values to be above 1e-8
    
        q_a = MultivariateNormal(self.a_mu, torch.diag_embed(a_var)) # BS X T X a_dim
        
        # pdf of a given q_a 
        latent_ll = q_a.log_prob(self.a_sample).mean(dim=0).sum().to(self.device) # summed across all time steps, averaged in batch 
        return latent_ll

    def compute_z_marginal_loglikelihood(self):
        """ Compute q(z).

        Arguments: 
            smoothed_z: torch Distribution
            z_sample: sample of dimension 

        Returns: 
            loss: torch.float64 
        """
        loss_qz = self.smoothed_z.log_prob(self.z_sample).mean(dim=0).sum().to(self.device)
        return loss_qz

    def compute_z_conditional_loglikelihood(self):
        decoder_z0 = MultivariateNormal(self.mu_z0, scale_tril=torch.linalg.cholesky(self.sigma_z0))
        decoder_z = MultivariateNormal(torch.zeros(4).to(self.device), scale_tril=torch.linalg.cholesky(self.Q))
        
        loss_z0 = decoder_z0.log_prob(self.z_sample[0]).mean(dim=0) 
        loss_zt_ztminus1 = decoder_z.log_prob((self.z_sample[:, 1:, :] - self.z_next[:,:-1, :])).mean(dim=0).sum().to(self.device) # averaged across batches, summed over time

        loss_z = loss_z0 + loss_zt_ztminus1
        print("q(z_0) is ", loss_z0.item())

        return loss_z

    def compute_a_conditional_loglikelihood(self):
        decoder_a = MultivariateNormal(torch.zeros(2).to(self.device), scale_tril=torch.linalg.cholesky(self.R))
        loss_a = decoder_a.log_prob((self.a_sample - self.a_pred)).mean(dim=1).sum().to(self.device)
        
        # loss_a = torch.clamp(loss_a, -10000, 10000)
        
        return loss_a



