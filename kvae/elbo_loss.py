import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ELBO():
    def __init__(self, x, x_hat, a_sample, a_mu, a_log_var, smoothed, \
            A_t, C_t, beta = 1):
        """ Object to compute ELBO loss for Kalman VAE. 

        Arguments: 
            x: Dim [B X T X NC X H X W]
            x_hat: Dim [B X T X NC X H X W]
            a_sample: Dim [BS X Time X a_dim]
            a_mu: Dim [BS X Time X a_dim]
            a_log_var: Dim [BS X Time X a_dim]
            smoothed: tuple containing mu_z_smoothed and sigma_z_smoothed
            A_t: 
            C_t: 
        """
        self.x = x
        self.x_hat = x_hat
        self.a_sample = a_sample 
        self.a_mu = a_mu
        self.a_log_var = a_log_var

        self.mu_z_smoothed, self.sigma_z_smoothed = smoothed
        self.A_t = A_t
        self.C_t = C_t 

        self.beta = beta 
        self.z_dim = self.mu_z_smoothed.size(2)
        self.a_dim = self.a_mu.size(2)

        # Fixed covariance matrices 
        self.Q = 0.08*torch.eye(self.z_dim).to(float) 
        self.R = 0.03*torch.eye(self.a_dim).to(float) 

        # Initialise p(z_1) 
        self.mu_z0 = (torch.zeros(self.z_dim)).float()
        self.sigma_z0 = (20*torch.eye(self.z_dim)).float()

    def compute_loss(self): 
        """
        min -ELBO =  - log p(x_t|a_t) + log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)

        Returns: 
            loss: - recon_loss + kld
            elbo: - recon_loss + beta * kld
            recon_loss: log p(x_t|a_t)
            kld: log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)
        """

        (B, T, *_) = self.x.size()

        recon_loss = self.compute_reconstruction_loss()
        kld = 0 
        loss_qa = self.compute_a_marginal_loglikelihood()

        self.smoothed_z = MultivariateNormal(self.mu_z_smoothed.squeeze(-1), 
                                        scale_tril=torch.linalg.cholesky(self.sigma_z_smoothed))
        self.z_sample = self.smoothed_z.sample()
        self.z_sample = self.z_sample.reshape(B, T, -1)
        loss_qz = self.compute_z_marginal_loglikelihood()

        ### LGSSM - p(zt|zt-1) and p(at|zt)
        a_pred, z_next = self._decode_latent(self.z_sample, self.A_t, self.C_t) 
        self.a_pred = a_pred.reshape(B, T, -1)
        self.z_next = z_next.reshape(B, T, -1)

        loss_z_cond = self.compute_z_conditional_loglikelihood()
        loss_a_cond = self.compute_a_conditional_loglikelihood()

        kld = loss_qa + loss_qz - loss_a_cond - loss_z_cond
        loss = - recon_loss + self.beta * kld

        return loss, kld, recon_loss  

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
        z_next = torch.matmul(A, z_sample.unsqueeze(-1)).squeeze(-1)
        a_pred = torch.matmul(C, z_sample.unsqueeze(-1)).squeeze(-1)
        
        return a_pred, z_next

    def compute_reconstruction_loss(self): 
        """ Compute reconstruction loss of x_hat against x. 
        
        Arguments: 
            x: dim [B X T X NC X H X W]
            x_hat: dim [B X T X NC X H X W]

        Returns: 
            mse: Mean Squared Error summed across all pixels and all time steps, 
                 averaged over batch size
        """
        calc_mse_loss = nn.MSELoss(reduction = 'sum') # MSE over all pixels
        mse = calc_mse_loss(self.x, self.x_hat)
        mse = mse / self.x.size(0)
        return mse 

    def compute_a_marginal_loglikelihood(self):
        """ Compute q(a). 

        We define the distribution q(.) given its mean and variance. We then 
        find its pdf given a_sample. 

        Arguments: 
            a_sample: Dim [BS X Time X a_dim ]
            a_mu: Dim [BS X Time X a_dim ]
            a_log_var: Dim [BS X Time X a_dim ]

        Returns: 
            loss_qa: float 
        """
        q_a = MultivariateNormal(self.a_mu, torch.diag_embed(torch.exp(self.a_log_var))) # BS X T X a_dim
        
        # pdf of a given q_a 
        loss_qa = q_a.log_prob(self.a_sample).mean(dim=0).sum() # summed across all time steps, averaged in batch 
        return loss_qa

    def compute_z_marginal_loglikelihood(self):
        """ Compute q(z).

        Arguments: 
            smoothed_z: torch Distribution
            z_sample: sample of dimension 

        Returns: 
            loss: float 
        """
        loss_qz = self.smoothed_z.log_prob(self.z_sample).mean(dim=0).sum()
        return loss_qz

    def compute_z_conditional_loglikelihood(self):
        decoder_z0 = MultivariateNormal(self.mu_z0, scale_tril=torch.linalg.cholesky(self.sigma_z0))
        decoder_z = MultivariateNormal(torch.zeros(4), scale_tril=torch.linalg.cholesky(self.Q))
        
        loss_z0 = decoder_z0.log_prob(self.z_sample[0]).mean(dim=0)
        loss_zt_ztminus1 = decoder_z.log_prob((self.z_sample[:, 1:] - self.z_next[:,:-1])).mean(dim=0).sum() # averaged across batches, summed over time

        return loss_z0 + loss_zt_ztminus1

    def compute_a_conditional_loglikelihood(self):
        decoder_a = MultivariateNormal(torch.zeros(2), scale_tril=torch.linalg.cholesky(self.R))
        loss_a = decoder_a.log_prob((self.a_sample - self.a_pred)).mean(dim=1).sum()
        return loss_a



