import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

def test_z_cond_ll(z_next_value):
    z_sample = torch.zeros((32, 10, 4))
    z_next = torch.full_like(z_sample, z_next_value)

    Q = 0.08*torch.eye(4).to(torch.float64)
    decoder_z = MultivariateNormal(torch.zeros(4), scale_tril=torch.linalg.cholesky(Q))
    
    log_prob_zt_ztminus1 = decoder_z.log_prob((z_sample - z_next)).mean(dim=0).sum() # averaged across batches, summed over time
    print(log_prob_zt_ztminus1.item())


# So the more different z_t+1 is from z_t, the more negative the LL 
# But if they are close, then the LL is supposed to be positive 

def test_z0(z0_sample_value):
    mu_z0 = (torch.zeros(4)).double()
    sigma_z0 = (20*torch.eye(4)).double()

    decoder_z0 = MultivariateNormal(mu_z0, scale_tril=torch.linalg.cholesky(sigma_z0))
    z_sample = torch.full_like(mu_z0, z0_sample_value)
    loss_z0 = decoder_z0.log_prob(z_sample)
    print(loss_z0.item())

if __name__ == "__main__":
    test_z_cond_ll(0.01)
    test_z_cond_ll(0.1)
    test_z_cond_ll(0.3)
    test_z_cond_ll(1)
    
    test_z0(0.01)
    test_z0(0.1)
    test_z0(1)
    test_z0(10)
    test_z0(100)

