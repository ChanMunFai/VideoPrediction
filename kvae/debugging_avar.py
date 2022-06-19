import torch 
from torch.distributions import MultivariateNormal

a_var = torch.load("kvae/a_var.pt", map_location=torch.device('cpu'))
a_var_diag = torch.load("kvae/diag_a_var.pt", map_location=torch.device('cpu'))

# print(a_var.size())
# print(a_var_diag.size())

# Check for negative values 

# print(torch.any(a_var < 0)) # False 
# print(torch.all(a_var >= 0)) # True 

# print(torch.any(a_var_diag < 0)) # False 
# print(torch.all(a_var_diag >= 0)) # True 

print(torch.any(a_var == 0)) # True
a_var2 = a_var + 1e-8 

# print(a_var_diag)
a_var_diag_2 = a_var_diag + 1e-8

# e, v = torch.linalg.eig(a_var_diag)
# print(e)
# print(e.size())

# chol_a_var = torch.linalg.cholesky(a_var_diag_2)

a_mu = torch.full_like(a_var, 1)
q_a = MultivariateNormal(a_mu, torch.diag_embed(a_var2))

