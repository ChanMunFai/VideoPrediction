import torch 

sigma_z = torch.load("sigma_z.pt")
sigma_z = torch.clamp(sigma_z, min = 0.1, max = 10) 
print(torch.min(sigma_z))
print(torch.max(sigma_z))
print(torch.isnan(sigma_z).any())

example = sigma_z[1]
# print(example)

L, info = torch.linalg.cholesky_ex(example)
print(L)
print(info)
# torch.linalg.cholesky(example)

# torch.linalg.cholesky(sigma_z)