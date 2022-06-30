# Comments 

Training differed from what original paper recommended. 

Paper recommended 
0. Pretrain CDNA architecture 
1. Train same network for 1st stage but instead of using a latent variable sampled from posterior, use a latent sampled from prior 
2. Use latent variable sampled from posterior, but do not include KL divergence loss 
3. Include KL divergence loss 

### What I did 
1. Train CDNA architecture without even including any stochastic elements 
2. Add in stochastic element and use latent variable from posterior but do not include KL divergence loss 
3. Include KL divergence loss

In summary, I just did not have Step 1 done properly, but it should still work out. 

