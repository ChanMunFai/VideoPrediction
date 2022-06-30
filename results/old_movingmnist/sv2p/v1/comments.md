# Comments on current results

state_dict_path = "saves/sv2p/stage3/final_beta=0.1/sv2p_state_dict_499.pth"

Used decay from beta = 1 to beta = 0. 

Results 
- Got good reconstructions 
- But bad predictions
- This is with both prior and true posterior (done wrongly)

I forgot to save the posterior network. Hence, using the true posterior is going to give me bad results because this posterior is not even regulated to approach the prior whilst the SV2P architecture expects the latent samples to come from a posterior that does approach the prior.

Hence, what I can do is to freeze the best network that I have, and just train my posterior network and save that. That will give me the true posterior. 

On the other hand, I do not actually need the true posterior for my results, so there is something bad going on here. I will try the better model in v2, and if it does not work well, then there is 
1) something wrong with my training 
2) something wrong with my inference 

To investigate that, I will 
a) retrain the posterior net 
b) look through code for inference, and look at generated images to see if they are any different 