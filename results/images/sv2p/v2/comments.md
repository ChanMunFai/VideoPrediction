# Comments on current results

state_dict_path = "saves/sv2p/stage3/final_beta=0.001/sv2p_state_dict_550.pth"

Used decay from beta = 1 to beta = 0. 

Results 
- Got good reconstructions 
- But bad predictions
    - No changes in prediction even though prior use is different 
- Exactly the same as v1 

### Things to troubleshoot 
1. Train posterior network only so that we can see what happens when posterior is done correctly 

We hope that when true posterior is available, prediction network learns to use the latent variables properly and gives good predictions. 

--------------------

However, my suspicion is that the network is not using the latent variables well. It seems to be heavily relying on x_t to predict x_t-1 instead of the global variable z. 

2. Check the code for CDNA architecture. 

Ideally, we can think of a way to be able to test if code is using the latent variables well. I see no reason why it isn't, but there may be a bug. 

Another simple reason why this is not working may be that I have performed the prediction wrongly. 

3. Check through the code again, and replace the prior code with the code I used for my new KL divergence. 

4. I had problems previously with regards to whether I am actually using the standard Gaussian for my prior during training. I do think that I fixed it but I am not very sure. Nevertheless, even if this is not exactly standard Gaussian, if I do use a standard Gaussian for inference, I shouldn't get such bad results and results that do not change at all. 

Hence, there is indeed something wrong with my network or something wrong with my inference.
