# Stochastic model

This uses the state dict "/vol/bitbucket/mc821/VideoPrediction/saves/v1/important/vrnn_state_dict_v1_beta=0.4_step=1000000_149.pth"

This has been trained in 3 stages. 
1) Beta = 1 with constant LR = 1e-4 for 150 epochs
2) Beta = 0 with constant LR = 1e-4 for 300 epochs 
3) Beta = 0.1 with constant LR = 1e-4 for 350 epochs 
4) Beta = 0.5 for 400 steps 
5) Beta = 0.4 for 150 steps 

#### Performance on training set 
- KLD: 0.00906295
- MSE: 245.26462411

Training loss logs can be found in "/logs/VRNN/v1/important/VRNN_v1_beta=0.4_step=1000000_149.log"

#### Performance on test set 
- MSE: 1085 

MSE is calculated for entire image and for whole sequence. 
KLD is calculated for all time steps. 

--------------------------------------





