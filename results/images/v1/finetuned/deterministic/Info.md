# Deterministic model

This uses the state dict "/vol/bitbucket/mc821/VideoPrediction/saves/v1/important/vrnn_state_dict_v1_beta=0.0_step=1000000_299.pth"

This has been trained in 2 stages. 
1) Beta = 1 with constant LR = 1e-4 for 150 epochs
2) Beta = 0 with constant LR = 1e-4 for 300 epochs 

#### Performance on training set 
- Total loss: 119.41995354, 
- KLD: 307709.35954861, 
- MSE: 119.41995354

Training loss logs can be found in "/logs/VRNN/v1/important/VRNN_v1_beta=0.0_step=1000000_300.log"

#### Performance on test set 
- KLD = 6193
- MSE = 26.07533109 

** A bit buggy as to why test performance is better in terms of MSE but worse in KLD. Need to investigate**

--------------------------------------





