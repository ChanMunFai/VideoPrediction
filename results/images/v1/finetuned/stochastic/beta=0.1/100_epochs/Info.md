# Stochastic model

This uses the state dict "/vol/bitbucket/mc821/VideoPrediction/saves/v1/important/vrnn_state_dict_v1_beta=1.0_step=1000000_99.pth"

This has been trained in 3 stages. 
1) Beta = 1 with constant LR = 1e-4 for 150 epochs
2) Beta = 0 with constant LR = 1e-4 for 300 epochs 
3) Beta = 0.1 with constant LR = 1e-4 for 100 epochs 

#### Performance on training set 
- Total loss = 406.37303653
- KLD = 186.00270479
- MSE =  387.77276594

Training loss logs can be found in "/logs/VRNN/v1/important/VRNN_v1_beta=0.1_step=1000000_99.log"

#### Performance on test set 
- KLD = 3.76144352
- MSE = 52.78745972  

--------------------------------------





