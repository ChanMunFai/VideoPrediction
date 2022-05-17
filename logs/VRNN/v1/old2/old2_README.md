This subdirectory contains log files where I tried to change the MSE to use a reduction over sum. 

However, this implementation is wrong as it simply did not take the average loss across all items in a batch (i.e. batch dependent). 

I had wanted the MSE algorithm to find the MSE over all pixels - this is already implemented. 