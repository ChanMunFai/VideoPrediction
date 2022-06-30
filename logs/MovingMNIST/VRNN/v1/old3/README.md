This subdirectory contains all the logs for when MSE is scaled. 

i.e. I used reduction = None = mean, which divides the MSE across all items in a batch and across all pixels. These are the pixel-wise MSE. 

However, in the ELBO of the paper, we use image-wise MSE instead. 