# Video Prediction 

This is the repository for my MSc in Artificial Intelligence individual project (dissertation) at Imperial College. 

# Table of Contents 
   * [Variational Recurrent Neural Network(VRNN)](#ariational-Recurrent-Neural-Network)
       * [ELBO](#VRNN-ELBO)
       * [Results](#Results)

# 1. Variational Recurrent Neural Network (VRNN)

The code is adapted from [here](https://github.com/emited/VariationalRecurrentNeuralNetwork) for the paper [*A Recurrent Latent Variable Model for Sequential Data*](https://arxiv.org/abs/1506.02216).

```
@inproceedings{NIPS2015_b618c321,
 author = {Chung, Junyoung and Kastner, Kyle and Dinh, Laurent and Goel, Kratarth and Courville, Aaron C and Bengio, Yoshua},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {C. Cortes and N. Lawrence and D. Lee and M. Sugiyama and R. Garnett},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {A Recurrent Latent Variable Model for Sequential Data},
 url = {https://proceedings.neurips.cc/paper/2015/file/b618c3210e934362ac261db280128c22-Paper.pdf},
 volume = {28},
 year = {2015}
}
```

## VRNN ELBO 

![png](images/fig_1_vrnn.png)

## Results

Reconstructed frames and predictions for the Moving MNIST dataset can be found (here)[https://github.com/ChanMunFai/VideoPrediction/tree/master/results/images/v1/stochastic/stage_c]. 

An example of a prediction is given below. 

![png](results/images/v1/stochastic/stage_c/train/predictions_6.jpeg)

The first row are ground truth frames (Frames 1 -5)  which the model has seen, the second row are ground truth frames (Frames 6 - 10) which the model does not see and is trying to predict, and the third row (Frames 6 - 10) are the predicted frames. 



