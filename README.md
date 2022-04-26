

### 1. Variational Recurrent Neural Network

Code adapted from [https://github.com/emited/VariationalRecurrentNeuralNetwork](https://github.com/emited/VariationalRecurrentNeuralNetwork)

Variational RNN (VRNN), from *A Recurrent Latent Variable Model for Sequential Data*.


The paper is available [here](https://arxiv.org/abs/1506.02216).

![png](images/fig_1_vrnn.png)

##### Things to figure out
1. What does it mean to predict future frames? Is there a ground truth that exists, but that the model does not see? How then do we generate new frames?

We have xt_hat during our forward model. That can be our predicted frame of the current frame.

But what about the "future future" (as in what we do for images)?

2. What about sampling? What does that actually mean?

In the sampling script, we initialise an empty latent state h; specifically, h_0 is a matrix of zeros.

The prior for z is sampled from this h and the learned prior distributions. What this means is that, given a random variable h, we can sample a z that makes sense based on our trained prior distributions.

We then decode this z (and h_0) into a predicted video frame x_hat0. What this means is that given an arbitrary choice of latent variable h_0, we can eventually produce an image.

**Note** this is different from training, where z_t is an input of x_t (which is known to us). Here, we are using xt_hat instead.


3. How does prediction work in an image VAE?

4. How will prediction work here?
Since we already have a sequence of test data (e.g. of sequence length 10), we can split this into $x_{0:4}$ and $x_{5:9}$, where the first sub-sequence is what the model has access to and the second sub-sequence is what the model has to completely predict.

In the generative model, xt depends on h_t-1 and z_t. We have access to h_4 and z_5 to generate x5_hat. Similarly, for x6, we have access to h_5 and z_6, where h_5 is produced based on x5_hat.




#### Questions to ask during meeting
1. What do we do with the posterior distribution given that it is now a transposed deconvolution layer instead of a Normal distribution with mean and variance?


