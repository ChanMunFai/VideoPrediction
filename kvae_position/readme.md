This directory is for KVAE on the Bouncing Ball positions dataset. 

In other words, instead of having a video of bouncing balls, we receive the 2D coordinates of the ball instead. 

This means that we do not need an encoder and a decoder. 

The observations are our a_t and we can use the Kalman filter on them directly. 
