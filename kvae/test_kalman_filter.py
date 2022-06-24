from pykalman import KalmanFilter
import torch 
import numpy as np

from kvae.model_kvae import KalmanVAE

T = 10 
z_dim = 4
a_dim = 2

def test_kalman_vae():
    kalman_vae = KalmanVAE(
        x_dim = 1, 
        a_dim = a_dim, 
        z_dim = z_dim, 
        K = 3, 
        device = "cpu"
        )
    
    x = torch.rand(1, 10, 1, 64, 64)
    a_sample, _, _ = kalman_vae._encode(x) 
    A, C = kalman_vae._interpolate_matrices(a_sample)
    filtered, pred = kalman_vae.filter_posterior(a_sample, A, C)
    smoothed = kalman_vae.smooth_posterior(A, filtered, pred)
    
    return a_sample, A, C, filtered, smoothed

def test_closeness_kalman_filter():
    
    a_sample, A, C, filtered, smoothed = test_kalman_vae()
    a_sample = a_sample.detach().numpy().squeeze()
    A = A.cpu().detach().numpy().squeeze() # remove batch size 
    C = C.cpu().detach().numpy().squeeze()

    filtered_mean, filtered_cov = filtered
    filtered_mean = filtered_mean.detach().numpy().squeeze()
    filtered_cov = filtered_cov.detach().numpy().squeeze()

    smoothed_mean, smoothed_cov = smoothed
    smoothed_mean = smoothed_mean.detach().numpy().squeeze()
    smoothed_cov = smoothed_cov.detach().numpy().squeeze()

    mu_0 = np.zeros(z_dim) # initial_state_mean
    sigma_0 = 20*np.eye(z_dim) # initial_state_covariance
    Q = 0.08*np.eye(z_dim) # transition_covariance
    R = 0.03*np.eye(a_dim) # observation_covariance 

    kf = KalmanFilter(
        initial_state_mean=mu_0,
        initial_state_covariance = sigma_0, 
        transition_matrices = A, 
        observation_matrices = C, 
        transition_covariance = Q, 
        observation_covariance = R)

    filtered_mean_kf, filtered_cov_kf = kf.filter(a_sample)

    print(np.allclose(filtered_cov, filtered_cov_kf))
    print(np.allclose(filtered_mean, filtered_mean_kf))

if __name__ == "__main__":
    for i in range(10):
        test_closeness_kalman_filter()