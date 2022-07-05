from pykalman import KalmanFilter
import torch 
import numpy as np
import argparse

from kvae.model_kvae import KalmanVAE

T = 10 
z_dim = 4
a_dim = 2

def test_kalman_vae(args):
    kalman_vae = KalmanVAE(args = args)
    
    x = torch.rand(1, 10, 1, 64, 64)
    a_sample, _, _ = kalman_vae._encode(x) 
    A, C, weights = kalman_vae._interpolate_matrices(a_sample)
    filtered, pred = kalman_vae.filter_posterior(a_sample, A, C)
    smoothed = kalman_vae.smooth_posterior(A, filtered, pred)

    filtered_new, pred_new = kalman_vae.new_filter_posterior(a_sample.transpose(1, 0), A, C)
    smoothed_new = kalman_vae.new_smooth_posterior(A, filtered_new, pred_new)

    return a_sample, A, C, filtered, smoothed, filtered_new, smoothed_new

def test_closeness_kalman_filter():
    
    a_sample, A, C, filtered, smoothed, filtered_new, smoothed_new = test_kalman_vae(args)
    a_sample = a_sample.detach().numpy().squeeze()
    A = A.cpu().detach().numpy().squeeze() # remove batch size 
    C = C.cpu().detach().numpy().squeeze()

    filtered_mean, filtered_cov = filtered
    filtered_mean = filtered_mean.detach().numpy().squeeze()
    filtered_cov = filtered_cov.detach().numpy().squeeze()

    smoothed_mean, smoothed_cov = smoothed
    smoothed_mean = smoothed_mean.detach().numpy().squeeze()
    smoothed_cov = smoothed_cov.detach().numpy().squeeze()

    filtered_mean_new, filtered_cov_new = filtered_new 
    filtered_mean_new = filtered_mean_new.detach().numpy().squeeze()
    filtered_cov_new = filtered_cov_new.detach().numpy().squeeze()

    smoothed_mean_new, smoothed_cov_new = smoothed_new 
    smoothed_mean_new = smoothed_mean_new.detach().numpy().squeeze()
    smoothed_cov_new = smoothed_cov_new.detach().numpy().squeeze()

    mu_0 = np.zeros(z_dim) # initial_state_mean
    sigma_0 = 20*np.eye(z_dim) # initial_state_covariance
    Q = 0.08*np.eye(z_dim) # transition_covariance
    R = 0.03*np.eye(a_dim) # observation_covariance 

    # print(a_sample.shape, A.shape, C.shape, filtered_mean.shape, smoothed_mean.shape)

    kf = KalmanFilter(
        initial_state_mean=mu_0,
        initial_state_covariance = sigma_0, 
        transition_matrices = A, 
        observation_matrices = C, 
        transition_covariance = Q, 
        observation_covariance = R)

    filtered_mean_kf, filtered_cov_kf = kf.filter(a_sample)
    smoothed_mean_kf, smoothed_cov_kf = kf.smooth(a_sample)

    print(np.allclose(filtered_mean, filtered_mean_kf)) 
    print(np.allclose(filtered_mean_new, filtered_mean_kf))

    print(np.allclose(smoothed_mean, smoothed_mean_kf))
    print(np.allclose(smoothed_mean_new, smoothed_mean_kf))

    # print("=====> KF Mean:") 
    # print(smoothed_mean_kf.round(5))
    # print("=====> MF's Mean:")
    # print(smoothed_mean.round(5))
    # print("=====> Carles's Mean:")
    # print(smoothed_mean_new.round(5))

    ### Covariane 
    # print(np.allclose(smoothed_cov, smoothed_cov_kf))
    # print(np.allclose(smoothed_cov_new, smoothed_cov_kf))

    # print(np.allclose(filtered_cov, filtered_cov_kf))
    # print(np.allclose(filtered_cov_new, filtered_cov_kf))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = "MovingMNIST", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall]")
    parser.add_argument('--x_dim', default=1, type=int)
    parser.add_argument('--a_dim', default=2, type=int)
    parser.add_argument('--z_dim', default=4, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--scale', default=0.3, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--alpha', default="rnn", type=str, 
                    help = "choose between [mlp, rnn]")
    
    args = parser.parse_args()

    for i in range(10):
        test_closeness_kalman_filter()