""" Source: https://github.com/kkew3/cse291g-sv2p/blob/master/src/sv2p/model.py """

import typing

import torch
import torch.distributions
import torch.nn as nn


class Squeeze(nn.Module):
    """
    Merely for convenience to use in ``torch.nn.Sequential``.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    # pylint: disable=arguments-differ
    def forward(self, tensor):
        return tensor.squeeze(self.dim)


class PosteriorInferenceNet(nn.Module):
    """ q_phi(z|x_{0:T})
    z can be a global z or z_t
    """
    def __init__(self, tbatch: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(10), # changed from 32 to 10
            nn.Conv2d(10, 64, 3, stride=2, padding=1), # changed from 32 to 10
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, 3, stride=2, padding=1, bias=False),
        )

        self.find_mu = nn.Sequential(
            nn.Conv2d(2, 1, 3, stride=2, padding=1, bias=False),
        )


        self.find_sigma = nn.Sequential(
            nn.Conv2d(2, 1, 3, stride=2, padding=1, bias=False),
        )

    # pylint: disable=arguments-differ
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        assert len(frames.shape) == 5, str(frames.shape) # Batch X Seq Len X Num of Channels X Width X Height

        # Batch X (Seq Len * Num of Channels) X Width X Height
        stacked_frames = frames.view(frames.size(0), -1,
                        frames.size(-2), frames.size(-1))

        output = self.features(stacked_frames)
        mu = self.find_mu(output)

        m = nn.Softplus()
        sigma = self.find_sigma(output)
        sigma = m(sigma) # constrain sigma to positive

        sigma = torch.nan_to_num(sigma, nan = 1.0) # catch nan problems

        return mu, sigma

class LatentVariableSampler:
    def __init__(self):
        pass

    def sample(self, mu, sigma) -> torch.Tensor:
        """
        :param mu: the Gaussian parameter tensor of shape (B, 1, 8, 8)
        :return: of shape (B, 1, 8, 8)
        """
        z = torch.normal(mu, sigma)
        return z

    def sample_prior(self, shape): 
        prior = torch.distributions.Normal(0, 1)
        z = prior.sample(shape)
        return z


if __name__ == "__main__":
    # Create tensor of size (BS X Seq Len X Channels X NC X W X H)
    input_tensor = torch.rand(28, 10, 1, 64, 64)

    stacked_tensor = input_tensor.view(input_tensor.size(0), -1,
                                input_tensor.size(-2), input_tensor.size(-1))

    q_net = PosteriorInferenceNet(tbatch = 10) # temporal batch = 10
    mu, sigma = q_net(input_tensor)

    sampler = LatentVariableSampler()
    z = sampler.sample(mu, sigma)

    print(mu.shape)
    print(sigma.shape)
    print(z.shape) # batch size X 1 X 8 X 8