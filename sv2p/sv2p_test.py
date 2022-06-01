import os
import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_msssim import ssim
import numpy as np
import torch 
from utils import checkdir

from sv2p.cdna import CDNA 
from sv2p.model_sv2p import PosteriorInferenceNet, LatentVariableSampler

seed = 128
torch.manual_seed(seed)

batch_size = 3

def test_prior(): 
    sampler = LatentVariableSampler()
    sampler.using_prior = True

    prior_mean = torch.zeros(batch_size, 1, 8, 8)
    prior_std = torch.ones(batch_size, 1, 8, 8) # wrong - should be isotropic 

    for i in range(3): 
        z = sampler.sample_prior((batch_size, 1, 8, 8))
        print(z[0][0][0][0])


