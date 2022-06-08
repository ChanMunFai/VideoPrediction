import os
import math
import logging
import argparse
from pprint import pprint
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from data.MovingMNIST import MovingMNIST
from sv2p.cdna import CDNA # network for CDNA
from sv2p.model_sv2p import PosteriorInferenceNet, LatentVariableSampler
from scheduler import LinearScheduler

class SV2PPosteriorTrainer:
    """
    state_dict_path_stoc: path for state dictionary for a stochastic model 

    Assume that we are already in Stage 3, and have a trained CDNA network. Freeze 
    the CDNA network and only train the posterior. 
    """
    def __init__(self, 
                state_dict_path_stoc = None,
                *args, **kwargs):

        self.args = kwargs['args']
        self.writer = SummaryWriter(f"runs/sv2p/Posterior/")

        self.stoc_model = CDNA(in_channels = 1, cond_channels = 1,
            n_masks = 10).to(self.args.device) # stochastic

        state_dict = torch.load(state_dict_path_stoc, map_location = self.args.device)
        self.stoc_model.load_state_dict(state_dict)

        # Freeze CDNA 
        for param in self.stoc_model.parameters():
            param.requires_grad = False
        
        # Posterior network
        self.q_net = PosteriorInferenceNet(tbatch = 10).to(self.args.device) # figure out what tbatch is again (seqlen?)
        self.sampler = LatentVariableSampler()
 
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                        lr=self.args.learning_rate) # only optimise posterior 

        self.criterion = nn.MSELoss(reduction = 'sum').to(self.args.device) # image-wise MSE

    def _split_data(self, data):
        """ Splits sequence of video frames into inputs and targets

        Both have shapes (Batch_Size X Seq_Len–1 X
                            Num_Channels X Height X Width)

        data: Batch Size X Seq Length X Channels X Height X Width

        Inputs: x_0 to x_T-1
        Targets: x_1 to x_T
        """

        inputs = data[:, :-1, :, :, :]
        targets = data[:, 1:, :, :, :]

        assert inputs.shape == targets.shape
        return inputs, targets

    def train(self, train_loader):
        """
        """

        logging.info(f"Starting SV2P training on Posterior Network.")
        logging.info("Train Loss, KLD, MSE") # header for losses

        steps = 0

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_kld = 0
            running_recon = 0

            for data, _ in tqdm(train_loader):
                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                self.optimizer.zero_grad()

                # Separate data into inputs and targets
                # inputs, targets are both of size: Batch Size X Seq Length - 1 X Channels X Height X Width
                inputs, targets = self._split_data(data)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                hidden = None
                recon_loss = 0.0
                total_loss = 0.0

                # Sample latent variable z from posterior - same z for all time steps
                mu, sigma = self.q_net(data) 
                z = self.sampler.sample(mu, sigma).to(self.args.device) # to be updated with global z 
                
                prior_mean = torch.full_like(mu, 0).to(self.args.device)
                prior_std = torch.full_like(sigma, 1).to(self.args.device) 
                
                p = torch.distributions.Normal(mu,sigma)
                q = torch.distributions.Normal(prior_mean,prior_std)

                kld_loss = torch.distributions.kl_divergence(p, q).sum()/self.args.batch_size
                print("KLD Divergence is", kld_loss)

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    predictions_t, hidden, _, _ = self.stoc_model(
                                                inputs = x_t,
                                                conditions = z,
                                                hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1

                    recon_loss += loss_t/inputs.size(0) # image-wise MSE summed over all time steps

                print("recon_loss", recon_loss)
                total_loss += recon_loss
                total_loss += self.args.beta * kld_loss

                print("Total loss after KLD", total_loss)
                    
                self.optimizer.zero_grad()
                total_loss.backward() 
                self.optimizer.step()

                running_loss += total_loss.item()

                self.writer.add_scalar('Loss/Total Loss',
                                        total_loss,
                                        steps)

                running_kld += kld_loss.item()
                running_recon +=  recon_loss.item()
                self.writer.add_scalar('Loss/MSE',
                                    recon_loss,
                                    steps)
                self.writer.add_scalar('Loss/KLD Loss',
                                    kld_loss,
                                    steps)
                steps += 1

            training_loss = running_loss/len(train_loader)
            training_kld = running_kld/len(train_loader)
            training_recon = running_recon/len(train_loader)
  
            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n KLD Loss: {training_kld}\
                    \n Reconstruction Loss: {training_recon}")
                
            logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_recon:.8f}")

            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

        logging.info('Finished training')
        
        self._save_model(epoch)
        logging.info('Saved model. Final Checkpoint.')

    def _save_model(self, epoch):
        checkpoint_path = f'saves/sv2p/posterior/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_filename = f'sv2p_posterior_state_dict_beta={self.args.beta}_{epoch}.pth'
        checkpoint_name = checkpoint_path + checkpoint_filename

        torch.save(self.q_net.state_dict(), checkpoint_name)
        print('Saved Posterior model to '+checkpoint_name)
        

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)

parser.add_argument('--model', default="cdna", type=str)
parser.add_argument('--beta', default=0.001, type=float)

parser.add_argument('--save_every', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-6, type=float)
parser.add_argument('--batch_size', default=52, type=int)
parser.add_argument('--clip', default=10, type=int)

# Load in model
state_dict_path_stoc = "saves/sv2p/stage3/final_beta=0.001/sv2p_state_dict_550.pth"

def main():
    seed = 128
    torch.manual_seed(seed)
    EPS = torch.finfo(torch.float).eps # numerical logs

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # Set up logging
    log_fname = f'{args.model}_posterior_beta={args.beta}_{args.epochs}.log'
    log_dir = f"logs/{args.model}/posterior/"
    
    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)

    # Datasets
    train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True)


    if args.model == "cdna":
        trainer = SV2PPosteriorTrainer(state_dict_path_stoc, args=args)  
        trainer.train(train_loader)
        
    logging.info(f"Completed posterior training")

if __name__ == "__main__":
    main()



