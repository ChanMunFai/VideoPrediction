import os
import math
import logging
import argparse
from pprint import pprint

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from kvae.model_kvae import KalmanVAE
from data.MovingMNIST import MovingMNIST

class KVAETrainer:
    def __init__(self, state_dict_path = None, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()
        print(self.args)

        self.model = KalmanVAE(self.args.x_dim, self.args.a_dim, self.args.z_dim, self.args.K, self.args.device, self.args.scale).to(self.args.device)
        
        parameters = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) \
                    + list(self.model.parameter_net.parameters()) + list(self.model.alpha_out.parameters()) \
                    + [self.model.a1, self.model.A, self.model.C]
        
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.learning_rate)
    
        if state_dict_path: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            logging.info(f"Loaded State Dict from {state_dict_path}.")
            
            self.model.load_state_dict(state_dict)

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting KVAE training for {self.args.epochs} epochs.")

        logging.info("Train Loss, Reconstruction Loss, log q(a), ELBO_KF, MSE") # header for losses

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_recon = 0
            running_latent_ll = 0 
            running_elbo_kf = 0
            running_mse = 0 

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                # Binarise input data 
                data = torch.where(data > 0.5, 1.0, 0.0)

                #forward + backward + optimize
                self.optimizer.zero_grad()
                loss, recon_loss, latent_ll, elbo_kf, mse  = self.model(data)
                loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                # forward pass
                print(f"Loss: {loss}")
                print(f"Reconstruction Loss: {recon_loss}") 
                print(f"Latent Loglikelihood: {latent_ll}")
                print(f"ELBO KF: {elbo_kf}")
                print(f"MSE: {mse}")

                print("First value of A", self.model.A[0, 0, 0].item())

                n_iterations += 1
                running_loss += loss.item()
                running_recon +=  recon_loss.item() 
                running_latent_ll += latent_ll.item()
                running_elbo_kf += elbo_kf.item()
                running_mse += mse.item()

            training_loss = running_loss/len(train_loader)
            training_recon = running_recon/len(train_loader)
            training_latent_ll = running_latent_ll/len(train_loader)
            training_elbo_kf = running_elbo_kf/len(train_loader)
            training_mse = running_mse/len(train_loader)

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n Reconstruction Loss: {training_recon}\
                    \n Latent Log-likelihood: {training_latent_ll}\
                    \n ELBO Kalman Filter: {training_elbo_kf}\
                    \n MSE: {training_mse}")

            logging.info(f"{training_loss:.8f}, {training_recon:.8f}, {training_latent_ll:.8f}, {training_elbo_kf:.8f}, {training_mse:.8f}")

            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

        logging.info("Finished training.")

        final_checkpoint = self._save_model(epoch)
        logging.info(f'Saved model. Final Checkpoint {final_checkpoint}')

    def _save_model(self, epoch):  
        checkpoint_path = f'saves/kvae/{self.args.subdirectory}/scale={self.args.scale}/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        filename = f'kvae_state_dict_scale={self.args.scale}_{epoch}.pth'
        checkpoint_name = checkpoint_path + filename

        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to ', checkpoint_name)
        
        return checkpoint_name


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default="KVAE", type=str)
parser.add_argument('--subdirectory', default="finetuned3", type=str)

parser.add_argument('--x_dim', default=1, type=int)
parser.add_argument('--a_dim', default=2, type=int)
parser.add_argument('--z_dim', default=4, type=int)
parser.add_argument('--K', default=3, type=int)

parser.add_argument('--clip', default=150, type=int)
parser.add_argument('--scale', default=0.3, type=float)

parser.add_argument('--save_every', default=10, type=int) 
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--batch_size', default=32, type=int)

state_dict_path = "saves/kvae/finetuned2/scale=0.5/kvae_state_dict_scale=0.5_40.pth"
# state_dict_path = None 

def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set up logging
    log_fname = f'{args.model}_scale={args.scale}_{args.epochs}.log'
    log_dir = f"logs/{args.model}/{args.subdirectory}/"
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

    if args.model == "KVAE":
        trainer = KVAETrainer(state_dict_path= state_dict_path, args=args)

    trainer.train(train_loader)

if __name__ == "__main__":
    main()



