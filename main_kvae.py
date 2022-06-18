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

        self.model = KalmanVAE(self.args.x_dim, self.args.a_dim, self.args.z_dim, self.args.K, self.args.device, self.args.beta).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
    
        # for name, param in self.model.named_parameters(): 
        #     if param.requires_grad:
        #         print(name)

        if state_dict_path: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            self.model.load_state_dict(state_dict)

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting KVAE training for {self.args.epochs} epochs.")

        logging.info("Train Loss, KLD, MSE") # header for losses

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_kld = 0
            running_recon = 0

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                #forward + backward + optimize
                self.optimizer.zero_grad()
                loss, kld_loss, recon_loss, *_ = self.model(data)
                loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                # forward pass
                print(f"Loss: {loss}")
                print(f"KLD: {kld_loss}")
                print(f"Reconstruction Loss: {recon_loss}") 

                n_iterations += 1
                running_loss += loss.item()
                running_kld += kld_loss.item()
                running_recon +=  recon_loss.item() # non-weighted by beta

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

        logging.info("Finished training.")

        final_checkpoint = self._save_model(epoch)
        logging.info(f'Saved model. Final Checkpoint {final_checkpoint}')

    def _save_model(self, epoch):  
        checkpoint_path = f'saves/kvae/{self.args.subdirectory}/beta={self.args.beta}/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        filename = f'kvae_state_dict_beta={self.args.beta}_{epoch}.pth'
        checkpoint_name = checkpoint_path + filename

        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to ', checkpoint_name)
        
        return checkpoint_name


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default="KVAE", type=str)
parser.add_argument('--subdirectory', default="finetuned2", type=str)

parser.add_argument('--x_dim', default=1, type=int)
parser.add_argument('--a_dim', default=2, type=int)
parser.add_argument('--z_dim', default=4, type=int)
parser.add_argument('--K', default=3, type=int)

parser.add_argument('--clip', default=10, type=int)
parser.add_argument('--beta', default=1, type=float)

parser.add_argument('--save_every', default=10, type=int) 
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=32, type=int)

state_dict_path = "saves/kvae/finetuned1/beta=0.0/kvae_state_dict_beta=0.0_25.pth" 

def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set up logging
    log_fname = f'{args.model}_beta={args.beta}_{args.epochs}.log'
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



