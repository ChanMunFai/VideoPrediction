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
from vrnn.model_vrnn import VRNN
from data.MovingMNIST import MovingMNIST

class VRNNTrainer:
    def __init__(self, state_dict_path = None, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()

        self.model = VRNN(self.args.xdim, self.args.hdim, self.args.zdim, self.args.nlayers).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.5)

        if state_dict: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            self.model.load_state_dict(state_dict)

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting VRNN training for {self.args.epochs} epochs.")

        logging.info("Train Loss, KLD, MSE") # header for losses

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_kld = 0
            running_nll = 0

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                #forward + backward + optimize
                self.optimizer.zero_grad()
                kld_loss, nll_loss, _ = self.model(data)
                loss = self.args.beta * kld_loss + nll_loss
                # loss = self.args.beta * kld_loss # ignore reconstruction loss (for debugging only)
                loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                # forward pass
                print(f"Loss: {loss}")
                print(f"KLD: {kld_loss}")
                print(f"Reconstruction Loss: {nll_loss}") # non-weighted by beta
                print(f"Learning rate: {self.scheduler.get_last_lr()}") 

                n_iterations += 1
                running_loss += loss.item()
                running_kld += kld_loss.item()
                running_nll +=  nll_loss.item() # non-weighted by beta

                self.scheduler.step()

            training_loss = running_loss/len(train_loader)
            training_kld = running_kld/len(train_loader)
            training_nll = running_nll/len(train_loader)

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n KLD Loss: {training_kld}\
                    \n Reconstruction Loss: {training_nll}")
            logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_nll:.8f}")

            if epoch % self.args.save_every == 0:
                checkpoint_name = f'saves/{self.args.version}/finetuned4/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_step={self.args.step_size}_{epoch}.pth'
                torch.save(self.model.state_dict(), checkpoint_name)
                print('Saved model to '+checkpoint_name)

        logging.info("Finished training.")

        # Save model
        checkpoint_name = f'saves/VRNN/{self.args.version}/{self.args.subdirectory}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_step={self.args.step_size}_{epoch}.pth'
        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to '+checkpoint_name)
        logging.info('Saved model to '+checkpoint_name)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default="VRNN", type=str)
parser.add_argument('--version', default="v1", type=str)
parser.add_argument('--subdirectory', default="finetuned4", type=str)

# These arguments can only be changed if we use a model version that is not v1 or v0
parser.add_argument('--xdim', default=64, type=int)
parser.add_argument('--hdim', default=1024, type=int)
parser.add_argument('--zdim', default=32, type=int)
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--clip', default=10, type=int)

parser.add_argument('--beta', default=1, type=float)

parser.add_argument('--save_every', default=100, type=int) 
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--step_size', default = 1000000, type = int)

state_dict_path = "saves/v1/important/vrnn_state_dict_v1_beta=0.5_400.pth"  # None otherwise 

def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set up logging
    log_fname = f'{args.model}_{args.version}_beta={args.beta}_step={args.step_size}_{args.epochs}.log'
    log_dir = f"logs/{args.model}/{args.version}/{args.subdirectory}/"
    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)

    # Default values for VRNN model versions
    if args.model == "VRNN":
        if args.version == "v0":
            args.xdim = 64
            args.hdim = 1024
            args.zdim = 32
            args.nlayers = 3
            args.clip = 10
        elif args.version == "v1":
            args.xdim = 64
            args.hdim = 1024
            args.zdim = 32
            args.nlayers = 1
            args.clip = 10
        elif args.version == "v2": 
            args.xdim = 64
            args.hdim = 64
            args.zdim = 32
            args.nlayers = 1
            args.clip = 10
        else:
            pass

    logging.info(args)

    # Datasets
    train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True)

    if args.model == "VRNN":
        trainer = VRNNTrainer(state_dict_path= state_dict_path, args=args)

    trainer.train(train_loader)

if __name__ == "__main__":
    main()



