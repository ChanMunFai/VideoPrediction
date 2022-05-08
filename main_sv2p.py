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
from model import VRNN
from data.MovingMNIST import MovingMNIST

class SV2PTrainer:
    """
    In the midst of major editing without beiung too complicated.

    Let me try to train both CDNA and everything together
    """
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.optimizer = kwargs['optimizer']
        self.writer = SummaryWriter()
        self.model = kwargs["model"]

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting SV2P training for {self.args.epochs} epochs.")

        logging.info("Train Loss, KLD, Reconstruction Loss") # header for losses






class VRNNTrainer:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.optimizer = kwargs['optimizer']
        self.writer = SummaryWriter()
        self.model = kwargs["model"]

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting VRNN training for {self.args.epochs} epochs.")

        logging.info("Train Loss, KLD, Reconstruction Loss") # header for losses

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
                loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                # forward pass
                print(f"Loss: {loss}")
                print(f"KLD: {kld_loss}")
                print(f"Reconstruction Loss: {nll_loss}") # non-weighted by beta

                n_iterations += 1
                running_loss += loss.item()
                running_kld += kld_loss.item()
                running_nll +=  nll_loss.item() # non-weighted by beta

            training_loss = running_loss/len(train_loader)
            training_kld = running_kld/len(train_loader)
            training_nll = running_nll/len(train_loader)

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n KLD Loss: {training_kld}\
                    \n Reconstruction Loss: {training_nll}")
            logging.info(f"{training_loss:.3f}, {training_kld:.3f}, {training_nll:.3f}")

            if epoch % self.args.save_every == 0:
                checkpoint_name = f'saves/{self.args.version}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_{epoch}.pth'
                torch.save(self.model.state_dict(), checkpoint_name)
                print('Saved model to '+checkpoint_name)

        logging.info("Finished training.")

        # Save model
        checkpoint_name = f'saves/{self.args.version}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_{epoch}.pth'
        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to '+checkpoint_name)
        logging.info('Saved model to '+checkpoint_name)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default="VRNN", type=str)
parser.add_argument('--version', default="v1", type=str)

# These arguments can only be changed if we use a model version that is not v1 or v0
parser.add_argument('--xdim', default=64, type=int)
parser.add_argument('--hdim', default=1024, type=int)
parser.add_argument('--zdim', default=32, type=int)
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--clip', default=10, type=int)

parser.add_argument('--beta', default=1, type=float)

parser.add_argument('--save_every', default=100, type=int) # seems like not working

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=50, type=int)


def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set up logging
    log_fname = f'{args.model}_{args.version}_beta={args.beta}_{args.epochs}.log'
    log_dir = f"logs/{args.model}/{args.version}/"
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
        else:
            pass

    logging.info(args)

    # Datasets
    train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True)

    # Load in model
    if args.model == "VRNN":
        VRNNTrainer
        model = VRNN(args.xdim, args.hdim, args.zdim, args.nlayers)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        trainer = VRNNTrainer(args=args, model=model, optimizer=optimizer)

    trainer.train(train_loader)

if __name__ == "__main__":
    main()



