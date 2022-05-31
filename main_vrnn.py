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
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter("runs/VRNN")

        self.model = VRNN(self.args.xdim, self.args.hdim, self.args.zdim, self.args.nlayers).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.5)

    def train(self, train_loader):
        logging.info(f"Starting VRNN training for {self.args.epochs} epochs.")
        logging.info("Train Loss, KLD, MSE") # header for losses

        steps = 0

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_kld = 0
            running_recon = 0

            for data, _ in tqdm(train_loader):
                steps += 1

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                #forward + backward + optimize
                self.optimizer.zero_grad()
                kld_loss, recon_loss, _ = self.model(data)
                loss = self.args.beta * kld_loss + recon_loss
                loss.backward()

                # Debug 
                if steps >= self.args.warmup: 
                    self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                # forward pass
                print(f"Loss: {loss}")
                print(f"KLD: {kld_loss}")
                print(f"Reconstruction Loss: {recon_loss}") # non-weighted by beta
                print(f"Learning rate: {self.scheduler.get_last_lr()}") 

                learning_rate = self.scheduler.get_last_lr()

                # Tensorboard integration
                self.writer.add_scalar('Loss/Train Loss',
                                    training_loss,
                                    steps)

                self.writer.add_scalar('Loss/KLD',
                                    kld_loss,
                                    steps)

                self.writer.add_scalar('Loss/Reconstruction Loss',
                                    recon_loss,
                                    steps)

                self.writer.add_scalar('Learning rate',
                                    learning_rate,
                                    steps)

                running_loss += loss.item()
                running_kld += kld_loss.item()
                running_recon +=  recon_loss.item() # non-weighted by beta

                self.scheduler.step()

            training_loss = running_loss/len(train_loader)
            training_kld = running_kld/len(train_loader)
            training_recon = running_recon/len(train_loader)

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n KLD Loss: {training_kld}\
                    \n Reconstruction Loss: {training_recon}")
            logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_recon:.8f}")

            if epoch % self.args.save_every == 0:
                checkpoint_name = f'saves/{self.args.version}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_{epoch}.pth'
                torch.save(self.model.state_dict(), checkpoint_name)
                print('Saved model to '+checkpoint_name)

        logging.info("Finished training.")

        # Save model
        if self.args.warmup > 0: 
            checkpoint_name = f'saves/{self.args.version}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_step={self.args.step_size}_warmup={self.args.warmup}_{epoch}.pth'
        else: 
            checkpoint_name = f'saves/{self.args.version}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_step={self.args.step_size}_{epoch}.pth'
        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to '+checkpoint_name)
        logging.info('Saved model to '+checkpoint_name)


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default="VRNN", type=str)
parser.add_argument('--version', default="v2", type=str)

# These arguments can only be changed if we use a model version that is not v1 or v0
parser.add_argument('--xdim', default=64, type=int)
parser.add_argument('--hdim', default=1024, type=int)
parser.add_argument('--zdim', default=32, type=int)
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--clip', default=10, type=int)

parser.add_argument('--beta', default=1, type=float)

parser.add_argument('--save_every', default=100, type=int) # seems like not working

parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--step_size', default = 30, type = int)
parser.add_argument('--warmup', default = 0, type = int)

def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set up logging
    if args.warmup > 0: 
        log_fname = f'{args.model}_{args.version}_beta={args.beta}_step={args.step_size}_warmup={args.warmup}_{args.epochs}.log'
    else: 
        log_fname = f'{args.model}_{args.version}_beta={args.beta}_step={args.step_size}_{args.epochs}.log'
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
        elif args.version == "v2": 
            args.xdim = 64
            args.hdim = 64
            args.zdim = 32
            args.nlayers = 1
            args.clip = 10
        else:
            pass

    logging.info(args)
    print(args)

    # Datasets
    train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True)

    # Load in model
    if args.model == "VRNN":
        trainer = VRNNTrainer(args=args)

    trainer.train(train_loader)

if __name__ == "__main__":
    main()



