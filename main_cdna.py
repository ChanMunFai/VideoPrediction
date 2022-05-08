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
from data.MovingMNIST import MovingMNIST
from sv2p.cdna import CDNA # network for CDNA


class CDNATrainer:
    """ In the midst of major refactoring

    CDNATrainer is only used to train the generator.

    We need an argument to determine if network is stochastic (Stage 2)
    or deterministic (Stage 1)

    We can have another trainer to train all 3 Stages at once
    """

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.optimizer = kwargs['optimizer']
        self.writer = SummaryWriter()
        self.model = kwargs["model"] # cdna

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting CDNA training for {self.args.epochs} epochs.")

        logging.info("Train Loss") # header for losses

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                #forward + backward + optimize
                self.optimizer.zero_grad()

                # Output of CDNA model
                gen_images, hidden, cdna_kerns_t, masks_t = self.model(data) # gen_images are defined as predictions_t
                recon_loss = self._nll_bernoulli(gen_images, data) # may be wrong

                loss = recon_loss
                loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                # forward pass
                print(f"Loss: {loss}")

                n_iterations += 1
                running_loss += loss.item()

            training_loss = running_loss/len(train_loader)

            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")

            logging.info(f"{training_loss:.3f}")

            # if epoch % self.args.save_every == 0:
            #     checkpoint_name = f'saves/{self.args.version}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_{epoch}.pth'
            #     torch.save(self.model.state_dict(), checkpoint_name)
            #     print('Saved model to '+checkpoint_name)

        logging.info("Finished training CDNA")

        # Save model
        # checkpoint_name = f'saves/{self.args.version}/vrnn_state_dict_{self.args.version}_beta={self.args.beta}_{epoch}.pth'
        # torch.save(self.model.state_dict(), checkpoint_name)
        # print('Saved model to '+checkpoint_name)
        # logging.info('Saved model to '+checkpoint_name)

    def _nll_bernoulli(self, theta, x):
        EPS = torch.finfo(torch.float).eps # numerical logs
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default="cdna", type=str)
parser.add_argument('--version', default="v1", type=str)

parser.add_argument('--save_every', default=100, type=int) # seems like not working

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=4, type=int)


def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    # set up logging
    log_fname = f'{args.model}_{args.version}_{args.epochs}.log'
    log_dir = f"logs/{args.model}/{args.version}/"
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
        model = CDNA(in_channels = 1, cond_channels = 0,
        n_masks = 10)
        print(model)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        trainer = CDNATrainer(args=args, model=model, optimizer=optimizer)

    trainer.train(train_loader)

if __name__ == "__main__":
    main()



