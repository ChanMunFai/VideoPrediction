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

        self.criterion = nn.MSELoss().to(self.args.device)

    def train_once(self, inputs, targets):
        """
        Adapted from: https://github.com/kkew3/cse291g-sv2p/blob/master/src/train/train_cdna.py

        :param inputs: of shape (self.batch_size, self.seqlen-1, 1,  64, 64)
        :param targets: of shape (self.batch_size, self.seqlen-1, 1,  64, 64)
        """

        batch_size, seqlen = inputs.size(0), inputs.size(1) # swapped accd to my own implementation

        hidden = None
        loss = 0.0

        for t in range(seqlen):
            x_t = inputs[:,t,:,:,:]
            # targets_t = targets[:,t,:,:,:]

            predictions_t, hidden, cdna_kerns_t, masks_t = self.model(
                    x_t, hidden_states=hidden)

            loss_t, _ , _ = self.__compute_loss(
                predictions_t, cdna_kerns_t, masks_t, targets_t)

            loss += loss_t

        total_loss = loss / seqlen
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item() / seqlen

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


    def train_1(self, train_loader):
        logging.info(f"Starting CDNA training for {self.args.epochs} epochs.")
        logging.info("Train Loss") # header for losses

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                self.optimizer.zero_grad()

                # Separate data into inputs and targets
                # inputs, targets are both of size: Batch Size X Seq Length - 1 X Channels X Height X Width
                inputs, targets = self._split_data(data)

                hidden = None
                loss = 0.0

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    # Output x_t+1 hat given x_t
                    predictions_t, hidden, _, _ = self.model(
                                                x_t, hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    print(loss_t)
                    loss += loss_t

                total_loss = loss / inputs.size(1)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                running_loss += total_loss.item()

            training_loss = running_loss/len(train_loader)
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.3f}")

    def train_2(self, train_loader):
        logging.info(f"Starting CDNA training for {self.args.epochs} epochs.")
        logging.info("Train Loss") # header for losses

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                self.optimizer.zero_grad()
                inputs, targets = self._split_data(data)

                hidden = None
                loss = 0.0

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    # Output x_t+1 hat given x_t
                    predictions_t, hidden, _, _ = self.model(
                                                x_t, hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    print(loss_t)
                    loss += loss_t

                total_loss = loss / inputs.size(1)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                running_loss += total_loss.item()

            training_loss = running_loss/len(train_loader)
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.3f}")



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



