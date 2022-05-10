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
from sv2p.model import PosteriorInferenceNet, LatentVariableSampler


class CDNATrainer:
    """
    CDNATrainer is only used to train the generator.

    We need an argument to determine if network is stochastic (Stage 2)
    or deterministic (Stage 1)

    We can have another trainer to train all 3 Stages at once
    """

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()

        # Define model directly in trainer class
        self.model = CDNA(in_channels = 1, cond_channels = 0,
            n_masks = 10).to(self.args.device) # deterministic

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=self.args.learning_rate)
        self.criterion = nn.MSELoss().to(self.args.device)

        # Posterior network
        self.q_net = PosteriorInferenceNet(tbatch = 10).to(self.args.device) # figure out what tbatch is again (seqlen?)
        self.sampler = LatentVariableSampler()

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


    def train_stage1(self, train_loader):
        """ Trains generator deterministically.

        Generator (CDNA architecture) does not use any latent variables.

        KL divergence is not used.
        """
        try:
            self.args.stage1_epochs
        except:
            self.args.stage1_epochs = self.args.epochs

        logging.info(f"Starting SV2P training on Stage 1 for {self.args.stage1_epochs} epochs.")
        logging.info("Train Loss") # header for losses

        for epoch in range(self.args.stage1_epochs):
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
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

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

                # Add in gradient clipping
                running_loss += total_loss.item()

            training_loss = running_loss/len(train_loader)
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.3f}")

            # Save model
            checkpoint_name = f'saves/cdna_state_dict_{self.args.stage1_epochs}.pth'
            torch.save(self.model.state_dict(), checkpoint_name)
            print('Saved model to '+checkpoint_name)
            logging.info('Saved model to '+checkpoint_name)

    def copy_state_dict(self, model1, model2):
        model1_state_dict = model1.state_dict()
        model2_state_dict = model2.state_dict()

        for name, param in model1_state_dict.items():
            if name == "u_lstm.conv4.weight" or "u_lstm.conv4.bias":
                pass
            else:
                # param = param.data
                model2.state_dict()[name].copy_(param)

        # Does not seem to copy exactly
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            print(torch.equal(p1, p2))


    def train_stage2(self, train_loader):
        """ Trains generator stochastically.

        Generator (CDNA architecture) uses latent variables which are
        sampled from the posterior at training time.

        KL divergence is not used.

        # Not yet implemented
        - Sample latent variable for each time step
        """

        # create a new model - model 2
        # copy parameters from model 1 - everything is the same except self.conv4
        self.model2 = CDNA(in_channels = 1, cond_channels = 1,
            n_masks = 10).to(self.args.device) # stochastic
        self.copy_state_dict(self.model, self.model2)

        # Reinitialise optimiser - not sure if this is right
        self.optimizer = torch.optim.Adam(self.model2.parameters(),
                                            lr=self.args.learning_rate)

        try:
            self.args.stage2_epochs
        except:
            self.args.stage2_epochs = self.args.epochs

        logging.info(f"Starting SV2P training on Stage 2 for {self.args.stage2_epochs} epochs.")
        logging.info("Train Loss") # header for losses

        for epoch in range(self.args.stage2_epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                self.optimizer.zero_grad()
                inputs, targets = self._split_data(data)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                hidden = None
                loss = 0.0

                # Sample latent variable z from posterior - same z for all time steps
                mu, sigma = self.q_net(data) # input is full sequence of video frames

                z = self.sampler.sample(mu, sigma).to(self.args.device)

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    # Output x_t+1 hat given x_t
                    predictions_t, hidden, _, _ = self.model2(
                                                inputs = x_t,
                                                conditions = z,
                                                hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    loss += loss_t

                total_loss = loss / inputs.size(1)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # add in gradient clipping in future

                running_loss += total_loss.item()

            training_loss = running_loss/len(train_loader)
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.3f}")

    def train_stage3(self, train_loader):

        # Use model 2 - stochastic
        try:
            self.model2
        except:
            self.model2 = CDNA(in_channels = 1, cond_channels = 1,
                                n_masks = 10) # stochastic

        self.model2 = self.model2.to(self.args.device)

        # Reinitialise optimiser - not sure if this is right
        self.optimizer = torch.optim.Adam(self.model2.parameters(),
                                            lr=self.args.learning_rate)

        try:
            self.args.stage3_epochs
        except:
            self.args.stage3_epochs = self.args.epochs

        logging.info(f"SV2P training on Stage 3 for {self.args.stage3_epochs} epochs.")
        logging.info("Train Loss") # header for losses

        for epoch in range(self.args.stage3_epochs):
            print("Epoch:", epoch)
            running_loss = 0 # loss per epoch

            for data, _ in tqdm(train_loader):

                data = data.to(self.args.device)
                data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
                data = (data - data.min()) / (data.max() - data.min())

                self.optimizer.zero_grad()
                inputs, targets = self._split_data(data)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                hidden = None
                loss = 0.0 # loss per batch

                # Sample latent variable z from posterior - same z for all time steps
                mu, sigma = self.q_net(data) # input is full sequence of video frames
                z = self.sampler.sample(mu, sigma).to(self.args.device)
                prior_mean = torch.full_like(mu, 0)
                prior_std = torch.full_like(sigma, 1)

                kld_loss = self._kld_gauss(mu, sigma, prior_mean, prior_std)

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    # Output x_t+1 hat given x_t
                    predictions_t, hidden, _, _ = self.model2(
                                                inputs = x_t,
                                                conditions = z,
                                                hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    loss += loss_t

                total_loss = loss / inputs.size(1)
                total_loss += kld_loss
                print(total_loss) # loss normalised by seq len per batch
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # add in gradient clipping in future

                running_loss += total_loss.item()

            training_loss = running_loss/len(train_loader) # loss per epoch
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.3f}")


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        EPS = torch.finfo(torch.float).eps # numerical logs

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--stage1_epochs', default=50, type=int)
parser.add_argument('--stage2_epochs', default=1, type=int)
parser.add_argument('--stage3_epochs', default=1, type=int)

parser.add_argument('--model', default="cdna", type=str)
parser.add_argument('--version', default="v1", type=str)

parser.add_argument('--save_every', default=100, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=52, type=int)


def main():
    seed = 128
    torch.manual_seed(seed)
    EPS = torch.finfo(torch.float).eps # numerical logs

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
        trainer = CDNATrainer(args=args)
        trainer.train_stage1(train_loader) # train CDNA only
        # trainer.train_stage2(train_loader)
        # trainer.train_stage3(train_loader)

    logging.info("Completed training")

if __name__ == "__main__":
    main()



