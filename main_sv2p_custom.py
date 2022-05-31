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
from sv2p.model import PosteriorInferenceNet, LatentVariableSampler


class SV2PTrainer:
    """
    state_dict_path_det: path for state dictionary for a deterministic model 
    state_dict_path_stoc: path for state dictionary for a stochastic model 
    """
    def __init__(self, 
                state_dict_path_det = None, 
                state_dict_path_stoc = None, 
                *args, **kwargs):

        self.args = kwargs['args']
        self.writer = SummaryWriter(f"runs/sv2p/Stage{self.args.stage}/")

        assert state_dict_path_det = None or state_dict_path_stoc = None

        self.det_model = CDNA(in_channels = 1, cond_channels = 0,
            n_masks = 10).to(self.args.device) # deterministic
        if state_dict_path_det: 
            state_dict = torch.load(state_dict_path_det, map_location = self.args.device)
            self.det_model.load_state_dict(state_dict)

        self.stoc_model = CDNA(in_channels = 1, cond_channels = 1,
            n_masks = 10).to(self.args.device) # stochastic
        if state_dict_path_stoc: 
            state_dict = torch.load(state_dict_path_stoc, map_location = self.args.device)
            self.stoc_model.load_state_dict(state_dict)
        elif state_dict_path_det: 
            self.load_stochastic_model() # load deterministic layers into stochastic model 
        
        if self.args.stage == 1: 
            self.optimizer = torch.optim.Adam(self.det_model.parameters(),
                                            lr=self.args.learning_rate)
        else: 
            self.optimizer = torch.optim.Adam(self.stoc_model.parameters(),
                                            lr=self.args.learning_rate)

        self.criterion = nn.MSELoss(reduction = 'sum').to(self.args.device) # image-wise MSE

        # Posterior network
        self.q_net = PosteriorInferenceNet(tbatch = 10).to(self.args.device) # figure out what tbatch is again (seqlen?)
        self.sampler = LatentVariableSampler()

        ## Add in logging settings 

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
        stage: 1, 2, or 3
        """

        logging.info(f"Starting SV2P training on Stage {self.args.stage} for {self.args.epochs} epochs.")
        if self.args.stage == 1 or self.args.stage == 2: 
            logging.info("Train Loss") # header for losses
        elif self.args.stage == 3: 
            logging.info("Train Loss, MSE, KLD") # header for losses

        steps = 0

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
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                hidden = None
                loss = 0.0

                # Sample latent variable z from posterior - same z for all time steps
                mu, sigma = self.q_net(data) 
                z = self.sampler.sample(mu, sigma).to(self.args.device) # to be updated with global z 
                prior_mean = torch.full_like(mu, 0)
                prior_std = torch.full_like(sigma, 1)
                kld_loss = self._kld_gauss(mu, sigma, prior_mean, prior_std) # 1 KLD across all time steps

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    if self.args.stage == 1: 
                        predictions_t, hidden, _, _ = self.det_model(
                                                x_t, hidden_states=hidden)

                    else: 
                        predictions_t, hidden, _, _ = self.stoc_model(
                                                    inputs = x_t,
                                                    conditions = z,
                                                    hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    print(f"Image-wise MSE at time step {t} is {loss_t/inputs.size(0)}.")
                    loss += loss_t/inputs.size(0) # image-wise MSE summed over all time steps

                if self.args.stage == 3: 
                    loss += kld_loss

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()

                if self.args.stage == 1: 
                    nn.utils.clip_grad_norm_(self.det_model.parameters(), self.args.clip)
                else: 
                    nn.utils.clip_grad_norm_(self.stoc_model.parameters(), self.args.clip)
                    
                running_loss += loss.item()
                self.writer.add_scalar('Loss/MSE',
                                        loss,
                                        steps)

                steps += 1

            training_loss = running_loss/len(train_loader)
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.8f}")

            if epoch % self.args.save_every == 0:
                checkpoint_name = f'saves/cdna/cdna_state_dict_{epoch}.pth'
                torch.save(self.det_model.state_dict(), checkpoint_name)
                print('Saved model to '+checkpoint_name)

        logging.info('Finished training')
        
        checkpoint_name = f'saves/cdna/cdna_state_dict_{epoch}.pth'
        torch.save(self.det_model.state_dict(), checkpoint_name)
        logging.info('Saved model to '+checkpoint_name)


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

        steps = 0

        for epoch in range(self.args.stage1_epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            steps = 0 

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
                    predictions_t, hidden, _, _ = self.det_model(
                                                x_t, hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    print(f"Image-wise MSE at time step {t} is {loss_t}.")
                    loss += loss_t/inputs.size(0) # divide by batch size 

                self.optimizer.zero_grad()
                loss.backward() # image-wise MSE summed over all time steps
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.det_model.parameters(), self.args.clip)

                running_loss += loss.item()
                self.writer.add_scalar('Loss/MSE',
                                        loss,
                                        steps)

                steps += 1

            training_loss = running_loss/len(train_loader)
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.8f}")

            if epoch % self.args.save_every == 0:
                checkpoint_name = f'saves/cdna/cdna_state_dict_{epoch}.pth'
                torch.save(self.det_model.state_dict(), checkpoint_name)
                print('Saved model to '+checkpoint_name)

        logging.info('Finished training')
        
        checkpoint_name = f'saves/cdna/cdna_state_dict_{epoch}.pth'
        torch.save(self.det_model.state_dict(), checkpoint_name)
        logging.info('Saved model to '+checkpoint_name)
        
    def copy_state_dict(self, model1, model2):

        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 != "u_lstm.conv4.weight" and name1 != "u_lstm.conv4.bias":
                dict_params2[name1].data.copy_(param1.data)

        def test_copying():
            f = open("param_model1.txt", "w")
            f.write("#### Model 1 Parameters ####")

            for name, param in model1.named_parameters():
                f.write("\n")
                f.write(str(name))
                f.write(str(param))
                f.write("\n")

            f.close()

            f = open("param_model2.txt", "w")
            f.write("#### Model 2 Parameters ####")

            for name, param in model2.named_parameters():
                f.write("\n")
                f.write(str(name))
                f.write(str(param))
                f.write("\n")
                
            f.close()

        # test_copying()

    def load_stochastic_model(self): 

        self.copy_state_dict(self.det_model, self.stoc_model)
        
    def train_stage2(self, train_loader):
        """ Trains generator stochastically.

        Generator (CDNA architecture) uses latent variables which are
        sampled from the posterior at training time.

        KL divergence is not used.

        # Not yet implemented
        - Sample latent variable for each time step
        """
        self.writer = SummaryWriter("runs/sv2p/Stage2")

        self.stoc_model = CDNA(in_channels = 1, cond_channels = 1,
            n_masks = 10).to(self.args.device) # stochastic

        self.load_stochastic_model()
        self.optimizer = torch.optim.Adam(self.stoc_model.parameters(),
                                            lr=self.args.learning_rate)

        try:
            self.args.stage2_epochs
        except:
            self.args.stage2_epochs = self.args.epochs

        logging.info(f"Starting SV2P training on Stage 2 for {self.args.stage2_epochs} epochs.")
        logging.info("Train Loss") # header for losses

        steps = 0

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
                    predictions_t, hidden, _, _ = self.stoc_model(
                                                inputs = x_t,
                                                conditions = z,
                                                hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    loss += loss_t/inputs.size(0) 

                self.optimizer.zero_grad()
                loss.backward() # image-wise MSE for all time steps 
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.stoc_model.parameters(), self.args.clip)

                running_loss += loss.item()
                self.writer.add_scalar('Loss/MSE',
                                        loss,
                                        steps)

                steps += 1

            training_loss = running_loss/len(train_loader)
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.8f}")

            if epoch % self.args.save_every == 0:
                checkpoint_name = f'saves/sv2p/stage2/sv2p_state_dict_{epoch}.pth' # sv2p instead of cdna
                torch.save(self.stoc_model.state_dict(), checkpoint_name)
                print('Saved model to '+checkpoint_name)

        logging.info('Finished training')
        
        checkpoint_name = f'saves/sv2p/stage2/sv2p_state_dict_{epoch}.pth'
        torch.save(self.stoc_model.state_dict(), checkpoint_name)
        logging.info('Saved model to '+checkpoint_name)

    def train_stage3(self, train_loader):
        self.writer = SummaryWriter("runs/sv2p/Stage3s")
        self.stoc_model = CDNA(in_channels = 1, cond_channels = 1,
                                n_masks = 10).to(self.args.device) 

        self.optimizer = torch.optim.Adam(self.stoc_model.parameters(),
                                            lr=self.args.learning_rate)

        try:
            self.args.stage3_epochs
        except:
            self.args.stage3_epochs = self.args.epochs

        logging.info(f"SV2P training on Stage 3 for {self.args.stage3_epochs} epochs.")
        logging.info("Train Loss") # header for losses

        steps = 0 

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
                    loss += loss_t/inputs.size(0)

                loss += kld_loss
                print(loss) # loss for entire sequence
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.stoc_model.parameters(), self.args.clip)

                running_loss += loss.item()
                self.writer.add_scalar('Loss/MSE',
                                        loss,
                                        steps)
                steps += 1

                running_loss += loss.item()

            training_loss = running_loss/len(train_loader) # loss per epoch
            print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
            logging.info(f"{training_loss:.8f}")

            if epoch % self.args.save_every == 0:
                checkpoint_name = f'saves/sv2p/stage3/sv2p_state_dict_{epoch}.pth' # sv2p instead of cdna
                torch.save(self.stoc_model.state_dict(), checkpoint_name)
                print('Saved model to '+checkpoint_name)

        logging.info('Finished training')
        
        checkpoint_name = f'saves/sv2p/stage3/sv2p_state_dict_{epoch}.pth'
        torch.save(self.stoc_model.state_dict(), checkpoint_name)
        logging.info('Saved model to '+checkpoint_name)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        EPS = torch.finfo(torch.float).eps # numerical logs

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--stage1_epochs', default=0, type=int)
parser.add_argument('--stage2_epochs', default=0, type=int)
parser.add_argument('--stage3_epochs', default=0, type=int)

parser.add_argument('--model', default="cdna", type=str)
parser.add_argument('--version', default="v1", type=str)
parser.add_argument('--stage', default=2, type=int)

parser.add_argument('--save_every', default=100, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=52, type=int)

# Load in model
state_dict_path = "saves/cdna/cdna_state_dict_50.pth" 

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
    log_fname = f'{args.model}_stage={args.stage}_{args.epochs}.log'
    log_dir = f"logs/{args.model}/stage{args.stage}/"
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
        args.clip = 10 # gradient clipping 
        trainer = CDNATrainer(state_dict_path, args=args)

        if args.stage == 1: 
            tainer.train_stage1(train_loader)

        elif args.stage == 2: 
            # trainer.load_stochastic_model()
            trainer.train_stage2(train_loader)

        elif args.stage == 3:
            # Load in stochastic model trained from Stage 2
            trainer.train_stage3(train_loader)

        else: 
            print("Invalid stage. Only integers 1, 2, 3 are supported.")

    logging.info(f"Completed {args.stage} training")

if __name__ == "__main__":
    main()



