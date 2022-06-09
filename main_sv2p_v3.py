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

class SV2PTrainer:
    """
    state_dict_path_det: path for state dictionary for a deterministic model 
    state_dict_path_stoc: path for state dictionary for a stochastic model 
    state_dict_posterior: path for state dictionary for a posterior model 

    This is differently defined from the original paper. Here, 
        Deterministic model: cond channels = 0 (saved from Stage 0) 
        Stochastic model: cond channels = 1 (saved and used in all other Stages)

    Stages (new): 
        0: train CDNA architecture only 
        1: update CDNA architecture to include Z from prior variables 
        2: Use Z from posterior but do not include KL divergence
        3. Use Z from posterior and include KL divergence 
    """
    def __init__(self, 
                state_dict_path_det = None, 
                state_dict_path_stoc = None,
                state_dict_path_posterior = None, 
                beta_scheduler = None,  
                *args, **kwargs):

        self.args = kwargs['args']
        self.writer = SummaryWriter(f"runs/sv2p/Stage{self.args.stage}/")
        self.beta_scheduler = beta_scheduler

        assert state_dict_path_det == None or state_dict_path_stoc == None

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
        
        # Posterior network
        self.q_net = PosteriorInferenceNet(tbatch = 10).to(self.args.device) # figure out what tbatch is again (seqlen?)
        if state_dict_path_posterior: 
            state_dict_posterior = torch.load(state_dict_path_posterior, map_location = self.args.device)
            self.q_net.load_state_dict(state_dict_posterior)

        self.sampler = LatentVariableSampler()

        if self.args.stage == 2 or self.args.stage == 3: 
            if not state_dict_path_posterior: 
                print("WARNING!: State dict for posterior is not loaded")
                loggin.info("WARNING!: State dict for posterior is not loaded")

        if self.args.stage == 0: 
            self.optimizer = torch.optim.Adam(self.det_model.parameters(),
                                            lr=self.args.learning_rate)
        elif self.args.stage == 1: 
            self.optimizer = torch.optim.Adam(self.stoc_model.parameters(),
                                            lr=self.args.learning_rate)
        elif self.args.stage == 2 or self.args.stage == 3: # Stage 2 trains posterior to give good latents but is unregulated 
            self.optimizer = torch.optim.Adam(list(self.stoc_model.parameters()) + list(self.q_net.parameters()),
                                            lr=self.args.learning_rate)

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
        stage: 1, 2, or 3
        """

        logging.info(f"Starting SV2P training on Stage {self.args.stage} for {self.args.epochs} epochs.")
        if self.args.stage == 0: 
            logging.info("Train Loss") # header for losses
        else: 
            logging.info("Train Loss, KLD, MSE") # only Stage 3 uses KLD but we track KLD for the rest

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
                z = self.sampler.sample(mu, sigma).to(self.args.device) # to be updated with time-variant z 
                
                prior_mean = torch.full_like(mu, 0).to(self.args.device)
                prior_std = torch.full_like(sigma, 1).to(self.args.device) # check if correct 

                if self.args.stage == 1: # use z from prior 
                    mu = prior_mean
                    sigma = prior_std
                
                p = torch.distributions.Normal(mu,sigma)
                q = torch.distributions.Normal(prior_mean,prior_std)

                kld_loss = torch.distributions.kl_divergence(p, q).sum()/self.args.batch_size
                print("KLD Divergence is", kld_loss)

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    if self.args.stage == 0: 
                        predictions_t, hidden, _, _ = self.det_model(
                                                x_t, hidden_states=hidden)

                    else: 
                        predictions_t, hidden, _, _ = self.stoc_model(
                                                    inputs = x_t,
                                                    conditions = z,
                                                    hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    recon_loss += loss_t/inputs.size(0) # image-wise MSE summed over all time steps

                print("recon_loss", recon_loss)
                total_loss += recon_loss

                if self.args.stage == 3: 
                    beta_value = self.beta_scheduler.step()
                    total_loss += beta_value * kld_loss

                print("Total loss after KLD", total_loss)
                    
                self.optimizer.zero_grad()
                total_loss.backward() 
                self.optimizer.step()

                if self.args.stage == 0: 
                    nn.utils.clip_grad_norm_(self.det_model.parameters(), self.args.clip)
                else: 
                    nn.utils.clip_grad_norm_(self.stoc_model.parameters(), self.args.clip)
                    
                running_loss += total_loss.item()

                self.writer.add_scalar('Loss/Total Loss',
                                        total_loss,
                                        steps)

                if self.args.stage != 0: 
                    running_kld += kld_loss.item()
                    running_recon +=  recon_loss.item()
                    self.writer.add_scalar('Loss/MSE',
                                        recon_loss,
                                        steps)
                    self.writer.add_scalar('Loss/KLD Loss',
                                        kld_loss,
                                        steps)
                    if self.args.stage == 3: 
                        self.writer.add_scalar('Beta value',
                                        beta_value,
                                        steps)

                steps += 1

            training_loss = running_loss/len(train_loader)
            training_kld = running_kld/len(train_loader)
            training_recon = running_recon/len(train_loader)

            if self.args.stage == 0: 
                print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
                logging.info(f"{training_loss:.8f}")
            else:
                print(f"Epoch: {epoch}\
                        \n Train Loss: {training_loss}\
                        \n KLD Loss: {training_kld}\
                        \n Reconstruction Loss: {training_recon}")
                
                if self.args.stage != 3:     
                    logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_recon:.8f}")
                else: # only stage 3 needs beta values
                    logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_recon:.8f}, {beta_value:.8f}")

            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

        logging.info('Finished training')
        
        self._save_model(epoch)
        logging.info('Saved model. Final Checkpoint.')

    def _save_model(self, epoch):
        if self.args.stage != 3:  
            checkpoint_path = f'saves/sv2p/stage{self.args.stage}/finetuned2/'
        else: 
            checkpoint_path = f'saves/sv2p/stage{self.args.stage}/final_beta={self.args.beta_end}/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        
        if self.args.stage == 0 or self.args.stage == 1: 
            cdna_filename = f'sv2p_cdna_state_dict_{epoch}.pth'
            checkpoint_name_cdna = checkpoint_path + cdna_filename

            if self.args.stage == 0: 
                torch.save(self.det_model.state_dict(), checkpoint_name_cdna)
            elif self.args.stage == 1:
                torch.save(self.stoc_model.state_dict(), checkpoint_name_cdna)

            print('Saved model to '+checkpoint_name_cdna)
        else: 
            cdna_filename = f'sv2p_cdna_state_dict_{epoch}.pth'
            posterior_filename = f'sv2p_posterior_state_dict_{epoch}.pth'
            checkpoint_name_cdna = checkpoint_path + cdna_filename
            checkpoint_name_posterior = checkpoint_path + posterior_filename

            torch.save(self.stoc_model.state_dict(), checkpoint_name_cdna)
            torch.save(self.q_net.state_dict(), checkpoint_name_posterior)

            print('Saved CDNA model to '+checkpoint_name_cdna)
            print('Saved Posterior model to '+checkpoint_name_posterior)

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
        
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)

parser.add_argument('--model', default="cdna", type=str)
parser.add_argument('--stage', default=1, type=int)
# parser.add_argument('--beta', default=1, type=float)

parser.add_argument('--save_every', default=25, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=52, type=int)
parser.add_argument('--clip', default=10, type=int)

parser.add_argument('--beta_start', default=0, type=float) # should not change generally
parser.add_argument('--beta_end', default=0.001, type=float)

# Load in model
# state_dict_path_det = "saves/sv2p/stage0/finetuned2/sv2p_cdna_state_dict_299.pth"
# state_dict_path_det = "saves/sv2p/v2/stage1/finetuned/sv2p_state_dict_199.pth" 
# state_dict_path_det = "/vol/bitbucket/mc821/VideoPrediction/saves/sv2p/stage0/finetuned3/sv2p_cdna_state_dict_25.pth"
state_dict_path_det = None 
state_dict_path_stoc = "saves/sv2p/stage2/finetuned1/sv2p_cdna_state_dict_99.pth" 
state_dict_posterior = "saves/sv2p/stage2/finetuned1/sv2p_posterior_state_dict_99.pth"

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
    log_fname = f'{args.model}_stage={args.stage}_{args.epochs}.log'
    if args.stage == 3: 
        log_dir = f"logs/{args.model}/stage{args.stage}/finalB={args.beta_end}/"
    else: 
        log_dir = f"logs/{args.model}/stage{args.stage}/finetuned2/"

    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)

    if state_dict_path_det: 
        logging.info(f"State Dictionary Path (Deterministic) is: {state_dict_path_det}")
    elif state_dict_path_stoc: 
        logging.info(f"State Dictionary Path (Stochastic) is: {state_dict_path_stoc}")
    else: 
        logging.info("Training network from scratch")

    # Datasets
    train_set = MovingMNIST(root='.dataset/mnist', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True)

    training_steps = len(train_loader) * args.epochs
    beta_scheduler = LinearScheduler(training_steps, args.beta_start, args.beta_end)

    if args.model == "cdna":
        trainer = SV2PTrainer(state_dict_path_det, state_dict_path_stoc, state_dict_posterior, beta_scheduler, args=args)  
        trainer.train(train_loader)
        
    logging.info(f"Completed {args.stage} training")

if __name__ == "__main__":
    main()



