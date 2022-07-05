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
import torchvision 
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt
from kvae_position.model_kvae_pos import KalmanVAE
from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader

import wandb


class KVAETrainer:
    def __init__(self, state_dict_path = None, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()
        print(self.args)

        # Change out encoder and decoder 
        self.model = KalmanVAE(args = self.args).to(self.args.device)
        
        parameters = list(self.model.parameter_net.parameters()) + list(self.model.alpha_out.parameters()) \
                    + [self.model.a1, self.model.A, self.model.C]

        self.optimizer = torch.optim.Adam(parameters, lr=self.args.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)
    
        if state_dict_path: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            logging.info(f"Loaded State Dict from {state_dict_path}.")
            
            self.model.load_state_dict(state_dict)

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting KVAE training for {self.args.epochs} epochs.")

        logging.info("Train Loss") # header for losses

        # Save a copy of data to use for evaluation 
        # example_data, example_target = next(iter(train_loader))

        # example_data = example_data[0].clone().to(self.args.device)
        # example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        # example_data = torch.where(example_data > 0.5, 1.0, 0.0).unsqueeze(0)

        # example_target = example_target[0].clone().to(self.args.device)
        # example_target = (example_target - example_target.min()) / (example_target.max() - example_target.min())
        # example_target = torch.where(example_target > 0.5, 1.0, 0.0).unsqueeze(0)
        
        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            
            for data, _ in tqdm(train_loader):
                
                data = data.to(self.args.device)
                data = data.double()

                #forward + backward + optimize
                self.optimizer.zero_grad()
                loss = self.model(data)
                loss.backward()
                self.optimizer.step()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                # forward pass
                print(f"Loss: {loss}")

                metrics = {"train/train_loss": loss}
                if wandb_on == True:
                    wandb.log(metrics)

                n_iterations += 1
                running_loss += loss.item()

            training_loss = running_loss/len(train_loader)
            current_lr = self.scheduler.get_last_lr()[0]

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}")

            logging.info(f"{training_loss:.8f}, {current_lr}")

            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

            # if epoch % 5 == 0: 
            #     predictions, ground_truth = self._plot_predictions(example_data, example_target)
            #     wandb.log({"Ground Truth": [ground_truth]})
            #     wandb.log({"Predictions": [predictions]})

            #     plt.bar([1, 2, 3], averaged_weights)
            #     wandb.log({"Averaged Weights": plt})

            if epoch % 10 == 0 and epoch != 0: # since I have doubled the training examples 
                self.scheduler.step() 

        logging.info("Finished training.")

        final_checkpoint = self._save_model(epoch)
        logging.info(f'Saved model. Final Checkpoint {final_checkpoint}')

    def _save_model(self, epoch):  
        checkpoint_path = f'saves/{self.args.dataset}/kvae/{self.args.subdirectory}/scale={self.args.scale}/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        filename = f'kvae_state_dict_scale={self.args.scale}_{epoch}.pth'
        checkpoint_name = checkpoint_path + filename

        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to ', checkpoint_name)
        
        return checkpoint_name

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "BouncingBall_Pos", type = str)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--subdirectory', default="finetuned2", type=str)
parser.add_argument('--model', default="KVAE", type=str)

parser.add_argument('--x_dim', default=1, type=int)
parser.add_argument('--a_dim', default=2, type=int)
parser.add_argument('--z_dim', default=4, type=int)
parser.add_argument('--K', default=3, type=int)

parser.add_argument('--clip', default=150, type=int)
parser.add_argument('--scale', default=1, type=float)

parser.add_argument('--save_every', default=10, type=int) 
parser.add_argument('--learning_rate', default=0.007, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--initial', default="False", type=str, help = "Does not optimise parameters of DPN for first few epochs.")


def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    state_dict_path = None 
       
    # set up logging
    log_fname = f'{args.model}_scale={args.scale}_{args.epochs}.log'
    log_dir = f"logs/{args.dataset}/{args.model}/{args.subdirectory}/"
    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)

    global wandb_on
    wandb_on = False
    if wandb_on == True:
        wandb.init(project="KVAE_bouncing_ball_position")
        wandb.config.update(args)

    # Datasets
    train_set = BouncingBallDataLoader('dataset/bouncing_ball/v2/train', image = False) # positions
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set, 
                batch_size=args.batch_size, 
                shuffle=True)

    trainer = KVAETrainer(state_dict_path= state_dict_path, args=args)
    trainer.train(train_loader)

if __name__ == "__main__":
    main()



