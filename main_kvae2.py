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
from kvae.model_kvae import KalmanVAE
from data.MovingMNIST import MovingMNIST
from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader

import wandb

class KVAETrainer:
    def __init__(self, state_dict_path = None, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()
        print(self.args)

        # Change out encoder and decoder 
        self.model = KalmanVAE(args = self.args).to(self.args.device)
        
        
        parameters = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) \
                    + [self.model.a1, self.model.A, self.model.C]
        
        self.optimizer = torch.optim.Adam(parameters, lr=self.args.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)
    
        if state_dict_path: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            # logging.info(f"Loaded State Dict from {state_dict_path}.")
            
            self.model.load_state_dict(state_dict)

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting KVAE training for {self.args.epochs} epochs.")

        logging.info("Train Loss, Reconstruction Loss, log q(a), ELBO_KF, MSE") # header for losses

        # Save a copy of data to use for evaluation 
        example_data, example_target = next(iter(train_loader))

        example_data = example_data[0].clone().to(self.args.device)
        example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        example_data = torch.where(example_data > 0.5, 1.0, 0.0).unsqueeze(0)

        example_target = example_target[0].clone().to(self.args.device)
        example_target = (example_target - example_target.min()) / (example_target.max() - example_target.min())
        example_target = torch.where(example_target > 0.5, 1.0, 0.0).unsqueeze(0)

        # columns_wandb = ["Ground Truth", "Predictions"]
        # predictions_table = wandb.Table(columns = columns_wandb)
        
        for epoch in range(self.args.epochs):

            # if epoch >= 1: # Otherwise train KVAE only 
            #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            #     self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)

            if epoch == 1: # Otherwise train KVAE only 
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
                self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)

            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_recon = 0
            running_latent_ll = 0 
            running_elbo_kf = 0
            running_mse = 0 

            for data, _ in tqdm(train_loader):
                
                data = data.to(self.args.device)
                data = (data - data.min()) / (data.max() - data.min())

                # Binarise input data 
                data = torch.where(data > 0.5, 1.0, 0.0)

                #forward + backward + optimize
                self.optimizer.zero_grad()
                loss, recon_loss, latent_ll, elbo_kf, mse, averaged_weights, var_diff = self.model(data)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                # forward pass
                print(f"Loss: {loss}")
                print(f"Reconstruction Loss: {recon_loss}") 
                print(f"Latent Loglikelihood: {latent_ll}")
                print(f"ELBO KF: {elbo_kf}")
                print(f"MSE: {mse}")

                metrics = {"train/train_loss": loss, 
                            "train/reconstruction_loss": recon_loss, 
                            "train/q(a)": latent_ll, 
                            "train/elbo kf": elbo_kf,
                            "train/mse": mse,
                            "variance of weights": var_diff
                }
                if wandb_on: 
                    wandb.log(metrics)

                n_iterations += 1
                running_loss += loss.item()
                running_recon +=  recon_loss.item() 
                running_latent_ll += latent_ll.item()
                running_elbo_kf += elbo_kf.item()
                running_mse += mse.item()

            training_loss = running_loss/len(train_loader)
            training_recon = running_recon/len(train_loader)
            training_latent_ll = running_latent_ll/len(train_loader)
            training_elbo_kf = running_elbo_kf/len(train_loader)
            training_mse = running_mse/len(train_loader)
            current_lr = self.scheduler.get_last_lr()[0]

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n Reconstruction Loss: {training_recon}\
                    \n Latent Log-likelihood: {training_latent_ll}\
                    \n ELBO Kalman Filter: {training_elbo_kf}\
                    \n MSE: {training_mse}")

            logging.info(f"{training_loss:.8f}, {training_recon:.8f}, {training_latent_ll:.8f}, {training_elbo_kf:.8f}, {training_mse:.8f}, {current_lr}")
            # wandb.log({"train/learning rate": current_lr})

            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

            if epoch % 5 == 0: 
                if wandb_on: 
                    predictions, ground_truth = self._plot_predictions(example_data, example_target)
                    wandb.log({"Ground Truth": [ground_truth]})
                    wandb.log({"Predictions": [predictions]})

                    plt.bar([1, 2, 3], averaged_weights)
                    wandb.log({"Averaged Weights": plt})

            if epoch % 20 == 0 and epoch != 0: 
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

    def _plot_predictions(self, input, target):
        predicted, _, _ = self.model.predict(input, 20)
        predicted = predicted.squeeze(0)
        target = target.squeeze(0)
        predicted_frames = torchvision.utils.make_grid(predicted,predicted.size(0))
        ground_truth_frames = torchvision.utils.make_grid(target,target.size(0))
        predicted_wandb = wandb.Image(predicted_frames)
        ground_truth_wandb = wandb.Image(ground_truth_frames)

        return predicted_wandb, ground_truth_wandb


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "BouncingBall_20", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall_20, BouncingBall_50]")
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--subdirectory', default="v2", type=str)
parser.add_argument('--model', default="KVAE", type=str)
parser.add_argument('--alpha', default="rnn", type=str, 
                    help = "choose between [mlp, rnn]")

parser.add_argument('--x_dim', default=1, type=int)
parser.add_argument('--a_dim', default=2, type=int)
parser.add_argument('--z_dim', default=4, type=int)
parser.add_argument('--K', default=3, type=int)

parser.add_argument('--clip', default=150, type=int)
parser.add_argument('--scale', default=0.3, type=float)

parser.add_argument('--save_every', default=10, type=int) 
parser.add_argument('--learning_rate', default=0.007, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--wandb_on', default=None, type=str)

def main():
    seed = 128
    torch.manual_seed(seed)
    args = parser.parse_args()

    global wandb_on 
    wandb_on = args.wandb_on 
    if wandb_on: 
        wandb.init(project="KVAE_bouncing_ball")

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.dataset == "MovingMNIST": 
        # state_dict_path = "saves/MovingMNIST/kvae/v3/finetuned1/scale=0.2/kvae_state_dict_scale=0.2_99.pth"
        state_dict_path = None 
    elif args.dataset == "BouncingBall_20": 
        # state_dict_path = "saves/BouncingBall/kvae/v2/scale=0.3/kvae_state_dict_scale=0.3_30.pth" 
        state_dict_path = None 
       
    # set up logging
    log_fname = f'{args.model}_scale={args.scale}_{args.epochs}.log'
    log_dir = f"logs/{args.dataset}/{args.model}/{args.subdirectory}/"
    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)
    if wandb_on: 
        wandb.config.update(args)

    # Datasets
    if args.dataset == "MovingMNIST": 
        train_set = MovingMNIST(root='dataset/mnist', train=True, download=True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=args.batch_size,
                    shuffle=True)
    elif args.dataset == "BouncingBall_20": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/v2/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

        val_set = BouncingBallDataLoader('dataset/bouncing_ball/v2/val')
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)
    else: 
        raise NotImplementedError

    trainer = KVAETrainer(state_dict_path= state_dict_path, args=args)
    trainer.train(train_loader)

if __name__ == "__main__":
    main()



