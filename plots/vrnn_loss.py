from ast import Pass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os import listdir
from os.path import isfile, join

mypath = './logs/VRNN/v1/'
fpaths_list = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
print(fpaths_list)

def plot_losses_VRNN(log_path):
    """
    Log files for other models may be formatted differently
    """
    out_path = fpath.split("/")[-1]
    out_path = out_path.strip(".log")

    losses = []

    with open(log_path) as f:
        # ignore first line
        for idx, line in enumerate(f):
            if idx == 0 or idx == 1 or idx == 2:
                pass
            elif "Finished" in line or "Saved" in line:
                pass
            else:
                x = line.split(":")[-1]
                x = x.strip("\n")
                losses.append(x)

    train_loss = []
    kld = []
    reconstruction_loss = []

    for line in losses:
        line = line.split(",")
        train_loss.append(float(line[0]))
        kld.append(float(line[1]))
        reconstruction_loss.append(float(line[2]))

    train_loss = np.array(train_loss)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(kld, label='KL Divergence')
    plt.plot(reconstruction_loss, label='Reconstruction Loss')
    plt.legend()

    plt.savefig(f"plots/VRNN/{out_path}_losses.jpg")
    plt.figure().clear()

def plot_VRNN_together(fpaths_list):
    train_loss = []
    kld = []
    mse = []
    models = ["Beta = 2.0","Beta = 4.0", "Beta = 3.0", "Beta = 5.0", "Beta = 1.0"]

    for fpath in fpaths_list: # loop through all files in directory 
        # model_path = fpath.split("/")[-1]
        # model_path = model_path.strip(".log")
        # models.append(model_path)

        losses_for_each_model = []
        train_loss_model = []
        kld_model = []
        mse_model = []

        with open(fpath) as f:
            # ignore first line
            for idx, line in enumerate(f):
                if idx == 0 or idx == 1 or idx == 2:
                    pass
                elif "Finished" in line or "Saved" in line:
                    pass
                else:
                    x = line.split(":")[-1]
                    x = x.strip("\n")
                    losses_for_each_model.append(x)

        for line in losses_for_each_model:
            line = line.split(",")
            train_loss_model.append(float(line[0]))
            kld_model.append(float(line[1]))
            mse_model.append(float(line[2]))

        train_loss.append(train_loss_model)
        kld.append(kld_model)
        mse.append(mse_model)

    # Divide MSE by seq length and number of pixels 

    # for loss, model in zip(kld, models):
    #     print(loss[0], model)

    for loss in kld: 
        plt.plot(loss)
    
    plt.legend(models)
    plt.savefig(f"plots/VRNN/kld_losses.png")
    plt.figure().clear()

    for loss in mse: 
        plt.plot(loss)
    
    plt.legend(models)
    plt.savefig(f"plots/VRNN/kld_losses.png")
    plt.figure().clear()

    for loss in train_loss: 
        plt.plot(loss)
    
    plt.legend(models)
    plt.savefig(f"plots/VRNN/total_losses.png")
    plt.figure().clear()


if __name__ == "__main__":
    # for fpath in fpaths_list:
    #     plot_losses_VRNN(fpath)

    plot_VRNN_together(fpaths_list)
    # print(2250/5)
