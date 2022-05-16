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

def plot_VRNN_together(log_path):
    out_path = fpath.split("/")[-1]
    out_path = out_path.strip(".log")


if __name__ == "__main__":
    # for fpath in fpaths_list:
    #     plot_losses_VRNN(fpath)

    for fpath in fpaths_list:
        plot_VRNN_together(fpath)
