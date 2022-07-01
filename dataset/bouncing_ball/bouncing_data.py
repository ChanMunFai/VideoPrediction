""" Code from https://github.com/charlio23/bouncing-ball/blob/main/dataloaders/bouncing_data.py"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt, animation
import cv2
from glob import glob

class BouncingBallDataLoader(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        """
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """
        sample = np.load(os.path.join(
            self.root_dir, self.file_list[i]))
        im = sample['images']
        if len(im.shape) == 3:
            im = im[:,np.newaxis,:,:]
        else:
            im = im.transpose((0,3,1,2))

        seq_len = len(im)//2

        seq, target = im[:seq_len], im[seq_len:]

        return seq, target

def visualize_rollout(rollout, interval=50, show_step=False, save=False):
    """Visualization for a single sample rollout of a physical system.
    Args:
        rollout (numpy.ndarray): Numpy array containing the sequence of images. It's shape must be
            (seq_len, height, width, channels).
        interval (int): Delay between frames (in millisec).
        show_step (bool): Whether to draw the step number in the image
    """
    fig = plt.figure()
    img = []
    for i, im in enumerate(rollout):
        if show_step:
            black_img = np.zeros(list(im.shape))
            cv2.putText(
                black_img, str(i), (0, 30), fontScale=0.22, color=(255, 255, 255), thickness=1,
                fontFace=cv2.LINE_AA)
            res_img = (im + black_img / 255.) / 2
        else:
            res_img = im
        img.append([plt.imshow(res_img, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=interval,
                                    blit=True,
                                    repeat_delay=100)
    if save:
        writergif = animation.PillowWriter(fps=30)
        ani.save('dataset/bouncing_ball/v1/bouncing_sequence.gif', writergif)
    plt.show()


if __name__ == '__main__':
    dl = BouncingBallDataLoader('dataset/bouncing_ball/v1/train')
    print(len(dl))
    train_loader = torch.utils.data.DataLoader(dl, batch_size=10, shuffle=False)
    sample, target = next(iter(train_loader))
    print(torch.max(sample))
    print(torch.min(sample))
    print(sample.size())
    print(target.size())

    sample_np = sample.detach().cpu().numpy()
    sample_np = sample_np.transpose((0, 1, 3, 4, 2))

    # visualize_rollout(sample_np[8], interval=50, show_step=False, save=True)
