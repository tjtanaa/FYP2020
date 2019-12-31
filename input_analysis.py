import torch
import torch.nn as nn
from lib.model import ARTN, ARCNN, FastARCNN
# from lib import load_dataset
import os
import numpy as np
import torch.optim as optim
import lib.pytorch_ssim as pytorch_ssim
from datetime import datetime
import time
import math
import cv2
from skimage.measure import compare_ssim as ssim
import ipdb

'''
    Command

    python train_ARTN.py --load_from_ckpt "D:\\Github\\tecogan_video_data\\ARTN\\12-20-2019=17-24-35"

    experiment residual 1: D:\\Github\\tecogan_video_data\\ARTN\\12-20-2019=17-24-35
'''

import argparse

cur_time = datetime.now().strftime("%m-%d-%Y=%H-%M-%S")

# set all seeds
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ensure that the split of the dataset is the same each time
np.random.seed(0)

# argparser not working WHY!
# ------------------------------------parameters------------------------------#

# model type
# max epoch
# mini_batch_size
# dataset_directory
# output_dir 
# - model
# - - ckpt
# - - best model
# - test output
# - log
# load_from_ckpt: input the directory path to the model

parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser = argparse.ArgumentParser()
parser.add_argument('--model', default="ARCNN", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=1000, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\input_analysis", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\input_analysis", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
# parser.add_argument('--disk_path', default="D:\\Github\\tecogan_video_data", help='the path to save the dataset')

Flags = parser.parse_args()

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def load_dataset(model = 'ARTN'):
    # save_path = os.path.join(Flags.input_dir, 'ARTN')
    save_path = Flags.input_dir
    input_path = os.path.join(save_path, 'input.npy')
    gt_path = os.path.join(save_path, 'gt.npy')
    # [T x N x C x H  x W]
    return np.load(input_path), np.load(gt_path)

                                

summary_dir = os.path.join(Flags.input_dir, 'images')
# os.path.isdir(summary_dir) or 
if not(os.path.exists(summary_dir)):
    os.makedirs(summary_dir)


# cuda devide
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: ", device)

# define training parameters
best_model_loss = 999999
batch_size = Flags.mini_batch
st_epoch = 0 # starting epoch


# load dataset
X, Y = load_dataset(model = Flags.model)

# split into train, validation and test
# 70, 20, 10

input_size = X.shape
T, N, C, H, W = input_size

indices = np.arange(N)
np.random.shuffle(indices)

train_indices = indices[: int(N * 0.7)]
val_indices = indices[int(N * 0.7): int(N * 0.9)]
test_indices = indices[int(N * 0.9):]


for i in range(len(train_indices)):
    # print("Epoch: %i \t Iteration: %i" % (epoch, i))
    filename = os.path.join(summary_dir, "training_%i.png"%(i))

    tindices = train_indices[i]
    Xtrain = X[:,tindices,:,:,:]
    Ytrain = Y[tindices,:,:,:]
    mask = Ytrain-Xtrain[1]
    print(mask)
    exit()
    reconstructed = mask + Xtrain[1]

    img_pair1 = np.hstack(([Xtrain[0], Xtrain[1], Xtrain[2], Ytrain-Xtrain[1], reconstructed, Ytrain, reconstructed - Ytrain]))
    cv2.imwrite(filename, img_pair1.astype(np.uint8)) 
    
