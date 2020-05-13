import torch
import torch.nn as nn
from lib.model import ARTN, ARCNN, FastARCNN, VRCNN
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
import h5py

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
parser.add_argument('--model', default="ARTN", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=200, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
parser.add_argument('--channel', default=1, type=int, help='image channel dimension')
Flags = parser.parse_args()

def psnr(img1, img2):
    size = img1.shape

    PIXEL_MAX = 1.0

    if len(size) == 2:
        mse = np.mean( (img1 - img2) ** 2 )
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    else: # len(size) == 3
        h,w,c = size
        psnr = 0
        for i in range(c):
            mse = np.mean((img1[:,:,i] - img2[:,:,i]) ** 2)
            if mse == 0:
                psnr += 100
            else:
                psnr += 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
        psnr /= c

    return psnr

def store_many_hdf5(input_images, gt_images, out_dir):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 3, 32, 32) to be stored don't have to transpose
        gt_images       labels array, (N, 3, 32, 32)  to be stored
    """
    num_images = len(input_images)

    # Create a new HDF5 file
    file = h5py.File(out_dir, "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "input", np.shape(input_images), h5py.h5t.STD_U8BE, data=input_images
    )
    meta_set = file.create_dataset(
        "gt", np.shape(gt_images), h5py.h5t.STD_U8BE, data=gt_images
    )
    file.close()

def read_many_hdf5(input_dir):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []

    # Open the HDF5 file
    file = h5py.File(input_dir, "r+")

    input_images = np.array(file["/input"]).astype("uint8")
    gt_images_labels = np.array(file["/gt"]).astype("uint8")

    return input_images, gt_images_labels

def load_dataset(model = 'ARTN'):
    if model == 'ARTN':
        save_path = os.path.join(Flags.input_dir, 'ARTN')
        input_path = os.path.join(save_path, 'input.npy')
        gt_path = os.path.join(save_path, 'gt.npy')
        # [T x N x C x H  x W]
        return np.transpose(np.load(input_path), [0, 1, 4, 2, 3])/255.0, np.transpose(np.load(gt_path), [0, 2, 3, 1])/255.0
    if model == 'ARCNN' or model == 'FastARCNN' or model == 'VRCNN':    
        save_path = os.path.join(Flags.input_dir, 'ARCNN')
        input_path = os.path.join(save_path, 'data.h5')
        input_images, gt_images = read_many_hdf5(input_path)
        print("input_images.shape: ", str(input_images.shape), "\t gt_images.shape: " + str(gt_images.shape))

        return np.transpose(input_images, [0, 3, 1, 2])/255.0, np.transpose(gt_images, [0, 3, 1, 2])/255.0
    # elif model == 'FastARCNN':
    #     save_path = os.path.join(Flags.input_dir, 'ARTN')
    #     input_path = os.path.join(save_path, 'input.npy')
    #     gt_path = os.path.join(save_path, 'gt.npy')
    #     # [T x N x C x H  x W]
    #     return np.transpose(np.load(input_path), [0, 1, 4, 2, 3])/255.0, np.transpose(np.load(gt_path), [0, 2, 3, 1])/255.0
    # elif model == 'VRCNN':
    #     save_path = os.path.join(Flags.input_dir, 'ARTN')
    #     input_path = os.path.join(save_path, 'input.npy')
    #     gt_path = os.path.join(save_path, 'gt.npy')
    #     # [T x N x C x H  x W]
    #     return np.transpose(np.load(input_path), [0, 1, 4, 2, 3])/255.0, np.transpose(np.load(gt_path), [0, 2, 3, 1])/255.0
                                                                



# cuda devide
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: ", device)

# define training parameters
best_model_loss = 999999
batch_size = Flags.mini_batch
st_epoch = 0 # starting epoch


C = Flags.channel
# create model

if Flags.model == 'ARTN':
    model = ARTN(C, C).to(device)
    # print(model)
elif Flags.model == 'ARCNN':
    model = ARCNN(C, C).to(device)
elif Flags.model == 'FastARCNN':
    model = FastARCNN(C, C).to(device)
elif Flags.model == 'VRCNN':
    model = VRCNN(C, C).to(device)



# criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
criterion = nn.MSELoss()
lr = 1e-4
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer = optim.Adam([
    {'params': model.pre_branch.parameters()},
    {'params': model.cur_branch.parameters()},
    {'params': model.post_branch.parameters()},
    {'params': model.inception_block1.parameters()},
    {'params': model.inception_block2.parameters()},
    {'params': model.conv5.parameters(), 'lr': lr * 0.1},
], lr=lr)
# if the checkpoint dir is not null refers to load checkpoint
if Flags.load_from_ckpt != "":
    summary_dir = Flags.load_from_ckpt
    # checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/ckpt_model.pth'))
    checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    st_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    best_model_loss = checkpoint['val_loss']
    model.train()
    # override training output, continue training
else:
    summary_dir = os.path.join(Flags.output_dir, Flags.model)
    # os.path.isdir(summary_dir) or 

    if not(os.path.exists(summary_dir)):
        os.makedirs(summary_dir)

    summary_dir = os.path.join(summary_dir, cur_time)
    # os.path.isdir(summary_dir) or 
    if not(os.path.exists(summary_dir)):
        os.makedirs(summary_dir)
# print("summary_dir: ", summary_dir)
# exit()
# create the output log folder
log_dir = os.path.join(summary_dir, "log")
# print(summary_dir)
# print(log_dir)
# exit()

# os.path.isdir(log_dir) or 
if not(os.path.exists(log_dir)):
    os.makedirs(log_dir)
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)
# os.path.isfile(log_dir+ '/' + cur_time + '.log') or touch(log_dir+ '/' + cur_time + '.log')

if not(os.path.exists(log_dir+ '/' + cur_time + '.log')):
    touch(log_dir+ '/' + cur_time + '.log')

import logging
logging.basicConfig(level=logging.INFO)
# Create a custom logger
logger = logging.getLogger(__name__)


# Create handlers
# c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(log_dir+ '/' + cur_time + '.log')
# c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
# logger.addHandler(c_handler)
logger.addHandler(f_handler)

# logger.warning('This is a warning')
# logger.error('This is an error')

# create the model folder
model_dir = os.path.join(summary_dir, "model")
# os.path.isdir(model_dir) or 
if not(os.path.exists(model_dir)):
    os.makedirs(model_dir)

# create test output_model
test_dir = os.path.join(summary_dir, "test")
# os.path.isdir(test_dir) or
if not(os.path.exists(test_dir)):
    os.makedirs(test_dir)

# ensure that the shuffle is random different from the random indices in st_epoch = 0
np.random.seed(st_epoch)

logger.info(cur_time)


from torch.utils import data
from lib.dataloader import HDF5Dataset
from torch.utils.data.sampler import SubsetRandomSampler


dataset = HDF5Dataset(os.path.join(Flags.input_dir, 'ARTN'), recursive=False, load_data=False, 
   data_cache_size=100, transform=None)

shuffle_dataset = True
# Creating data indices for training and validation splits:
validation_split = 0.2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(0)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader_params = {'batch_size': 44, 'num_workers': 6,'sampler': train_sampler}
validation_loader_params = {'batch_size': 4, 'num_workers': 6,'sampler': valid_sampler}

train_loader = data.DataLoader(dataset, **train_loader_params)
validation_loader = data.DataLoader(dataset, **validation_loader_params)


# training loop
for epoch in range(st_epoch,Flags.epoch):
    start_timing_epoch = time.time()
    running_loss = 0.0
    validation_loss = 0.0
    train_length = 0
    for X,Y in train_loader:
        # here comes your training loop
        # print("train_x. shape", X.shape)
        # print(len(train_loader))
        train_length = len(Y)
        for i in range(train_length):
            # X_ = X[t,:,:,:,:].permute([0, 3, 1, 2])
            # print(X_.shape)

            Xtrain = (X[i,:,:,:,:,:].permute([0, 1, 4, 2, 3])/255.0).float().to(device)
            Ytrain = (Y[i,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print("forward")
            outputs = model(Xtrain)
            # print("outputs.shape: ", outputs.shape)
            # print("Xtrain.shape: ", Xtrain.shape)
            
            if( i == 0):
                y = np.transpose(Ytrain[0].detach().cpu().numpy(), [1,2,0])
                x = np.transpose((outputs[0].detach()).cpu().numpy(), [1,2,0])
                x_input = np.transpose((Xtrain[1,0].detach()).cpu().numpy(), [1,2,0])
                logger.info("Training: input_psnr: %.5f \t train_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
            # print("loss")
            loss = criterion(outputs, Ytrain)
            # print("backward")
            loss.backward()
            # print("optimize")
            optimizer.step()

            # print statistics
            running_loss += loss.item()
    # if i % 1000 == 999:    # print every 2000 mini-batches
    # if i % 50 == 49:
    # np.random.shuffle(val_indices)
    # print("max validation iteration: ",len(val_indices)//batch_size )
    with torch.set_grad_enabled(False):
        validation_loss = 0.0
        val_length = 0
        for X,Y in validation_loader:
            # here comes your training loop
            # print("ValX.shape", X.shape)
            # print(len(validation_loader))
            val_length = len(Y)
            for j in range(val_length):
                Xval = (X[j,:,:,:,:,:].permute([0, 1, 4, 2, 3])/255.0).float().to(device)
                Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
                val_outputs = model(Xval)
                # print("Xval[1] size ", Xval[1].shape, "val_outputs size ", val_outputs.shape)
                # ssim = pytorch_ssim.ssim(Yval, val_outputs)
                # print("ssim ", ssim)
                # exit()
                if(j == 0):
                    y = np.transpose(Yval[0].detach().cpu().numpy(), [1,2,0])
                    x = np.transpose((val_outputs[0].detach()).cpu().numpy(), [1,2,0])
                    x_input = np.transpose((Xval[1,0].detach()).cpu().numpy(), [1,2,0])
                    logger.info("Validation: input_psnr: %.5f \t val_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
                val_loss = criterion(val_outputs , Yval)
                validation_loss += val_loss.item()
        validation_loss /= (val_length)
        logger.info('Epoch: %i \t Iteration: %i training_running_loss: %.6e \t validation_loss: %.6e' %
              (epoch + 1, i + 1, running_loss / train_length, validation_loss))


    # save checkpoint
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'val_loss': validation_loss
    }, os.path.join(model_dir, 'ckpt_model.pth'))

    # save best model
    if validation_loss < best_model_loss:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_loss': validation_loss
        }, os.path.join(model_dir, 'best_model.pth'))
        best_model_loss = validation_loss

        running_loss = 0.0

    end_timing_epoch = time.time()
    logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))
    print('Finished Training')
