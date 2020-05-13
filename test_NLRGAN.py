import torch
import torch.nn as nn
from lib.model import NLRGAN_G, NLRGAN_D
from torchvision.utils import save_image
from torch.autograd import Variable
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
# import ipdb
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
parser.add_argument('--model', default="NLRGAN", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=1000, type=int, help='number of training epochs')
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
parser.add_argument("--sample_interval", type=int, default=30, help="interval between image sampling")

Flags = parser.parse_args()
Flags.rel_avg_gan = True

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

# cuda devide
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: ", device)

# define training parameters
best_model_loss = [999999, 99999, 99999] # d, g, rec
batch_size = Flags.mini_batch
st_epoch = 0 # starting epoch
# number of input channel
C = 1

# create model

if Flags.model == 'RGAN':
    generator = RGAN_G(C,C).to(device)
    discriminator = RGAN_D(C,C).to(device)
elif Flags.model == 'NLRGAN':
    generator = NLRGAN_G(C,C).to(device)
    discriminator = NLRGAN_D(C,C).to(device)

# criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
criterion = nn.MSELoss()

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

if(Flags.model == 'RGAN'):
    # # Optimizers
    # optimizer_G = torch.optim.Adam([
    #                                 {'params': generator.base.parameters()},
    #                                 {'params': generator.last.parameters(), 'lr': Flags.lr * 0.1},
    #                                 ], lr=Flags.lr, betas=(Flags.b1, Flags.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))
elif(Flags.model == 'NLRGAN'):
    # # Optimizers
    # optimizer_G = torch.optim.Adam([
    #                                 {'params': generator.base.parameters()},
    #                                 {'params': generator.last.parameters(), 'lr': Flags.lr * 0.1},
    #                                 ], lr=Flags.lr, betas=(Flags.b1, Flags.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))

# if the checkpoint dir is not null refers to load checkpoint
if Flags.load_from_ckpt != "":
    summary_dir = Flags.load_from_ckpt
    # checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/ckpt_model.pth'))
    checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/best_model.pth'))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    st_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    # best_model_loss = checkpoint['val_loss']
    best_model_loss[0] = checkpoint['val_d_loss']
    best_model_loss[1] = checkpoint['val_g_loss']
    best_model_loss[2] = checkpoint['val_rec_loss']
    generator.train()
    discriminator.train()
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

log_dir = os.path.join(summary_dir, "log")


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

# test loop
model.eval()
with torch.set_grad_enabled(False):
    test_loss = 0.0
    avg_predicted_psnr = 0.0
    avg_input_psnr = 0.0
    avg_predicted_ssim = 0.0
    avg_input_ssim = 0.0
    count = 0
    better_count = 0
    for X,Y in validation_loader:
        # here comes your training loop
        # print("ValX.shape", X.shape)
        print(len(validation_loader))
        for j in range(len(X)):
            Xtest = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
            Ytest = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
            # [N x C x H x W]
            test_outputs = model(Xtest)

            # print(test_outputs * 255)
            # exit()
            np_images = test_outputs.cpu().numpy()

            N, _, _, _ = np_images.shape
            for n in range(N):
                filename = os.path.join(test_dir, "test_batch_best_%i.png"%(count))
                # residual
                cur_img1 = np.transpose(Xtest[n].cpu().numpy(), [1,2,0])
                pred_mask1 = np.transpose(np_images[n], [1,2,0])
                pred_img1 = pred_mask1  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
                gt_img1 = np.transpose(Ytest[n].cpu().numpy(), [1,2,0])

                # print("test_outputs size ", test_outputs.shape, "\t Ytest size ", Ytest.shape)
                # Stats of Predicted Image

                ssim1 = np.mean(np.mean(ssim(gt_img1, pred_img1, full=True,multichannel=True)))
                psnr1 = psnr(gt_img1, pred_img1)
                avg_predicted_psnr += (psnr1 )
                avg_predicted_ssim += (ssim1 )
                count += 1

                # Stats of Input Image
                ssim2 = np.mean(np.mean(ssim(gt_img1, cur_img1, full=True,multichannel=True)))
                psnr2 = psnr(gt_img1, cur_img1)
                avg_input_psnr += (psnr2)
                avg_input_ssim += (ssim2)
                print("issim: ", ssim2,  "\t pssim: ", ssim1, "\t ipsnr: ", psnr2,  "\t ppsnr: ", psnr1)

                if(psnr2 < psnr1):
                    better_count += 1

                img_pair1 = np.hstack(([cur_img1, pred_mask1, pred_img1, gt_img1])) * 255
                # img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
                display_img = np.vstack([img_pair1])
                # print(pred_mask1.shape)
                # print(pred_img1.shape)
                # print(gt_img1.shape)
                # ipdb.set_trace()
                # print(display_img.shape)
                # cv2.imshow('sample image', display_img.astype(np.uint8))
                # cv2.waitKey(0) # waits until a key is pressed
                # cv2.destroyAllWindows()
                cv2.imwrite(filename, display_img.astype(np.uint8)) 

            t_loss = criterion(test_outputs, Ytest)
            test_loss += t_loss.item()
        print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
                " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
                " better count: ", better_count)

    test_loss /= count
    logger.info('Test loss: %.5f' % (test_loss))
