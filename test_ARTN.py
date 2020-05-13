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

# test loop
model.eval()
with torch.set_grad_enabled(False):
    test_loss = 0.0
    avg_psnr = 0.0
    count = 0
    for k in range(len(test_indices)//2-1):
        filename = os.path.join(test_dir, "test_batch_best_%i.png"%(k))
        tindices = test_indices[2*k: 2*(k+1)] 
        Xtest = torch.from_numpy(X[:,tindices,:,:,:]).float().to(device)
        Ytest = torch.from_numpy(Y[tindices,:,:,:]).float().to(device)
        # [N x C x H x W]
        test_outputs = model(Xtest)
        np_images = test_outputs.cpu().numpy()
        pre_img1 = np.transpose(Xtest[0,0].cpu().numpy(), [1,2,0])
        cur_img1 = np.transpose(Xtest[1,0].cpu().numpy(), [1,2,0])
        post_img1 = np.transpose(Xtest[2,0].cpu().numpy(), [1,2,0])

        # residual
        # pred_mask1 = np.transpose(np_images[0], [1,2,0])
        # pred_img1 =  np.transpose(Xtest[1,0].cpu().numpy(), [1,2,0]) + pred_mask1
        # gt_img1 = np.transpose(Ytest[0].cpu().numpy(), [1,2,0])

        # pre_img2 = np.transpose(Xtest[0,1].cpu().numpy(), [1,2,0])
        # cur_img2 = np.transpose(Xtest[1,1].cpu().numpy(), [1,2,0])
        # post_img2 = np.transpose(Xtest[2,1].cpu().numpy(), [1,2,0])
        # pred_mask2 = np.transpose(np_images[1], [1,2,0])
        # pred_img2 =  np.transpose(Xtest[1,1].cpu().numpy(), [1,2,0]) + pred_mask2 
        # gt_img2 = np.transpose(Ytest[1].cpu().numpy(), [1,2,0])

        # reconstruction
        pred_mask1 = np.transpose(np_images[0], [1,2,0])
        pred_img1 = pred_mask1  #+ np.transpose(Xtest[1,0].cpu().numpy(), [1,2,0])
        gt_img1 = np.transpose(Ytest[0].cpu().numpy(), [1,2,0])

        pre_img2 = np.transpose(Xtest[0,1].cpu().numpy(), [1,2,0])
        cur_img2 = np.transpose(Xtest[1,1].cpu().numpy(), [1,2,0])
        post_img2 = np.transpose(Xtest[2,1].cpu().numpy(), [1,2,0])
        pred_mask2 = np.transpose(np_images[1], [1,2,0])
        pred_img2 = pred_mask2 # + np.transpose(Xtest[1,1].cpu().numpy(), [1,2,0])
        gt_img2 = np.transpose(Ytest[1].cpu().numpy(), [1,2,0])

        # print("test_outputs size ", test_outputs.shape, "\t Ytest size ", Ytest.shape)
        ssim1 = np.mean(np.mean(ssim(gt_img1, pred_img1, full=True,multichannel=True)))
        psnr1 = psnr(gt_img1, pred_img1)
        ssim2 = np.mean(np.mean(ssim(gt_img2, pred_img2, full=True,multichannel=True)))
        psnr2 = psnr(gt_img2, pred_img2)
        avg_psnr += (psnr1 + psnr2)
        count += 2
        print("ssim1 ", ssim1, "\t psnr1: ", psnr1, "\t ssim2 ", ssim2, "\t psnr2: ", psnr2)

        img_pair1 = np.hstack(([pre_img1, cur_img1, post_img1, pred_mask1, pred_img1, gt_img1])) * 255
        img_pair2 = np.hstack(([pre_img2, cur_img2, post_img2, pred_mask2, pred_img2, gt_img2])) * 255
        display_img = np.vstack([img_pair1, img_pair2])
        # print(pred_mask1.shape)
        # print(pred_img1.shape)
        # print(gt_img1.shape)
        # ipdb.set_trace()
        # print(display_img.shape)
        # cv2.imshow('sample image', display_img.astype(np.uint8))
        # cv2.waitKey(0) # waits until a key is pressed
        # cv2.destroyAllWindows()
        cv2.imwrite(filename, display_img.astype(np.uint8)) 
#         
        # exit()
            # test_dir

        t_loss = criterion(test_outputs, Ytest)
        test_loss += t_loss.item()
    print(Flags.model, " avg_psnr: ", avg_psnr/count)

test_loss /= ((len(test_indices)//2-1)*2)
logger.info('Test loss: %.5f' % (test_loss))