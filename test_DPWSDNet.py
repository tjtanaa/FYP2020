import torch
import torch.nn as nn
from lib.model import DPWSDNet
# from lib import load_dataset
from torchvision.utils import save_image
from torch.autograd import Variable
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
from lib import MyLogger
import h5py
from torchsummary import summary

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
parser.add_argument('--model', default="DPWSDNet", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=10, type=int, help='number of training epochs')
parser.add_argument('--max_iteration', default=100000, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=0.1, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx265", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=1, type=int, help='scene duration')
parser.add_argument('--depth', default=19, type=int, help='depth')
parser.add_argument("--sample_interval", type=int, default=30, help="interval between image sampling")

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

# cuda devide
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: ", device)

# define training parameters
best_model_loss = 999999
batch_size = Flags.mini_batch
st_epoch = 0 # starting epoch
iteration_count = 0 # starting iteration

C = Flags.channel
# create model

if Flags.model == 'DPWSDNet':
    model = DPWSDNet(C, C, P = Flags.depth).to(device)
    print(summary(model, (C, 256, 256)))


# optimizer = optim.SGD([
#     {'params': model.base.parameters()}
# ], lr=Flags.lr, momentum = 0.9)

optimizer = torch.optim.SGD(model.parameters(), lr=Flags.lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=True)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

# criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
criterion = nn.MSELoss()

# if the checkpoint dir is not null refers to load checkpoint
if Flags.load_from_ckpt != "":
    summary_dir = Flags.load_from_ckpt
    checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/ckpt_model.pth'))
    # checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    st_epoch = checkpoint['epoch']
    # loss = checkpoint['loss']
    scheduler = checkpoint['scheduler']
    best_model_loss = checkpoint['val_loss']
    iteration_count = checkpoint['iteration']
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

# create the output log folder
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

# define logger
logger = MyLogger(log_dir, cur_time).getLogger()
logger.info(cur_time)
logger.info(Flags)
# exit()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

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
    for k in range(len(test_indices)//2-1):
        filename = os.path.join(test_dir, "test_batch_best_%i.png"%(k))
        tindices = test_indices[2*k: 2*(k+1)] 
        Xtest = torch.from_numpy(X[tindices,:,:,:]).float().to(device)
        Ytest = torch.from_numpy(Y[tindices,:,:,:]).float().to(device)
        # [N x C x H x W]
        test_outputs = model(Xtest)

        # print(test_outputs * 255)
        # exit()
        np_images = test_outputs.cpu().numpy()

        # residual
        cur_img1 = np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
        pred_mask1 = np.transpose(np_images[0], [1,2,0])
        pred_img1 = pred_mask1  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
        gt_img1 = np.transpose(Ytest[0].cpu().numpy(), [1,2,0])

        cur_img2 = np.transpose(Xtest[1].cpu().numpy(), [1,2,0])
        pred_mask2 = np.transpose(np_images[1], [1,2,0])
        pred_img2 = pred_mask2 #+ np.transpose(Xtest[1].cpu().numpy(), [1,2,0])
        gt_img2 = np.transpose(Ytest[1].cpu().numpy(), [1,2,0])

        # print("test_outputs size ", test_outputs.shape, "\t Ytest size ", Ytest.shape)
        # Stats of Predicted Image

        ssim1 = np.mean(np.mean(ssim(gt_img1, pred_img1, full=True,multichannel=True)))
        psnr1 = psnr(gt_img1, pred_img1)
        ssim2 = np.mean(np.mean(ssim(gt_img2, pred_img2, full=True,multichannel=True)))
        psnr2 = psnr(gt_img2, pred_img2)
        avg_predicted_psnr += (psnr1 + psnr2)
        avg_predicted_ssim += (ssim1 + ssim2)
        count += 2
        print("PREDICTED: ssim1 ", ssim1, "\t psnr1: ", psnr1, "\t ssim2 ", ssim2, "\t psnr2: ", psnr2)
        ppsnr1 = psnr1
        ppsnr2 = psnr2

        # Stats of Input Image
        ssim1 = np.mean(np.mean(ssim(gt_img1, cur_img1, full=True,multichannel=True)))
        psnr1 = psnr(gt_img1, cur_img1)
        ssim2 = np.mean(np.mean(ssim(gt_img2, cur_img2, full=True,multichannel=True)))
        psnr2 = psnr(gt_img2, cur_img2)
        avg_input_psnr += (psnr1 + psnr2)
        avg_input_ssim += (ssim1 + ssim2)
        print("INPUT ssim1 ", ssim1, "\t psnr1: ", psnr1, "\t ssim2 ", ssim2, "\t psnr2: ", psnr2)
        ipsnr1 = psnr1
        ipsnr2 = psnr2

        if(ipsnr1 < psnr1):
            better_count += 1
        if(ipsnr2 < psnr2):
            better_count += 1


        img_pair1 = np.hstack(([cur_img1, pred_mask1, pred_img1, gt_img1])) * 255
        img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
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
    print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
            " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
            " better count: ", better_count)

test_loss /= ((len(test_indices)//2-1)*2)
logger.info('Test loss: %.5f' % (test_loss))