import torch
import torch.nn as nn
from lib.model import ARTN, ARCNN, FastARCNN, VRCNN
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

import sys
import os.path as osp
import math
import torchvision.utils

# sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from lib.data import create_dataloader, create_dataset  # noqa: E402

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
parser.add_argument('--max_iteration', default=100000, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="./model/REDS", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="./model/REDS", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx265", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=3, type=int, help='scene duration')
parser.add_argument("--sample_interval", type=int, default=30, help="interval between image sampling")

Flags = parser.parse_args()

dataset_name = 'REDS'  # REDS | Vimeo90K | DIV2K800_sub
opt = {}
opt['dist'] = False
opt['gpu_ids'] = [0]
if dataset_name == 'REDS':
    opt['name'] = 'test_REDS'
    # opt['dataroot_GT'] = '../../datasets/REDS/train_sharp_wval.lmdb'
    # opt['dataroot_LQ'] = '../../datasets/REDS/train_sharp_bicubic_wval.lmdb'
    opt['dataroot_GT'] = '/media/data3/tjtanaa/REDS/train_sharp_wval.lmdb'
    opt['dataroot_LQ'] = '/media/data3/tjtanaa/REDS/train_blur_comp_wval.lmdb'
    opt['mode'] = 'REDS'
    opt['N_frames'] = 5
    opt['phase'] = 'train'
    opt['use_shuffle'] = True
    opt['n_workers'] = 8
    opt['batch_size'] = 32
    opt['GT_size'] = 256
    # opt['LQ_size'] = 64
    opt['LQ_size'] = 256
    opt['scale'] = 4
    opt['use_flip'] = True
    opt['use_rot'] = True
    opt['interval_list'] = [1]
    opt['random_reverse'] = False
    opt['border_mode'] = False
    opt['cache_keys'] = None
    opt['data_type'] = 'lmdb'  # img | lmdb | mc
elif dataset_name == 'Vimeo90K':
    opt['name'] = 'test_Vimeo90K'
    opt['dataroot_GT'] = '../../datasets/vimeo90k/vimeo90k_train_GT.lmdb'
    opt['dataroot_LQ'] = '../../datasets/vimeo90k/vimeo90k_train_LR7frames.lmdb'
    opt['mode'] = 'Vimeo90K'
    opt['N_frames'] = 7
    opt['phase'] = 'train'
    opt['use_shuffle'] = True
    opt['n_workers'] = 8
    opt['batch_size'] = 16
    opt['GT_size'] = 256
    opt['LQ_size'] = 64
    opt['scale'] = 4
    opt['use_flip'] = True
    opt['use_rot'] = True
    opt['interval_list'] = [1]
    opt['random_reverse'] = False
    opt['border_mode'] = False
    opt['cache_keys'] = None
    opt['data_type'] = 'lmdb'  # img | lmdb | mc
elif dataset_name == 'DIV2K800_sub':
    opt['name'] = 'DIV2K800'
    opt['dataroot_GT'] = '../../datasets/DIV2K/DIV2K800_sub.lmdb'
    opt['dataroot_LQ'] = '../../datasets/DIV2K/DIV2K800_sub_bicLRx4.lmdb'
    opt['mode'] = 'LQGT'
    opt['phase'] = 'train'
    opt['use_shuffle'] = True
    opt['n_workers'] = 8
    opt['batch_size'] = 16
    opt['GT_size'] = 128
    opt['scale'] = 4
    opt['use_flip'] = True
    opt['use_rot'] = True
    opt['color'] = 'RGB'
    opt['data_type'] = 'lmdb'  # img | lmdb
else:
    raise ValueError('Please implement by yourself.')   



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

if Flags.model == 'ARCNN':
    model = ARCNN(C, C).to(device)
elif Flags.model == 'FastARCNN':
    model = FastARCNN(C, C).to(device)


optimizer = optim.Adam([
    {'params': model.base.parameters()},
    {'params': model.last.parameters(), 'lr': Flags.lr * 0.1},
], lr=Flags.lr)


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

train_set = create_dataset(opt)
train_loader = create_dataloader(train_set, opt, opt, None)

# for i, train_data in enumerate(train_loader):
#     X = train_data['LQs'][0,0,0,0,0]
#     print(X)
#     exit()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


for epoch in range(st_epoch,Flags.epoch):
    if iteration_count > Flags.max_iteration:
                break
    start_timing_epoch = time.time()
    running_g_loss = 0.0
    running_d_loss = 0.0
    running_rec_loss = 0.0
    val_g_loss = 0.0
    val_d_loss = 0.0
    val_rec_loss = 0.0
    num_train_batches = 0
    for m, train_data in enumerate(train_loader):
        if iteration_count > Flags.max_iteration:
            break
        # here comes your training loop
        X = train_data['LQs']
        Y = train_data['GT']
        num_train_batches += 1

        Xtrain = X[:,opt['N_frames']//2+1,:,:,:].float().to(device)
        Ytrain = Y.float().to(device)
        # print(Xtrain.shape)
        # print(Ytrain.shape)
        # exit()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # print("forward")
        outputs = model(Xtrain)
        # print("loss")
        reconstruction_loss = criterion(outputs, Ytrain)
        # print("backward")
        reconstruction_loss.backward()
        # print("optimize")
        optimizer.step()

        if num_train_batches % Flags.sample_interval == 0:
            y = np.transpose(Ytrain[0].detach().cpu().numpy(), [1,2,0])
            x = np.transpose((outputs[0].detach()).cpu().numpy(), [1,2,0])
            x_input = np.transpose((Xtrain[0].detach()).cpu().numpy(), [1,2,0])
            logger.info("Training: input_psnr: %.5f \t train_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
            save_image(outputs.data[:25],  os.path.join(test_dir,"train_%d.png") % num_train_batches, nrow=5, normalize=True)


        # running_g_loss += g_loss.item()
        # running_d_loss += d_loss.item()
        running_rec_loss += reconstruction_loss.item()
        iteration_count += 1

        logger.info("[Epoch %d/%d] [tRec loss: %.5e]" 
            % (epoch, Flags.epoch, running_rec_loss / num_train_batches))

    # with torch.set_grad_enabled(False):
    #     num_val_batches = 0
    #     for k, (X,Y) in enumerate(validation_loader):
    #         val_length = len(Y)
    #         num_val_batches += val_length
    #         for j in range(val_length):
    #             # here comes your validation loop

    #             # ------------------------------------
    #             #
    #             #   Train Generator
    #             #
    #             #-------------------------------------

    #             Xval = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
    #             Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
    #             val_outputs = model(Xval)
    #             # print("Xval[1] size ", Xval[1].shape, "val_outputs size ", val_outputs.shape)
    #             reconstruction_loss = criterion(val_outputs, Yval)
    #             ssim = pytorch_ssim.ssim(Yval, val_outputs)
    #             logger.info("ssim ", ssim)

    #             if(j == 0):
    #                 y = np.transpose(Yval[0].detach().cpu().numpy(), [1,2,0])
    #                 x = np.transpose((val_outputs[0].detach()).cpu().numpy(), [1,2,0])
    #                 x_input = np.transpose((Xval[0].detach()).cpu().numpy(), [1,2,0])
    #                 logger.info("Validation: input_psnr: %.5f \t val_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
    #                 save_image(val_outputs.data[:25],  os.path.join(test_dir,"val_%d_%d.png") % (epoch, j), nrow=5, normalize=True)

    #             val_rec_loss += reconstruction_loss.item()
                
    #     val_rec_loss /= num_val_batches
    #     val_g_loss /= num_val_batches
    #     val_d_loss /= num_val_batches

        # logger.info("[Epoch %d/%d] [tRec loss: %.5e][vRec loss: %.5e]" 
        #     % (epoch, Flags.epoch, running_rec_loss / num_train_batches, val_rec_loss))


    # save checkpoint
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_rec_loss / num_train_batches,
    'val_loss': val_rec_loss,
    'iteration': iteration_count,
    }, os.path.join(model_dir, 'ckpt_model.pth'))

    # save best model
    if val_rec_loss < best_model_loss:
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_rec_loss / num_train_batches,
        'val_loss': val_rec_loss,
        'iteration': iteration_count,
        }, os.path.join(model_dir, 'best_model.pth'))
        best_model_loss = val_rec_loss

        running_loss = 0.0

    end_timing_epoch = time.time()
    logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))
    if iteration_count > Flags.max_iteration:
                break