import torch
import torch.nn as nn
from lib.model import DPWSDNet, Pixel_Net, Wavelet_Net
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
from lib import TecoGANDataset

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
parser.add_argument('--mini_batch', default=4, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="/media/data3/tjtanaa/tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="/media/data3/tjtanaa/tecogan_video_data", type=str, help='dataset directory')
# parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--tseq_length", type=int, default=1, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx264", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=3, type=int, help='scene duration')
parser.add_argument('--depth', default=19, type=int, help='depth')
parser.add_argument('--crop_size', default=512, type=int, help='scene duration')
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
    # print(summary(model, (C, 256, 256)))
elif Flags.model == 'PixelNet':
    model = Pixel_Net(C, C, P = Flags.depth).to(device)
    # print(summary(model, (C, 256, 256)))
elif Flags.model == 'WaveletNet':
    model = Wavelet_Net(C, C, P = Flags.depth).to(device)
    # print(summary(model, (C, 256, 256)))    


# optimizer = optim.SGD([
#     {'params': model.base.parameters()}
# ], lr=Flags.lr, momentum = 0.9)

optimizer = torch.optim.SGD(model.parameters(), lr=Flags.lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=True)
# optimizer = torch.optim.Adam(model.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.1)

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

from torch.utils import data
from lib.dataloader import HDF5Dataset
from torch.utils.data.sampler import SubsetRandomSampler

if 'drive' in Flags.input_dir:
    input_dir = '.'
else:
    save_dir = os.path.join(Flags.input_dir, "dataset_{}_qp{}".format(Flags.vcodec,Flags.qp))
    # input_dir = os.path.join(os.path.join(save_dir, Flags.model), '{}_qp{}'.format(Flags.vcodec,str(Flags.qp)))
    input_dir = os.path.join(os.path.join(save_dir, 'DPWSDNet'), '{}_qp{}'.format(Flags.vcodec,str(Flags.qp)))

# dataset = HDF5Dataset(input_dir, recursive=False, load_data=False, 
#    data_cache_size=100, transform=None)
dataset = TecoGANDataset(Flags.input_dir, Flags.vcodec, Flags.qp, Flags.tseq_length, Flags.crop_size)

shuffle_dataset = True
# Creating data indices for training and validation and test splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
if shuffle_dataset :
    np.random.seed(0)
    np.random.shuffle(indices)
# indices = np.arange(N)
# np.random.shuffle(indices)

train_indices = indices[: int(dataset_size * 0.98)]
val_indices = indices[int(dataset_size * 0.98): int(dataset_size * 0.99)]
test_indices = indices[int(dataset_size * 0.99):]
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader_params = {'batch_size': Flags.mini_batch, 'num_workers': 4,'sampler': train_sampler}
validation_loader_params = {'batch_size': 4, 'num_workers': 4,'sampler': valid_sampler}

train_loader = data.DataLoader(dataset, **train_loader_params)
validation_loader = data.DataLoader(dataset, **validation_loader_params)


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
    for m, (X,Y) in enumerate(train_loader):
        num_train_batches =  (m + 1)
        if iteration_count > Flags.max_iteration:
                break


        Xtrain = (X[:,Flags.tseq_length//2,:,:]).float().to(device)
        Ytrain = (Y[:,Flags.tseq_length//2,:,:]).float().to(device)
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


        logger.info(
            "[Epoch %d/%d] [Batch %d/%d] [Rec loss: %.5e]"
            % (epoch, Flags.epoch, m, len(train_loader), reconstruction_loss.item())
        )

        if m % Flags.sample_interval == 0:
            y = np.transpose(Ytrain[0].detach().cpu().numpy(), [1,2,0])
            x = np.transpose((outputs[0].detach()).cpu().numpy(), [1,2,0])
            x_input = np.transpose((Xtrain[0].detach()).cpu().numpy(), [1,2,0])
            logger.info("Training: input_psnr: %.5f \t train_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
            save_image(outputs.data[:25],  os.path.join(test_dir,"train_%d.png") % m, nrow=5, normalize=True)


        # running_g_loss += g_loss.item()
        # running_d_loss += d_loss.item()
        running_rec_loss += reconstruction_loss.item()
        iteration_count += 1
        scheduler.step()

        if m % 1000 == 0:
            with torch.set_grad_enabled(False):
                num_val_batches = 0
                for k, (X,Y) in enumerate(validation_loader):
                    num_val_batches = k + 1
                    
                    # here comes your validation loop

                    # ------------------------------------
                    #
                    #   Train Generator
                    #
                    #-------------------------------------

                    Xval = (X[:,Flags.tseq_length//2,:,:]).float().to(device)
                    Yval = (Y[:,Flags.tseq_length//2,:,:]).float().to(device)
                    val_outputs = model(Xval)
                    # print("Xval[1] size ", Xval[1].shape, "val_outputs size ", val_outputs.shape)
                    reconstruction_loss = criterion(val_outputs, Yval)
                    ssim = pytorch_ssim.ssim(Yval, val_outputs)
                    # logger.info("ssim ", ssim)

                    if(k == 0):
                        y = np.transpose(Yval[0].detach().cpu().numpy(), [1,2,0])
                        x = np.transpose((val_outputs[0].detach()).cpu().numpy(), [1,2,0])
                        x_input = np.transpose((Xval[0].detach()).cpu().numpy(), [1,2,0])
                        logger.info("Validation: input_psnr: %.5f \t val_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
                        save_image(val_outputs.data[:25],  os.path.join(test_dir,"val_%d_%d.png") % (epoch, k), nrow=5, normalize=True)

                    val_rec_loss += reconstruction_loss.item()
                    
                val_rec_loss /= num_val_batches
                val_g_loss /= num_val_batches
                val_d_loss /= num_val_batches

                logger.info("[Epoch %d/%d] [tRec loss: %.5e][vRec loss: %.5e]" 
                    % (epoch, Flags.epoch, running_rec_loss / num_train_batches, val_rec_loss))


        # save checkpoint
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_rec_loss / num_train_batches,
        'val_loss': val_rec_loss,
        'iteration': iteration_count,
        # 'scheduler': scheduler,
        }, os.path.join(model_dir, 'ckpt_model.pth'))

        # save best model
        if val_rec_loss < best_model_loss:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_rec_loss / num_train_batches,
            'val_loss': val_rec_loss,
            # 'scheduler': scheduler,
            'iteration': iteration_count,
            }, os.path.join(model_dir, 'best_model.pth'))
            best_model_loss = val_rec_loss

            running_loss = 0.0

    end_timing_epoch = time.time()
    logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))
    if iteration_count > Flags.max_iteration:
        break
