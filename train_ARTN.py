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
parser.add_argument('--input_dir', default="D:\\Github\\tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\tecogan_video_data", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="D:\\Github\\tecogan_video_data\\ARCNN\\12-30-2019=09-11-03", type=str, help='ckpt model directory')
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
    if model == 'ARTN':
        save_path = os.path.join(Flags.input_dir, 'ARTN')
        input_path = os.path.join(save_path, 'input.npy')
        gt_path = os.path.join(save_path, 'gt.npy')
        # [T x N x C x H  x W]
        return np.transpose(np.load(input_path), [0, 1, 4, 2, 3])/255.0, np.transpose(np.load(gt_path), [0, 3, 1, 2])/255.0
    elif model == 'ARCNN':
        save_path = os.path.join(Flags.input_dir, 'ARTN')
        input_path = os.path.join(save_path, 'input.npy')
        gt_path = os.path.join(save_path, 'gt.npy')
        # [T x N x C x H  x W]
        return np.transpose(np.load(input_path), [0, 1, 4, 2, 3])/255.0, np.transpose(np.load(gt_path), [0, 3, 1, 2])/255.0
    elif model == 'FastARCNN':
        save_path = os.path.join(Flags.input_dir, 'ARTN')
        input_path = os.path.join(save_path, 'input.npy')
        gt_path = os.path.join(save_path, 'gt.npy')
        # [T x N x C x H  x W]
        return np.transpose(np.load(input_path), [0, 1, 4, 2, 3])/255.0, np.transpose(np.load(gt_path), [0, 3, 1, 2])/255.0
    elif model == 'VRCNN':
        save_path = os.path.join(Flags.input_dir, 'ARTN')
        input_path = os.path.join(save_path, 'input.npy')
        gt_path = os.path.join(save_path, 'gt.npy')
        # [T x N x C x H  x W]
        return np.transpose(np.load(input_path), [0, 1, 4, 2, 3])/255.0, np.transpose(np.load(gt_path), [0, 3, 1, 2])/255.0
                                                                



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
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
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

# # training loop
# for epoch in range(st_epoch,Flags.epoch):  # loop over the dataset multiple times
#     start_timing_epoch = time.time()
#     running_loss = 0.0
#     validation_loss = 0.0
#     np.random.shuffle(train_indices)
#     # print("max iteration: ", len(train_indices)//batch_size)
#     for i in range(len(train_indices)//batch_size-1):
#         # print("Epoch: %i \t Iteration: %i" % (epoch, i))

#         tindices = train_indices[batch_size*i: batch_size*(i+1)]
#         Xtrain = torch.from_numpy(X[:,tindices,:,:,:]).float().to(device)
#         Ytrain = torch.from_numpy(Y[tindices,:,:,:]).float().to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         # print("forward")
#         outputs = model(Xtrain)
#         print("outputs.shape: ", outputs.shape)
#         print("Xtrain.shape: ", Xtrain.shape)
#         
#         # print("loss")
#         loss = criterion(outputs, Ytrain - Xtrain[1])
#         # print("backward")
#         loss.backward()
#         # print("optimize")
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         # if i % 1000 == 999:    # print every 2000 mini-batches
#         if i % 50 == 49:
#             np.random.shuffle(val_indices)
#             # print("max validation iteration: ",len(val_indices)//batch_size )
#             with torch.set_grad_enabled(False):
#                 validation_loss = 0.0
#                 for j in range(len(val_indices)//batch_size-1):
#                     vindices = val_indices[batch_size*j: batch_size*(j+1)] 
#                     Xval = torch.from_numpy(X[:,vindices,:,:,:]).float().to(device)
#                     Yval = torch.from_numpy(Y[vindices,:,:,:]).float().to(device)
#                     val_outputs = model(Xval)
#                     # print("Xval[1] size ", Xval[1].shape, "val_outputs size ", val_outputs.shape)
#                     # ssim = pytorch_ssim.ssim(Yval, val_outputs)
#                     # print("ssim ", ssim)
#                     # exit()
#                     val_loss = criterion(val_outputs , Yval - Xval[1])
#                     validation_loss += val_loss.item()
#             validation_loss /= ((len(val_indices)//batch_size-1)*batch_size)
#             logger.info('Epoch: %i \t Iteration: %i training_running_loss: %.6e \t validation_loss: %.6e' %
#                   (epoch + 1, i + 1, running_loss / 2000, validation_loss))


#             # save checkpoint
#             torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             'val_loss': validation_loss
#             }, os.path.join(model_dir, 'ckpt_model.pth'))

#             # save best model
#             if validation_loss < best_model_loss:
#                 torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'loss': loss,
#                 'val_loss': validation_loss
#                 }, os.path.join(model_dir, 'best_model.pth'))
#                 best_model_loss = validation_loss

#             running_loss = 0.0

#     end_timing_epoch = time.time()
#     logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))

# print('Finished Training')

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
