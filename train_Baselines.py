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
parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx265", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=1, type=int, help='scene duration')
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
# exit()

from torch.utils import data
from lib.dataloader import HDF5Dataset
from torch.utils.data.sampler import SubsetRandomSampler

if 'drive' in Flags.input_dir:
    input_dir = '.'
else:
    save_dir = os.path.join(Flags.input_dir, "dataset_{}_qp{}".format(Flags.vcodec,Flags.qp))
    input_dir = os.path.join(os.path.join(save_dir, Flags.model), '{}_qp{}'.format(Flags.vcodec,str(Flags.qp)))

dataset = HDF5Dataset(input_dir, recursive=False, load_data=False, 
   data_cache_size=100, transform=None)

shuffle_dataset = True
# Creating data indices for training and validation and test splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
if shuffle_dataset :
    np.random.seed(0)
    np.random.shuffle(indices)
# indices = np.arange(N)
# np.random.shuffle(indices)

train_indices = indices[: int(dataset_size * 0.7)]
val_indices = indices[int(dataset_size * 0.7): int(dataset_size * 0.9)]
test_indices = indices[int(dataset_size * 0.9):]
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader_params = {'batch_size': 100, 'num_workers': 6,'sampler': train_sampler}
validation_loader_params = {'batch_size': 20, 'num_workers': 6,'sampler': valid_sampler}

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
        if iteration_count > Flags.max_iteration:
                break
        # here comes your training loop
        train_length = len(Y)
        num_train_batches += train_length
        for i in range(train_length):
            if iteration_count > Flags.max_iteration:
                break

            Xtrain = (X[i,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
            Ytrain = (Y[i,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
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

            batches_done = epoch * len(train_loader) + m + i 

            if batches_done % Flags.sample_interval == 0:
                y = np.transpose(Ytrain[0].detach().cpu().numpy(), [1,2,0])
                x = np.transpose((outputs[0].detach()).cpu().numpy(), [1,2,0])
                x_input = np.transpose((Xtrain[0].detach()).cpu().numpy(), [1,2,0])
                logger.info("Training: input_psnr: %.5f \t train_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
                save_image(outputs.data[:25],  os.path.join(test_dir,"train_%d.png") % batches_done, nrow=5, normalize=True)


            # running_g_loss += g_loss.item()
            # running_d_loss += d_loss.item()
            running_rec_loss += reconstruction_loss.item()
            iteration_count += 1

    with torch.set_grad_enabled(False):
        num_val_batches = 0
        for k, (X,Y) in enumerate(validation_loader):
            val_length = len(Y)
            num_val_batches += val_length
            for j in range(val_length):
                # here comes your validation loop

                # ------------------------------------
                #
                #   Train Generator
                #
                #-------------------------------------

                Xval = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
                Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
                val_outputs = model(Xval)
                # print("Xval[1] size ", Xval[1].shape, "val_outputs size ", val_outputs.shape)
                reconstruction_loss = criterion(val_outputs, Yval)
                ssim = pytorch_ssim.ssim(Yval, val_outputs)
                logger.info("ssim ", ssim)

                if(j == 0):
                    y = np.transpose(Yval[0].detach().cpu().numpy(), [1,2,0])
                    x = np.transpose((val_outputs[0].detach()).cpu().numpy(), [1,2,0])
                    x_input = np.transpose((Xval[0].detach()).cpu().numpy(), [1,2,0])
                    logger.info("Validation: input_psnr: %.5f \t val_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
                    save_image(val_outputs.data[:25],  os.path.join(test_dir,"val_%d_%d.png") % (epoch, j), nrow=5, normalize=True)

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




# for epoch in range(st_epoch,Flags.epoch):
#     start_timing_epoch = time.time()
#     running_loss = 0.0
#     validation_loss = 0.0
#     for X,Y in train_loader:
#         # here comes your training loop
#         # print("train_x. shape", X.shape)
#         # print(len(train_loader))
#         for i in range(len(X)):
#             # X_ = X[t,:,:,:,:].permute([0, 3, 1, 2])
#             # print(X_.shape)

#             Xtrain = (X[i,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#             Ytrain = (Y[i,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             # print("forward")
#             outputs = model(Xtrain)
#             # print("outputs.shape: ", outputs.shape)
#             # print("Xtrain.shape: ", Xtrain.shape)
#             if( i == 0):
#                 y = np.transpose(Ytrain[0].detach().cpu().numpy(), [1,2,0])
#                 x = np.transpose((outputs[0].detach()).cpu().numpy(), [1,2,0])
#                 x_input = np.transpose((Xtrain[0].detach()).cpu().numpy(), [1,2,0])
#                 logger.info("Training: input_psnr: %.5f \t train_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
#             # img_pair1 = np.hstack(([x_input, y, x])) * 255
#             # display_img = np.vstack([img_pair1, img_pair2])
#             # ipdb.set_trace()
#             # print(display_img.shape)
#             # cv2.imshow('sample image', img_pair1.astype(np.uint8))
#             # cv2.waitKey(0) # waits until a key is pressed
#             # cv2.destroyAllWindows()
        
#             # print("loss")
#             loss = criterion(outputs, Ytrain - Xtrain)
#             # print("backward")
#             loss.backward()
#             # print("optimize")
#             optimizer.step()


#     # print statistics
#     running_loss += loss.item()
#     # if i % 1000 == 999:    # print every 2000 mini-batches
#     # if i % 50 == 49:
#     # np.random.shuffle(val_indices)
#     # print("max validation iteration: ",len(val_indices)//batch_size )
#     with torch.set_grad_enabled(False):
#         validation_loss = 0.0
#         for X,Y in validation_loader:
#             # here comes your training loop
#             # print("ValX.shape", X.shape)
#             # print(len(validation_loader))
#             for j in range(len(X)):
#                 Xval = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#                 Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#                 val_outputs = model(Xval)
#                 # print("Xval[1] size ", Xval[1].shape, "val_outputs size ", val_outputs.shape)
#                 # ssim = pytorch_ssim.ssim(Yval, val_outputs)
#                 # print("ssim ", ssim)
#                 # exit()
#                 if(j == 0):
#                     y = np.transpose(Yval[0].detach().cpu().numpy(), [1,2,0])
#                     x = np.transpose((val_outputs[0].detach()).cpu().numpy(), [1,2,0])
#                     x_input = np.transpose((Xval[0].detach()).cpu().numpy(), [1,2,0])
#                     logger.info("Validation: input_psnr: %.5f \t val_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
#                 val_loss = criterion(val_outputs , Yval - Xval)
#                 validation_loss += val_loss.item()
#         validation_loss /= ((len(val_indices)//batch_size-1)*batch_size)
#         logger.info('Epoch: %i \t Iteration: %i training_running_loss: %.6e \t validation_loss: %.6e' %
#               (epoch + 1, i + 1, running_loss / 2000, validation_loss))


#     # save checkpoint
#     torch.save({
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,
#     'val_loss': validation_loss
#     }, os.path.join(model_dir, 'ckpt_model.pth'))

#     # save best model
#     if validation_loss < best_model_loss:
#         torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         'val_loss': validation_loss
#         }, os.path.join(model_dir, 'best_model.pth'))
#         best_model_loss = validation_loss

#         running_loss = 0.0

#     end_timing_epoch = time.time()
#     logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))

# # test loop
# model.eval()
# with torch.set_grad_enabled(False):
#     test_loss = 0.0
#     avg_predicted_psnr = 0.0
#     avg_input_psnr = 0.0
#     avg_predicted_ssim = 0.0
#     avg_input_ssim = 0.0
#     count = 0
#     better_count = 0
#     for X,Y in validation_loader:
#         # here comes your training loop
#         # print("ValX.shape", X.shape)
#         print(len(validation_loader))
#         for j in range(len(X)):
#             Xtest = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#             Ytest = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#             # [N x C x H x W]
#             test_outputs = model(Xtest)

#             # print(test_outputs * 255)
#             # exit()
#             np_images = test_outputs.cpu().numpy()

#             N, _, _, _ = np_images.shape
#             for n in range(N):
#                 filename = os.path.join(test_dir, "test_batch_best_%i.png"%(count))
#                 # residual
#                 cur_img1 = np.transpose(Xtest[n].cpu().numpy(), [1,2,0])
#                 pred_mask1 = np.transpose(np_images[n], [1,2,0])
#                 pred_img1 = pred_mask1  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
#                 gt_img1 = np.transpose(Ytest[n].cpu().numpy(), [1,2,0])

#                 # print("test_outputs size ", test_outputs.shape, "\t Ytest size ", Ytest.shape)
#                 # Stats of Predicted Image

#                 ssim1 = np.mean(np.mean(ssim(gt_img1, pred_img1, full=True,multichannel=True)))
#                 psnr1 = psnr(gt_img1, pred_img1)
#                 avg_predicted_psnr += (psnr1 )
#                 avg_predicted_ssim += (ssim1 )
#                 count += 1

#                 # Stats of Input Image
#                 ssim2 = np.mean(np.mean(ssim(gt_img1, cur_img1, full=True,multichannel=True)))
#                 psnr2 = psnr(gt_img1, cur_img1)
#                 avg_input_psnr += (psnr2)
#                 avg_input_ssim += (ssim2)
#                 print("issim: ", ssim2,  "\t pssim: ", ssim1, "\t ipsnr: ", psnr2,  "\t ppsnr: ", psnr1)

#                 if(psnr2 < psnr1):
#                     better_count += 1

#                 img_pair1 = np.hstack(([cur_img1, pred_mask1, pred_img1, gt_img1])) * 255
#                 # img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
#                 display_img = np.vstack([img_pair1])
#                 # print(pred_mask1.shape)
#                 # print(pred_img1.shape)
#                 # print(gt_img1.shape)
#                 # ipdb.set_trace()
#                 # print(display_img.shape)
#                 # cv2.imshow('sample image', display_img.astype(np.uint8))
#                 # cv2.waitKey(0) # waits until a key is pressed
#                 # cv2.destroyAllWindows()
#                 cv2.imwrite(filename, display_img.astype(np.uint8)) 

#             t_loss = criterion(test_outputs, Ytest)
#             test_loss += t_loss.item()
#         print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
#                 " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
#                 " better count: ", better_count)

#     test_loss /= count
#     logger.info('Test loss: %.5f' % (test_loss))











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
#         Xtrain = torch.from_numpy(X[tindices,:,:,:]).float().to(device)
#         Ytrain = torch.from_numpy(Y[tindices,:,:,:]).float().to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         # print("forward")
#         outputs = model(Xtrain)
#         # print("outputs.shape: ", outputs.shape)
#         # print("Xtrain.shape: ", Xtrain.shape)

        
#         y = np.transpose(Ytrain[0].detach().cpu().numpy(), [1,2,0])
#         x = np.transpose((outputs[0].detach()).cpu().numpy(), [1,2,0])
#         x_input = np.transpose((Xtrain[0].detach()).cpu().numpy(), [1,2,0])
#         print("Training: input_psnr", psnr(y,x_input), "val_psnr", psnr(y,x))
#         img_pair1 = np.hstack(([x_input, y, x])) * 255
#         # display_img = np.vstack([img_pair1, img_pair2])
#         # ipdb.set_trace()
#         # print(display_img.shape)
#         cv2.imshow('sample image', img_pair1.astype(np.uint8))
#         cv2.waitKey(0) # waits until a key is pressed
#         cv2.destroyAllWindows()
    
#         # print("loss")
#         loss = criterion(outputs, Ytrain - Xtrain)
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
#                     Xval = torch.from_numpy(X[vindices,:,:,:]).float().to(device)
#                     Yval = torch.from_numpy(Y[vindices,:,:,:]).float().to(device)
#                     val_outputs = model(Xval)
#                     # print("Xval[1] size ", Xval[1].shape, "val_outputs size ", val_outputs.shape)
#                     # ssim = pytorch_ssim.ssim(Yval, val_outputs)
#                     # print("ssim ", ssim)
#                     # exit()
#                     if(j == 0):
#                         y = np.transpose(Yval[0].detach().cpu().numpy(), [1,2,0])
#                         x = np.transpose((val_outputs[0].detach()).cpu().numpy(), [1,2,0])
#                         x_input = np.transpose((Xval[0].detach()).cpu().numpy(), [1,2,0])
#                         print("Validation: input_psnr", psnr(y,x_input), "val_psnr", psnr(y,x))
#                     val_loss = criterion(val_outputs , Yval - Xval)
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

# # test loop
# model.eval()
# with torch.set_grad_enabled(False):
#     test_loss = 0.0
#     avg_predicted_psnr = 0.0
#     avg_input_psnr = 0.0
#     avg_predicted_ssim = 0.0
#     avg_input_ssim = 0.0
#     count = 0
#     better_count = 0
#     for k in range(len(test_indices)//2-1):
#         filename = os.path.join(test_dir, "test_batch_best_%i.png"%(k))
#         tindices = test_indices[2*k: 2*(k+1)] 
#         Xtest = torch.from_numpy(X[tindices,:,:,:]).float().to(device)
#         Ytest = torch.from_numpy(Y[tindices,:,:,:]).float().to(device)
#         # [N x C x H x W]
#         test_outputs = model(Xtest)

#         # print(test_outputs * 255)
#         # exit()
#         np_images = test_outputs.cpu().numpy()

#         # residual
#         cur_img1 = np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
#         pred_mask1 = np.transpose(np_images[0], [1,2,0])
#         pred_img1 = pred_mask1  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
#         gt_img1 = np.transpose(Ytest[0].cpu().numpy(), [1,2,0])

#         cur_img2 = np.transpose(Xtest[1].cpu().numpy(), [1,2,0])
#         pred_mask2 = np.transpose(np_images[1], [1,2,0])
#         pred_img2 = pred_mask2 #+ np.transpose(Xtest[1].cpu().numpy(), [1,2,0])
#         gt_img2 = np.transpose(Ytest[1].cpu().numpy(), [1,2,0])

#         # print("test_outputs size ", test_outputs.shape, "\t Ytest size ", Ytest.shape)
#         # Stats of Predicted Image

#         ssim1 = np.mean(np.mean(ssim(gt_img1, pred_img1, full=True,multichannel=True)))
#         psnr1 = psnr(gt_img1, pred_img1)
#         ssim2 = np.mean(np.mean(ssim(gt_img2, pred_img2, full=True,multichannel=True)))
#         psnr2 = psnr(gt_img2, pred_img2)
#         avg_predicted_psnr += (psnr1 + psnr2)
#         avg_predicted_ssim += (ssim1 + ssim2)
#         count += 2
#         print("PREDICTED: ssim1 ", ssim1, "\t psnr1: ", psnr1, "\t ssim2 ", ssim2, "\t psnr2: ", psnr2)
#         ppsnr1 = psnr1
#         ppsnr2 = psnr2

#         # Stats of Input Image
#         ssim1 = np.mean(np.mean(ssim(gt_img1, cur_img1, full=True,multichannel=True)))
#         psnr1 = psnr(gt_img1, cur_img1)
#         ssim2 = np.mean(np.mean(ssim(gt_img2, cur_img2, full=True,multichannel=True)))
#         psnr2 = psnr(gt_img2, cur_img2)
#         avg_input_psnr += (psnr1 + psnr2)
#         avg_input_ssim += (ssim1 + ssim2)
#         print("INPUT ssim1 ", ssim1, "\t psnr1: ", psnr1, "\t ssim2 ", ssim2, "\t psnr2: ", psnr2)
#         ipsnr1 = psnr1
#         ipsnr2 = psnr2

#         if(ipsnr1 < psnr1):
#             better_count += 1
#         if(ipsnr2 < psnr2):
#             better_count += 1


#         img_pair1 = np.hstack(([cur_img1, pred_mask1, pred_img1, gt_img1])) * 255
#         img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
#         display_img = np.vstack([img_pair1, img_pair2])
#         # print(pred_mask1.shape)
#         # print(pred_img1.shape)
#         # print(gt_img1.shape)
#         # ipdb.set_trace()
#         # print(display_img.shape)
#         # cv2.imshow('sample image', display_img.astype(np.uint8))
#         # cv2.waitKey(0) # waits until a key is pressed
#         # cv2.destroyAllWindows()
#         cv2.imwrite(filename, display_img.astype(np.uint8)) 
# #         
#         # exit()
#             # test_dir

#         t_loss = criterion(test_outputs, Ytest)
#         test_loss += t_loss.item()
#     print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
#             " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
#             " better count: ", better_count)

# test_loss /= ((len(test_indices)//2-1)*2)
# logger.info('Test loss: %.5f' % (test_loss))
