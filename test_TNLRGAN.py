import torch
import torch.nn as nn
from lib.model import TNLRGAN_G, TNLRGAN_D, TDNLRGAN_G, TDNLRGAN_D, TDKNLRGAN_G, TDKNLRGAN_D
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
import cv2 as cv
from skimage.measure import compare_ssim as ssim
# import ipdb
import h5py

'''
    Command

    python train_ARTN.py --load_from_ckpt "D:\\Github\\tecogan_video_data\\ARTN\\12-20-2019=17-24-35"
    TNL : ../content/drive/My Drive/FYP/TNLRGAN/01-18-2020=11-08-24
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
parser.add_argument('--model', default="TDKNLRGAN", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=100, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data\\TNLRGAN", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
parser.add_argument('--test_dir', default="D:\\Github\\FYP2020\\test_sequence", help='the path to save the dataset')
# parser.add_argument('--input_dir', default=".", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="D:\\Github\\FYP2020\\tecogan_video_data/TDKNLRGAN/01-20-2020=16-59-08", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
parser.add_argument("--sample_interval", type=int, default=30, help="interval between image sampling")
parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx264", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')

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
print("model: ",  Flags.model)
# create model

if Flags.model == 'RGAN':
    generator = RGAN_G(C,C).to(device)
    discriminator = RGAN_D(C,C).to(device)
elif Flags.model == 'TNLRGAN':
    generator = TNLRGAN_G(C,C,Flags.tseq_length>0, Flags.tseq_length).to(device)
    discriminator = TNLRGAN_D(C,C,img_size=512).to(device)
elif Flags.model == 'TDNLRGAN':
    generator = TDNLRGAN_G(C,C,Flags.tseq_length>0, Flags.tseq_length).to(device)
    discriminator = TDNLRGAN_D(C,C,img_size=512).to(device)

elif Flags.model == 'TDKNLRGAN':
    generator = TDKNLRGAN_G(C,C,Flags.tseq_length>0, Flags.tseq_length,device=device).to(device)
    discriminator = TDKNLRGAN_D(C,C,img_size=512).to(device)

criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
# criterion = nn.MSELoss()

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

if(Flags.model == 'RGAN'):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Flags.lr/4, betas=(Flags.b1, Flags.b2))
    # optimizer_G = torch.optim.Adam([
    #                                 {'params': generator.base.parameters()},
    #                                 {'params': generator.last.parameters(), 'lr': Flags.lr * 0.1},
    #                                 ], lr=Flags.lr) # , betas=(Flags.b1, Flags.b2)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))
elif(Flags.model == 'TNLRGAN' or Flags.model == 'TDKNLRGAN'):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Flags.lr/4, betas=(Flags.b1, Flags.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))

# if the checkpoint dir is not null refers to load checkpoint
if Flags.load_from_ckpt != "":
    summary_dir = Flags.load_from_ckpt
    checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/ckpt_model.pth'))
    # checkpoint = torch.load(os.path.join(Flags.load_from_ckpt, 'model/best_model.pth'))
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


from torch.utils import data
from lib.dataloader import HDF5Dataset
from torch.utils.data.sampler import SubsetRandomSampler


# dataset = HDF5Dataset(Flags.input_dir, recursive=False, load_data=False, 
#    data_cache_size=128, transform=None)

# shuffle_dataset = True
# # Creating data indices for training and validation and test splits:
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# if shuffle_dataset :
#     np.random.seed(0)
#     np.random.shuffle(indices)
# # indices = np.arange(N)
# # np.random.shuffle(indices)

# train_indices = indices[: int(dataset_size * 0.7)]
# val_indices = indices[int(dataset_size * 0.7): int(dataset_size * 0.9)]
# test_indices = indices[int(dataset_size * 0.9):]

# # Creating PT data samplers and loaders:
# # train_sampler = SubsetRandomSampler(train_indices)
# # valid_sampler = SubsetRandomSampler(val_indices)
# test_sampler = SubsetRandomSampler(test_indices)

# # train_loader_params = {'batch_size': 64, 'num_workers': 6,'sampler': train_sampler}
# # validation_loader_params = {'batch_size': 32, 'num_workers': 6,'sampler': valid_sampler}
# test_loader_params = {'batch_size': 2, 'num_workers': 6,'sampler': test_sampler}

# # train_loader = data.DataLoader(dataset, **train_loader_params)
# # validation_loader = data.DataLoader(dataset, **validation_loader_params)
# test_loader = data.DataLoader(dataset, **test_loader_params)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


start_timing_epoch = time.time()
running_g_loss = 0.0
running_d_loss = 0.0
running_rec_loss = 0.0

test_rec_loss = 0.0
num_train_batches = 0
cumulative_psnr_codec = 0.0
cumulative_psnr_pred = 0.0
num_samples = 0

def test_image_loader(stride=128):
    if Flags.vcodec == 'libx264':
        hr_test_dir = os.path.join(os.path.join(Flags.test_dir, "AVC"), 'original')
        hr_test_frames_dir = os.path.join(os.path.join(Flags.test_dir, "AVC"), 'test_frames')
        lr_test_dir = os.path.join(os.path.join(Flags.test_dir, "AVC"), 'qp'+str(Flags.qp))
        lr_test_frames_dir = os.path.join(os.path.join(Flags.test_dir, "AVC"), 'test_frames_qp'+str(Flags.qp))
    if Flags.vcodec == 'libx265':
        hr_test_dir = os.path.join(os.path.join(Flags.test_dir, "HEVC"), 'original')
        hr_test_frames_dir = os.path.join(os.path.join(Flags.test_dir, "HEVC"), 'test_frames')
        lr_test_dir = os.path.join(os.path.join(Flags.test_dir, "HEVC"), 'qp'+str(Flags.qp))
        lr_test_frames_dir = os.path.join(os.path.join(Flags.test_dir, "HEVC"), 'test_frames_qp'+str(Flags.qp))
    # process the number of images in the file
    print(lr_test_frames_dir)
    total_test_images = 0
    test_image_name_list = []
    for filename in os.listdir(lr_test_frames_dir):
        if('.png' in filename):
            test_image_name_list.append(filename)
            total_test_images += 1
    # print(len(test_image_name_list))

    if(Flags.model == 'TDKNLRGAN'):
        dt = Flags.tseq_length//2
        for t in range(dt, len(test_image_name_list) - dt):
            # load image
            temp_input_img_patch_list = []
            temp_gt_img_patch_list = []
            # load three lr images and one gt image
            lr_images_name = test_image_name_list[t-dt:t+dt+1]
            # print("read image: ", image_path)
            # print(len(lr_images_name))
            # exit()
            
            for name in lr_images_name:
                input_image_path = os.path.join(lr_test_frames_dir, name)
                input_image = cv.imread(input_image_path, cv.IMREAD_UNCHANGED)
                h,w,c = input_image.shape
                input_yuv_image = cv.cvtColor(input_image, cv.COLOR_RGB2YUV)
                y_input, _, _ = cv.split(input_yuv_image)
                input_image = np.expand_dims(y_input, axis=2)
                temp_input_img_patch_list.append(input_image)

            gt_image_path = os.path.join(hr_test_frames_dir, test_image_name_list[t])

            gt_image = cv.imread(gt_image_path, cv.IMREAD_UNCHANGED)
            h,w,c = gt_image.shape
            gt_yuv_image = cv.cvtColor(gt_image, cv.COLOR_RGB2YUV)

            y_gt, _, _ = cv.split(gt_yuv_image)
            gt_image = np.expand_dims(y_gt, axis=2)

            temp_gt_img_patch_list.append(gt_image)

            if len(temp_input_img_patch_list) == 0:
                continue
            # T x H x W x C
            temporal_input_img_patch_list = np.stack(temp_input_img_patch_list, axis=0)
            temporal_gt_img_patch_list = np.stack(temp_gt_img_patch_list, axis=0)

            # print(temporal_input_img_patch_list.shape)
            # print(temporal_gt_img_patch_list.shape)

            h_length = h // stride
            w_length = w // stride

            input_img_patch = None
            gt_img_patch = None

            for hind in range(h_length):
                for wind in range(w_length):
                    x1 = stride * wind
                    x2 = stride * wind + stride
                    y1 = stride * hind
                    y2 = stride * hind + stride

                    temporal_input = np.array(temporal_input_img_patch_list[:,y1:y2, x1:x2,:])
                    temporal_gt = np.array(temporal_gt_img_patch_list[:,y1:y2, x1:x2,:])
                    # print(temporal_input.shape)
                    # print(temporal_gt.shape)
                    if input_img_patch is None:
                        input_img_patch = np.expand_dims(temporal_input, axis=1)
                    else:
                        input_img_patch = np.concatenate([input_img_patch, np.expand_dims(temporal_input, axis=1)], axis = 1)

                    if gt_img_patch is None:
                        gt_img_patch = np.expand_dims(temporal_gt, axis=0)
                        # gt_img_patch = np.expand_dims(temporal_gt, axis=1)
                    else:
                        gt_img_patch = np.concatenate([gt_img_patch, np.expand_dims(temporal_gt, axis=1)], axis = 1)
            #         print(input_img_patch.shape)
            #         print(gt_img_patch.shape)
            # print(input_img_patch.shape)
            # print(gt_img_patch.shape)

            yield h_length, w_length, input_img_patch, gt_img_patch

# test_image_loader()

validation_loss = 0.0
num_test_batches = 0

for epoch, (h_length, w_length, X, Y) in enumerate(test_image_loader(stride=512)):
    with torch.set_grad_enabled(False):
        test_g_loss = 0.0
        test_d_loss = 0.0
        num_test_batches+=1
        # here comes your test loop

        # ------------------------------------
        #
        #   Train Generator
        #
        #-------------------------------------
        X = Tensor(X)
        Y = Tensor(Y)
        # Xval = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
        # Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
        Xtest = (X[:,:,:,:,:].permute([1,4,0,2,3])/255.0).float().to(device)
        Ytest = (Y[0,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
        # print(Xtest.shape)
        # exit()

        # Configure input
        real_imgs = Ytest

        # Adversarial ground truths
        valid = Variable(Tensor(Xtest.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(Ytest.shape[0], 1).fill_(0.0), requires_grad=False)

        # zero the parameter gradients
        optimizer_G.zero_grad()

        # Generate a batch of images
        gen_imgs = generator(Xtest)

        real_pred = discriminator(real_imgs).detach()
        fake_pred = discriminator(gen_imgs)

        if Flags.rel_avg_gan:
            g_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), valid)
        else:
            g_loss = adversarial_loss(fake_pred - real_pred, valid)

        # Loss measures generator's ability to fool the discriminator
        reconstruction_loss = criterion(Ytest, gen_imgs)
        g_loss += reconstruction_loss

        # ---------------------
        #
        #  Train Discriminator
        #
        # ---------------------

        optimizer_D.zero_grad()

        # Predict validity
        real_pred = discriminator(real_imgs)
        fake_pred = discriminator(gen_imgs.detach())

        if Flags.rel_avg_gan:
            real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), valid)
            fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), fake)
        else:
            real_loss = adversarial_loss(real_pred - fake_pred, valid)
            fake_loss = adversarial_loss(fake_pred - real_pred, fake)

        d_loss = (real_loss + fake_loss) / 2

        for p in range(len(Ytest)):
            y = np.transpose(Ytest[p].detach().cpu().numpy(), [1,2,0])
            x = np.transpose((gen_imgs[p].detach()).cpu().numpy(), [1,2,0])
            x_input = np.transpose((Xtest[p,:,Flags.tseq_length//2].detach()).cpu().numpy(), [1,2,0])
            psnr_codec = psnr(y,x_input)
            psnr_pred =  psnr(y,x)
            logger.info("test_patch: input_psnr: %.5f \t test_psnr: %.5f"%(psnr_codec, psnr_pred))
        save_image(Ytest.data[:],  os.path.join(test_dir,"%d_test_seq_gt.png") % (epoch), nrow=w_length, normalize=True,padding=0)
        save_image(Xtest[:,:,Flags.tseq_length//2].data[:],  os.path.join(test_dir,"%d_test_seq_lr.png") % (epoch), nrow=w_length, normalize=True,padding=0)
        save_image(gen_imgs.data[:],  os.path.join(test_dir,"%d_test_seq_hr.png") % (epoch), nrow=w_length, normalize=True,padding=0)

        gt_image = cv.imread(os.path.join(test_dir,"%d_test_seq_gt.png") % (epoch))
        lr_image = cv.imread(os.path.join(test_dir,"%d_test_seq_lr.png") % (epoch))
        pred_image = cv.imread(os.path.join(test_dir,"%d_test_seq_hr.png") % (epoch))
        
        # cv.imshow('SUCCESS sample image',pred_image)
        # cv.waitKey(0) # waits until a key is pressed
        # cv.destroyAllWindows()    
        psnr_codec = psnr(gt_image/255.0,lr_image/255.0)
        psnr_pred =  psnr(gt_image/255.0,pred_image/255.0)
        cumulative_psnr_codec += psnr_codec
        cumulative_psnr_pred += psnr_pred
        num_samples += 1        
        logger.info("test_full_image: input_psnr: %.5f \t test_psnr: %.5f"%(psnr_codec, psnr_pred))
        # exit()

        running_rec_loss += reconstruction_loss.item()
        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

    test_rec_loss =  running_rec_loss / num_test_batches
    test_g_loss = running_g_loss / num_test_batches
    test_d_loss =  running_d_loss/ num_test_batches

    logger.info("[Epoch %d] [tD loss: %.5e] [tG loss: %.5e] [tRec loss: %.5e] [tpsnr_codec : %.5f] [tpsnr_pred : %.5f]" 
        % (epoch, test_d_loss, test_g_loss, test_rec_loss, cumulative_psnr_codec/ num_samples, cumulative_psnr_pred/ num_samples))



    end_timing_epoch = time.time()
    logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))

# for epoch in range(1):
#     with torch.set_grad_enabled(False):
#         validation_loss = 0.0
#         num_test_batches = 0
#         for k, (X,Y) in enumerate(test_loader):
#             test_length = len(Y)
#             num_test_batches += test_length
#             for j in range(test_length):
#                 # here comes your validation loop

#                 # ------------------------------------
#                 #
#                 #   Train Generator
#                 #
#                 #-------------------------------------

#                 # Xval = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#                 # Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#                 Xtest = (X[j,:,:,:,:,:].permute([1,4,0,2,3])/255.0).float().to(device)
#                 Ytest = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#                 print(Xtest.shape)
#                 exit()

#                 # Configure input
#                 real_imgs = Ytest

#                 # Adversarial ground truths
#                 valid = Variable(Tensor(Xtest.shape[0], 1).fill_(1.0), requires_grad=False)
#                 fake = Variable(Tensor(Ytest.shape[0], 1).fill_(0.0), requires_grad=False)

#                 # zero the parameter gradients
#                 optimizer_G.zero_grad()

#                 # Generate a batch of images
#                 gen_imgs = generator(Xtest)

#                 real_pred = discriminator(real_imgs).detach()
#                 fake_pred = discriminator(gen_imgs)

#                 if Flags.rel_avg_gan:
#                     g_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), valid)
#                 else:
#                     g_loss = adversarial_loss(fake_pred - real_pred, valid)

#                 # Loss measures generator's ability to fool the discriminator
#                 reconstruction_loss = criterion(Ytest, gen_imgs)
#                 g_loss += reconstruction_loss

#                 # ---------------------
#                 #
#                 #  Train Discriminator
#                 #
#                 # ---------------------

#                 optimizer_D.zero_grad()

#                 # Predict validity
#                 real_pred = discriminator(real_imgs)
#                 fake_pred = discriminator(gen_imgs.detach())

#                 if Flags.rel_avg_gan:
#                     real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), valid)
#                     fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), fake)
#                 else:
#                     real_loss = adversarial_loss(real_pred - fake_pred, valid)
#                     fake_loss = adversarial_loss(fake_pred - real_pred, fake)

#                 d_loss = (real_loss + fake_loss) / 2

#                 for p in range(len(Ytest)):
#                     y = np.transpose(Ytest[p].detach().cpu().numpy(), [1,2,0])
#                     x = np.transpose((gen_imgs[p].detach()).cpu().numpy(), [1,2,0])
#                     x_input = np.transpose((Xtest[p,:,Flags.tseq_length//2].detach()).cpu().numpy(), [1,2,0])
#                     psnr_codec = psnr(y,x_input)
#                     psnr_pred =  psnr(y,x)
#                     cumulative_psnr_codec += psnr_codec
#                     cumulative_psnr_pred += psnr_pred
#                     num_samples += 1
#                     logger.info("test: input_psnr: %.5f \t test_psnr: %.5f"%(psnr_codec, psnr_pred))
#                 save_image(Ytest.data[:],  os.path.join(test_dir,"test_gt_%d_%d.png") % (epoch, j), nrow=4, normalize=True)
#                 save_image(Xtest[:,:,Flags.tseq_length//2].data[:25],  os.path.join(test_dir,"test_lr_%d_%d.png") % (epoch, j), nrow=4, normalize=True)
#                 save_image(gen_imgs.data[:],  os.path.join(test_dir,"test_hr_%d_%d.png") % (epoch, j), nrow=4, normalize=True)

#                 test_rec_loss += reconstruction_loss.item()
#                 test_g_loss += g_loss.item()
#                 test_d_loss += d_loss.item()
#         test_rec_loss /= num_test_batches
#         test_g_loss /= num_test_batches
#         test_d_loss /= num_test_batches

#         logger.info("[Epoch %d] [tD loss: %.5e] [tG loss: %.5e] [tRec loss: %.5e] [tpsnr_codec : %.7e] [tpsnr_pred : %.7e]" 
#             % (epoch, test_d_loss, test_g_loss, test_rec_loss, cumulative_psnr_codec/ num_samples, cumulative_psnr_pred/ num_samples))

#     end_timing_epoch = time.time()
#     logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))

# test loop
# model.eval()
# with torch.set_grad_enabled(False):
#     test_loss = 0.0
#     avg_predicted_psnr = 0.0
#     avg_input_psnr = 0.0
#     avg_predicted_ssim = 0.0
#     avg_input_ssim = 0.0
#     count = 0
#     better_count = 0
#     for X,Y in :
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
