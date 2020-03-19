#----------------------- Description -----------------------------------------
#   The model is trained with Relativistic GAN loss and L1 reconstruction loss




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
import cv2
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
# ../content/drive/My Drive/FYP/TDKNLRGAN/01-20-2020=16-59-08
parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser = argparse.ArgumentParser()
parser.add_argument('--model', default="TNLGAN", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=100, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default=".", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--adv_coeff", type=float, default=0.999, help="coefficient of adversarial loss")
parser.add_argument("--rec_coeff", type=float, default=0.999, help="coefficient of reconstruction loss")
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
parser.add_argument("--tseq_length", type=int, default=11, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx265", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=1, type=int, help='scene duration')
parser.add_argument("--sample_interval", type=int, default=30, help="interval between image sampling")
parser.add_argument("--img_size", type=int, default=256, help="interval between image sampling")

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

if Flags.model == 'GAN':
    generator = RGAN_G(C,C).to(device)
    discriminator = RGAN_D(C,C).to(device)
elif Flags.model == 'TNLGAN':
    generator = TNLRGAN_G(C,C,Flags.tseq_length>0, Flags.tseq_length).to(device)
    discriminator = TNLRGAN_D(C,C,img_size=Flags.img_size).to(device)
elif Flags.model == 'TDNLGAN':
    generator = TDNLRGAN_G(C,C,Flags.tseq_length>0, Flags.tseq_length).to(device)
    discriminator = TDNLRGAN_D(C,C,img_size=Flags.img_size).to(device)

elif Flags.model == 'TDKNLGAN':
    generator = TDKNLRGAN_G(C,C,Flags.tseq_length>0, Flags.tseq_length,device=device).to(device)
    discriminator = TDKNLRGAN_D(C,C,img_size=Flags.img_size).to(device)

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
    # optimizer_G = torch.optim.Adam([
    #                                 {'params': generator.base.parameters()},
    #                                 {'params': generator.last.parameters(), 'lr': Flags.lr * 0.1},
    #                                 ], lr=Flags.lr) # , betas=(Flags.b1, Flags.b2)
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

if 'drive' in Flags.input_dir:
    input_dir = '.'
else:
    save_dir = os.path.join(Flags.input_dir, "dataset_{}_qp{}".format(Flags.vcodec,Flags.qp))
    # input_dir = os.path.join(os.path.join(save_dir, Flags.model), '{}_qp{}'.format(Flags.vcodec,str(Flags.qp)))
    input_dir = os.path.join(os.path.join(save_dir, 'TDKNLRGAN'), '{}_qp{}'.format(Flags.vcodec,str(Flags.qp)))

print(input_dir)

dataset = HDF5Dataset(input_dir, recursive=False, load_data=False, 
   data_cache_size=128, transform=None)

shuffle_dataset = True
# Creating data indices for training and validation and test splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
if shuffle_dataset :
    np.random.seed(0)
    np.random.shuffle(indices)
# indices = np.arange(N)
# np.random.shuffle(indices)

train_indices = indices[: int(dataset_size * 0.1)]
val_indices = indices[int(dataset_size * 0.7): int(dataset_size * 0.9)]
test_indices = indices[int(dataset_size * 0.9):]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader_params = {'batch_size': 64, 'num_workers': 6,'sampler': train_sampler}
validation_loader_params = {'batch_size': 32, 'num_workers': 6,'sampler': valid_sampler}

train_loader = data.DataLoader(dataset, **train_loader_params)
validation_loader = data.DataLoader(dataset, **validation_loader_params)

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


for epoch in range(st_epoch,Flags.epoch):
    start_timing_epoch = time.time()
    running_g_loss = 0.0
    running_d_loss = 0.0
    running_rec_loss = 0.0
    val_g_loss = 0.0
    val_d_loss = 0.0
    val_rec_loss = 0.0
    num_train_batches = 0
    for m, (X,Y) in enumerate(train_loader):
        # here comes your training loop
        train_length = len(Y)
        num_train_batches += train_length
        for i in range(train_length):

            # ------------------------------------
            #
            #   Train Generator
            #
            #-------------------------------------
            # Xtrain = (X[i,:,:,:,:,:].permute([1,4,0,2,3]))

            Xtrain = (X[i,:,:,:,:,:].permute([1,4,0,2,3])/255.0).float().to(device)
            Ytrain = (Y[i,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
            # print(Xtrain.shape)
            # print(Ytrain.shape)
            # exit()

            # Configure input
            real_imgs = Ytrain

            # Adversarial ground truths
            valid = Variable(Tensor(Xtrain.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(Ytrain.shape[0], 1).fill_(0.0), requires_grad=False)

            # zero the parameter gradients
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(Xtrain)
            # print(gen_imgs.shape)
            # print(Ytrain.shape)
            # exit()


            real_pred = discriminator(real_imgs).detach()
            fake_pred = discriminator(gen_imgs)

            g_loss = adversarial_loss(fake_pred, valid)

            # Loss measures generator's ability to fool the discriminator
            reconstruction_loss = criterion(Ytrain, gen_imgs)
            g_loss = Flags.adv_coeff * g_loss + Flags.rec_coeff * reconstruction_loss 


            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #
            #  Train Discriminator
            #
            # ---------------------

            optimizer_D.zero_grad()

            # Predict validity
            real_pred = discriminator(real_imgs)
            fake_pred = discriminator(gen_imgs.detach())

            real_loss = adversarial_loss(real_pred, valid)
            fake_loss = adversarial_loss(fake_pred, fake)

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()
            logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.5e] [G loss: %.5e] [Rec loss: %.5e]"
                % (epoch, Flags.epoch, m, len(train_loader), d_loss.item(), g_loss.item(), reconstruction_loss.item())
            )

            batches_done = epoch * len(train_loader) + m + i 

            if batches_done % Flags.sample_interval == 0:
                y = np.transpose(Ytrain[0].detach().cpu().numpy(), [1,2,0])
                x = np.transpose((gen_imgs[0].detach()).cpu().numpy(), [1,2,0])
                x_input = np.transpose((Xtrain[0,:,Flags.tseq_length//2].detach()).cpu().numpy(), [1,2,0])
                logger.info("Training: input_psnr: %.5f \t train_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
                save_image(gen_imgs.data[:25],  os.path.join(test_dir,"train_%d.png") % batches_done, nrow=5, normalize=True)


    running_g_loss += g_loss.item()
    running_d_loss += d_loss.item()
    running_rec_loss += reconstruction_loss.item()

    with torch.set_grad_enabled(False):
        validation_loss = 0.0
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

                # Xval = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
                # Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
                Xval = (X[j,:,:,:,:,:].permute([1,4,0,2,3])/255.0).float().to(device)
                Yval = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
                # print(Xtrain.shape)
                # exit()

                # Configure input
                real_imgs = Yval

                # Adversarial ground truths
                valid = Variable(Tensor(Xval.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(Yval.shape[0], 1).fill_(0.0), requires_grad=False)

                # zero the parameter gradients
                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(Xval)

                real_pred = discriminator(real_imgs).detach()
                fake_pred = discriminator(gen_imgs)

                g_loss = adversarial_loss(fake_pred, valid)

                # Loss measures generator's ability to fool the discriminator
                reconstruction_loss = criterion(Yval, gen_imgs)
                g_loss = Flags.adv_coeff * g_loss + Flags.rec_coeff * reconstruction_loss 

                # ---------------------
                #
                #  Train Discriminator
                #
                # ---------------------

                optimizer_D.zero_grad()

                # Predict validity
                real_pred = discriminator(real_imgs)
                fake_pred = discriminator(gen_imgs.detach())

                real_loss = adversarial_loss(real_pred, valid)
                fake_loss = adversarial_loss(fake_pred, fake)

                d_loss = (real_loss + fake_loss) / 2

                if(j == 0):
                    y = np.transpose(Yval[0].detach().cpu().numpy(), [1,2,0])
                    x = np.transpose((gen_imgs[0].detach()).cpu().numpy(), [1,2,0])
                    x_input = np.transpose((Xval[0,:,Flags.tseq_length//2].detach()).cpu().numpy(), [1,2,0])
                    logger.info("Validation: input_psnr: %.5f \t val_psnr: %.5f"%(psnr(y,x_input), psnr(y,x)))
                    save_image(gen_imgs.data[:25],  os.path.join(test_dir,"val_%d_%d.png") % (epoch, j), nrow=5, normalize=True)

                val_rec_loss += reconstruction_loss.item()
                val_g_loss += g_loss.item()
                val_d_loss += d_loss.item()
        val_rec_loss /= num_val_batches
        val_g_loss /= num_val_batches
        val_d_loss /= num_val_batches

        logger.info("[Epoch %d/%d] [tD loss: %.5e] [tG loss: %.5e] [tRec loss: %.5e] [vD loss: %.5e] [vG loss: %.5e] [vRec loss: %.5e]" 
            % (epoch, Flags.epoch, running_d_loss / num_train_batches, running_g_loss / num_train_batches,
            running_rec_loss / num_train_batches, val_d_loss, val_g_loss, val_rec_loss))


    # save checkpoint
    torch.save({
    'epoch': epoch,
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_state_dict': optimizer_D.state_dict(),
    'rec_loss': running_rec_loss / num_train_batches,
    'd_loss': running_d_loss / num_train_batches,
    'g_loss': running_g_loss / num_train_batches,
    'val_rec_loss': val_rec_loss,
    'val_d_loss': val_d_loss,
    'val_g_loss': val_g_loss,
    }, os.path.join(model_dir, 'ckpt_model.pth'))

    # save best model
    if((val_d_loss < best_model_loss[0]) and (val_g_loss < best_model_loss[1]) and (val_rec_loss < best_model_loss[2])):
        torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'rec_loss': running_rec_loss / num_train_batches,
        'd_loss': running_d_loss / num_train_batches,
        'g_loss': running_g_loss / num_train_batches,
        'val_rec_loss': val_rec_loss,
        'val_d_loss': val_d_loss,
        'val_g_loss': val_g_loss,
        }, os.path.join(model_dir, 'best_model.pth'))
        best_model_loss = [val_d_loss, val_g_loss, val_rec_loss]
        running_loss = 0.0

    end_timing_epoch = time.time()
    logger.info("Epoch %i runtime: %.3f"% (epoch+1, end_timing_epoch - start_timing_epoch))

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
