import torch
import torch.nn as nn
from lib.model import  RGAN_D, RGAN_G
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

# parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # parser = argparse.ArgumentParser()
# parser.add_argument('--model', default="ARCNN", type=str, help='the path to save the dataset')
# parser.add_argument('--epoch', default=1000, type=int, help='number of training epochs')
# parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
# parser.add_argument('--disk_path', default="D:\\Github\\FYP2020\\tecogan_video_data", help='the path to save the dataset')
# parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
# # parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# # parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
# parser.add_argument('--load_from_ckpt', default="D:/Github/FYP2020/tecogan_video_data/ARCNN/01-28-2020=15-25-51", type=str, help='ckpt model directory')
# parser.add_argument('--duration', default=120, type=int, help='scene duration')
# parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
# parser.add_argument('--vcodec', default="libx264", help='the path to save the dataset')
# parser.add_argument('--qp', default=37, type=int, help='scene duration')
# parser.add_argument('--channel', default=1, type=int, help='scene duration')


parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser = argparse.ArgumentParser()
parser.add_argument('--model', default="RGAN", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=1000, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="/media/data3/tjtanaa/tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="/media/data3/tjtanaa/tecogan_video_data", type=str, help='output and log directory')
parser.add_argument('--test_dir', default="/home/tan/tjtanaa/tjtanaa/DeepVideoPrior/data/artifact_removal/send/blur_003", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="/media/data3/tjtanaa/tecogan_video_data/RGAN/04-30-2020=14-39-42", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
# parser.add_argument("--sample_interval", type=int, default=30, help="interval between image sampling")
parser.add_argument("--tseq_length", type=int, default=11, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx265", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=3, type=int, help='scene duration')
parser.add_argument("--sample_interval", type=int, default=30, help="interval between image sampling")
parser.add_argument('--max_iteration', default=100000, type=int, help='number of training epochs')
parser.add_argument("--mse_lambda", type=float, default=1, help="interval between image sampling")
parser.add_argument("--rec_lambda", type=float, default=1e-3, help="interval between image sampling")


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

# def load_dataset(model = 'ARCNN'):
#     if model == 'ARCNN' or model == 'FastARCNN' or model == 'VRCNN':    
#         save_path = os.path.join(Flags.input_dir, 'ARCNN')
#         input_path = os.path.join(save_path, 'backup/data_yuv.h5')
#         input_images, gt_images = read_many_hdf5(input_path)
#         print("input_images.shape: ", str(input_images.shape), "\t gt_images.shape: " + str(gt_images.shape))

#         return np.transpose(input_images, [0, 3, 1, 2])/255.0, np.transpose(gt_images, [0, 3, 1, 2])/255.0



# cuda devide
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: ", device)

# define training parameters
best_model_loss = [999999, 99999, 99999] # d, g, rec
batch_size = Flags.mini_batch
st_epoch = 0 # starting epoch
iteration_count = 0
# number of input channel
C = Flags.channel

# create model

if Flags.model == 'RGAN':
    generator = RGAN_G(C,C).to(device)
    discriminator = RGAN_D(C,C,img_size=256).to(device)
elif Flags.model == 'RGANn':
    Flags.rel_avg_gan = False
    generator = RGAN_G(C,C).to(device)
    discriminator = RGAN_D(C,C,img_size=256).to(device)

criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
# criterion = nn.MSELoss()

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

if(Flags.model == 'RGAN' or Flags.model == 'RGANn'):
    # # Optimizers
    # optimizer_G = torch.optim.Adam([
    #                                 {'params': generator.base.parameters()},
    #                                 {'params': generator.last.parameters(), 'lr': Flags.lr * 0.1},
    #                                 ], lr=Flags.lr, betas=(Flags.b1, Flags.b2))
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
    iteration_count = checkpoint['iteration']
    best_model_loss[0] = checkpoint['val_d_loss']
    best_model_loss[1] = checkpoint['val_g_loss']
    best_model_loss[2] = checkpoint['val_rec_loss']
    generator.train()
    discriminator.train()
    print("iteration: " + str(iteration_count))
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

# # load the test_dir image
# # generator
# def get_girl_frames(test_dir):

#     for k, fname in enumerate(os.listdir(test_dir)):
#         if fname.find('.png') != -1:
#             # print("read image: ", image_path)
#             input_image_path = os.path.join(test_dir, fname)
#             gt_image_path = os.path.join(os.path.join(test_dir, 'gt'), fname)
#             # print('input_image_path: ', input_image_path)
#             # print("gt_image_path: ", gt_image_path)
#             # read current image
#             input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
#             gt_image = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
#             h,w,c = input_image.shape
#             # print(input_image.shape)
#             # if w >3840-1:
#             #     # do not load 2k videos
#             #     break
#             input_yuv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YUV)
#             gt_yuv_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2YUV)
#             # print(img_yuv.shape)
#             y_input, _, _ = cv2.split(input_yuv_image)
#             y_gt, _, _ = cv2.split(gt_yuv_image)
#             # convert (H,W) to (1,H,W,1)
#             input_image = np.expand_dims(np.expand_dims(y_input, axis=2), axis=0)
#             gt_image = np.expand_dims(np.expand_dims(y_gt, axis=2),axis=0)
#             block_size = 256
#             for h_ind in range(0,h//block_size-1):
#                 for w_ind in range(0,w//block_size-1):
#                     yield input_image[:,h_ind*block_size : (h_ind+1)*block_size, w_ind*block_size: (w_ind+1)*block_size,:], gt_image[:,h_ind*block_size : (h_ind+1)*block_size, w_ind*block_size: (w_ind+1)*block_size,:]

# load the test_dir image
# generator
def get_girl_frames(test_dir):

    for k, fname in enumerate(os.listdir(test_dir)):
        if fname.find('.png') != -1:
            # print("read image: ", image_path)
            input_image_path = os.path.join(test_dir, fname)
            gt_image_path = os.path.join(os.path.join(test_dir, 'gt'), fname)
            # print('input_image_path: ', input_image_path)
            # print("gt_image_path: ", gt_image_path)
            # read current image
            input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
            gt_image = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
            h,w,c = input_image.shape
            # print(input_image.shape)
            # if w >3840-1:
            #     # do not load 2k videos
            #     break
            input_yuv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YUV)
            gt_yuv_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2YUV)
            # print(img_yuv.shape)
            y_input, _, _ = cv2.split(input_yuv_image)
            y_gt, _, _ = cv2.split(gt_yuv_image)
            # convert (H,W) to (1,H,W,1)
            input_image = np.expand_dims(np.expand_dims(y_input, axis=2), axis=0)
            gt_image = np.expand_dims(np.expand_dims(y_gt, axis=2),axis=0)
            block_size = 256
            for h_ind in range(0,h//block_size-1):
                for w_ind in range(0,w//block_size-1):
                    yield input_image[:,h_ind*block_size : (h_ind+1)*block_size, w_ind*block_size: (w_ind+1)*block_size,:], gt_image[:,h_ind*block_size : (h_ind+1)*block_size, w_ind*block_size: (w_ind+1)*block_size,:]



def get_blur_frames(test_dir):
    # fname_list = ['col_high_0097.png', 'col_high_0098.png', 'col_high_0099.png', 'col_high_0100.png', 'col_high_0101.png']
    for k, fname in enumerate(os.listdir(test_dir)):
    # for k, fname in enumerate(fname_list):
        # print(fname)
        if fname.find('.png') != -1:
            # print("read image: ", image_path)
            print("read file: ", fname)
            input_image_path = os.path.join(test_dir, fname)
            gt_image_path = os.path.join(os.path.join(test_dir, 'gt'), fname)
            # print('input_image_path: ', input_image_path)
            # print("gt_image_path: ", gt_image_path)
            # read current image
            input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
            gt_image = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
            input_image = cv2.resize(input_image, (512,512), interpolation = cv2.INTER_AREA)
            # gt_image = cv2.resize(gt_image, (512,512), interpolation = cv2.INTER_AREA)

            print("input_image.size ", input_image.shape)
            yield input_image, gt_image, 'blur_003', fname

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# get_girl_frames(Flags.test_dir)

start_timing_epoch = time.time()
running_g_loss = 0.0
running_d_loss = 0.0
running_rec_loss = 0.0
val_g_loss = 0.0
val_d_loss = 0.0
val_rec_loss = 0.0
num_train_batches = 0
# avg_psnr = 0.0
# avg_psnr_model = 0.0
# avg_ssim = 0.0
# avg_ssim_model = 0.0
avg_predicted_psnr = 0.0
avg_input_psnr = 0.0
avg_predicted_ssim = 0.0
avg_input_ssim = 0.0
count = 0
count = 0
better_count = 0
with torch.set_grad_enabled(False):
    validation_loss = 0.0
    num_val_batches = 0

    for k, (X, Y, folder_name, fname) in enumerate(get_blur_frames(Flags.test_dir)):
        val_length = len(Y)
        num_val_batches += val_length
    # for k, (X,Y) in enumerate(get_girl_frames(Flags.test_dir)):
    #     val_length = len(Y)
    #     num_val_batches += val_length
        # here comes your validation loop

        # ------------------------------------
        #
        #   Train Generator
        #
        #-------------------------------------

        X = Tensor(X).unsqueeze(0)
        Y = Tensor(Y).unsqueeze(0)

        Xtest = (X[:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
        Ytest = (Y[:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
        test_outputs = generator(Xtest)

        # X = Tensor(X)
        # Y = Tensor(Y)

        # Xtest = (X[:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
        # Ytest = (Y[:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
        # # print(Xtrain.shape)
        # # exit()

        # # Configure input
        # real_imgs = Ytest

        # # Adversarial ground truths
        # valid = Variable(Tensor(Xtest.shape[0], 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(Ytest.shape[0], 1).fill_(0.0), requires_grad=False)

        # zero the parameter gradients
        # optimizer_G.zero_grad()

        # Generate a batch of images
        # gen_imgs = generator(Xtest)

        # real_pred = discriminator(real_imgs).detach()
        # fake_pred = discriminator(gen_imgs)

        # if Flags.rel_avg_gan:
        #     g_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), valid)
        # else:
        #     g_loss = adversarial_loss(fake_pred - real_pred, valid)

        # # Loss measures generator's ability to fool the discriminator
        # reconstruction_loss = criterion(Ytest, gen_imgs)
        # g_loss = (Flags.mse_lambda * reconstruction_loss) + (Flags.rec_lambda * g_loss)

        # # ---------------------
        # #
        # #  Train Discriminator
        # #
        # # ---------------------

        # optimizer_D.zero_grad()

        # # Predict validity
        # real_pred = discriminator(real_imgs)
        # fake_pred = discriminator(gen_imgs.detach())

        # if Flags.rel_avg_gan:
        #     real_loss = adversarial_loss(real_pred - fake_pred.mean(0, keepdim=True), valid)
        #     fake_loss = adversarial_loss(fake_pred - real_pred.mean(0, keepdim=True), fake)
        # else:
        #     real_loss = adversarial_loss(real_pred - fake_pred, valid)
        #     fake_loss = adversarial_loss(fake_pred - real_pred, fake)

        # d_loss = (real_loss + fake_loss) / 2


        # np_images = gen_imgs.cpu().numpy()

        # N, _, _, _ = np_images.shape
        # try:
    #     for n in range(N):
    #         filename = os.path.join(test_dir, "test_batch_best_%i.png"%(count))
    #         print(filename)
    #         # residual
    #         cur_img1 = np.transpose(Xtest[n].cpu().numpy(), [1,2,0])
    #         pred_mask1 = np.transpose(np_images[n], [1,2,0])
    #         pred_img1 = np.maximum(np.minimum(pred_mask1, 1.0), 0.0)  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
    #         gt_img1 = np.transpose(Ytest[n].cpu().numpy(), [1,2,0])

    #         # print("test_outputs size ", test_outputs.shape, "\t Ytest size ", Ytest.shape)
    #         # Stats of Predicted Image

    #         ssim1 = np.mean(np.mean(ssim(gt_img1, pred_img1, full=True,multichannel=True)))
    #         psnr1 = psnr(gt_img1, pred_img1)
    #         avg_psnr_model += (psnr1 )
    #         avg_ssim_model += (ssim1 )
    #         count += 1

    #         # Stats of Input Image
    #         ssim2 = np.mean(np.mean(ssim(gt_img1, cur_img1, full=True,multichannel=True)))
    #         psnr2 = psnr(gt_img1, cur_img1)
    #         if (np.abs(psnr2 - 100) < 1e-6):
    #             continue
    #         avg_psnr += (psnr2)
    #         avg_ssim += (ssim2)
    #         print("issim: ", ssim2,  "\t pssim: ", ssim1, "\t ipsnr: ", psnr2,  "\t ppsnr: ", psnr1)
    #         logger.info("PSNR= Original %.5f\t Predicted: %.5f\t SSIM= Original %.5f\t Predicted: %.5f \n"%(psnr2,psnr1, ssim2, ssim1))
            
    #         if(psnr2 < psnr1):
    #             better_count += 1

    #         img_pair1 = np.hstack(([cur_img1, pred_img1, gt_img1])) * 255
    #         # img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
    #         display_img = np.vstack([img_pair1])
    #         # print(pred_mask1.shape)
    #         # print(pred_img1.shape)
    #         # print(gt_img1.shape)
    #         # ipdb.set_trace()
    #         # print(display_img.shape)
    #         # cv2.imshow('sample image', display_img.astype(np.uint8))
    #         # cv2.waitKey(0) # waits until a key is pressed
    #         # cv2.destroyAllWindows()
    #         cv2.imwrite(filename, display_img.astype(np.uint8)) 
    #     # val_rec_loss += reconstruction_loss.item()
    #     # val_g_loss += g_loss.item()
    #     # val_d_loss += d_loss.item()
    # # val_rec_loss /= num_val_batches
    # # val_g_loss /= num_val_batches
    # # val_d_loss /= num_val_batches

    # avg_psnr /= num_val_batches
    # avg_psnr_model /= num_val_batches
    # avg_ssim /= num_val_batches
    # avg_ssim_model /= num_val_batches

    # # logger.info("[testRec loss: %.5e][testPsnr loss: %.5f][testPsnrModel loss: %.5f][testSsim loss: %.5f][testSsimModel loss: %.5f]" 
    # #     % ( val_rec_loss, avg_psnr, avg_psnr_model, avg_ssim, avg_ssim_model))
    # logger.info("[testPsnr loss: %.5f][testPsnrModel loss: %.5f][testSsim loss: %.5f][testSsimModel loss: %.5f]" 
    #     % (avg_psnr, avg_psnr_model, avg_ssim, avg_ssim_model))
        np_images = test_outputs.cpu().numpy()

        N, _, _, _ = np_images.shape
        # try:
        for n in range(N):
            # print(test_dir)
            write_to_folder = os.path.join(test_dir + '/test_imgs' , folder_name)
            if not(os.path.exists(write_to_folder)):
                os.makedirs(write_to_folder)
            # filename = os.path.join(test_dir, os.path.join("test_batch_best_%i.png"%(count))
            filename = os.path.join(write_to_folder, fname)
            print("OUtput path: ", filename)
            # residual
            cur_img1 = np.transpose(Xtest[n].cpu().numpy(), [1,2,0])
            cur_img1 = cv2.resize(cur_img1, (640,360), interpolation = cv2.INTER_AREA)
            pred_mask1 = np.transpose(np_images[n], [1,2,0])
            pred_mask1 = cv2.resize(pred_mask1, (640,360), interpolation = cv2.INTER_AREA)
            # print("pred_mask1.shape: ", pred_mask1.shape)
            pred_img1 = np.maximum(np.minimum(pred_mask1, 1.0), 0.0)  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
            # print("pred_img1.shape: ", pred_img1.shape)
            gt_img1 = np.transpose(Ytest[n].cpu().numpy(), [1,2,0])
            # print("gt_img1.shape: ", gt_img1.shape)
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
            if (np.abs(psnr2 - 100) < 1e-6):
                continue
            avg_input_psnr += (psnr2)
            avg_input_ssim += (ssim2)
            print("issim: ", ssim2,  "\t pssim: ", ssim1, "\t ipsnr: ", psnr2,  "\t ppsnr: ", psnr1)
            logger.info("PSNR= Original %.5f\t Predicted: %.5f\t SSIM= Original %.5f\t Predicted: %.5f \n"%(psnr2,psnr1, ssim2, ssim1))

            if(psnr2 < psnr1):
                better_count += 1

            # img_pair1 = np.hstack(([cur_img1, pred_img1, gt_img1])) * 255
            # img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
            # display_img = np.vstack([img_pair1])
            display_img = np.vstack([pred_img1]) * 255
            display_img = np.maximum(np.minimum(display_img, 255.0), 0.0)
            # print(pred_mask1.shape)
            # print(pred_img1.shape)
            # print(gt_img1.shape)
            # ipdb.set_trace()
            # print(display_img.shape)
            # cv2.imshow('sample image', display_img.astype(np.uint8))
            # cv2.waitKey(0) # waits until a key is pressed
            # cv2.destroyAllWindows()
            cv2.imwrite(filename, display_img.astype(np.uint8)) 
        # except:
        #     pass

        # val_rec_loss += reconstruction_loss.item()
            
    # val_rec_loss /= num_val_batches
    # val_g_loss /= num_val_batches
    # val_d_loss /= num_val_batches

    logger.info("Test: [vRec loss: %.5e]" 
        % (val_rec_loss))
running_loss = 0.0
print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
        " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
        " better count: ", better_count," count: ", count)

end_timing_epoch = time.time()
logger.info("Epoch %i runtime: %.3f"% (1, end_timing_epoch - start_timing_epoch))












# # load dataset
# from torch.utils import data
# from lib.dataloader import HDF5Dataset
# from torch.utils.data.sampler import SubsetRandomSampler

# if 'drive' in Flags.input_dir:
#     input_dir = '.'
# else:
#     save_dir = os.path.join(Flags.input_dir, "dataset_{}_qp{}".format(Flags.vcodec,Flags.qp))
#     input_dir = os.path.join(os.path.join(save_dir, 'ARCNN'), '{}_qp{}'.format(Flags.vcodec,str(Flags.qp)))

# print(input_dir)

# dataset = HDF5Dataset(input_dir, recursive=False, load_data=False, 
#    data_cache_size=100, transform=None)

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
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
# test_sampler = SubsetRandomSampler(test_indices)

# train_loader_params = {'batch_size': 128, 'num_workers': 6,'sampler': train_sampler}
# validation_loader_params = {'batch_size': 32, 'num_workers': 6,'sampler': valid_sampler}
# test_loader_params = {'batch_size': 32, 'num_workers': 6,'sampler': test_sampler}

# train_loader = data.DataLoader(dataset, **train_loader_params)
# validation_loader = data.DataLoader(dataset, **validation_loader_params)
# test_loader = data.DataLoader(dataset, **test_loader_params)

# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# # loaders = [train_loader, validation_loader, test_loader]

# loaders = [test_loader]


# count = 0
# f_psnr = open(os.path.join(Flags.output_dir,"RGANn\\01-30-2020=01-49-41\\test\\psnr_yuv_test.txt"),"w")
# for loader in loaders:
#     val_rec_loss = 0.0
#     val_g_loss = 0.0
#     val_d_loss = 0.0
#     test_loss = 0.0
#     avg_predicted_psnr = 0.0
#     avg_input_psnr = 0.0
#     avg_predicted_ssim = 0.0
#     avg_input_ssim = 0.0
#     # count = 0
#     better_count = 0
#     start_timing_epoch = time.time()

#     with torch.set_grad_enabled(False):
#         validation_loss = 0.0
#         num_val_batches = 0
#         for k, (X,Y) in enumerate(loader):
#             val_length = len(Y)
#             num_val_batches += val_length
#             for j in range(val_length):
#                 # here comes your validation loop

#                 # ------------------------------------
#                 #
#                 #   Train Generator
#                 #
#                 #-------------------------------------

#                 Xtest = (X[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#                 Ytest = (Y[j,:,:,:,:].permute([0, 3, 1, 2])/255.0).float().to(device)
#                 # print(Xtrain.shape)
#                 # exit()

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
#                 g_loss = (Flags.mse_lambda * reconstruction_loss) + (Flags.rec_lambda * g_loss)

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


#                 np_images = gen_imgs.cpu().numpy()

#                 N, _, _, _ = np_images.shape
#                 # try:
#                 for n in range(N):
#                     filename = os.path.join(test_dir, "test_batch_best_%i.png"%(count))
#                     print(filename)
#                     # residual
#                     cur_img1 = np.transpose(Xtest[n].cpu().numpy(), [1,2,0])
#                     pred_mask1 = np.transpose(np_images[n], [1,2,0])
#                     pred_img1 = np.maximum(np.minimum(pred_mask1, 1.0), 0.0)  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
#                     gt_img1 = np.transpose(Ytest[n].cpu().numpy(), [1,2,0])

#                     # print("test_outputs size ", test_outputs.shape, "\t Ytest size ", Ytest.shape)
#                     # Stats of Predicted Image

#                     ssim1 = np.mean(np.mean(ssim(gt_img1, pred_img1, full=True,multichannel=True)))
#                     psnr1 = psnr(gt_img1, pred_img1)
#                     avg_predicted_psnr += (psnr1 )
#                     avg_predicted_ssim += (ssim1 )
#                     count += 1

#                     # Stats of Input Image
#                     ssim2 = np.mean(np.mean(ssim(gt_img1, cur_img1, full=True,multichannel=True)))
#                     psnr2 = psnr(gt_img1, cur_img1)
#                     if (np.abs(psnr2 - 100) < 1e-6):
#                         continue
#                     avg_input_psnr += (psnr2)
#                     avg_input_ssim += (ssim2)
#                     print("issim: ", ssim2,  "\t pssim: ", ssim1, "\t ipsnr: ", psnr2,  "\t ppsnr: ", psnr1)
#                     f_psnr.write("PSNR= Original %.5f\t Predicted: %.5f\t SSIM= Original %.5f\t Predicted: %.5f \n"%(psnr2,psnr1, ssim2, ssim1))
#                     f_psnr.flush()
#                     if(psnr2 < psnr1):
#                         better_count += 1

#                     img_pair1 = np.hstack(([cur_img1, pred_img1, gt_img1])) * 255
#                     # img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
#                     display_img = np.vstack([img_pair1])
#                     # print(pred_mask1.shape)
#                     # print(pred_img1.shape)
#                     # print(gt_img1.shape)
#                     # ipdb.set_trace()
#                     # print(display_img.shape)
#                     # cv2.imshow('sample image', display_img.astype(np.uint8))
#                     # cv2.waitKey(0) # waits until a key is pressed
#                     # cv2.destroyAllWindows()
#                     cv.imwrite(filename, display_img.astype(np.uint8)) 
#                 val_rec_loss += reconstruction_loss.item()
#                 val_g_loss += g_loss.item()
#                 val_d_loss += d_loss.item()
#         val_rec_loss /= num_val_batches
#         val_g_loss /= num_val_batches
#         val_d_loss /= num_val_batches

#         logger.info("Test: [vD loss: %.5e] [vG loss: %.5e] [vRec loss: %.5e]" 
#             % ( val_d_loss, val_g_loss, val_rec_loss))

#     print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
#             " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
#             " better count: ", better_count," count: ", count)
#     f_psnr.write("Mode:%s Better: %.3f avgPSNR= Original %.5f \t Predicted: %.5f\n avgSSIM= Original %.5f \t Predicted: %.5f \n"%(Flags.model, float(better_count/count), float(avg_input_psnr/count), float(avg_predicted_psnr/count), float(avg_input_ssim/count), avg_predicted_ssim/count))
#     f_psnr.flush()

#     end_timing_epoch = time.time()
#     logger.info("Epoch %i runtime: %.3f"% (1, end_timing_epoch - start_timing_epoch))
