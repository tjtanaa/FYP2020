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
# Wavelet Net
# D:\\Github\\FYP2020\\tecogan_video_data\\WaveletNet\\02-25-2020=15-44-34
# Pixel Net
# D:\\Github\\FYP2020\\tecogan_video_data\\PixelNet\\02-26-2020=16-35-21

parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser = argparse.ArgumentParser()
parser.add_argument('--model', default="DPWSDNet", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=10, type=int, help='number of training epochs')
parser.add_argument('--max_iteration', default=100000, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--input_dir', default="/media/data3/tjtanaa/tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="/media/data3/tjtanaa/tecogan_video_data", type=str, help='output and log directory')
parser.add_argument('--test_dir', default="/home/tan/tjtanaa/tjtanaa/DeepVideoPrior/data/artifact_removal/send/blur_000", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
# parser.add_argument('--load_from_ckpt', default="/media/data3/tjtanaa/tecogan_video_data/PixelNet/04-16-2020=11-59-26", type=str, help='ckpt model directory')
# parser.add_argument('--load_from_ckpt', default="/media/data3/tjtanaa/tecogan_video_data/WaveletNet/05-07-2020=00-05-14", type=str, help='ckpt model directory')
parser.add_argument('--load_from_ckpt', default="/media/data3/tjtanaa/tecogan_video_data/DPWSDNet/05-07-2020=22-00-20", type=str, help='ckpt model directory')

parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx265", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=3, type=int, help='scene duration')
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
elif Flags.model == 'PixelNet':
    model = Pixel_Net(C, C, P = Flags.depth).to(device)
    print(summary(model, (C, 256, 256)))
elif Flags.model == 'WaveletNet':
    model = Wavelet_Net(C, C, P = Flags.depth).to(device)
    print(summary(model, (C, 256, 256)))    


# optimizer = optim.SGD([
#     {'params': model.base.parameters()}
# ], lr=Flags.lr, momentum = 0.9)

# optimizer = torch.optim.SGD(model.parameters(), lr=Flags.lr, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=True)
optimizer = torch.optim.Adam(model.parameters(), lr=Flags.lr, betas=(Flags.b1, Flags.b2))

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

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
    # scheduler = checkpoint['scheduler']
    best_model_loss = checkpoint['val_loss']
    iteration_count = checkpoint['iteration']
    model.eval()
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
            yield input_image, gt_image



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
            yield input_image, gt_image, 'blur_000', fname

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
            print("input_image.size ", input_image.shape)
            yield input_image, gt_image, 'blur_000', fname


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


start_timing_epoch = time.time()
running_g_loss = 0.0
running_d_loss = 0.0
running_rec_loss = 0.0
val_g_loss = 0.0
val_d_loss = 0.0
val_rec_loss = 0.0
num_train_batches = 0
avg_predicted_psnr = 0.0
avg_input_psnr = 0.0
avg_predicted_ssim = 0.0
avg_input_ssim = 0.0
count = 0
better_count = 0
with torch.set_grad_enabled(False):
    num_val_batches = 0
    
    for k, (X, Y, folder_name, fname) in enumerate(get_blur_frames(Flags.test_dir)):
        val_length = len(Y)
        num_val_batches += val_length

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
        test_outputs = model(Xtest)
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
            print("Output path: ", filename)
            # residual
            cur_img1 = np.transpose(Xtest[n].cpu().numpy(), [1,2,0])
            pred_mask1 = np.transpose(np_images[n], [1,2,0])
            pred_img1 = np.maximum(np.minimum(pred_mask1, 1.0), 0.0)  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
            gt_img1 = np.transpose(Ytest[n].cpu().numpy(), [1,2,0])

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

            display_img = np.vstack([pred_img1]) * 255
            display_img = np.maximum(np.minimum(display_img, 255.0), 0.0)

            cv2.imwrite(filename, display_img.astype(np.uint8)) 


    logger.info("Test: [vRec loss: %.5e]" 
        % (val_rec_loss))
running_loss = 0.0
print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
        " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
        " better count: ", better_count," count: ", count)
logger.info("Mode:%s Better: %.3f avgPSNR= Original %.5f \t Predicted: %.5f\n avgSSIM= Original %.5f \t Predicted: %.5f \n"%(Flags.model, float(better_count/count), float(avg_input_psnr/count), float(avg_predicted_psnr/count), float(avg_input_ssim/count), avg_predicted_ssim/count))

end_timing_epoch = time.time()
logger.info("Epoch %i runtime: %.3f"% (1, end_timing_epoch - start_timing_epoch))

