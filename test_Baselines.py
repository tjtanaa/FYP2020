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
import cv2 as cv
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

parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser = argparse.ArgumentParser()
parser.add_argument('--model', default="ARCNN", type=str, help='the path to save the dataset')
parser.add_argument('--epoch', default=1000, type=int, help='number of training epochs')
parser.add_argument('--mini_batch', default=32, type=int, help='mini_batch size')
parser.add_argument('--disk_path', default="D:\\Github\\FYP2020\\tecogan_video_data", help='the path to save the dataset')
parser.add_argument('--input_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='dataset directory')
parser.add_argument('--output_dir', default="D:\\Github\\FYP2020\\tecogan_video_data", type=str, help='output and log directory')
# parser.add_argument('--input_dir', default="../content/drive/My Drive/FYP", type=str, help='dataset directory')
# parser.add_argument('--output_dir', default="../content/drive/My Drive/FYP", type=str, help='output and log directory')
parser.add_argument('--load_from_ckpt', default="D:\\Github\\FYP2020\\tecogan_video_data\\ARCNN\\01-07-2020=10-11-27_reconstruction", type=str, help='ckpt model directory')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--tseq_length", type=int, default=3, help="interval between image sampling")
parser.add_argument('--vcodec', default="libx264", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')
parser.add_argument('--channel', default=1, type=int, help='scene duration')

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
best_model_loss = 999999
batch_size = Flags.mini_batch
st_epoch = 0 # starting epoch

C = Flags.channel

# create model

if Flags.model == 'ARCNN':
    model = ARCNN(C, C).to(device)
elif Flags.model == 'FastARCNN':
    model = FastARCNN(C, C).to(device)
elif Flags.model == 'VRCNN':
    model = VRCNN(C, C).to(device)

lr = 5e-4

optimizer = optim.Adam([
    {'params': model.base.parameters()},
    {'params': model.last.parameters(), 'lr': Flags.lr * 0.1},
], lr=Flags.lr)


# criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
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

# load dataset
from torch.utils import data
from lib.dataloader import HDF5Dataset
from torch.utils.data.sampler import SubsetRandomSampler

save_dir = os.path.join(Flags.disk_path, "dataset_{}_qp{}".format(Flags.vcodec,Flags.qp))
input_dir = os.path.join(os.path.join(save_dir, Flags.model), '{}_qp{}'.format(Flags.vcodec,str(Flags.qp)))

dataset = HDF5Dataset(input_dir, recursive=False, load_data=False, 
   data_cache_size=100, transform=None)

shuffle_dataset = True
# Creating data indices for training and validation splits:
validation_split = 0.2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(0)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader_params = {'batch_size': 100, 'num_workers': 6,'sampler': train_sampler}
validation_loader_params = {'batch_size': 4, 'num_workers': 6,'sampler': valid_sampler}

train_loader = data.DataLoader(dataset, **train_loader_params)
validation_loader = data.DataLoader(dataset, **validation_loader_params)

data_loader = data.DataLoader(dataset, **loader_params)



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
    gt_dir = "D:\\Github\\FYP2020\\tecogan_video_data\\train_video_frames"
    input_dir = "D:\\Github\\FYP2020\\tecogan_video_data\\train_compressed_video_frames"

    f_psnr = open(os.path.join(Flags.output_dir,"ARCNN\\01-07-2020=10-11-27_reconstruction\\test\\psnr_yuv_test.txt"),"w")
    # search for scene_ folder
    for subfolder in os.listdir(input_dir):
        input_subfolder_dir = os.path.join(input_dir, subfolder)
        gt_subfolder_dir = os.path.join(gt_dir, subfolder)

        temp_input_img_patch_list = []
        temp_gt_img_patch_list = []

        image_counter = 0

        if os.path.isdir(input_subfolder_dir) and os.path.isdir(gt_subfolder_dir):
            print(subfolder)

            # # for each subfolder start loading all 120 frames, then crop them randomly
            dataset_per_folder_count = 0
            # load frames

            # load the input frames
            for n, image_path in enumerate(os.listdir(input_subfolder_dir)):
                if n < 119:
                    continue
                if image_counter == 60:
                    break
                if image_path.find('.png') != -1:
                    # print("read image: ", image_path)
                    input_image_path = os.path.join(input_subfolder_dir, image_path)
                    gt_image_path = os.path.join(gt_subfolder_dir, image_path)
                    # print('input_image_path: ', input_image_path)
                    # print("gt_image_path: ", gt_image_path)
                    # read current image
                    input_image = cv.imread(input_image_path, cv.IMREAD_UNCHANGED)
                    gt_image = cv.imread(gt_image_path, cv.IMREAD_UNCHANGED)
                    h,w,c = input_image.shape
                    # print(input_image.shape)
                    if(h == 2160 or w == 4096):
                        break
                    input_yuv_image = cv.cvtColor(input_image, cv.COLOR_RGB2YUV)
                    gt_yuv_image = cv.cvtColor(gt_image, cv.COLOR_RGB2YUV)
                    # print(img_yuv.shape)
                    y_input, _, _ = cv.split(input_yuv_image)
                    y_gt, _, _ = cv.split(gt_yuv_image)
                    # convert (H,W) to (H,W,1)
                    input_image = np.expand_dims(y_input, axis=2)
                    gt_image = np.expand_dims(y_gt, axis=2)
                    # y = np.expand_dims(y, axis=0)

                    # print(y.shape)
                    # cv.imshow('y', y)
                    # cv.imshow('u', u)
                    # cv.imshow('v', v)
                    # cv.waitKey(0)
                    
                    temp_input_img_patch_list.append(input_image)
                    temp_gt_img_patch_list.append(gt_image)
                    # psnrValue = psnr(gt_image, input_image)
                    # running_mean_psnr += psnrValue
                    # running_count_psnr += 1
                    # f_psnr.write("PSRN: " + str(psnrValue) + " \t Avg Psnr: " + str(running_mean_psnr / running_count_psnr) + "\n")
                    # f_psnr.write("PSRN: %.5f \t Running_PSNR: %.5f\n"%(psnrValue,(running_mean_psnr / running_count_psnr)))
                    # f_psnr.flush()
        

        if len(temp_input_img_patch_list) == 0:
            continue
        # T x H x W x C
        X = np.stack(temp_input_img_patch_list, axis=0)
        Y = np.stack(temp_gt_img_patch_list, axis=0)

        list_size = X.shape
        T, H, W, C = list_size


        Xtest = (torch.from_numpy(X).permute([0, 3, 1, 2])/255.0).float().to(device)
        Ytest = (torch.from_numpy(Y).permute([0, 3, 1, 2])/255.0).float().to(device)
        # [N x C x H x W]
        test_outputs = model(Xtest)

        # print(test_outputs * 255)
        # exit()
        np_images = test_outputs.cpu().numpy()

        N, _, _, _ = np_images.shape
        try:
            for n in range(N):
                filename = os.path.join(test_dir, "test_batch_best_%i.png"%(count))
                # residual
                cur_img1 = np.transpose(Xtest[n].cpu().numpy(), [1,2,0])
                pred_mask1 = np.transpose(np_images[n], [1,2,0])
                pred_img1 = pred_mask1  #+ np.transpose(Xtest[0].cpu().numpy(), [1,2,0])
                gt_img1 = np.transpose(Ytest[n].cpu().numpy(), [1,2,0])

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
                f_psnr.write("PSNR= Original %.5f\t Predicted: %.5f\t SSIM= Original %.5f\t Predicted: %.5f \n"%(psnr2,psnr1, ssim2, ssim1))
                f_psnr.flush()
                if(psnr2 < psnr1):
                    better_count += 1

                img_pair1 = np.hstack(([cur_img1, pred_img1, gt_img1])) * 255
                # img_pair2 = np.hstack(([cur_img2, pred_mask2, pred_img2, gt_img2])) * 255
                display_img = np.vstack([img_pair1])
                # print(pred_mask1.shape)
                # print(pred_img1.shape)
                # print(gt_img1.shape)
                # ipdb.set_trace()
                # print(display_img.shape)
                # cv2.imshow('sample image', display_img.astype(np.uint8))
                # cv2.waitKey(0) # waits until a key is pressed
                # cv2.destroyAllWindows()
                cv.imwrite(filename, display_img.astype(np.uint8)) 
        except:
            pass

        t_loss = criterion(test_outputs, Ytest)
        test_loss += t_loss.item()
    print(Flags.model, " avg_input_psnr: ", avg_input_psnr/count , " avg_predicted_psnr: ", avg_predicted_psnr/count, \
            " avg_input_ssim: ", avg_input_ssim/count , " avg_predicted_ssim: ", avg_predicted_ssim/count, \
            " better count: ", better_count," count: ", count)
    f_psnr.write("Mode:%s Better: %.3f avgPSNR= Original %.5f \t Predicted: %.5f\n avgSSIM= Original %.5f \t Predicted: %.5f \n"%(Flags.model, float(better_count/count), float(avg_input_psnr/count), float(avg_predicted_psnr/count), float(avg_input_ssim/count), avg_predicted_ssim/count))
    f_psnr.flush()

    test_loss /= count
    logger.info('Test loss: %.5f' % (test_loss))
