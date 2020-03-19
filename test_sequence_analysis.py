import os
import cv2
import ipdb
import torch
import numpy as np 
import argparse
from matplotlib import interactive
interactive(True)
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start_id', default=2000, type=int, help='starting scene index')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument('--disk_path', default="D:\\Github\\FYP2020\\tecogan_video_data", help='the path to save the dataset')
parser.add_argument('--test_dir', default="D:\\Github\\FYP2020\\test_sequence", help='the path to save the dataset')
# parser.add_argument('--disk_path', default="../content/drive/My Drive/FYP/tecogan_video_data", help='the path to save the dataset')
parser.add_argument('--summary_dir', default="", help='the path to save the log')
parser.add_argument('--REMOVE', action='store_true', help='whether to remove the original video file after data preparation')
parser.add_argument('--TEST', action='store_true', help='verify video links, save information in log, no real video downloading!')
parser.add_argument('--gt_dir', default="train_video", help='the path to save the dataset')
parser.add_argument('--compressed_dir', default="train_compressed_video", help='the path to save the dataset')
parser.add_argument('--compressed_frame_dir', default="train_compressed_video_frames", help='the path to save the dataset')
parser.add_argument('--gt_frames_dir', default="train_video_frames", help='the path to save the dataset')
parser.add_argument('--model', default="ARCNN", help='the path to save the dataset')
parser.add_argument('--vcodec', default="libx264", help='the path to save the dataset')
parser.add_argument('--qp', default=37, type=int, help='scene duration')

Flags = parser.parse_args()

def psnr(img1, img2):
    size = img1.shape

    PIXEL_MAX = 255.0

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

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

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
    log_file = os.path.join(lr_test_dir, "stats.txt")
    touch(log_file)
    print(log_file)
    psnr_stats_list = []
    ssim_stats_list = []

    if (Flags.model == 'ARCNN'):
        with open(log_file, 'w') as f:
            for t in range( len(test_image_name_list) ):
                input_image_path = os.path.join(lr_test_frames_dir, test_image_name_list[t])
                input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
                h,w,c = input_image.shape
                input_yuv_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2YUV)
                y_input, _, _ = cv2.split(input_yuv_image)
                # input_image = np.expand_dims(y_input, axis=2)

                gt_image_path = os.path.join(hr_test_frames_dir, test_image_name_list[t])
                gt_image = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
                h,w,c = gt_image.shape
                gt_yuv_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2YUV)
                y_gt, _, _ = cv2.split(gt_yuv_image)

                psnr_value = psnr(y_gt,y_input)
                ssim_value = ssim(y_gt,y_input)

                logmsg = ("[" + input_image_path + "]\t[PSNR:" + str(psnr_value) + "]\t[SSIM:"+ str(ssim_value) + "]\n")
                print(logmsg)
                f.write(logmsg)
                f.flush()
                if(len(psnr_stats_list) < 30):
                    psnr_stats_list.append(psnr_value)
                    ssim_stats_list.append(ssim_value)
                else:
                    plt.plot(range(len(psnr_stats_list)), np.array(psnr_stats_list))
                    plt.show()
                    input('press return to continue')
                    plt.plot(range(len(ssim_stats_list)), np.array(ssim_stats_list))
                    input('press return to continue')
                    plt.show()
                    psnr_stats_list = []
                    ssim_stats_list = []

        # gt_image = np.expand_dims(y_gt, axis=2)    

    # if(Flags.model == 'TDKNLRGAN'):
    #     dt = Flags.tseq_length//2
    #     for t in range(dt, len(test_image_name_list) - dt):
    #         # load image
    #         temp_input_img_patch_list = []
    #         temp_gt_img_patch_list = []
    #         # load three lr images and one gt image
    #         lr_images_name = test_image_name_list[t-dt:t+dt+1]
    #         # print("read image: ", image_path)
    #         # print(len(lr_images_name))
    #         # exit()
            
    #         for name in lr_images_name:
    #             input_image_path = os.path.join(lr_test_frames_dir, name)
    #             input_image = cv.imread(input_image_path, cv.IMREAD_UNCHANGED)
    #             h,w,c = input_image.shape
    #             input_yuv_image = cv.cvtColor(input_image, cv.COLOR_RGB2YUV)
    #             y_input, _, _ = cv.split(input_yuv_image)
    #             input_image = np.expand_dims(y_input, axis=2)
    #             temp_input_img_patch_list.append(input_image)

    #         gt_image_path = os.path.join(hr_test_frames_dir, test_image_name_list[t])

    #         gt_image = cv.imread(gt_image_path, cv.IMREAD_UNCHANGED)
    #         h,w,c = gt_image.shape
    #         gt_yuv_image = cv.cvtColor(gt_image, cv.COLOR_RGB2YUV)

    #         y_gt, _, _ = cv.split(gt_yuv_image)
    #         gt_image = np.expand_dims(y_gt, axis=2)

    #         temp_gt_img_patch_list.append(gt_image)

    #         if len(temp_input_img_patch_list) == 0:
    #             continue
    #         # T x H x W x C
    #         temporal_input_img_patch_list = np.stack(temp_input_img_patch_list, axis=0)
    #         temporal_gt_img_patch_list = np.stack(temp_gt_img_patch_list, axis=0)

    #         # print(temporal_input_img_patch_list.shape)
    #         # print(temporal_gt_img_patch_list.shape)

    #         h_length = h // stride
    #         w_length = w // stride

    #         input_img_patch = None
    #         gt_img_patch = None

    #         for hind in range(h_length):
    #             for wind in range(w_length):
    #                 x1 = stride * wind
    #                 x2 = stride * wind + stride
    #                 y1 = stride * hind
    #                 y2 = stride * hind + stride

    #                 temporal_input = np.array(temporal_input_img_patch_list[:,y1:y2, x1:x2,:])
    #                 temporal_gt = np.array(temporal_gt_img_patch_list[:,y1:y2, x1:x2,:])
    #                 # print(temporal_input.shape)
    #                 # print(temporal_gt.shape)
    #                 if input_img_patch is None:
    #                     input_img_patch = np.expand_dims(temporal_input, axis=1)
    #                 else:
    #                     input_img_patch = np.concatenate([input_img_patch, np.expand_dims(temporal_input, axis=1)], axis = 1)

    #                 if gt_img_patch is None:
    #                     gt_img_patch = np.expand_dims(temporal_gt, axis=0)
    #                     # gt_img_patch = np.expand_dims(temporal_gt, axis=1)
    #                 else:
    #                     gt_img_patch = np.concatenate([gt_img_patch, np.expand_dims(temporal_gt, axis=1)], axis = 1)
    #         #         print(input_img_patch.shape)
    #         #         print(gt_img_patch.shape)
    #         # print(input_img_patch.shape)
    #         # print(gt_img_patch.shape)

    #         yield h_length, w_length, input_img_patch, gt_img_patch


if __name__ == '__main__':
    test_image_loader()
