import torch
# from lib.ops import *

import os, sys, datetime
import argparse

import cv2 as cv
import collections, math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import ipdb
from skimage.measure import compare_ssim as ssim
import h5py

# Mean Absolute distance
def MAD(img1, img2):
    return np.mean(np.abs(img1-img2))

# Mean Square distance
def MSD(img1, img2):
    return np.mean(np.square((img1-img2)))

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

def full_search(img_patch_ref, img_target, x1, x2, y1, y2, p=15, s=40):
    ref_img_size = img_patch_ref.shape
    rH, rW, rC = ref_img_size
    tar_img_size = img_target.shape
    tH, tW, tC = tar_img_size
    minx = np.max([x1 - rW + p, 0])
    maxx = np.min([x2 + rW - p, tW - rW])
    miny = np.max([y1 - rH + p, 0])
    maxy = np.min([y2 + rH - p, tH - rH])

    # match_score = 99999999
    match_score = -99999999
    match_position = None
    match_patch = None
    # match_ssim = scores

    for x in range(minx, maxx):
        for y in range(miny, maxy):
            sx1 = x
            sx2 = x + rW
            sy1 = y
            sy2 = y + rH

            if sx2 - sx1 > rW or sy2-sy1 > rH or sx1 < 0 or sy1 <0 or sx2 > tW or sy2 > tH:
                continue  

            img_patch_target = img_target[sy1:sy2, sx1:sx2, :]

            # compute the mad
            # print("img_patch_ref.shape: ", img_patch_ref.shape)
            # mad_score = MAD(img_patch_ref, img_patch_target)
            (mad_score, _) = ssim(img_patch_ref, img_patch_target, full=True)
            # mad_score = MSD(img_patch_ref, img_patch_target)
            if match_score < mad_score:
                match_patch = img_patch_target
                match_position = [sx1, sy1]
                match_score = mad_score
                start_coord = np.array([match_position])
                end_coord = np.array(np.array([match_position]) + np.array([[y2-y1, x2-x1]]))            
    
    display_img = np.hstack(([img_patch_ref, match_patch]))  
    cv.imshow('match patches',display_img)
    cv.imshow('img_target',img_target)
    cv.waitKey(0) # waits until a key is pressed
    cv.destroyAllWindows()
    return match_score, match_position, match_patch

# Three Step Search Algorithm (TSS)
def tss(img_patch_ref, img_target, x1, x2, y1, y2, p=15, s=7):
    '''
        Input
        Output:
        return the patch that is most similar to the img_patch_ref
    '''
    img_size = img_target.shape
    H, W, C = img_size

    start_coord = np.array([[x1,y1]])
    end_coord = np.array([[x2,y2]])
    # print(type(img_target))
    # print(img_size)
    # cv.imshow('img target',img_target)
    
    # cv.waitKey(0) # waits until a key is pressed
    # cv.destroyAllWindows()

    match_score = -9999999
    match_position = None
    match_patch = None
    match_ssim = -999999
    # s = p
    while(s != 1):
        corner_coordinates = np.array([[-s, -s], [0, -s], [s, -s], 
                                [-s,0], [0,0], [s, 0],
                                [-s,s], [0,s], [s, s]])

        # sum to get the coordinates
        start_anchor_coord = corner_coordinates + start_coord
        end_anchor_coord = corner_coordinates + end_coord

        # ipdb.set_trace()
        # validity mask
        start_mask = np.sum(start_anchor_coord >=0, axis=1) >= 2
        start_mask_w = start_anchor_coord[:,0] <=W
        start_mask_h = start_anchor_coord[:,1] <=H

        end_mask = np.sum(end_anchor_coord >=0) >0
        end_mask_w = end_anchor_coord[:,0] <=W
        end_mask_h = end_anchor_coord[:,1] <=H

        valid_mask_s = np.logical_and( np.logical_and( start_mask, start_mask_w), start_mask_h) 
        valid_mask_e = np.logical_and( np.logical_and( end_mask, end_mask_w), end_mask_h) 
        valid_mask = np.logical_and( valid_mask_s, valid_mask_e)

        # print("valid_mask: ", valid_mask)
        # start_anchor_coord = start_anchor_coord * valid_mask[:, np.newaxis]
        # end_anchor_coord = end_anchor_coord * valid_mask[:, np.newaxis]

        # loop through all valid position
        for ind, valid in enumerate(valid_mask):
            if valid:
                sx1, sy1 = start_anchor_coord[ind] 
                sx2, sy2 = end_anchor_coord[ind]
                # print("coordinates ", sx1, sy1, sx2, sy2)

                img_patch_target = img_target[sy1:sy2, sx1:sx2, :]

                # compute the mad
                # print("img_patch_ref.shape: ", img_patch_ref.shape)
                # mad_score = MAD(img_patch_ref, img_patch_target)
                mad_score = psnr(img_patch_ref, img_patch_target)
                (ssim_score, _) = ssim(img_patch_ref, img_patch_target, full=True,multichannel=True)
                if match_score < mad_score and match_ssim < ssim_score:
                    match_patch = img_patch_target
                    match_position = [sx1, sy1]
                    match_score = mad_score
                    match_ssim = ssim_score
                    start_coord = np.array([match_position])
                    end_coord = np.array(np.array([match_position]) + np.array([[y2-y1, x2-x1]]))


        s = int(s//2)
        # print("match_score ", match_score)

        # display_img = np.hstack(([img_patch_ref, match_patch]))  
        # cv.imshow('match patches',display_img)
        # cv.imshow('img_target',img_target)
        # cv.waitKey(0) # waits until a key is pressed
        # cv.destroyAllWindows()



        return (match_score, match_ssim), match_position, match_patch

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

def preprocess_Generic_YUV_dataset(input_dir, gt_dir, save_dir="D:\\Github\\FYP2020\\tecogan_video_data\\dataset",\
                                 tseq = 11, maxseqlen=120, \
                                 dataset_threshold=200000, \
                                 dataset_threshold_per_folder=150-1, \
                                 seed=2020):

    dataset_count = 0
    running_mean_psnr = 0;
    running_count_psnr = 0;
    if not(os.path.exists(save_dir)):
        os.makedirs(save_dir)
        
    f_psnr = open(os.path.join(save_dir, "psnr_yuv.txt"),"w")
    dt = tseq //2

    np.random.seed(seed)
    # search for scene_ folder
    for subfolder in os.listdir(input_dir):
        if dataset_count > dataset_threshold:
            print("Skip from scene: ", subfolder)
            break
        input_subfolder_dir = os.path.join(input_dir, subfolder)
        gt_subfolder_dir = os.path.join(gt_dir, subfolder)
        # # load dataset for ARTN
        input_img_patch = None
        gt_img_patch = None

        temp_input_img_patch_list = []
        temp_gt_img_patch_list = []
        # print(input_subfolder_dir)
        # print(gt_subfolder_dir)

        if os.path.isdir(input_subfolder_dir) and os.path.isdir(gt_subfolder_dir):
            print(subfolder)
            # # for each subfolder start loading all 120 frames, then crop them randomly

            # load frames

            # load the input frames
            for n, image_path in enumerate(os.listdir(input_subfolder_dir)):
                if n > 2*tseq:
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
                    # if w >3840-1:
                    #     # do not load 2k videos
                    #     break
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
                    psnrValue = psnr(gt_image, input_image)
                    running_mean_psnr += psnrValue
                    running_count_psnr += 1
                    # f_psnr.write("PSRN: " + str(psnrValue) + " \t Avg Psnr: " + str(running_mean_psnr / running_count_psnr) + "\n")
                    f_psnr.write("PSRN: %.5f \t Running_PSNR: %.5f\n"%(psnrValue,(running_mean_psnr / running_count_psnr)))
                    f_psnr.flush()

            if len(temp_input_img_patch_list) == 0:
                continue
            # T x H x W x C
            temporal_input_img_patch_list = np.stack(temp_input_img_patch_list, axis=0)
            temporal_gt_img_patch_list = np.stack(temp_gt_img_patch_list, axis=0)

            list_size = temporal_input_img_patch_list.shape
            T, H, W, C = list_size

            dataset_per_folder_count = 0

            macro_block_size = 256
            macro_block_stride = 128

            for t in range(dt,T-dt,2):
                if dataset_count > dataset_threshold:
                    break
                if(dataset_per_folder_count > dataset_threshold_per_folder):
                    break
                for wind in range(0, W // macro_block_stride):
                    if dataset_count > dataset_threshold:
                        break
                    if(dataset_per_folder_count > dataset_threshold_per_folder):
                        break
                    for hind in range(0, H//macro_block_stride):

                        # randomly skip blocks
                        if np.random.rand() < 0.5:
                            continue

                        x1 = macro_block_stride * wind
                        x2 = macro_block_stride * wind + macro_block_size
                        y1 = macro_block_stride * hind
                        y2 = macro_block_stride * hind + macro_block_size
                        # print("pre img coordinates: ", x1, x2, y1, y2)
                        if x1 < 0 or y1 < 0 or x2 >= W or y2 >= H or (y2-y1)!=macro_block_size or (x2-x1)!=macro_block_size:
                            # skip invalid anchor position
                            continue

                        gt_patch = np.array(temporal_gt_img_patch_list[t, y1:y2, x1:x2,:])


                        var_ = np.var(gt_patch/255)
                        if var_ <0.002:
                            continue

                        # display_img = np.hstack(([pre_patch, cur_patch, post_patch]))
                        # cv.imshow('SUCCESS sample image',display_img)
                        # cv.waitKey(0) # waits until a key is pressed
                        # cv.destroyAllWindows()    
                        # [T x H x W x C]
                        # temporal_input = np.stack([pre_img,cur_patch,post_img ], axis = 0)
                        # temporal_gt = np.stack([pre_patch, cur_patch, post_patch], axis = 0)
                        # H, W, C
                        temporal_input = np.array(temporal_input_img_patch_list[t-dt:t+dt+1,y1:y2, x1:x2,:])
                        temporal_gt = np.array(temporal_gt_img_patch_list[t-dt:t+dt+1,y1:y2, x1:x2,:])

                        # [Tx N x H x W x C]
                        if input_img_patch is None:
                            input_img_patch = np.expand_dims(temporal_input, axis=1)
                        else:
                            input_img_patch = np.concatenate([input_img_patch, np.expand_dims(temporal_input, axis=1)], axis = 1)

                        if gt_img_patch is None:
                            gt_img_patch = np.expand_dims(temporal_gt, axis=1)
                            # gt_img_patch = np.expand_dims(temporal_gt, axis=1)
                        else:
                            gt_img_patch = np.concatenate([gt_img_patch, np.expand_dims(temporal_gt, axis=1)], axis = 1)
                            # gt_img_patch = np.concatenate([gt_img_patch, np.expand_dims(temporal_gt, axis=1)], axis = 1)
                        dataset_count = dataset_count + 1
                        print("dataset_count :", dataset_count)
                        dataset_per_folder_count += 1
                        if dataset_count > dataset_threshold:
                            break
                        if(dataset_per_folder_count > dataset_threshold_per_folder):
                            break
                        # print("gt_img_patch.shape: ", gt_img_patch.shape)
                        
                        # res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
                        # threshold = 0.8
                        # loc = np.where( res >= threshold)
                        # for pt in zip(*loc[::-1]):
                        #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            # save the image patch as data_yuv_scenefolder.h5
            if input_img_patch is None:
                continue
            hdf5_path = os.path.join(save_dir, 'data_yuv_{}.h5'.format(subfolder))
            store_many_hdf5(input_img_patch, gt_img_patch, hdf5_path)
    # return input_img_patch,  gt_img_patch 

def split_hdf5(input_dir, out_dir, model='TNLRGAN',tseq=11):

    # find all the hdf5 files in the 
    # for f in os.listdir(input_dir):
    #     print('scene_' in f)
        # if (os.path.isfile(f) and 'scene_' in f):
    files = [f for f in os.listdir(input_dir) if ( 'scene' in f and '.h5' in f)]
    count = 0
    # print(files)
    if not(os.path.exists(out_dir)):
        os.makedirs(out_dir)
    for f in files:
        input_images, gt_images = read_many_hdf5(os.path.join(input_dir, f))
        _, N, _ , _,_ = input_images.shape
        batch_size = 4
        for i in range(N//batch_size):
            if model == 'ARCNN' or model == 'RGAN' or model == 'NLRGAN' or model == 'NLGAN':
                i_img = input_images[tseq//2, i*(batch_size):(i+1)*batch_size]
                gt_img = gt_images[tseq//2, i*(batch_size):(i+1)*batch_size]
            if model == 'TNLRGAN' or model == 'TDNLRGAN' or model == 'TDNLGAN' or model == 'TNLGAN':
                i_img = input_images[:,i*(batch_size):(i+1)*batch_size]
                gt_img = gt_images[tseq//2, i*(batch_size):(i+1)*batch_size]
            out_path = os.path.join(out_dir, 'data_yuv_{}.h5'.format(count))
            store_many_hdf5(i_img, gt_img, out_path)   
            count += 1
            print(count)

    
