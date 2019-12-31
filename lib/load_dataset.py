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


# ------------------------------------parameters------------------------------#
parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start_id', default=2000, type=int, help='starting scene index')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument('--disk_path', default="D:\\Github\\tecogan_video_data", help='the path to save the dataset')
parser.add_argument('--summary_dir', default="", help='the path to save the log')
parser.add_argument('--REMOVE', action='store_true', help='whether to remove the original video file after data preparation')
parser.add_argument('--TEST', action='store_true', help='verify video links, save information in log, no real video downloading!')
parser.add_argument('--gt_dir', default="train_video", help='the path to save the dataset')
parser.add_argument('--compressed_dir', default="train_compressed_video", help='the path to save the dataset')
parser.add_argument('--compressed_frame_dir', default="train_compressed_video_frames", help='the path to save the dataset')
parser.add_argument('--gt_frames_dir', default="train_video_frames", help='the path to save the dataset')
parser.add_argument('--resize_gt_frame_dir', default="train_video_resized_frames", help='the path to save the dataset')
parser.add_argument('--resize_dir', default="train_resized_video", help='the path to save the dataset')
parser.add_argument('--resize_by_4_dir', default="train_resized_video_by_4", help='the path to save the dataset')
parser.add_argument('--video_bitrate', default="40k", help='video_bitrate')
parser.add_argument('--process', default=1, type=int, help='run process 0: download video 1: compress video 2: generate frames')
Flags = parser.parse_args()

if Flags.summary_dir == "":
    Flags.summary_dir = os.path.join(Flags.disk_path, "log/")
os.path.isdir(Flags.disk_path) or os.makedirs(Flags.disk_path)
os.path.isdir(Flags.summary_dir) or os.makedirs(Flags.summary_dir)


# Mean Absolute distance
def MAD(img1, img2):
    return np.sum(np.abs(img1-img2))

# Mean Square distance
def MSD(img1, img2):
    return np.mean(np.square((img1-img2)))

def full_search(img_patch_ref, img_target, x1, x2, y1, y2, p=15, s=40):
    ref_img_size = img_patch_ref.shape
    rH, rW, rC = ref_img_size
    tar_img_size = img_target.shape
    tH, tW, tC = tar_img_size
    minx = np.max([x1 - rW + p, 0])
    maxx = np.min([x2 + rW - p, tW - rW])
    miny = np.max([y1 - rH + p, 0])
    maxy = np.min([y2 + rH - p, tH - rH])

    match_score = 99999999
    match_position = None
    match_patch = None

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
            if match_score > mad_score:
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
def tss(img_patch_ref, img_target, x1, x2, y1, y2, p=15, s=40):
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

    match_score = -99999999
    match_position = None
    match_patch = None

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
                # mad_score = MSD(img_patch_ref, img_patch_target)
                (mad_score, _) = ssim(img_patch_ref, img_patch_target, full=True,multichannel=True)
                if match_score < mad_score:
                    match_patch = img_patch_target
                    match_position = [sx1, sy1]
                    match_score = mad_score
                    start_coord = np.array([match_position])
                    end_coord = np.array(np.array([match_position]) + np.array([[y2-y1, x2-x1]]))


        s = int(s//2)
        # print("match_score ", match_score)

        # display_img = np.hstack(([img_patch_ref, match_patch]))  
        # cv.imshow('match patches',display_img)
        # cv.imshow('img_target',img_target)
        # cv.waitKey(0) # waits until a key is pressed
        # cv.destroyAllWindows()



        return match_score, match_position, match_patch


def preprocess_ARTN_dataset(input_dir, gt_dir):

    # # load dataset for ARTN
    input_img_patch = None
    gt_img_patch = None

    dataset_count = 0
    dataset_threshold = 50000

    # search for scene_ folder
    for subfolder in os.listdir(input_dir):
        if dataset_count > dataset_threshold:
            print("Skip from scene: ", subfolder)
            break
        input_subfolder_dir = os.path.join(input_dir, subfolder)
        gt_subfolder_dir = os.path.join(gt_dir, subfolder)

        temp_input_img_patch_list = []
        temp_gt_img_patch_list = []


        if os.path.isdir(input_subfolder_dir) and os.path.isdir(gt_subfolder_dir):
            print(subfolder)
            # # for each subfolder start loading all 120 frames, then crop them randomly

            # load frames

            # load the input frames
            for image_path in os.listdir(input_subfolder_dir):
                if image_path.find('.png') != -1:
                    # print("read image: ", image_path)
                    input_image_path = os.path.join(input_subfolder_dir, image_path)
                    gt_image_path = os.path.join(gt_subfolder_dir, image_path)
                    # print('input_image_path: ', input_image_path)
                    # print("gt_image_path: ", gt_image_path)
                    # read current image
                    input_image = cv.imread(input_image_path, cv.IMREAD_UNCHANGED)
                    gt_image = cv.imread(gt_image_path, cv.IMREAD_UNCHANGED)
                    temp_input_img_patch_list.append(input_image)
                    temp_gt_img_patch_list.append(gt_image)

                    # display_img = np.hstack(([input_image, gt_image]))
                    # cv.imshow('sample image',display_img)
                    
                    # cv.waitKey(0) # waits until a key is pressed
                    # cv.destroyAllWindows()

            # T x H x W x C
            temporal_input_img_patch_list = np.stack(temp_input_img_patch_list, axis=0)
            temporal_gt_img_patch_list = np.stack(temp_gt_img_patch_list, axis=0)

            list_size = temporal_input_img_patch_list.shape
            T, H, W, C = list_size

            # tss

            # input_img_patch_list_pre = []
            # input_img_patch_list_cur = []
            # input_img_patch_list_post = []

            # gt_img_patch_list_pre = []
            # gt_img_patch_list_cur = []
            # gt_img_patch_list_post = []

            macro_block_size = 64
            macro_block_stride = 48
            for t in range(1,T-1,3):
                if dataset_count > dataset_threshold:
                    break
                for wind in range(0, W // macro_block_stride):
                    if dataset_count > dataset_threshold:
                        break
                    for hind in range(0, H//macro_block_stride):

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
                        pre_img = np.array(temporal_gt_img_patch_list[t-1])
                        post_img = np.array(temporal_gt_img_patch_list[t+1])
                        cur_patch = np.array(temporal_gt_img_patch_list[t, y1:y2, x1:x2,:])

                        blur_patch = np.array(temporal_input_img_patch_list[t, y1:y2, x1:x2,:])


                        var_ = np.var(blur_patch/255)
                        if var_ <0.002:
                            print("var_current patch: ", var_)
                            # cv.imshow('low var image',blur_patch)
                            
                            # cv.waitKey(0) # waits until a key is pressed
                            # cv.destroyAllWindows()
                            continue

                        # cv.imshow('pre img ',current_img_patch)
                        
                        # cv.waitKey(0) # waits until a key is pressed
                        # cv.destroyAllWindows()

                        # if cur_patch_size[0] != macro_block_size or cur_patch_size[1] != macro_block_size:
                        #     continue
                        
                        # pre_score , pre_position , pre_patch= tss(cur_patch, pre_img, x1,x2, y1,y2)
                        # post_score , post_position , post_patch = tss(cur_patch, post_img, x1,x2, y1,y2)

                        pre_score , pre_position , pre_patch= tss(cur_patch, pre_img, x1,x2, y1,y2)
                        post_score , post_position , post_patch = tss(cur_patch, post_img, x1,x2, y1,y2)
                        
                        # mad_score = MAD(pre_patch, post_patch)
                        # if pre_score > mad_score or post_score > mad_score:
                        #     print("pre and post patch doesn't match")
                        #     # display_img = np.hstack(([pre_patch, cur_patch, post_patch]))
                        #     # cv.imshow('sample image',display_img)
                            
                        #     # cv.waitKey(0) # waits until a key is pressed
                        #     # cv.destroyAllWindows()
                        #     continue

                        # display_img = np.hstack(([pre_patch, cur_patch, post_patch]))
                        # cv.imshow('sample image',display_img)
                        
                        # cv.waitKey(0) # waits until a key is pressed
                        # cv.destroyAllWindows()

                        # gt_img_patch_list_pre.append(pre_patch)
                        # gt_img_patch_list_cur.append(cur_patch)
                        # gt_img_patch_list_post.append(post_patch)

                        pre_x1, pre_y1 = pre_position
                        pre_x2 = pre_x1 + macro_block_size
                        pre_y2 = pre_y1 + macro_block_size

                        post_x1, post_y1 = post_position
                        post_x2 = post_x1 + macro_block_size
                        post_y2 = post_y1 + macro_block_size

                        # input_img_patch_list_pre.append(temporal_input_img_patch_list[t, pre_y1:pre_y2, pre_x1:pre_x2,:])
                        # input_img_patch_list_cur.append(temporal_input_img_patch_list[t, y1:y2, x1:x2,:])
                        # input_img_patch_list_post.append(temporal_input_img_patch_list[t, post_y1:post_y2, post_x1:post_x2,:])

                        # use SSIM metric
                        if pre_score < 0.9 or post_score < 0.9:
                            continue
                            print(" ME fail")
                            input_cur_patch = temporal_input_img_patch_list[t, y1:y2, x1:x2,:]
                            input_pre_patch = input_cur_patch
                            input_post_patch = input_cur_patch
                        else:
                            # print(" ME success")
                            input_pre_patch = temporal_input_img_patch_list[t, pre_y1:pre_y2, pre_x1:pre_x2,:]
                            input_cur_patch = temporal_input_img_patch_list[t, y1:y2, x1:x2,:]
                            input_post_patch = temporal_input_img_patch_list[t, post_y1:post_y2, post_x1:post_x2,:]
                            # display_img = np.hstack(([pre_patch, cur_patch, post_patch]))
                            # cv.imshow('sample image',display_img)
                            # cv.waitKey(0) # waits until a key is pressed
                            # cv.destroyAllWindows()

                        # [T x H x W x C]
                        temporal_input = np.stack([input_pre_patch,input_cur_patch,input_post_patch ], axis = 0)
                        # temporal_gt = np.stack([pre_patch, cur_patch, post_patch], axis = 0)
                        # H, W, C
                        temporal_gt = cur_patch

                        # [Tx N x H x W x C]
                        if input_img_patch is None:
                            input_img_patch = np.expand_dims(temporal_input, axis=1)
                        else:
                            input_img_patch = np.concatenate([input_img_patch, np.expand_dims(temporal_input, axis=1)], axis = 1)

                        if gt_img_patch is None:
                            gt_img_patch = np.expand_dims(temporal_gt, axis=0)
                            # gt_img_patch = np.expand_dims(temporal_gt, axis=1)
                        else:
                            gt_img_patch = np.concatenate([gt_img_patch, np.expand_dims(temporal_gt, axis=0)], axis = 0)
                            # gt_img_patch = np.concatenate([gt_img_patch, np.expand_dims(temporal_gt, axis=1)], axis = 1)
                        dataset_count = dataset_count + 1
                        # print("dataset_count :", dataset_count)

                        if dataset_count > dataset_threshold:
                            break
                        # print("gt_img_patch.shape: ", gt_img_patch.shape)
                        # res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
                        # threshold = 0.8
                        # loc = np.where( res >= threshold)
                        # for pt in zip(*loc[::-1]):
                        #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    return input_img_patch,  gt_img_patch                

def prestore_dataset(model = 'ARTN'):

    if model == 'ARTN':    
        resize_hr_frames_output_dir = os.path.join(Flags.disk_path, Flags.resize_gt_frame_dir)
        compressed_frames_output_dir = os.path.join(Flags.disk_path, Flags.compressed_frame_dir)
        input_img_patch, gt_img_patch = preprocess_ARTN_dataset(compressed_frames_output_dir, resize_hr_frames_output_dir)
        save_path = os.path.join(Flags.disk_path, 'ARTN')
        input_path = os.path.join(save_path, 'input.npy')
        gt_path = os.path.join(save_path, 'gt.npy')
        np.save(input_path, input_img_patch)
        np.save(gt_path, gt_img_patch)

def load_dataset(model = 'ARTN'):
    if model == 'ARTN':
        save_path = os.path.join(Flags.disk_path, 'ARTN')
        input_path = os.path.join(save_path, 'input.npy')
        gt_path = os.path.join(save_path, 'gt.npy')
        # [T x N x C x H  x W]
        return np.transpose(np.load(input_path), [0, 1, 4, 2, 3]), np.transpose(np.load(gt_path), [0, 3, 1, 2])
        

if __name__ == '__main__':
    prestore_dataset()