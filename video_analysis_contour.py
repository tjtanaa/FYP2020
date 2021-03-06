"""

    This script is to run a test on a set of videos to determine
    the best possible compression rate such that the quality of the
    video doesn not deteriorate too much.

    The procedure is as follows:
    I thought of a way to make numerical analysis to plot a graph to show the relationship between resolution, compression ratio (bitrate) and visual quality.
    

    Procedure:
    1. there are only a finite set of commonly used resolution. Pick one resolution and fixed it. (e.g. lr 1280x720, hr4906x1980). Pick a metric (e.g.PSNR)
    2. Prepare the GT ( bilinear/bicubic downsampling the hr frames)
    3. Prepare the lr frames ,
    3.1 Pick a ratio, fix a small diff ,dx, small step_size, dz
    3.2 Compress the hr video to lr video to two different bitrate ( 1. ratio*ori_bitrate, 2. (ratio-dx)*ori_bitrate)
    3.3 Compute PSNR values for both lr videos.
    3.4 if 1 has higher psnr, then try increasing the ratio by dz, if 2 has higher psnr, decrease the ratio by dz. 
    3.5 if delta psnr =  abs(psnr video1 - psnr video2) < threshold, stop, save the ratio and psnr value, and resolution.
    4. Repeat the above steps for all other resolutions. e.g. 720x480

    Additional Notes.
    5. Each experiments consists of set of 10 HR videos, with variable number of bitrates (Take a note about the bitrates of these videos)

"""


import os, sys, datetime
import cv2 as cv
import argparse
import youtube_dl

from lib.data import video
import subprocess
from collections import defaultdict
import json
import subprocess, signal, shutil
import pandas
import numpy as np
import scipy.stats




"python dataPrepare.py --start_id 2000 --duration 120 --process 1"

# ------------------------------------parameters------------------------------#
parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start_id', default=2000, type=int, help='starting scene index')
parser.add_argument('--duration', default=10, type=int, help='scene duration')
parser.add_argument('--disk_path', default=".\\video_analysis\\original_videos", help='the path to save the dataset')
parser.add_argument('--summary_dir', default="", help='the path to save the log')
parser.add_argument('--REMOVE', action='store_true', help='whether to remove the original video file after data preparation')
parser.add_argument('--TEST', action='store_true', help='verify video links, save information in log, no real video downloading!')
parser.add_argument('--disk_compressed_path', default=".\\video_analysis\\compressed_videos", help='the path to save the dataset')
parser.add_argument('--disk_frames_path', default=".\\video_analysis\\LR_frames", help='the path to save the dataset')
parser.add_argument('--disk_resize_path', default=".\\video_analysis\\HR_frames", help='the path to save the dataset')
parser.add_argument('--video_bitrate', default="100k", help='video_bitrate')
parser.add_argument('--process', default=1, type=int, help='run process 0: download video 1: compress video 2: generate frames')
Flags = parser.parse_args()

if Flags.summary_dir == "":
    Flags.summary_dir = os.path.join(Flags.disk_path, "log/")
os.path.isdir(Flags.disk_path) or os.makedirs(Flags.disk_path)
os.path.isdir(Flags.summary_dir) or os.makedirs(Flags.summary_dir)


# ------------------------------------log------------------------------#
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    for name, value in FLAGS.__dict__.items():
        print('\t%s: %s'%(name, str(value)))
    print('End of configuration')
    
class MyLogger(object):
    def __init__(self):
        self.terminal = sys.stdout
        now_str = datetime.datetime.now().strftime("%m%d%H%M")
        self.log = open(Flags.summary_dir + "logfile_%s.txt"%now_str, "a") 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message) 

    def flush(self):
        self.log.flush()
        
sys.stdout = MyLogger()
print_configuration_op(Flags)

def preexec(): # Don't forward signals.
    os.setpgrp()
    
def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn = preexec)
    

# ------------------------------------tool------------------------------#
def gen_frames(infile, outdir, width, height, start, duration, savePNG=True):
    print("folder %s: %dx[%d,%d] at frame %d of %s"
        %(outdir, duration, width, height, start,infile,))
    
    if savePNG:
        vcap = cv.VideoCapture(infile) # 0=camera
        width = -1
        height = -1
        if vcap.isOpened():
            vcap.set(cv.CAP_PROP_POS_FRAMES, start) 
            # get vcap property 
            width = int(vcap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
            height = int(vcap.get(cv.CAP_PROP_FRAME_HEIGHT)) # float
            print("Resolution: {} x {}".format(int(width), int(height)))
        assert width >0 and height >0


        index = infile.find("compressed_")
        success,image = vcap.read()
        # count = init_count
        count = 0
        while success:
            # filename = os.path.join(output_dir,"%06d.png"%(count))
            filename = (outdir+("_%04d.png"%(count)))
            # filename = (outdir+'col_high'+("_%04d.png"%(count))) if index == -1 else (outdir+'col_high'+("_compressed_%04d.png"%(count)))
            cv.imwrite(filename, image)     # save frame as JPEG file      
            success,image = vcap.read()
            # print('Read a new frame: ', success)
            count += 1
            if count >=duration:
                break
        

def compress_videos(input_video_path=None, output_video_path=None, resolution = None, video_bitrate=None):

    if(not os.path.exists(input_video_path)): raise FileNotFoundError

    # if(not os.path.exists(output_video_path)): raise FileNotFoundError

    vcap = cv.VideoCapture(input_video_path) # 0=camera
    width = -1
    height = -1
    if vcap.isOpened(): 
        # get vcap property 
        width = int(vcap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
        height = int(vcap.get(cv.CAP_PROP_FRAME_HEIGHT)) # float
        print("Resolution: {} x {}".format(int(width), int(height)))
    assert width >0 and height >0

    if resolution is None:
        resolution = '{}x{}'.format(width//4, height//4)
    if video_bitrate is None:
        video_bitrate = '50k'
    video_codec = 'libx264'
    audio_codec = 'copy'

    cmd = ['ffmpeg', '-y', '-i', input_video_path, 
    '-s', resolution,
    '-b:v', video_bitrate,
    '-vcodec', video_codec,
    '-acodec', audio_codec,
    output_video_path]

    subprocess.call(cmd)

    print("Compressed a valid input video: %s to %s"%(input_video_path, output_video_path))

def prepare_frames(hr_input_video_path=None, output_dir=None, output_width=None, output_height=None, verbose=True):

    # check if the video exists
    if(not os.path.exists(hr_input_video_path)): raise FileNotFoundError

    # define the output directory
    output_res = str(output_width) + 'x' + str(output_height)

    split_name = hr_input_video_path.split(os.path.sep)
    # print(split_name)
    video_name = split_name[-1].split('.')[0]
    # print(video_name)


    # if directory does not exist create one
    if(not os.path.exists(output_dir)): os.makedirs(output_dir)
    tar_dir = os.path.join(output_dir, video_name)

    # do sanity check 
    if verbose:
        hr_vcap = cv.VideoCapture(hr_input_video_path) # 0=camera
        hr_width = -1
        hr_height = -1
        if hr_vcap.isOpened(): 
            # get vcap property 
            hr_width = int(hr_vcap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
            hr_height = int(hr_vcap.get(cv.CAP_PROP_FRAME_HEIGHT)) # float
            print("hr_input_video_path: {} Resolution: {} x {}".format(hr_input_video_path, int(hr_width), int(hr_height)))
        assert hr_width >0 and hr_height >0

    print("generate frames")
    gen_frames(hr_input_video_path, tar_dir, hr_width, hr_height, 0, Flags.duration)

    return output_dir



def resize_hr_frames(hr_input_video_path=None, output_dir=None, output_width=None, output_height=None, verbose=True):
    # check if the video exists
    if(not os.path.exists(hr_input_video_path)): raise FileNotFoundError

    # define the output directory
    output_res = str(output_width) + 'x' + str(output_height)

    output_dir = os.path.join(Flags.disk_resize_path, output_res) # store the resized hr frames

    # if directory does not exist create one
    if(not os.path.exists(output_dir)): os.makedirs(output_dir)

    for image_path in os.listdir(hr_input_video_path):
        if image_path.find('.png') != -1:
            input_img_path = os.path.join(hr_input_video_path, image_path)
            output_img_path = os.path.join(output_dir, image_path)
            img = cv.imread(input_img_path, cv.IMREAD_UNCHANGED)
            # resize image
            resized = cv.resize(img, (output_width, output_height), interpolation = cv.INTER_CUBIC)
            status = cv.imwrite(output_img_path, resized)
            # print(status)
            if(not(status)): 
                print("Failed to write resized image")
                exit()


def prepare_meta_data(input_video_path=None, verbose=True):
    # # check if the video exists
    # if(not os.path.exists(input_video_path)): raise FileNotFoundError

    # check if the video exists
    if (not os.path.isfile(input_video_path)):
        print("Skipped invalid link or other error:" + input_video_path)
        raise FileNotFoundError

    # define the output directory
    split_name = input_video_path.split(os.path.sep)
    output_dir = split_name[-4:-1]

    output_dir = os.path.join('.',os.path.join(*output_dir))
    video_name = split_name[-1]
    print('Video name: ', video_name)
    json_name = video_name.split('.')[0] + ".json"
    if(verbose):
        print("output_dir: ", output_dir, '\t output_file: ', json_name)
    # input_video_path = os.path.join(input_dir, video_name) 
    input_json_path = os.path.join(output_dir, json_name) 

    cmd = ['ffprobe', '-v', 'quiet', \
            '-print_format', 'json', \
            '-show_format', '-show_streams', '-select_streams', 'v', input_video_path]

    meta_data = subprocess.check_output(cmd)

    meta_data_json = json.loads(meta_data.decode('utf-8'))
    print(meta_data_json)
    print("Bitrate of {} : {}".format(video_name, meta_data_json["streams"][0]["bit_rate"]))
    # exit()
    with open(input_json_path, "w+") as outfile:
        json.dump(meta_data_json, outfile, indent=4)
        print("Obtained information from a valid input video: %s > %s"%(input_video_path, input_json_path))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def get_resolution(res_str):
    """
        convert the resolution in string to int
    """
    res = res_str.split('x')
    w = int(res[0])
    h = int(res[1])
    return w, h

def insert_dir(directory, sub_dir):
    split_name = directory.split(os.path.sep)
    # print(split_name)
    split_name.insert(-1, sub_dir)
    # print(split_name)
    return os.path.join(*split_name)

def main():

    now_str = datetime.datetime.now().strftime("%m%d%H%M")
    output_dir = os.path.join(Flags.summary_dir, 'stats_%s.txt'%(now_str))


    with open(output_dir, 'w') as f: 
        f.write("original_res, output_res, r, p, psnr_mean, psnr_lc, psnr_hc, psnr_std, ssim_mean, ssim_lc, ssim_hc, ssim_std, lpips_mean, lpips_lc, lpips_hc, lpips_std, tof_mean, tof_lc, tof_hc, tof_std, tlp100_mean, tlp100_lc, tlp100_hc, tlp100_std") 
        
    # define the commonly used resolutions
    cresolutions = ['3840x2160', '1920x1080', '1440x1080', '1280x720', '1280x1080' '960x540', '720x480', '640x360', '480x360']
    supported_video_extention = ['mov', 'mp4']

    cur_dir = os.getcwd()
    analysis_dir = os.path.join(os.path.join(cur_dir, 'video_analysis'), 'original_videos')
    

    # define dictionary of names of folers, which store the original video
    # the name of the folder indicates the original resolution
    # e.g. 2160x3840, 1980x1280
    # key is the resolution
    # string is the path to the videos in that subfolder
    video_folders = defaultdict(list)


    # get the subfolders in the video_analysis folder
    for subfolder in os.listdir(analysis_dir):
        subfolder_path = os.path.join(analysis_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.find('x') != -1:
            print(subfolder)
            for video in os.listdir(subfolder_path):
                video_path = os.path.join(subfolder_path, video)
                print(video.split('.'))
                if os.path.isfile(video_path) and (video.split('.')[-1] in supported_video_extention):
                    video_folders[subfolder].append(video_path)

    print(video_folders)

    ori_disk_resize_path = Flags.disk_resize_path
    ori_disk_compressed_path = Flags.disk_compressed_path
    ori_disk_frames_path = Flags.disk_frames_path


    for res in video_folders.keys():

        # update the Flags.disk_resize_path, Flags.disk_compressed_path, Flags.disk_frames_path

        Flags.disk_resize_path = insert_dir(ori_disk_resize_path, res)
        Flags.disk_compressed_path = insert_dir(ori_disk_compressed_path, res)
        Flags.disk_frames_path = insert_dir(ori_disk_frames_path, res)
        if(not os.path.exists(Flags.disk_resize_path)): os.makedirs(Flags.disk_resize_path)
        if(not os.path.exists(Flags.disk_compressed_path)): os.makedirs(Flags.disk_compressed_path)
        if(not os.path.exists(Flags.disk_frames_path)): os.makedirs(Flags.disk_frames_path)
        # print(Flags.disk_resize_path, Flags.disk_compressed_path, Flags.disk_frames_path)

        # parse the string resolution to integer
        w_ori, h_ori = get_resolution(res)
        # print('w_ori: ', w_ori, '\t h_ori: ', h_ori)
        original_res = str(w_ori) + 'x' + str(h_ori)
        hr_frames_output_dir = os.path.join(Flags.disk_resize_path, original_res) # store the resized hr frames


        # prepare the unscaled high resolution frames
        video_path_list = video_folders[res]
        for video_path in video_path_list:
            prepare_frames(video_path, hr_frames_output_dir, w_ori, h_ori, verbose=True)
            prepare_meta_data(video_path)

        # get the bitrate of the original videos and sotre them in a list
        bit_rate_list = []
        # read the metadata json file
        for video_path in video_path_list:
            extension = video_path.split('.')[-1]
            json_file = video_path.replace(extension, 'json')
            with open(json_file,'r') as f:
                meta_data_json = json.load(f)
                video_bitrate = float(int(meta_data_json["streams"][0]["bit_rate"]))
                bit_rate_list.append(video_bitrate)


        # get the configuration
        for res_c in cresolutions:
            w_c, h_c = get_resolution(res_c)
            print('w_c :', w_c, '\t h_c: ', h_c)

            if (w_ori > w_c) and (h_ori > h_c) and (w_ori % w_c == 0) and (h_ori % h_c ==0): # valid compression
                # pass
                # prepare the hr video frames once for a particular aspect_ratio as GT
                # through bilinear and bicubic algorithm
                # for each aspect_ratio only need to prepare once

                # define the output directory
                output_res = str(w_c) + 'x' + str(h_c)
                hr_frames_path = os.path.join(Flags.disk_resize_path, original_res) # store the hr frames
                # print("hr_frames_path: ", hr_frames_path)
                resized_hr_frames_path = os.path.join(Flags.disk_resize_path, output_res) # store the resized hr frames
                if(not os.path.exists(resized_hr_frames_path)): os.mkdir(resized_hr_frames_path)
                print('resized_hr_frames_path: ', resized_hr_frames_path)
                resize_hr_frames(hr_frames_path, resized_hr_frames_path, w_c, h_c, verbose=True)

                dmetric = 999999
                # unit of 
                #   * PSNR is dB
                #   * SSIM
                #   * toF
                #   * 
                # while the difference is larger than a threshold (stopping condition)
                # threshold, T = 0.01 for PSNR (in this case)
                # bitrate reduction ratio due to reduce in resolution, ra = 1/(original_w / aspect_ratio_w)^2
                # bitrate reduction ratio, r = ra * p
                # percentage, p = 0.3
                # TODO:
                # T = 0.01
                # p = 0.1
                ra = 1/ ((w_ori / w_c) * (h_ori / h_c)) 
                # r = ra * p
                # define a list to store the metrics value so that we could compute the
                # statistics of that particular metric, e.g. std, mean,
                # TODO
                # metric = [0] # the first 0 should be omitted when computing the statistic
                # psnr_list = []
                # ssim_list = []
                # lpips_list = []
                # tof_list = []
                # tlp100_list = []
                metric = defaultdict(list)
                metric['psnr'].append(0)
                metric['ssim'].append(0)
                metric['lpips'].append(0)
                metric['tof'].append(0)
                metric['tlp100'].append(0)

                # define a list of p's to be investigated
                p_values = [0.01, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7]

                for p in p_values:
                    invalid_bitrate = False
                    
                    # compresss the videos to a particular aspect ratio with r*bitrate
                    # TODO
                    # define the hr input video path
                    hr_video_path_list = video_path_list

                    # define the lr output video path
                    lr_video_path_list = []
                    for video_path in hr_video_path_list:
                        split_name = video_path.split(os.path.sep)
                        video_name = split_name[-1]
                        lr_dir = os.path.join(Flags.disk_compressed_path, output_res)

                        if(not os.path.exists(lr_dir)): os.makedirs(lr_dir)

                        lr_path = os.path.join(lr_dir, video_name)
                        lr_video_path_list.append(lr_path)

                    print(lr_video_path_list)

                    for index, (hr_path, lr_path) in enumerate(zip(hr_video_path_list, lr_video_path_list)):
                        print('original bit_rate: ', bit_rate_list[index])
                        bit_rate = bit_rate_list[index]
                        video_bitrate = int(bit_rate * ra * p)
                        if video_bitrate <= 50000:
                            invalid_bitrate = True
                            print('bitrate is too low')
                            break
                        video_bitrate = str(video_bitrate)
                        compress_videos(input_video_path=hr_path, output_video_path=lr_path, resolution = output_res, video_bitrate=video_bitrate)

                    
                        # slice the compressed video to frames and store into a folder
                        # TODO
                        lr_frames_path = os.path.join(Flags.disk_frames_path, output_res)
                        prepare_frames(lr_path, lr_frames_path, w_c, h_c, verbose=True)
                        

                    if(invalid_bitrate):
                        print('skip p value: {}'.format(p))
                        continue
                    # compute the metric of the compressed video frames with
                    # hr video frames
                    # output the metrics to the csv file in the same directory as
                    # the compressed video frames
                    # TODO
                    
                    out_list = os.path.join(Flags.disk_frames_path, output_res)
                    tar_list = resized_hr_frames_path

                    cmd1 = ["python", "metrics.py",
                        "--output", out_list+"_metric_log/",
                        "--results", out_list,
                        "--targets", tar_list,
                    ]
                    mycall(cmd1).communicate()
                    
                    stats = pandas.read_csv(out_list+ os.path.join("_metric_log", 'metrics.csv'))

                    print(stats[-6:])
                    psnr = stats.iloc[-5]['PSNR_00']
                    ssim = stats.iloc[-5]['SSIM_00']
                    lpips = stats.iloc[-5]['LPIPS_00']
                    tof = stats.iloc[-5]['tOF_00']
                    tlp100 = stats.iloc[-5]['tLP100_00']

                    metric['psnr'].append(float(psnr))
                    metric['ssim'].append(float(ssim))
                    metric['lpips'].append(float(lpips))
                    metric['tof'].append(float(tof))
                    metric['tlp100'].append(float(tlp100))

                    # compute the metric statistics
                    # Mean
                    # std
                    # TODO
                    # mean, lower confidence, higher confidence
                    psnr_mean, psnr_lc, psnr_hc = mean_confidence_interval(metric['psnr'][1:])
                    ssim_mean, ssim_lc, ssim_hc = mean_confidence_interval(metric['ssim'][1:])
                    lpips_mean, lpips_lc, lpips_hc = mean_confidence_interval(metric['lpips'][1:])
                    tof_mean, tof_lc, tof_hc = mean_confidence_interval(metric['tof'][1:])
                    tlp100_mean, tlp100_lc, tlp100_hc = mean_confidence_interval(metric['tlp100'][1:])

                    psnr_std = np.std(metric['psnr'][1:])
                    ssim_std = np.std(metric['ssim'][1:])
                    lpips_std = np.std(metric['lpips'][1:])
                    tof_std = np.std(metric['toff'][1:])
                    tlp100_std = np.std(metric['tlp100'][1:])

                    # store the [original resolution, aspect ratio, r, dr, dmetric, metric (str), metric mean, metric std]
                    # to a csv
                    # TODO

                    with open(output_dir, 'a+') as f:  
                        data = [original_res, output_res, ra*p, p,\
                                psnr_mean, psnr_lc, psnr_hc, psnr_std, \
                                ssim_mean, ssim_lc, ssim_hc, ssim_std, \
                                lpips_mean, lpips_lc, lpips_hc, lpips_std,\
                                tof_mean, tof_lc, tof_hc, tof_std,\
                                tlp100_mean, tlp100_lc, tlp100_hc, tlp100_std]
                        for i, d in enumerate(data):
                            if i < len(data) - 1:
                                f.write(str(d) + ',')
                            else:
                                f.write(str(d) + '\n')


                
def plot_contour(input_file):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    def surface_plot(X,Y,Z,**kwargs):
        """ WRITE DOCUMENTATION
        """
        xlabel, ylabel, zlabel, title = kwargs.get('xlabel',""), kwargs.get('ylabel',""), kwargs.get('zlabel',""), kwargs.get('title',"")
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X,Y,Z)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()