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

video_data_dict = { 
# Videos and frames are hard-coded. 
# We select frames to make sure that there is no scene switching in the data
# We assume that the Flags.duration is 120
    "121649159" : [0, 310,460,720,860], #1
    # "40439273"  : [90,520,700,1760,2920,3120,3450,4750,4950,5220,6500,6900,9420,9750], #2
    # "87389090"  : [100,300,500,800,1000,1200,1500,1900,2050,2450,2900], #3
    # "335874600" : [287, 308, 621, 1308, 1538, 1768, 2036, 2181, 2544, 2749, 2867, 3404, 3543, 3842, 4318, 4439,
    #                 4711, 4900, 7784, 8811, 9450],  # new, old #[4,6,13,14,19] 404
    # "114053015" : [30,1150,2160,2340,3190,3555], #5 
    # "160578133" : [550,940,1229,1460,2220,2900, 3180, 4080, 4340, 4612, 4935, 
    #                 5142, 5350, 5533, 7068], # new, old #[20,21,27,29,30,35] 404
    # "148058982" : [80,730,970,1230,1470,1740], #7
    # "150225201" : [0,560,1220,1590,1780], #8
    # "145096806" : [0,300,550,800,980,1500], #9
    # "125621327" : [240,900,1040,1300,1970,2130,2530,3020,3300,3620,3830,4300,4700,4960], #10
    # "162166758" : [120,350,540,750,950,1130,1320,1530,1730,1930], #11
    # "115829238" : [140,450,670,910,1100,1380,1520,1720], #12
    # "159455925" : [40,340,490,650,850,1180,1500,1800,2000,2300,2500,2800,3200], #15
    # "193873193" : [0,280,1720], #16
    # "133842385" : [300,430,970,1470,1740,2110,2240,2760,3080,3210,3400,3600], #17
    # "97692560"  : [0,210,620,930,1100,1460,1710,2400,2690,3200,3400,3560,3780], #18
    # "142480565" : [835,1380,1520,1700,2370,4880], #22
    # "174952003" : [480,680,925,1050,1200,1380,1600,1800,2100,2350,2480,2680,3000,3200,3460,4500,4780,
    #                 5040,5630,5830,6400,6680,7300,7500,7800], #23
    # "165643973" : [300,600,1000,1500,1700,1900,2280,2600,2950,3200,3500,3900,4300,4500], #24
    # "163736142" : [120,400,700,1000,1300,1500,1750,2150,2390,2550,3100,3400,3800,4100,4400,4800,5100,5500,5800,6300], #25
    # "189872577" : [0,170,340,4380,4640,5140,7300,7470,7620,7860,9190,9370], #26
    # "181180995" : [30,160,400,660,990,2560,2780,3320,3610,5860,6450,7260,7440,8830,9020,9220,9390,], #28
    # "167892347" : [220,1540,2120,2430,5570,6380,6740],  #31
    # "146484162" : [1770,2240,3000,4800,4980,5420,6800],  #32
    # "204313990" : [110],   #33
    # "169958461" : [140,700,1000,1430,1630,1900,2400,2600,2800,3000,3200,3600,3900,4200,4600,5000,5700,
    #                 6000,6400,6800,7100,7600,7900,8200],   #34
    # "198634890" : [200,320,440,1200,1320,1560,1680,1800,1920,3445],   #36
    # "89936769"  : [1260,1380,1880], #37
}


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
        f.write('original_res, output_res, r, p, dp, dmetric,\
                            psnr_mean, psnr_lc, psnr_hc, psnr_std, \
                            ssim_mean, ssim_lc, ssim_hc, ssim_std, \
                            lpips_mean, lpips_lc, lpips_hc, lpips_std,\
                            tof_mean, tof_lc, tof_hc, tof_std,\
                            tlp100_mean, tlp100_lc, tlp100_hc, tlp100_std') 
        
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
                T = 0.01
                p = 0.1
                ra = 1/ ((w_ori / w_c) * (h_ori / h_c)) 
                r = ra * p
                # define a list to store the metrics value so that we could compute the
                # statistics of that particular metric, e.g. std, mean,
                # TODO
                metric = [0] # the first 0 should be omitted when computing the statistic
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
                # define a step size for the change in the ratio, dp
                # this step size should be reduced slowly after each iteration in the
                # while loop
                # TODO
                dp = 0.01
                decay = 0.99

                while dmetric > 0.01:
                    
                    # # compresss the videos to a particular aspect ratio with r*bitrate
                    # # TODO
                    # # define the hr input video path
                    # hr_video_path_list = video_path_list

                    # # define the lr output video path
                    # lr_video_path_list = []
                    # for video_path in hr_video_path_list:
                    #     split_name = video_path.split(os.path.sep)
                    #     video_name = split_name[-1]
                    #     lr_dir = os.path.join(Flags.disk_compressed_path, output_res)

                    #     if(not os.path.exists(lr_dir)): os.makedirs(lr_dir)

                    #     lr_path = os.path.join(lr_dir, video_name)
                    #     lr_video_path_list.append(lr_path)

                    # print(lr_video_path_list)

                    # for index, (hr_path, lr_path) in enumerate(zip(hr_video_path_list, lr_video_path_list)):
                    #     print('original bit_rate: ', bit_rate_list[index])
                    #     bit_rate = bit_rate_list[index]
                    #     video_bitrate = int(bit_rate * ra * p)
                    #     # if video_bitrate <= 50000:
                    #     #     video_bitrate = 50000
                    #     #     print('bitrate is too low')
                    #     video_bitrate = str(video_bitrate)
                    #     compress_videos(input_video_path=hr_path, output_video_path=lr_path, resolution = output_res, video_bitrate=video_bitrate)

                    
                    #     # slice the compressed video to frames and store into a folder
                    #     # TODO
                    #     lr_frames_path = os.path.join(Flags.disk_frames_path, output_res)
                    #     prepare_frames(lr_path, lr_frames_path, w_c, h_c, verbose=True)
                        

                    # # compute the metric of the compressed video frames with
                    # # hr video frames
                    # # output the metrics to the csv file in the same directory as
                    # # the compressed video frames
                    # # TODO
                    
                    out_list = os.path.join(Flags.disk_frames_path, output_res)
                    # tar_list = resized_hr_frames_path

                    # cmd1 = ["python", "metrics.py",
                    #     "--output", out_list+"_metric_log/",
                    #     "--results", out_list,
                    #     "--targets", tar_list,
                    # ]
                    # mycall(cmd1).communicate()
                    
                    stats = pandas.read_csv(out_list+ os.path.join("_metric_log", 'metrics.csv'))

                    print(stats[-6:])
                    psnr = stats.iloc[-5]['PSNR_00']
                    ssim = stats.iloc[-5]['SSIM_00']
                    lpips = stats.iloc[-5]['LPIPS_00']
                    tof = stats.iloc[-5]['tOF_00']
                    tlp100 = stats.iloc[-5]['tLP100_00']

                    metric['psnr'].append(psnr)
                    metric['ssim'].append(ssim)
                    metric['lpips'].append(lpips)
                    metric['tof'].append(tof)
                    metric['tlp100'].append(tlp100)

                    exit()

                    # read the csv file to obtain the metrics, add the metric to the list
                    # TODO
                    # parse the txt file

                    # compute the difference between previous and current metric
                    # determine whether the ratio should be increased or decreased
                    # E.g. for metric PSNR, 
                    #       if the PSNR increases compared to the previous metric, 
                    #               ratio should be increased
                    #       if the PSNR decreases compared to the previous metric, 
                    #               ratio should be decreased
                    # TODO
                    dmetric = abs(metrics['psnr'][-2] - metrics['psnr'][-1])
                    if(metrics['psnr'][-2] < metrics['psnr'][-1]):
                        p += dp
                    else:
                        p -= dp
                    # Repeat while loop until convergence

                # compute the metric statistics
                # Mean
                # std
                # TODO
                # mean, lower confidence, higher confidence
                psnr_mean, psnr_lc, psnr_hc = mean_confidence_interval(metrics['psnr'][1:])
                ssim_mean, ssim_lc, ssim_hc = mean_confidence_interval(metrics['ssim'][1:])
                lpips_mean, lpips_lc, lpips_hc = mean_confidence_interval(metrics['lpips'][1:])
                tof_mean, tof_lc, tof_hc = mean_confidence_interval(metrics['toff'][1:])
                tlp100_mean, tlp100_lc, tlp100_hc = mean_confidence_interval(metrics['tlp100'][1:])

                psnr_std = np.std(metrics['psnr'][1:])
                ssim_std = np.std(metrics['ssim'][1:])
                lpips_std = np.std(metrics['lpips'][1:])
                tof_std = np.std(metrics['toff'][1:])
                tlp100_std = np.std(metrics['tlp100'][1:])

                # store the [original resolution, aspect ratio, r, dr, dmetric, metric (str), metric mean, metric std]
                # to a csv
                # TODO

                with open(output_dir, 'a+') as f:  
                    data = [original_res, output_res, r, p, dp, dmetric,\
                            psnr_mean, psnr_lc, psnr_hc, psnr_std, \
                            ssim_mean, ssim_lc, ssim_hc, ssim_std, \
                            lpips_mean, lpips_lc, lpips_hc, lpips_std,\
                            tof_mean, tof_lc, tof_hc, tof_std,\
                            tlp100_mean, tlp100_lc, tlp100_hc, tlp100_std]
                    for i, d in enumerate(data):
                        if i < len(data) - 1
                            f.write(str(d) + ',')
                        else:
                            f.write(str(d) + '\n')


                



if __name__ == '__main__':
    main()
    # if(Flags.process == 0):
    #     download_videos()
    # elif(Flags.process == 1):
    #     compress_videos()
    # elif(Flags.process == 2):
    #     prepare_frames()
    # elif(Flags.process == 3): # only resize the image
    #     resize_videos(output_dir = Flags.disk_resize_path)
    # elif(Flags.process == 4): # compress resize the image to 50k bitrate
    #     compress_videos(input_dir=Flags.disk_resize_path)
    # elif(Flags.process == 5):
    #     prepare_frames(hr_input_dir=Flags.disk_resize_path)
    # elif(Flags.process == 6): # Need to specify the bitrate
    #     compress_videos(input_dir=Flags.disk_resize_path, video_bitrate=Flags.video_bitrate)