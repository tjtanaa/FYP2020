import os, sys, datetime
import cv2 as cv
import argparse
import youtube_dl
import json

from lib.data import video
import subprocess

"python prepare_dataset_clean.py --start_id 2000 --duration 120 --process 7"
# D:\\Github\\tecogan_video_data\\train_video

# ------------------------------------parameters------------------------------#
parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start_id', default=2000, type=int, help='starting scene index')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
# parser.add_argument('--disk_path', default="D:\\Github\\tecogan_video_data", help='the path to save the dataset')
parser.add_argument('--disk_path', default="D:\\Github\\Vimeo90K", help='the path to save the dataset')
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
    

def download_videos():
    """
        this function download the videos to the disk_path
    """
    cur_id, valid_video, try_num = Flags.start_id, 0, 0

    for keys in video_data_dict:
        try_num += len(video_data_dict[keys])
    print("Try loading %dx%d."%(try_num, Flags.duration))
                
    ydl = youtube_dl.YoutubeDL( 
        {'format': 'bestvideo/best',
        'outtmpl': os.path.join(Flags.disk_path, '%(id)s.%(ext)s'),})
        
    saveframes = not Flags.TEST
    for keys in video_data_dict:
        tar_vid_input = link_path + keys
        print(tar_vid_input)
        info_dict = {"width":-1, "height": -1, "ext": "xxx", }
        
        # download video from vimeo
        try:
            info_dict = ydl.extract_info(tar_vid_input, download=saveframes)
            # we only need info_dict["ext"], info_dict["width"], info_dict["height"]
        except KeyboardInterrupt:
            print("KeyboardInterrupt!")
            exit()
        except:
            print("youtube_dl error:" + tar_vid_input)
            pass
        
        # check the downloaded video
        tar_vid_output = os.path.join(Flags.disk_path, keys+'.'+info_dict["ext"])
        if saveframes and (not os.path.exists(tar_vid_output)):
            print("Skipped invalid link or other error:" + tar_vid_input)
            continue
        if info_dict["width"] < 400 or info_dict["height"] < 400:
            print("Skipped videos of small size %dx%d"%(info_dict["width"] , info_dict["height"] ))
            print("remove ", tar_vid_output)
            os.remove(tar_vid_output)
            continue
        valid_video = valid_video + 1
        print("Downloaded valid video %d"%(valid_video))
    print("Done Downloading Video")

# ------------------------------------tool------------------------------#
def gen_frames(infile, outdir, width, height, start, duration, prefix=None, savePNG=True):
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


        # index = infile.find("compressed_")
        success,image = vcap.read()
        # count = init_count
        count = 0
        while success:
            # filename = os.path.join(output_dir,"%06d.png"%(count))
            # filename = (outdir+("_%04d.png"%(count)))
            filename = (outdir+'col_high'+("_%04d.png"%(count))) if prefix is None else (outdir+ prefix + '_col_high'+("_%04d.png"%(count)))
            cv.imwrite(filename, image)     # save frame as JPEG file      
            success,image = vcap.read()
            # print('Read a new frame: ', success)
            count += 1
            if count >=duration:
                break
        

def compress_videos(input_video_path=None, output_video_path=None, resolution = None, video_bitrate=None, qp=None):
    '''
        it takes in the a video specific path
        output to the output video specific path
    '''

    if(not os.path.exists(input_video_path)): raise FileNotFoundError


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

    if qp is None:
        cmd = ['ffmpeg', '-y', '-i', input_video_path, 
        '-s', resolution,
        '-b:v', video_bitrate,
        '-vcodec', video_codec,
        # '-acodec', audio_codec,
        output_video_path]
    else:
        cmd = ['ffmpeg', '-y', '-i', input_video_path, 
        '-s', resolution,
        # '-b:v', video_bitrate,
        '-vcodec', video_codec,
        '-qp', qp,
        # '-acodec', audio_codec,
        output_video_path]        

    subprocess.call(cmd)

    print("Compressed a valid input video: %s to %s"%(input_video_path, output_video_path))


def prepare_frames(input_dir=None, output_dir=None):
    '''
        Generates the frames of all the video in the input_dir to output_dir
    '''
# ------------------------------------main------------------------------#
    cur_id, valid_video, try_num = Flags.start_id, 0, 0

    if input_dir is None:
        input_dir = os.path.join(Flags.disk_path, Flags.train_path)
    if output_dir is None:
        output_dir = os.path.join(Flags.disk_path, Flags.gt_path)

    for keys in video_data_dict:
        try_num += len(video_data_dict[keys])
    print("Try loading %dx%d."%(try_num, Flags.duration))
    
    if(not os.path.exists(input_dir)): raise FileNotFoundError

    if(not os.path.exists(output_dir)): os.mkdir(output_dir)

    for key in sorted(video_data_dict.keys()):
        video_name = key + ".mp4"
        # input_video_path = os.path.join(input_dir, video_name)
        # check the downloaded video
        input_video_path = os.path.join(input_dir, video_name)
        if (not os.path.isfile(input_video_path)):
            print("prepare_frames ][ Skipped invalid link or other error:" + input_video_path)
            continue

        vcap = cv.VideoCapture(input_video_path) # 0=camera
        width = -1
        height = -1
        if vcap.isOpened(): 
            # get vcap property 
            width = int(vcap.get(cv.CAP_PROP_FRAME_WIDTH))   # float
            height = int(vcap.get(cv.CAP_PROP_FRAME_HEIGHT)) # float
            print("lr_ input_video_path: {} Resolution: {} x {}".format(input_video_path, int(width), int(height)))
        assert width >0 and height >0

        # get training frames
        for start_fr in video_data_dict[key]:
            tar_dir = os.path.join(output_dir, "scene_%04d/"% cur_id)
            os.path.isdir(tar_dir) or os.makedirs(tar_dir)
            print("generate lr frames")
            gen_frames(input_video_path, tar_dir, width, height, start_fr, Flags.duration, prefix=None)
            cur_id = cur_id+1 # important factor to determine the scene folder id




def _resize_hr_frames(hr_input_image_path=None, output_dir=None, output_width_ratio=None, output_height_ratio=None, verbose=True):
    '''
        only search for images in the hr_input_video_path then resize the image
        to the output_dir
    '''
    # check if the video exists
    if(not os.path.exists(hr_input_image_path)): raise FileNotFoundError

    # # define the output directory
    # output_res = str(output_width) + 'x' + str(output_height)

    # output_dir = os.path.join(Flags.resize_dir, output_res) # store the resized hr frames

    # if directory does not exist create one
    if(not os.path.exists(output_dir)): os.makedirs(output_dir)

    for image_path in os.listdir(hr_input_image_path):
        if image_path.find('.png') != -1:
            input_img_path = os.path.join(hr_input_image_path, image_path)
            output_img_path = os.path.join(output_dir, image_path)
            img = cv.imread(input_img_path, cv.IMREAD_UNCHANGED)
            # resize image
            # height, width, number of channels in image
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]
             
            output_width = (int)(output_width_ratio * width)
            output_height = (int)(output_height_ratio * height)
            # https://chadrick-kwag.net/cv2-resize-interpolation-methods/
            resized = cv.resize(img, (output_width, output_height), interpolation = cv.INTER_CUBIC)
            status = cv.imwrite(output_img_path, resized)
            # print(status)
            if(not(status)): 
                print("Failed to write resized image")
                exit()

def resize_hr_frames(hr_input_video_dir=None, output_dir=None, output_width_ratio=None, output_height_ratio=None, verbose=True):
    '''
        search for subdirectories
        resize all the images in the subdirectories to the corresponding output_dir
    '''

    # search through current folders to find subdirectories

    # get the subfolders in the video_analysis folder
    for subfolder in os.listdir(hr_input_video_dir):
        subfolder_path = os.path.join(hr_input_video_dir, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.find('scene_') != -1:
            print(subfolder)
            # preprocess to get output_dir
            output_scene_subfolder = os.path.join(output_dir,subfolder)

            # if directory does not exist create one
            if(not os.path.exists(output_scene_subfolder)): os.makedirs(output_scene_subfolder)
            _resize_hr_frames(subfolder_path, output_scene_subfolder, output_width_ratio, output_height_ratio, verbose=verbose)



def prepare_meta_data(input_video_path=None, verbose=True):
    # # check if the video exists
    # if(not os.path.exists(input_video_path)): raise FileNotFoundError

    # check if the video exists
    if (not os.path.isfile(input_video_path)):
        print("Skipped invalid link or other error:" + input_video_path)
        raise FileNotFoundError

    # define the output directory
    split_name = input_video_path.split(os.path.sep)
    print("splitname: " ,split_name)
    output_dir = split_name[:-1]

    output_dir = os.path.join('.',os.path.join(*output_dir))
    video_name = split_name[-1]
    print('Video name: ', video_name)
    json_name = video_name.split('.')[0] + ".json"
    if(verbose):
        print("output_dir: ", output_dir, '\t output_file: ', json_name)
    # input_video_path = os.path.join(input_dir, video_name)
    input_json_path = os.path.join(output_dir, json_name)
    input_json_path =  input_json_path[:2] + os.path.sep + input_json_path[2:]

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

supported_video_extention = ['mov', 'mp4']  

if __name__ == '__main__':
    if(Flags.process == 0):
        download_videos()
    elif(Flags.process == 1):
        compress_videos()
    elif(Flags.process == 2):
        prepare_frames()
    elif(Flags.process == 3): # only resize the image
        resize_videos(output_dir = Flags.disk_resize_path)
    elif(Flags.process == 4): # compress resize the image to 50k bitrate
        compress_videos(input_dir=Flags.disk_resize_path)
    elif(Flags.process == 5):
        prepare_frames(hr_input_dir=Flags.disk_resize_path)
    elif(Flags.process == 6): # Need to specify the bitrate
        compress_videos(input_dir=Flags.disk_resize_path, video_bitrate=Flags.video_bitrate)

    # # Prepare the dataset for the ARTN
    # Input :
    # frames of compressed video by a factor of 4
    # compress video -> generate frames
    # GT:
    # HR frames of the bilinear resized original video frames
    # generate frames -> resize the frames        
    elif(Flags.process == 7): 
        # # Prepare the GT for artifacts removal network AND
        # # Input of video super resolution
        # resize_videos_by_4(output_dir = Flags.disk_resize_by_4_path)

        # Input path to the original video
        # get all the video path
        GT_dir = os.path.join(Flags.disk_path, Flags.gt_dir)

        hr_frames_output_dir = os.path.join(Flags.disk_path, Flags.gt_frames_dir)
        resize_hr_frames_output_dir = os.path.join(Flags.disk_path, Flags.resize_gt_frame_dir)

        compressed_dir = os.path.join(Flags.disk_path, Flags.compressed_dir)
        compressed_frames_output_dir = os.path.join(Flags.disk_path, Flags.compressed_frame_dir)

        GT_video_path_list = []
        # get the subfolders in the video_analysis folder

        for video in os.listdir(GT_dir):
            video_path = os.path.join(GT_dir, video)
            if os.path.isfile(video_path) and (video.split('.')[-1] in supported_video_extention):
                GT_video_path_list.append(video_path)
        print(GT_video_path_list)

        # # generate the metadata
        # for GT_video_dir in GT_video_path_list:
        #     # generate the metadata data of GT videos
        #     prepare_meta_data(GT_video_dir)

        # # # Preparing target video frames for the ARTN
        # # prepare GT video frames
        # prepare_frames(GT_dir, hr_frames_output_dir)

        # resize the video frames
        # resize_hr_frames(hr_frames_output_dir, resize_hr_frames_output_dir, 0.25, 0.25, verbose =True)

        # # Preparing the input video frames for the ARTN
        # read the meta data of the original videos
        # read the metadata json file
        for GT_video_path in GT_video_path_list:
            extension = GT_video_path.split('.')[-1]
            json_file = GT_video_path.replace(extension, 'json')
            compressed_video_path = GT_video_path.replace(GT_dir, compressed_dir)
            print("compressed_video_path: ", compressed_video_path)
            print("GT_video_path: ", GT_video_path)
            with open(json_file,'r') as f:
                meta_data_json = json.load(f)
                video_bitrate = float(int(meta_data_json["streams"][0]["bit_rate"]))
                video_width = float(int(meta_data_json["streams"][0]["width"]))
                video_height = float(int(meta_data_json["streams"][0]["height"]))
                if video_width >= 3840:
                    video_bitrate = 3000000 # 3Mbps
                elif video_width >= 1920:
                    video_bitrate = 3000000 # 3Mbps
                elif video_width > 1280:
                    video_bitrate = 1500000 # 1.5Mbps
                else:
                    print("Skipped")
                    continue
                resolution = '{}x{}'.format(int(video_width), int(video_height))
                # compress the video by a factor of 4
                compress_videos(input_video_path=GT_video_path, 
                                output_video_path=compressed_video_path, 
                                resolution = resolution, 
                                video_bitrate=str(int(video_bitrate)),
                                qp = str(32))

        # preparing compressed frames
        prepare_frames(compressed_dir, compressed_frames_output_dir)
        


    elif(Flags.process == 8):
        prepare_frames(hr_input_dir=Flags.disk_resize_path)

