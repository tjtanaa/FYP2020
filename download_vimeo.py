import os, sys, datetime
import cv2 as cv
import argparse
import youtube_dl

from lib.data import video
import subprocess

"python download_vimeo.py --start_id 2000 --duration 120 --process 1"
# D:\\Github\\tecogan_video_data\\train_video

# ------------------------------------parameters------------------------------#
parser = argparse.ArgumentParser(description='Process parameters.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start_id', default=2000, type=int, help='starting scene index')
parser.add_argument('--duration', default=120, type=int, help='scene duration')
parser.add_argument('--disk_path', default="./Vimeo90K", help='the path to save the dataset')
parser.add_argument('--summary_dir', default="", help='the path to save the log')
parser.add_argument('--REMOVE', action='store_true', help='whether to remove the original video file after data preparation')
parser.add_argument('--TEST', action='store_true', help='verify video links, save information in log, no real video downloading!')
parser.add_argument('--disk_compressed_path', default="D:\\Github\\tecogan_video_data\\train_compressed_video", help='the path to save the dataset')
parser.add_argument('--disk_frames_path', default="D:\\Github\\tecogan_video_data\\train_frames", help='the path to save the dataset')
parser.add_argument('--disk_resize_path', default="D:\\Github\\tecogan_video_data\\train_resized_video", help='the path to save the dataset')
parser.add_argument('--disk_resize_by_4_path', default="D:\\Github\\tecogan_video_data\\train_resized_video_by_4", help='the path to save the dataset')
parser.add_argument('--video_bitrate', default="40k", help='video_bitrate')
parser.add_argument('--process', default=0, type=int, help='run process 0: download video 1: compress video 2: generate frames')
Flags = parser.parse_args()

if Flags.summary_dir == "":
    Flags.summary_dir = os.path.join(Flags.disk_path, "log/")
os.path.isdir(Flags.disk_path) or os.makedirs(Flags.disk_path)
os.path.isdir(Flags.summary_dir) or os.makedirs(Flags.summary_dir)

link_path = "https://vimeo.com/"


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

# ------------------------------------tool------------------------------#
def gen_frames(infile, outdir, width, height, start, duration, savePNG=True):
    print("folder %s: %dx[%d,%d]//2 at frame %d of %s"
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
        count = 0
        while success:
            filename = (outdir+'col_high'+("_%04d.png"%(count))) if index == -1 else (outdir+'col_high'+("_compressed_%04d.png"%(count)))
            cv.imwrite(filename, image)     # save frame as JPEG file      
            success,image = vcap.read()
            # print('Read a new frame: ', success)
            count += 1
            if count >=duration:
                break
        
def download_videos():
    """
        this function download the videos to the disk_path
    """
    import pprint
    cur_id, valid_video, try_num = Flags.start_id, 0, 0

    # create the youtube downloader
    ydl = youtube_dl.YoutubeDL( 
        {'format': 'bestvideo/best',
        'outtmpl': os.path.join(Flags.disk_path, '%(id)s.%(ext)s'),})
        
    saveframes = not Flags.TEST
    FROM = "00:00:00.00"
    TO = "00:00:15.00"
    # read from txt file
    original_video_list_txt = os.path.join(Flags.disk_path, 'original_video_links.txt')
    downloaded_video_list = os.path.join(Flags.disk_path, 'downloaded_video_links.txt')
    downloaded_keys = []

    with open(downloaded_video_list, 'r') as fr:
        print("downloaded_video_links open")
        for link in fr:
            downloaded_keys.append(link)


    with open(downloaded_video_list, 'a') as fw:
        with open(original_video_list_txt, 'r') as f:
            for tar_vid_input in f:
                print("tar_vid_input: ", tar_vid_input)
                keys = tar_vid_input.replace(link_path, '').replace('\n','')
                if tar_vid_input in downloaded_keys:
                    print("downloaded")
                    continue
                tar_vid_input = tar_vid_input.replace('\n','')

                print('keys: ', keys)
                info_dict = {"width":-1, "height": -1, "ext": "xxx", }
                # exit()
                # download video from vimeo
                try:
                    info_dict = ydl.extract_info(tar_vid_input, download=False)
                    info_dict_summary = pprint.pformat(info_dict)
                    # print("info dict: ",info_dict_summary)
                    if info_dict["width"] < 400 or info_dict["height"] < 400:
                        print("Skipped videos of small size %dx%d"%(info_dict["width"] , info_dict["height"] ))
                        continue
                    if info_dict["width"] > 1980 or info_dict["height"] > 2160:
                        print("Skipped videos of large size %dx%d"%(info_dict["width"] , info_dict["height"] ))
                        continue
                    info_dict = ydl.extract_info(tar_vid_input, download=saveframes)
                    # tar_vid_output = os.path.join(Flags.disk_path, keys+'.'+info_dict["ext"])

                    # video = info_dict['entries'][0] if 'entries' in info_dict else info_dict
                    # # print(info_dict.keys())
                    # # print(video.keys())
                    # url = video['url']
                    # print('url: ', url)
                    # print("tar_vid_output: ", tar_vid_output)
                    # subprocess.call('ffmpeg -i "%s" -ss %s -t %s -c copy %s' % (url , FROM, TO, tar_vid_output))

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
                if info_dict["width"] > 1980 or info_dict["height"] > 2160:
                    print("Skipped videos of small size %dx%d"%(info_dict["width"] , info_dict["height"] ))
                    print("remove ", tar_vid_output)
                    os.remove(tar_vid_output)
                    continue
                valid_video = valid_video + 1
                fw.write(tar_vid_input + '\n')
                print("Downloaded valid video %d"%(valid_video))
            print("Done Downloading Video")


if __name__ == '__main__':
    if(Flags.process == 0):
        download_videos()

