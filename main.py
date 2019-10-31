import numpy as np
import os, math, time, collections, numpy as np
import datetime
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
Disable Logs for now '''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import random as rn

# fix all randomness, except for multi-treading or GPU process
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)

# import tensorflow.contrib.slim as slim
import sys, shutil, subprocess

# from lib.ops import *
# from lib.dataloader import inference_data_loader, frvsr_gpu_data_loader
from lib.dataloader import video_frames_generator, loadLRHR, frvsr_gpu_data_loader
# from lib.frvsr import generator_F, fnet
# from lib.Teco import FRVSR, TecoGAN

import matplotlib.pyplot as plt
# import itertools


def folder_check(path):
    try_num = 1
    oripath = path[:-1] if path.endswith('/') else path
    while os.path.exists(path):
        print("Delete existing folder " + path + "?(Y/N)")
        decision = input()
        if decision == "Y":
            shutil.rmtree(path, ignore_errors=True)
            break
        else:
            path = oripath + "_%d/"%try_num
            try_num += 1
            print(path)
    
    return path

now_str = datetime.datetime.now().strftime("%m-%d-%H")
train_dir = folder_check("ex_TecoGAN%s/"%now_str)

Flags = tf.app.flags

Flags.DEFINE_integer('rand_seed', 1 , 'random seed' )

# Directories
Flags.DEFINE_string('input_dir_LR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_integer('input_dir_len', -1, 'length of the input for inference mode, -1 means all')
Flags.DEFINE_string('input_dir_HR', None, 'The directory of the input resolution input data, for inference mode')
Flags.DEFINE_string('mode', 'inference', 'train, or inference')
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('output_pre', '', 'The name of the subfolder for the images')
Flags.DEFINE_string('output_name', 'output', 'The pre name of the outputs')
Flags.DEFINE_string('output_ext', 'jpg', 'The format of the output when evaluating')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')

# Models
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
# Models for training
Flags.DEFINE_boolean('pre_trained_model', False, 'If True, the weight of generator will be loaded as an initial point'
                                                     'If False, continue the training')
Flags.DEFINE_string('vgg_ckpt', None, 'path to checkpoint file for the vgg19')

# Machine resources
Flags.DEFINE_string('cudaID', '0', 'CUDA devices')
Flags.DEFINE_integer('queue_thread', 6, 'The threads of the queue (More threads can speedup the training process.')
Flags.DEFINE_integer('name_video_queue_capacity', 512, 'The capacity of the filename queue (suggest large to ensure'
                                                  'enough random shuffle.')
Flags.DEFINE_integer('video_queue_capacity', 256, 'The capacity of the video queue (suggest large to ensure'
                                                   'enough random shuffle')
Flags.DEFINE_integer('video_queue_batch', 2, 'shuffle_batch queue capacity')
                                                   
# Training details
# The data preparing operation
Flags.DEFINE_integer('RNN_N', 10, 'The number of the rnn recurrent length')
Flags.DEFINE_integer('batch_size', 4, 'Batch size of the input batch')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
Flags.DEFINE_boolean('movingFirstFrame', False, 'Whether use constant moving first frame randomly.')
Flags.DEFINE_integer('crop_size', 32, 'The crop size of the training image')
# Training data settings
Flags.DEFINE_string('input_video_dir', '', 'The directory of the video input data, for training')
Flags.DEFINE_string('input_video_pre', 'scene', 'The pre of the directory of the video input data')
Flags.DEFINE_integer('str_dir', 2000, 'The starting index of the video directory')
Flags.DEFINE_integer('end_dir', 2001, 'The ending index of the video directory')
Flags.DEFINE_integer('end_dir_val', 2002, 'The ending index for validation of the video directory')
Flags.DEFINE_integer('max_frm', 119, 'The ending index of the video directory')
# The loss parameters
Flags.DEFINE_float('vgg_scaling', -0.002, 'The scaling factor for the VGG perceptual loss, disable with negative value')
Flags.DEFINE_float('warp_scaling', 1.0, 'The scaling factor for the warp')
Flags.DEFINE_boolean('pingpang', False, 'use bi-directional recurrent or not')
Flags.DEFINE_float('pp_scaling', 1.0, 'factor of pingpang term, only works when pingpang is True')
# Training parameters
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.5, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_float('adameps', 1e-8, 'The eps parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')
# Dst parameters
Flags.DEFINE_float('ratio', 0.01, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_boolean('Dt_mergeDs', True, 'Whether only use a merged Discriminator.')
Flags.DEFINE_float('Dt_ratio_0', 1.0, 'The starting ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dt_ratio_add', 0.0, 'The increasing ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dt_ratio_max', 1.0, 'The max ratio for the temporal adversarial loss')
Flags.DEFINE_float('Dbalance', 0.4, 'An adaptive balancing for Discriminators')
Flags.DEFINE_float('crop_dt', 0.75, 'factor of dt crop') # dt input size = crop_size*crop_dt
Flags.DEFINE_boolean('D_LAYERLOSS', True, 'Whether use layer loss from D')

FLAGS = Flags.FLAGS

# Set CUDA devices correctly if you use multiple gpu system
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.cudaID 
# Fix randomness
my_seed = FLAGS.rand_seed
rn.seed(my_seed)
np.random.seed(my_seed)
# tf.set_random_seed(my_seed)
tf.compat.v1.set_random_seed(my_seed)

FLAGS.output_dir = train_dir
FLAGS.input_video_dir = "D:\\Github\\TecoGAN\\tecogan_video_data\\train_frames"
FLAGS.summary_dir = os.path.join(train_dir,"log/")

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')
# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)
# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)





if __name__ == '__main__':
    # import itertools

    # dataset = loadLRHR(FLAGS)
    frvsr_gpu_data_loader(FLAGS, 1)
    # iterator = dataset.make_initializable_iterator()
    # s_inputs, s_targets = iterator.get_next()

    # for lr_images, hr_images in dataset.take(-1):
    
    #     print(lr_images.get_shape().as_list())




    
    # plt.figure(figsize=(12,12)) 
    # tf.InteractiveSession() 
    # #     lr_data, hr_data = sess.run([x,y])
    # #     print(1, lr_data.shape, hr_data.shape)
    # iterator = dataset.make_one_shot_iterator()
    # # for lr_images_list, hr_images_list in iterator:
    # lr_images_list, hr_images_list = iterator.get_next()
    # for i in range(FLAGS.RNN_N):
    #     image= lr_images_list[i]
    #     plt.subplot(FLAGS.RNN_N//2, 2, i+1)
    #     plt.imshow(image.eval())
    #     plt.title("lr_images_list")
    # plt.grid(False)                 

    # for i in range(FLAGS.RNN_N):
    #     image = hr_images_list[i]
    #     plt.subplot(FLAGS.RNN_N//2, 2, i+1)
    #     plt.imshow(image.eval())
    #     plt.title("hr_images_list")
    #     plt.grid(False)

    # batch_gen = video_frames_generator(FLAGS)
    # i = 0
    # for b in batch_gen:
        # if i > 5:
            # break
        # i += 1
        # print(b)


