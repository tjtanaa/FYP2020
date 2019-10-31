'''
    This script is used to run single evaluation cycle to
    calculate the metrics, and save the numbers in csv.

'''
import os, subprocess, sys, datetime, signal, shutil

hdd_dir = "../../../mnt/external2/tunjian"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# runcase = int(sys.argv[1])
# print ("Testing test case %d" % runcase)

def preexec(): # Don't forward signals.
    os.setpgrp()
    
def mycall(cmd, block=False):
    if not block:
        return subprocess.Popen(cmd)
    else:
        return subprocess.Popen(cmd, preexec_fn = preexec)
    
# def folder_check(path):
#     try_num = 1
#     oripath = path[:-1] if path.endswith('/') else path
#     while os.path.exists(path):
#         print("Delete existing folder " + path + "?(Y/N)")
#         decision = input()
#         if decision == "Y":
#             shutil.rmtree(path, ignore_errors=True)
#             break
#         else:
#             path = oripath + "_%d/"%try_num
#             try_num += 1
#             print(path)
    
#     return path

if __name__ == "__main__":
    # testpre = ["calendar"] # just put more scenes to evaluate all of them
    # dirstr = "/home/tunjian/oriTecoGAN/results/"  # the outputs
    # tarstr = hdd_dir+"/HR/"       # the GT

    tarstr = 'D:\\Github\\SuperResolutionOutputFromServer\\EDVR\\Vid4\\edvr150k'
    dirstr = 'D:\\Github\\SuperResolutionOutputFromServer\\EDVR\\Vid4\\edvr50k'

    # tar_list = [(tarstr+_) for _ in testpre]
    # out_list = [(dirstr+_) for _ in testpre]
    tar_list = tarstr
    out_list = dirstr
    cmd1 = ["python", "metrics.py",
        "--output", dirstr+"metric_log/",
        "--results", out_list,
        "--targets", tar_list,
    ]
    # cmd1 = ["python", "metrics.py",
    #     "--output", dirstr+"metric_log/",
    #     "--results", ",".join(out_list),
    #     "--targets", ",".join(tar_list),
    # ]
    mycall(cmd1).communicate()
 