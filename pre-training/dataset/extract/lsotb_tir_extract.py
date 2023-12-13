import shutil
import os
import cv2
import numpy as np
import glob

src_base_path = r'/LSOTB-TIR_TrainingData/TrainingData'
dst_path = r'./MSIP/lsotb_tir'
img_id = 0
frame_interval = 10 
video_num = 0

for set in os.listdir(src_base_path):
    src_set_path = os.path.join(src_base_path, set)
    for video in os.listdir(src_set_path):
        src_video_path = os.path.join(src_set_path, video)
        img_list = os.listdir(src_video_path)
        sorted_img_list = sorted(img_list, key=lambda name: int(name.split('.')[0]))
        for i in range(0, len(sorted_img_list), frame_interval):
            src_img_path = os.path.join(src_video_path, sorted_img_list[i])
            dst_img_path = os.path.join(dst_path, f"{img_id:07}.jpg")
            img_id += 1
            shutil.copyfile(src_img_path, dst_img_path)
        video_num += 1
        if video_num % 100 == 0:
            print(f"Parse {video_num} videos, extract {img_id} frames.")

print(f"Parse {video_num} videos, extract {img_id} frames.")

            