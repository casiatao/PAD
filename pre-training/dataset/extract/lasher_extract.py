import shutil
import os
import cv2
import numpy as np
import glob


src_base_path = r'/LasHeR/TrainingSet/trainingset'
dst_path = r'./MSIP/lasher'
thermal_img_id = 0
rgb_img_id = 0
frame_interval = 10 
video_num = 0

os.makedirs(os.path.join(dst_path, r'thermal'), exist_ok=True)
os.makedirs(os.path.join(dst_path, r'rgb'), exist_ok=True)


for video in os.listdir(src_base_path):
    src_video_path = os.path.join(src_base_path, video, r"infrared")
    img_list = os.listdir(src_video_path)
    for i in range(0, len(img_list), frame_interval):
        src_img_path = os.path.join(src_video_path, img_list[i])
        dst_img_path = os.path.join(dst_path, r'thermal', f"{thermal_img_id:07}.jpg")
        thermal_img_id += 1
        shutil.copyfile(src_img_path, dst_img_path)
    src_video_path = os.path.join(src_base_path, video, r"visible")
    img_list = os.listdir(src_video_path)
    for i in range(0, len(img_list), frame_interval):
        src_img_path = os.path.join(src_video_path, img_list[i])
        dst_img_path = os.path.join(dst_path, r'rgb', f"{rgb_img_id:07}.jpg")
        rgb_img_id += 1
        shutil.copyfile(src_img_path, dst_img_path)
    video_num += 1
    if video_num % 100 == 0:
        print(f"Parse {video_num} videos, extract {thermal_img_id} rgb frames and {rgb_img_id} rgb frames.")
        
print(f"Parse {video_num} videos, extract {thermal_img_id} rgb frames and {rgb_img_id} rgb frames totally.")

