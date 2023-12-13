import shutil
import os
import cv2
import numpy as np
import glob

src_base_path = r'/SCUT FIR Pedestrian Dataset/videos'
dst_path = r'./MSIP/scut_fir_pedestrian'
img_id = 0
frame_interval = 10  

for set in os.listdir(src_base_path):
    src_set_path = os.path.join(src_base_path, set)
    for video in os.listdir(src_set_path):
        src_video_path = os.path.join(src_set_path, video)
        if os.path.isdir(src_video_path):
            img_list = os.listdir(src_video_path)
            sorted_img_list = sorted(img_list, key=lambda name: int(name.split('.')[0]))
            for i in range(0, len(sorted_img_list), frame_interval):
                src_img_path = os.path.join(src_video_path, sorted_img_list[i])
                img = cv2.imread(src_img_path, 0)
                if np.max(img) > 20: 
                    dst_img_path = os.path.join(dst_path, f"{img_id:07}.jpg")
                    img_id += 1
                    shutil.copyfile(src_img_path, dst_img_path)
                    if img_id % 500 == 0:
                        print(f"extract {img_id} frames")
print(f"extract {img_id} frames totally")
