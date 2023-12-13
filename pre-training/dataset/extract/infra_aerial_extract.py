import shutil
import os
import glob
import cv2
import numpy as np

src_base_path = r"/infrared_aerial/"
dst_path = r'./MSIP/aerial'
img_id = 0
img_interval = 2

img_list = glob.glob(src_base_path + '\*.jpg')

for i in range(0, len(img_list), img_interval):
    src_img_path = img_list[i]
    dst_img_path = os.path.join(dst_path, f"{img_id:07}.jpg")
    img_id += 1
    shutil.copyfile(src_img_path, dst_img_path)
    if img_id % 500 == 0:
        print(f"extract {img_id} frames")

print(f"extract {img_id} frames totally")