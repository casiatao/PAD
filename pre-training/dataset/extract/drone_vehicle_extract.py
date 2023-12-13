import cv2
import os
import shutil


infrared_src_base_path = r'/DroneVehicle/train/trainimgr'
visible_src_base_path = r'/DroneVehicle/train/trainimg'
infrared_dst_path = r'../MSIP/thermal'
visible_dst_path = r'../MSIP/rgb'

os.makedirs(infrared_dst_path, exist_ok=True)
os.makedirs(visible_dst_path, exist_ok=True)

# infrared
img_list = os.listdir(infrared_src_base_path)
for img_path in img_list:
    src_img_path = os.path.join(infrared_src_base_path, img_path)
    dst_img_path = os.path.join(infrared_dst_path, img_path)
    img = cv2.imread(src_img_path, 0)
    cv2.imwrite(dst_img_path, img[100:-100, 100:-100])

# visible
img_list = os.listdir(visible_src_base_path)
for img_path in img_list:
    src_img_path = os.path.join(visible_src_base_path, img_path)
    dst_img_path = os.path.join(visible_dst_path, img_path)
    img = cv2.imread(src_img_path, 1)
    cv2.imwrite(dst_img_path, img[100:-100, 100:-100, :])