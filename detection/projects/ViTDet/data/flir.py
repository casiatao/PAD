import os
import numpy as np
import json
import cv2
from detectron2.structures import BoxMode

cls_list = ['person', 'bike', 'car', 'motor', 'bus', 
            'train', 'truck', 'light', 'hydrant', 'sign', 
            'dog', 'deer', 'skateboard', 'stroller', 'scooter', 'other vehicle']

cls_dict = {k: i for i, k in enumerate(cls_list)}


def get_flir_dicts(img_dir, json_file):

    with open (json_file) as f:
        imgs_anns = json.load(f)
        
    dataset_dicts = []
    images = imgs_anns['images']
    anns = imgs_anns['annotations']
    for image in images:
        record = {}
        
        filename = os.path.join(img_dir, image["file_name"])
        record['file_name'] = filename
        record['image_id'] = image["id"]
        record['height'] = image['height']
        record['width'] = image['width']
        record['annotations'] = []
        dataset_dicts.append(record)

    for ann in anns:
        if int(ann['category_id']) in [1, 2, 3]:
            # cls_num[str(ann['category_id'])] += 1
            category_id = ann['category_id'] - 1
            obj = {
                'bbox': ann['bbox'],
                'bbox_mode': BoxMode.XYWH_ABS,
                # 'segmentation': ann['segmentation'],
                'category_id': category_id,
            }
            dataset_dicts[ann['image_id']]['annotations'].append(obj)
    
    return dataset_dicts



