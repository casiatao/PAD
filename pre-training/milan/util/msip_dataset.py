import torch
from torch import Tensor
from torchvision.transforms import RandomResizedCrop, RandomHorizontalFlip, ToTensor, Normalize, Resize, Compose
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import os
import glob
from PIL import Image
import math

    
class RandomRoiCrop(RandomResizedCrop):
    
    def __init__(self, size, random=True, use_roi=True, p=0.85, scale=(0.2, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR, min_size=60):
        super().__init__(size, scale, ratio, interpolation)
        self.random = random
        self.use_roi = use_roi
        self.min_size = min_size
        self.p = p
        
    def get_roi(self, img, rois):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            rois (list): list of roi, roi is a list of [x, y, w, h]

        Returns:
            tuple: params (y, x, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        
        width, height = F.get_image_size(img)
        img_area = width * height
        larger_ratio = torch.rand(1) > self.p
        num_roi = len(rois)
        for _ in range(20):
            roi_index = torch.randint(0, num_roi, (1,)).item()
            # roi_index = torch.empty(1).uniform_(0, num_roi-1).item()
            roi = rois[roi_index]
            x, y, w, h = roi
            if x < 0 or y < 0 or x + w > width or y + h > height:
                continue
            if w < 40 or h < 40:
                continue
            if not self.random:
                return y, x, h, w
            
            # random adjust the center of roi
            x_c = int(round(x + w / 2.0))
            y_c = int(round(y + h / 2.0))
            x_c += int(round(w * torch.empty(1).uniform_(-0.4, 0.4).item()))
            y_c += int(round(h * torch.empty(1).uniform_(-0.4, 0.4).item()))
            # random adjust the width and height of the roi
            w_adjust = max(w * torch.empty(1).uniform_(0.7, 1.5).item(), self.min_size)
            h_adjust = max(h * torch.empty(1).uniform_(0.7, 1.5).item(), self.min_size)
            
            x1 = int(round(x_c - w_adjust / 2.0))
            x2 = int(round(x_c + w_adjust / 2.0))
            y1 = int(round(y_c - h_adjust / 2.0))
            y2 = int(round(y_c + h_adjust / 2.0))
            
            x1 = max(0, x1)
            x2 = min(x2, width -1)
            y1 = max(0, y1)
            y2 = min(y2, height -1)
            
            w = x2 - x1
            h = y2 - y1
            
            if y2 <= y1 or x2 <= x1 or w < self.min_size or h < self.min_size:
                # print(f"x1:{x1}, x2:{x2}, y1:{y1}, y2:{y2}, x_c:{x_c}, y_c:{y_c}, x:{x}, y:{y}, w:{w}, h:{h}, w_adjust:{w_adjust}, h_adjust:{h_adjust}")
                continue
            
            area_ratio = w * h / img_area
            if larger_ratio:
                if area_ratio < 0.15:
                    continue

            return y1, x1, h, w
        
        # random resized crop
        return self.get_params(img, self.scale, self.ratio)

        
    def forward(self, thermal_img, rois=None):
        if self.use_roi and rois is not None and len(rois) > 0:
            i, j, h, w = self.get_roi(thermal_img, rois)
        else:
            i, j, h, w = self.get_params(thermal_img, self.scale, self.ratio)
        thermal_img = F.resized_crop(thermal_img, i, j, h, w, self.size, self.interpolation)
        return thermal_img
    
    
class RoICompose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, thermal_img, rois=None):
        for t in self.transforms:
            if isinstance(t, RandomRoiCrop) and rois is not None:
                thermal_img= t(thermal_img, rois)
            else:
                thermal_img = t(thermal_img)
        return thermal_img
            

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
        


def mysort(dataset_path):
    types = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp']
    dataset_img_path = []
    for type in types:
        dataset_img_path.extend(glob.glob(os.path.join(dataset_path, type)))
    # dataset_img_path = glob.glob(os.path.join(dataset_path, r'*.(jpg|jpeg|png|bmp)'))
    try:
        dataset_img_path = sorted(dataset_img_path, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    except Exception as e:
        print(f"The name of images in {dataset_path} is not numerical!")
    return dataset_img_path


def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



class MSIP(Dataset):

    def __init__(self, root, transforms=None, spec_dataset=None, use_roi=True, data_ratio=1.0):
        """
        Args:
            root (_type_): _description_
            transforms (_type_, optional): _description_. Defaults to None.
            spec_dataset (list, optional): 指定数据集.
            use_roi (bool, optional): 是否使用roi_crop.
            use_rgb (bool, optional): 是否使用可见光图像.
            only_rgbt (bool, optional): 是否仅只用rgbt数据集.
            use_in1k (bool, optional): 是否使用IN1K图像作为配比.
        """
        self.transforms = transforms
        self.use_roi = use_roi
        self.spec_dataset = spec_dataset
        self.thermal_img_list = self._make_dataset(root)
        if data_ratio != 1.0:
            total_data_num = len(self.thermal_img_list)
            data_num = int(total_data_num * data_ratio)
            data_sample_interval = int(1 / data_ratio)
            self.thermal_img_list = self.thermal_img_list[::data_sample_interval]
            
        
    def _make_dataset(self, root):
        """
        root/
            dataset1_name/
                thermal/
                    image1.jpg
                    image2.jpg
                    ...
            dataset2_name/
                thermal/
                    ...
        """
        thermal_img_list = []
        if self.spec_dataset is not None:
            datasets = self.spec_dataset
        else:
            datasets = os.listdir(root)
        
        print(datasets)
        # 只使用红外图像
        for dataset in datasets:
            dataset_path = os.path.join(root, dataset)
            thermal_path = os.path.join(dataset_path, r'thermal')
            dataset_thermal_img_list = mysort(thermal_path)
            thermal_img_list.extend(dataset_thermal_img_list)

        return thermal_img_list
    
    
    def __len__(self):
        return len(self.thermal_img_list)
    
    
    def __getitem__(self, index):
        thermal_img_path = self.thermal_img_list[index]
        thermal_img = pil_loader(thermal_img_path)
        
        if self.transforms is not None:
            rois = None
            if self.use_roi:
                # read roi txt
                thermal_folder = os.path.dirname(thermal_img_path)
                filename = os.path.basename(thermal_img_path).split('.')[0]
                roi_txt_path = os.path.join(os.path.dirname(thermal_folder), 'roi', filename + '.txt')
                if os.path.exists(roi_txt_path):
                    rois = []
                    with open(roi_txt_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            coord = line.strip().split(',')
                            coord = [int(x) for x in coord]
                            rois.append(coord)

            thermal_img = self.transforms(thermal_img, rois)
            
        return thermal_img
        
        

    