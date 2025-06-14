
# import cv2
# import torch
# from torchvision import datasets
# import numpy as np


# def opencv_loader(path):
#     img = cv2.imread(path)
#     return img


# class DatasetFolderFT(datasets.ImageFolder):
#     def __init__(self, root, transform=None, target_transform=None,
#                  ft_width=10, ft_height=10, loader=opencv_loader):
#         super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
#         self.root = root
#         self.ft_width = ft_width
#         self.ft_height = ft_height

#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         # generate the FT picture of the sample
#         ft_sample = generate_FT(sample)
#         if sample is None:
#             print('image is None --> ', path)
#         if ft_sample is None:
#             print('FT image is None -->', path)
#         assert sample is not None

#         ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
#         ft_sample = torch.from_numpy(ft_sample).float()
#         ft_sample = torch.unsqueeze(ft_sample, 0)

#         if self.transform is not None:
#             try:
#                 sample = self.transform(sample)
#             except Exception as err:
#                 print('Error Occured: %s' % err, path)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         return sample, ft_sample, target


# def generate_FT(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#     fimg = np.log(np.abs(fshift)+1)
#     maxx = -1
#     minn = 100000
#     for i in range(len(fimg)):
#         if maxx < max(fimg[i]):
#             maxx = max(fimg[i])
#         if minn > min(fimg[i]):
#             minn = min(fimg[i])
#     fimg = (fimg - minn+1) / (maxx - minn+1)
#     return fimg

# -*- coding: utf-8 -*-
# @Time : 20-6-4 下午4:04
# @Author : zhuying
# @Company : Minivision
# @File : dataset_folder.py
# @Software : PyCharm

import cv2
import torch
from torchvision import datasets
import numpy as np


def opencv_loader(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image at path: {path}")
    return img


class DatasetFolderFT(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ft_width=10, ft_height=10, loader=opencv_loader):
        super(DatasetFolderFT, self).__init__(root, transform, target_transform, loader)
        self.root = root
        self.ft_width = ft_width
        self.ft_height = ft_height

    def __getitem__(self, index):
        try:
            path, target = self.samples[index]
            sample = self.loader(path)
            if sample is None:
                raise ValueError(f"Failed to load image: {path}")
            
            # Kiểm tra kích thước ảnh trước khi xử lý
            if sample.shape[0] * sample.shape[1] > 4096 * 4096:  # Giới hạn kích thước
                sample = cv2.resize(sample, (4096, 4096))
            
            ft_sample = generate_FT(sample)
            if ft_sample is None:
                print('FT image is None -->', path)
            assert sample is not None

            ft_sample = cv2.resize(ft_sample, (self.ft_width, self.ft_height))
            ft_sample = torch.from_numpy(ft_sample).float()
            ft_sample = torch.unsqueeze(ft_sample, 0)

            if self.transform is not None:
                try:
                    sample = self.transform(sample)
                except Exception as err:
                    print('Error Occured: %s' % err, path)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, ft_sample, target
        except Exception as e:
            print(f"Error processing image {path}: {str(e)}")
            return None


def generate_FT(image):
    try:
        # Giảm kích thước ảnh nếu quá lớn
        max_dim = 1024
        h, w = image.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        fimg = np.log(np.abs(fshift)+1)
        
        # Giải phóng bộ nhớ
        del f, fshift
        
        return fimg
    except Exception as e:
        print(f"Error in generate_FT: {str(e)}")
        return None


def monitor_cpu_memory():
    import psutil
    import os
    
    # Monitor CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # print(f"CPU Usage: {cpu_percent}%")
    # print(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    # print(f"Virtual Memory: {memory_info.vms / 1024 / 1024:.2f} MB")
    
    # Monitor swap usage
    swap = psutil.swap_memory()
    # print(f"Swap Usage: {swap.percent}%")