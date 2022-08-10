# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# The brain dataset for training StyleGAN

from io import BytesIO
import pdb

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
from os import listdir
import random
import itertools

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        super().__init__()
        # folders for micro CT images (tif images)
        self.microfolders = [
            '/dataT0/Free/tzheng/microCT/NU_017_6mm_Rec_down',
            '/dataT0/Free/tzheng/microCT/Lung036_5_65_20201014_Rec_down',
            # '/dataT0/Free/tzheng/microCT/Lung044_6.66um_60kV_166uA-2nd_Rec_down',
            '/dataT0/Free/tzheng/microCT/lung46_40k_250uA-20210609-2_Rec_down'            
        ]

        # here note the start files
        self.startfiles = [
            'NU_046_20mm_rec00001020.tif', # i misnote 17 as 46
            'Lung036_5_65_20201014_IR_rec00000091.tif',
            # 'Lung044_6.66um_60kV_166uA-2nd_IR_rec00000070.bmp',
            'lung15_40k_250uA-20210609-2_rec00000070.tif'
        ]
        # the startnum of each image folder
        self.startnums = [
            '1020',
            '0091',
            # '0070',
            '0070'
        ]
        
        # self._raw_shape = [int(len(self.movedfilenames) * self.batchs_percase)] + [1,128,128]
        print ('loaded dataset')
        # consider mix some paired downsample micro CT / original data into it
        self.all_imgnames = []
        for folder in self.microfolders:
            files = os.listdir(folder)
            for i in range(len(files)):
                files[i] = os.path.join(folder, files[i])
            self.all_imgnames.append(files.copy())

        self.all_imgnames = list(itertools.chain.from_iterable(self.all_imgnames))
        
        
    def __getitem__(self, index):        
        randinx = random.randint(0, len(self.all_imgnames)-1)
        img_arr = np.array(Image.open(self.all_imgnames[randinx]))
        img_arr = self.normalize(img_arr)
        img = self.random_crop(img_arr, 256)
        img = img[np.newaxis, :, :]
        img = np.repeat(img, 3, axis=0)
        return img.astype(np.float32).copy()

    def random_crop(self, array, size):
        x = random.randint(0, int(array.shape[0] - size - 1))
        y = random.randint(0, int(array.shape[1] - size - 1))
        
        croparray = array[x:x+size, y:y+size]
        return croparray
        
    def normalize(self, arr):
        arrmin = arr.min() + 0.1 * (arr.max() - arr.min())
        arrmax = arr.max() - 0.1 * (arr.max() - arr.min())
        arr[arr<arrmin] = arrmin
        arr[arr>arrmax] = arrmax
        arr = arr - arr.min()
        arr = arr / (arr.max() / 2) - 1
        return arr

    def __len__(self):
        return int(len(self.all_imgnames))