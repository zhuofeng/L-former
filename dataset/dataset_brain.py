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
import monai
import albumentations as albu
from scipy.ndimage.interpolation import zoom
from monai.transforms import (
    CastToTyped,
    LoadImaged,
    EnsureTyped,
    HistogramNormalized,
)

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        super().__init__()
        name = 'Oasisnew' # perhaps should reshape to size 256
        self.path = '/dataF0/Free/tzheng/OASIS/MRI_MASKED'
        self.batchs_percase = 100 # 从一个batch里面取的case数量
        self.patchsize = 256
        self.channel = 1
        
        self.MRIfilenames = [os.path.join(self.path, f) for f in listdir(self.path)]
        self.MRIfilenames = self.MRIfilenames[:300]
        
        self.keys = ("move", "fix") # 不能单独一个move或者fix
        train_files = [{self.keys[0]:move, self.keys[1]: fix} for move, fix in zip(self.MRIfilenames, self.MRIfilenames)]
        train_transforms = self.get_xforms("train", self.keys)
        
        self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, num_workers=8) # maybe this is not proper in this dataset
        # (-1, 1) noprmalize
        
        for i in range(len(self.train_ds)):
            self.train_ds[i][self.keys[0]] = self.train_ds[i][self.keys[0]] - self.train_ds[i][self.keys[0]].min()
            self.train_ds[i][self.keys[0]] = self.train_ds[i][self.keys[0]] / (self.train_ds[i][self.keys[0]].max() / 2)  - 1
        
        print ('loaded dataset')

    def __getitem__(self, index):
        casenum = index // self.batchs_percase
        # img = self.random_patches_singleslice(self.train_ds[casenum][self.keys[0]], patchsize=self.patchsize)
        img = self.random_slice(self.train_ds[casenum][self.keys[0]])
        # if fine-tune the pretrained model, 3-channels?
        img = np.repeat(img, 3, axis=0)
        return img.astype(np.float32).copy()

    def __len__(self):
        return int(len(self.MRIfilenames) * self.batchs_percase)
    
    def random_slice(self, array1):
        array1 = np.squeeze(array1)
        slicenum = random.randint(int(array1.shape[2]*0.1), int(array1.shape[2]*0.9))
        slice1 = array1[..., slicenum].cpu().detach().numpy()     
        #scaling and crop (to get a 256 * 256 patch)
        slice1 = np.pad(slice1,[(40,40),(24,24)], 'edge')
        slice1 = zoom(slice1, 1.3)
        slice1 = slice1[int((slice1.shape[0] - 256) / 2) : int((slice1.shape[0] - 256) / 2) + 256, int((slice1.shape[1] - 256) / 2) : int((slice1.shape[1] - 256) / 2) + 256]
        slice1 = slice1[np.newaxis,:,:]
        
        return slice1

    @staticmethod
    def get_xforms(self, mode="train", keys=("move","fix")):
        """returns a composed transform for train/val/infer."""
        xforms = [
            LoadImaged(keys, dtype=np.float32),
            # HistogramNormalized(keys, min=-1, max=1),
            # ResizeWithPadOrCropd(keys, spatial_size=(208, 208, 176)) # this is very slow
        ]
        dtype = (np.float32, np.float32)
        xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
        return monai.transforms.Compose(xforms)