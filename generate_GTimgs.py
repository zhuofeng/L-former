# generate imgs from trained model and evaluate FID score
from tqdm import tqdm
import os
import argparse
import pdb
from os import listdir
import random
import itertools

from PIL import Image
import torch
import torchvision
import monai
import numpy as np
from monai.transforms import (
    CastToTyped,
    LoadImaged,
    EnsureTyped
)
from scipy.ndimage.interpolation import zoom

def generateGTOASISimgs():
        
        def getitem(train_ds):
            index =  random.randint(0, len(train_ds))
            casenum = index // batchs_percase
            img = random_slice(train_ds[casenum][keys[0]])
            img = np.repeat(img, 3, axis=0)
            return img.astype(np.float32).copy()

        def random_slice(array1):
            array1 = np.squeeze(array1)
            slicenum = random.randint(int(array1.shape[2]*0.1), int(array1.shape[2]*0.9))
            slice1 = array1[..., slicenum].cpu().detach().numpy()     
            slice1 = np.pad(slice1,[(40,40),(24,24)], 'edge')
            slice1 = zoom(slice1, 1.3)
            slice1 = slice1[int((slice1.shape[0] - 256) / 2) : int((slice1.shape[0] - 256) / 2) + 256, int((slice1.shape[1] - 256) / 2) : int((slice1.shape[1] - 256) / 2) + 256]
            slice1 = slice1[np.newaxis,:,:]            
            return slice1

        path = '/dataF0/Free/tzheng/OASIS/MRI_MASKED'
        batchs_percase = 100 # 从一个batch里面取的case数量
        
        MRIfilenames = [os.path.join(path, f) for f in listdir(path)]
        MRIfilenames = MRIfilenames[300:350]
        
        keys = ("move", "fix") # 不能单独一个move或者fix
        train_files = [{keys[0]:move, keys[1]: fix} for move, fix in zip(MRIfilenames, MRIfilenames)]
        train_transforms = get_xforms("train", keys)
        
        train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, num_workers=8) # maybe this is not proper in this dataset
        
        for i in range(len(train_ds)):
            train_ds[i][keys[0]] = train_ds[i][keys[0]] - train_ds[i][keys[0]].min()
            train_ds[i][keys[0]] = train_ds[i][keys[0]] / (train_ds[i][keys[0]].max() / 2)  - 1
        cnt = 0
        for _ in tqdm(range(10000)):
            img = torch.from_numpy(getitem(train_ds))
            
            torchvision.utils.save_image(
                img,
                os.path.join('/dataT1/Free/tzheng/workdata/Styleswin/test/GT', "eval_" + f"{str(cnt).zfill(6)}.png"),
                nrow=1,
                padding=0,
                normalize=True,
                # range=(0, 1),
            )
            cnt += 1

def get_xforms(mode="train", keys=("move","fix")):
        """returns a composed transform for train/val/infer."""
        xforms = [
            LoadImaged(keys, dtype=np.float32)
        ]
        dtype = (np.float32, np.float32)
        xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
        return monai.transforms.Compose(xforms)

def generateGTmicroimgs():    
    def random_crop(array, size):
        x = random.randint(0, int(array.shape[0] - size - 1))
        y = random.randint(0, int(array.shape[1] - size - 1))
        
        croparray = array[x:x+size, y:y+size]
        return croparray

    def normalize(arr):
        arrmin = arr.min() + 0.1 * (arr.max() - arr.min())
        arrmax = arr.max() - 0.1 * (arr.max() - arr.min())
        arr[arr<arrmin] = arrmin
        arr[arr>arrmax] = arrmax
        arr = arr - arr.min()
        arr = arr / (arr.max() / 2) - 1
        return arr

    def getitem(all_imgnames):
        randinx = random.randint(0, len(all_imgnames)-1)
        img_arr = np.array(Image.open(all_imgnames[randinx]))
        img_arr = normalize(img_arr)
        img = random_crop(img_arr, 256)
        img = img[np.newaxis, :, :]
        img = np.repeat(img, 3, axis=0)
        return img.astype(np.float32).copy()

    microfolders = [
            '/dataT0/Free/tzheng/microCT/NU_017_6mm_Rec_down',
            '/dataT0/Free/tzheng/microCT/Lung036_5_65_20201014_Rec_down',
            # '/dataT0/Free/tzheng/microCT/Lung044_6.66um_60kV_166uA-2nd_Rec_down',
            '/dataT0/Free/tzheng/microCT/lung46_40k_250uA-20210609-2_Rec_down'            
        ]
        
    all_imgnames = []
    for folder in microfolders:
        files = os.listdir(folder)
        for i in range(len(files)):
            files[i] = os.path.join(folder, files[i])
        all_imgnames.append(files.copy())

    all_imgnames = list(itertools.chain.from_iterable(all_imgnames))

    cnt = 0
    for _ in tqdm(range(10000)):
        img = torch.from_numpy(getitem(all_imgnames))
        
        torchvision.utils.save_image(
            img,
            os.path.join('/dataT1/Free/tzheng/workdata/Styleswin/test/GT_micro', "eval_" + f"{str(cnt).zfill(6)}.png"),
            nrow=1,
            padding=0,
            normalize=True,
            # range=(0, 1),
        )
        cnt += 1


if __name__ == "__main__":
    # generateGTOASISimgs()
    generateGTmicroimgs()