# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from io import BytesIO
import pdb
import os
import numpy as np

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.resolution = resolution
        self.transform = transform
        
        ## my own code
        self._path = path
        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}

        Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in Image.EXTENSION)
        self.length = len(self._image_fnames)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self._xflip = np.zeros(self.length, dtype=np.uint8)
        self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self._load_raw_image(idx)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        image = image / 255
        return image.copy()

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = np.array(Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1)# HWC => CHW
        return image

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()