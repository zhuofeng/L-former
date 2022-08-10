import torch
import numpy as np

from utils import fid_score

if __name__ == "__main__":
    gt_path = '/dataT1/Free/tzheng/workdata/Styleswin/test/GT_micro'
    sample_path = '/dataT1/Free/tzheng/workdata/Styleswin/test/Styleswin_micro'
    device = torch.device('cuda:3')
    fid = fid_score.calculate_fid_given_paths([sample_path, gt_path], batch_size=16, device=device, dims=2048)
    
    print("Fid Score : {:.2f}".format(fid))
