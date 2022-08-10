# generate imgs from trained model and evaluate FID score
from tqdm import tqdm
import os
import argparse
import pdb
from os import listdir
import random
import time

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

from utils import fid_score
from models.generator import Generator
from models.generator_new import Generator_new
from models.generator_styleGAN import Generator_stylegan

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_xforms(mode="train", keys=("move","fix")):
        """returns a composed transform for train/val/infer."""
        xforms = [
            LoadImaged(keys, dtype=np.float32)
        ]
        dtype = (np.float32, np.float32)
        xforms.extend([CastToTyped(keys, dtype=dtype), EnsureTyped(keys)])
        return monai.transforms.Compose(xforms)

def generateimgs(args, generator):
    with torch.no_grad():
        cnt = 0

        for _ in tqdm(range(args.val_num_batches)):
            with torch.no_grad():
                noise = torch.randn((args.val_batch_size, 512)).cuda()
                out_sample, _ = generator(noise)
                for j in range(args.val_batch_size):
                    torchvision.utils.save_image(
                        out_sample[j],
                        os.path.join('/dataT1/Free/tzheng/workdata/Styleswin/test/tmp', "eval_" + f"{str(cnt).zfill(6)}.png"),
                        nrow=1,
                        padding=0,
                        normalize=True,
                        # range=(0, 1),
                    )
                    cnt += 1


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default=None, help="Path of training data")
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default='/dataT1/Free/tzheng/workdata/Styleswin/Stylegan/checkpoints/455000.pt')
    parser.add_argument("--G_lr", type=float, default=0.001)
    parser.add_argument("--D_lr", type=float, default=0.001)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--start_dim", type=int, default=512, help="Start dim of generator input dim")
    parser.add_argument("--D_channel_multiplier", type=int, default=2)
    parser.add_argument("--G_channel_multiplier", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=20000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument('--workers', default=8, type=int, help='Number of workers')

    parser.add_argument('--checkpoint_path', default='/tmp', type=str, help='Save checkpoints')
    parser.add_argument('--sample_path', default='/tmp', type=str, help='Save sample')
    parser.add_argument('--start_iter', default=0, type=int, help='Start iter number')
    parser.add_argument('--tf_log', action="store_true", help='If we use tensorboard file')
    parser.add_argument('--gan_weight', default=1, type=float, help='Gan loss weight')
    parser.add_argument('--val_num_batches', default=2500, type=int, help='Num of batches will be generated during evalution')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Batch size during evalution')
    parser.add_argument('--D_sn', action="store_true", help='If we use spectral norm in D')
    parser.add_argument('--ttur', action="store_true", help='If we use TTUR during training')
    parser.add_argument('--eval', action="store_true", help='Only do evaluation')
    parser.add_argument("--eval_iters", type=int, default=0, help="Iters of evaluation ckpt")
    parser.add_argument('--eval_gt_path', default='/tmp', type=str, help='Path to ground truth images to evaluate FID score')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='MLP ratio in swin')
    parser.add_argument("--lr_mlp", default=0.01, type=float, help='Lr mul for 8 * fc')
    parser.add_argument("--bcr", action="store_true", help='If we add bcr during training')
    parser.add_argument("--bcr_fake_lambda", default=10, type=float, help='Bcr weight for fake data')
    parser.add_argument("--bcr_real_lambda", default=10, type=float, help='Bcr weight for real data')
    parser.add_argument("--enable_full_resolution", default=8, type=int, help='Enable full resolution attention index')
    parser.add_argument("--auto_resume", action="store_true", help="Auto resume from checkpoint")
    parser.add_argument("--lmdb", action="store_true", help='Whether to use lmdb datasets')
    parser.add_argument("--use_checkpoint", action="store_true", help='Whether to use checkpoint')
    parser.add_argument("--use_flip", action="store_true", help='Whether to use random flip in training')
    parser.add_argument("--wandb", action="store_true", help='Whether to use wandb record training')
    parser.add_argument("--project_name", type=str, default='StyleSwin', help='Project name')
    parser.add_argument("--lr_decay", action="store_true", help='Whether to use lr decay')
    parser.add_argument("--lr_decay_start_steps", default=800000, type=int, help='Steps to start lr decay')

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    args.latent = 4096
    args.n_mlp = 8 
    args.g_reg_every = 10000000    # We do not apply regularization on G

    generator = Generator_new(
        args.size, args.style_dim, args.n_mlp, channel_multiplier=args.G_channel_multiplier, lr_mlp=args.lr_mlp,
        enable_full_resolution=args.enable_full_resolution, use_checkpoint=args.use_checkpoint
    ).to(device)
    generator.eval()

    paramnum = get_n_params(generator)
    
    # Load model checkpoint.
    if args.ckpt is not None:
        print("load model: ", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_name = os.path.basename(args.ckpt)
        try:
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except:
            pass
        
        generator.load_state_dict(ckpt["g"])
    
    if args.distributed:
        generator = torch.nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    
    generateimgs(args, generator)