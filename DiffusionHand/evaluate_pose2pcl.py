import numpy as np
import argparse
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dataloader.rhddataloader as rhddataloader
from util.misc import *
from dpc.autoencoderhand2pcl import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from util.general_pytorch import EvalUtil, calc_auc, np2pyt, pyt2np
from dpc.evaluation_metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_fn(worker_id):np.random.seed(worker_id)

colorlist_pred = ['#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
colorlist_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']


# Plots all 5 fingers
def plot_fingers(points, plt_specs, c, ax):
    for i in range(5):
        start, end = i*4+1, (i+1)*4+1
        to_plot = np.concatenate((points[start:end], points[0:1]), axis=0)
        ax.plot(to_plot[:,0], to_plot[:,1], to_plot[:,2], plt_specs, color=c[i])
        
# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
# parser.add_argument('--dataset_path', type=str, default='/media/arc/Nuclear/shapenet.hdf5') # Please set the path in /dataloader/rhddataloader.py
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
# parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)


#Training
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)

args = parser.parse_args()
seed_all(args.seed)


dataset = rhddataloader.RHDDateset3D(mode='evaluation', view_correction=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=32, shuffle=False, worker_init_fn=init_fn)


ckpt = torch.load('./logs_ae/AE_2023_04_13__20_27_24/ckpt_0.182003_233.pt')
model = AutoEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

all_refs = []
all_recons = []   

with torch.no_grad():
    for i, (image, data) in enumerate(tqdm(dataloader, desc='Validate')):   
        image = image.to(device)
        pose_gt = data['pose3d'].to(args.device) #.reshape(-1, 21, 3)
        cloud = data['cloud'].to(device).permute(0, 2, 1)
        scale = data['scale'].to(device)
        
        
        recons = model.decode(pose_gt, cloud.shape[1], flexibility=args.flexibility)
        all_refs.append(cloud * scale )
        all_recons.append(recons * scale)
        
        img_crop_np = pyt2np(image.squeeze(0).permute(1,2,0))
        cloud_vis = pyt2np(cloud.squeeze(0)*scale)
        cloud_recon= pyt2np(recons.squeeze(0) * scale)
        
    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=1)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    print('CD: {:.4f}, EMD: {:.4f}'.format(cd, emd))
 
                
 

        








