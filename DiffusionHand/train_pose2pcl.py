import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from util.misc import *
from util.data import *
from dpc.autoencoderhand2pcl import *
import os,torch,time

import dataloader.rhddataloader as rhddataloader
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D



# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=63)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default=' ')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
# parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300*THOUSAND)

#Training
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
# parser.add_argument('--num_val_batches', type=int, default=-1)
# parser.add_argument('--num_inspect_batches', type=int, default=1)
# parser.add_argument('--num_inspect_pointclouds', type=int, default=4)
args = parser.parse_args()
seed_all(args.seed)

colorlist_pred = ['#660000', '#b30000', '#ff0000', '#ff4d4d', '#ff9999']
colorlist_gt = ['#000066', '#0000b3', '#0000ff', '#4d4dff', '#9999ff']


# Plots all 5 fingers
def plot_fingers(points, plt_specs, c, ax):
    for i in range(5):
        start, end = i*4+1, (i+1)*4+1
        to_plot = np.concatenate((points[start:end], points[0:1]), axis=0)
        ax.plot(to_plot[:,0], to_plot[:,1], to_plot[:,2], plt_specs, color=c[i])



# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='AE_Pose2Pcl', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')
dataset = rhddataloader.RHDDateset3D(mode='training', view_correction=False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=12, drop_last=False, pin_memory=True)


# Model
logger.info('Building model...')
if args.resume is not None:
    logger.info('Resuming from checkpoint...')
    ckpt = torch.load(args.resume)
    model = AutoEncoder(ckpt['args']).to(args.device)
    model.load_state_dict(ckpt['state_dict'])
else:
    model = AutoEncoder(args).to(args.device)
logger.info(repr(model))


# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)

# Train
epochs = 700
for epoch in range(0, epochs):
    # **** star training ****
    start_time = time.time()
    
    for i, (image, data) in enumerate(tqdm(dataloader, desc='Training')):   
        image = image
        pose_gt = data['pose3d'].reshape(-1, 21, 3).to(args.device)
        scale = data['scale'].to(args.device)
        cloud = data['cloud'].permute(0,2,1).to(args.device)
        
        # Reset grad and model state
        optimizer.zero_grad()
        model.train()
        
        # Forward
        loss = model.get_loss(cloud, pose_gt)
        
        # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        logger.info('[Train] Epoch %04d | Loss %.6f | Grad %.4f ' % (epoch, loss.item(), orig_grad_norm))   
    writer.add_scalar('train/loss', loss, epoch)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('train/grad_norm', orig_grad_norm, epoch)
    writer.flush()
    
    opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
    ckpt_mgr.save(model, args, loss, opt_states, step=epoch)
  
        
        
        