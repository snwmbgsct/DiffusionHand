import torch,random,logging
import numpy as np
from enum import Enum
import dataloader.dataPreprocess.augment as augment



def meanEuclideanLoss(pred, gt, scale, jointNr=21):
    pred = pred.view([-1, jointNr, 3])
    gt = gt.reshape([-1, jointNr, 3])
    eucDistance = torch.squeeze(torch.sqrt(torch.sum((pred - gt) ** 2, dim=2)))
    meanEucDistance_normed = torch.mean(eucDistance)
    eucDistance = eucDistance * torch.squeeze(scale).view(scale.shape[0], 1)
    meanEucDistance = torch.mean(eucDistance)
    return meanEucDistance_normed, meanEucDistance

def pose_loss(p0, p1, scale,jointN=21):
    pose_loss_rgb = torch.sum((p0 - p1) ** 2, 1)
    _, eucLoss_rgb = meanEuclideanLoss(p0, p1, scale,jointN)
    return pose_loss_rgb, eucLoss_rgb

def kl_loss(z_mean, z_stddev, goalStd=1.0):
        latent_loss = 0.5 * torch.sum(z_mean ** 2 + z_stddev ** 2 - torch.log(z_stddev ** 2) - goalStd, 1)
        return latent_loss

# def cloud_dis(c0, c1):
#     c0 = c0.permute(0, 2, 1).contiguous()
#     c1 = c1.permute(0, 2, 1).contiguous()
#     dist1, dist2 = chamfer_dist(c0, c1)

#     cd0 = torch.max(torch.mean(dist1, dim=1), torch.mean(dist2, dim=1))
#     b = earth_mover_distance(c0, c1, transpose=False)/augment.POINT_SIZE
#     d0, d1 = torch.mean(cd0), torch.mean(b)
#     return d0+d1

def init_fn(worker_id):np.random.seed(worker_id)

class Mode(Enum):
    Train = 1
    Eval = 2
    Refine = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def setSeed(seed=None):
    import time
    if not isinstance(seed, int):
        seed = int(time.time())%10000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

import random,string
def rand_model_name() -> str:
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))