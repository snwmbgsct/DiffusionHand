from __future__ import print_function, unicode_literals
import sys
sys.path.append('..')
import pickle,os,imageio,torchvision,copy
from dataloader.dataPreprocess.preprocess import *
import dataloader.dataPreprocess.augment as augment
from torch.utils.data import Dataset

# SET THIS to where RHD is located on your machine
path_to_db = '/home/arc/vae-hands-3d/RHD_published_v2'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/yangl/dataset/RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/allusers/handdata/dataset/RHD_published_v2'


def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

class RHDDateset3D(Dataset):
    def __init__(self, mode='training', path_name=path_to_db, view_correction=True):
        self.mode = mode
        self.image_list = []
        self.mask_list = []
        self.depth_list = []
        self.image_uv = []
        self.image_xyz = []
        self.kp_visible_list = [] # visibility of the keypoints, boolean
        self.camera_intrinsic_matrix = []  # matrix containing intrinsic parameters
        self.row = 320
        self.col = 320
        self.num_samples=0
        self.path_name=path_name
        self.mode=mode
        self.view_correction = view_correction
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)
            self.num_samples = len(self.anno_all.items())
        print('self.num_samples',self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if(idx==20500 or idx==28140):idx=0
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            anno=self.anno_all[idx]
            image = imageio.imread(os.path.join(path_to_db, self.mode, 'color', '%.5d.png' % idx))
            mask = imageio.imread(os.path.join(path_to_db, self.mode, 'mask', '%.5d.png' % idx))
            depth = imageio.imread(os.path.join(path_to_db, self.mode, 'depth', '%.5d.png' % idx))
            depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])


            # get info from annotation dictionary
            kp_coord_uv = anno['uv_vis'][:, :2]  # u, v coordinates of 42 hand keypoints, pixel
            kp_visible = anno['uv_vis'][:, 2] == 1  # visibility of the keypoints, boolean
            kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters
            camera_intrinsic_matrix = anno['K']  # matrix containing intrinsic parameters

            image_crop, depth_crop, cloud_normed, pose3d_normed, cloud_vc_normed, \
            pose3d_vc_normed, viewRotation, scale, hand_side, heatmap \
                = preprocessSample(image, depth, mask, kp_coord_uv, kp_visible, kp_coord_xyz, camera_intrinsic_matrix)

            image_crop = image_crop.reshape([256, 256, 3])
            depth_crop = depth_crop.reshape([256, 256, 1])
            cloud_normed = cloud_normed.reshape([4000, 3])
            cloud_vc_normed = cloud_vc_normed.reshape([4000, 3])
            heatmap = heatmap.reshape([64, 64, 21])

            if self.view_correction:
                cloud_output = copy.deepcopy(cloud_vc_normed)
                pose3d_output = copy.deepcopy(pose3d_vc_normed)
            else:
                cloud_output = copy.deepcopy(cloud_normed)
                pose3d_output = copy.deepcopy(pose3d_normed)

            if(self.mode=='training'):
                image_crop, _, cloud_output, _, pose3d_output = \
                    augment.processing_augmentation(image_crop, depth_crop, cloud_output, heatmap, pose3d_output, hand_side)
                cloud_output = cloud_output.transpose(1, 0)
                cjitter=torchvision.transforms.ColorJitter(brightness=0.8, contrast=[0.4,1.6], saturation=[0.4,1.6], hue=0.1)
                image_trans = torchvision.transforms.Compose([cjitter,torchvision.transforms.ToTensor()])
            else:
                image_crop, _, cloud_output, _, pose3d_output = \
                    augment.processing(image_crop, depth_crop, cloud_output, heatmap, pose3d_output, hand_side)
                cloud_output = cloud_output.transpose(1, 0)
                image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image_crop = image_trans(torchvision.transforms.ToPILImage()((image_crop).astype(np.uint8)))


            target = {}
            target['cloud'] = cloud_output
            target['pose3d'] = pose3d_output
            target['scale'] = scale
            target['hand_side'] = hand_side
            target['vis'] = kp_visible
            target['viewRotation'] = viewRotation

            return image_crop, target



