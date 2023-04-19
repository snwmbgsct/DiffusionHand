# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import math
import numpy as np
import cv2

POINT_SIZE = 256

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def processing_augmentation(image, depth, cloud, heatmap, pose3d, hand_side):
    # Random scaling (can be different scaling factors for image and pointclou, because doesn't affect the normalized pose)
    # randScaleImage =  np.maximum(np.minimum(np.random.normal(1.0, 0.08),1.16),0.84)  #process image later together with warpAffine
    randScaleImage = np.random.uniform(low=0.8, high=1.0)
    #cloud = cloud*randScaleCloud

    # Random rotation around z-axis (must be same rotation for image and pointcloud, because it affects the normalized pose)
    pose3d = np.reshape(pose3d, [21, 3])
    randAngle = 2 * math.pi * np.random.rand(1)[0]
    rotMat = cv2.getRotationMatrix2D((128, 128), -180.0 * randAngle / math.pi,
                                     randScaleImage)  # change image later together with translation
    rotMatHeatMap = cv2.getRotationMatrix2D((32, 32), -180.0 * randAngle / math.pi,
                                            randScaleImage)  # change image later together with translation

    (cloud[:, 0], cloud[:, 1]) = rotate((0, 0), (cloud[:, 0], cloud[:, 1]), randAngle)
    (pose3d[:, 0], pose3d[:, 1]) = rotate((0, 0), (pose3d[:, 0], pose3d[:, 1]), randAngle)

    # # Random translation (can be different tranlsations for image and pointcloud, because doesn't affect the normalized pose)
    randTransX = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)
    randTransY = np.maximum(np.minimum(np.random.normal(0.0, 22.0), 40.0), -40.0)

    rotMat[0, 2] += randTransX
    rotMat[1, 2] += randTransY
    rotMatHeatMap[0, 2] += randTransX * 0.25
    rotMatHeatMap[1, 2] += randTransY * 0.25
    image = cv2.warpAffine(image, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=0.0)
    heatmap = cv2.warpAffine(heatmap, rotMatHeatMap, (64, 64), flags=cv2.INTER_LINEAR, borderValue=0.0)
    depth = cv2.warpAffine(depth, rotMat, (256, 256), flags=cv2.INTER_NEAREST, borderValue=10.0)

    randInidices = np.arange(len(cloud))
    np.random.shuffle(randInidices)

    cloud = cloud[randInidices[0:POINT_SIZE, ], :]

    # flipping
    if (hand_side[0] == 0.0):
        image = cv2.flip(image, 1)
        depth = cv2.flip(depth, 1)
        heatmap = cv2.flip(heatmap, 1)
        cloud[:, 0] = -cloud[:, 0]
        pose3d[:, 0] = -pose3d[:, 0]

    pose3d = np.reshape(pose3d, [63])
    image = np.reshape(image, [256, 256, 3])
    depth = np.reshape(depth, [256, 256, 1])

    heatmap = np.reshape(heatmap, [64, 64, 21])

    return image, depth, cloud, heatmap, pose3d

def processing(image, depth, cloud, heatmap, pose3d, hand_side):
    randInidices = np.arange(len(cloud))
    np.random.shuffle(randInidices)
    cloud = cloud[randInidices[0:POINT_SIZE, ], :]

    pose3d = np.reshape(pose3d, [21, 3])

    if (hand_side[0] == 0.0):
        image = cv2.flip(image, 1)
        depth = cv2.flip(depth, 1)
        cloud[:, 0] = -cloud[:, 0]
        pose3d[:, 0] = -pose3d[:, 0]

    pose3d = np.reshape(pose3d, [63])

    image = np.reshape(image, [256, 256, 3])
    depth = np.reshape(depth, [256, 256, 1])
    heatmap = np.reshape(heatmap, [64, 64, 21])

    return image, depth, cloud, heatmap, pose3d
