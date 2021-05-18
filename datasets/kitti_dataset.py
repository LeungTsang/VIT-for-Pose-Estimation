# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from scipy.spatial.transform import Rotation as R
import torch

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}


    def get_color(self, folder, frame_index, side):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        #if do_flip:
        #    color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

        self.gt_local_poses = {}
        self.gt_q = {}
        self.gt_p = {}
        if self.is_train:
            sequence_id = [0,1,2,3,4,5,6,7,8]

        else :
            sequence_id = [9,10]

        for s_id in sequence_id:
            gt_poses_path = os.path.join(self.data_path, "poses", "{:02d}.txt".format(s_id))
            gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
            gt_global_poses = np.concatenate(
                (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
            gt_global_poses[:, 3, 3] = 1
            gt_xyzs = gt_global_poses[:, :3, 3]

            self.gt_local_poses[s_id] = []
            self.gt_q[s_id] = []
            self.gt_p[s_id] = []
            for i in range(1, len(gt_global_poses)):
                self.gt_local_poses[s_id].append(
                    np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

                r = R.from_matrix(self.gt_local_poses[s_id][i-1][0:3,0:3])
                self.gt_q[s_id].append(torch.tensor(r.as_quat()))

                self.gt_p[s_id].append(torch.from_numpy(self.gt_local_poses[s_id][i-1][0:3,3]))

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path

    def get_pose(self, folder, frame_index):
        s_id = int(folder)
        return self.gt_q[s_id][frame_index], self.gt_p[s_id][frame_index]
