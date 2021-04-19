#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch
import torch.nn as nn
# import torch.nn.functional as F

from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, quaternion_to_matrix

from ...util_file import get_assets_path, join_path
from ...geom.sdf.robot_world import RobotWorldCollisionVoxel
from .gaussian_projection import GaussianProjection

class VoxelCollisionCost(nn.Module):
    def __init__(self, weight=None, robot_params=None,
                 gaussian_params={}, grid_resolution=0.05, distance_threshold=-0.01, 
                 batch_size=2, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(VoxelCollisionCost, self).__init__()
        self.tensor_args = tensor_args
        self.device = tensor_args['device']
        self.float_dtype = tensor_args['dtype']
        self.distance_threshold = distance_threshold
        self.weight = torch.as_tensor(weight, **self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)


        # load robot model:
        robot_collision_params = robot_params['collision_params']
        robot_collision_params['urdf'] = join_path(get_assets_path(),
                                                   robot_collision_params['urdf'])


        # load nn params:
        label_map = robot_params['world_collision_params']['label_map']
        bounds = robot_params['world_collision_params']['bounds']
        #model_path = robot_params['world_collision_params']['model_path']
        self.threshold = robot_params['collision_params']['threshold']
        self.batch_size = batch_size
        
        # initialize NN model:
        self.coll = RobotWorldCollisionVoxel(robot_collision_params, self.batch_size,
                                             label_map, bounds, grid_resolution=grid_resolution,
                                             tensor_args=self.tensor_args)

        #self.coll.set_robot_objects()
        self.coll.build_batch_features(self.batch_size, clone_pose=True, clone_points=True)
        
        self.COLL_INIT = False
        self.SCENE_INIT = False
        self.camera_data = None
        self.res = None
        self.t_mat = None
    def first_run(self, camera_data):
        
        # set world transforms:
        quat = camera_data['robot_camera_pose'][3:]
        rot = quaternion_to_matrix(torch.as_tensor([quat[3],quat[0], quat[1], quat[2]]).unsqueeze(0))

        robot_camera_trans = torch.tensor(camera_data['robot_camera_pose'][0:3]).unsqueeze(0)
        robot_camera_rot = torch.tensor(rot)

        robot_table_trans = torch.tensor([0.0,-0.35,-0.24]).unsqueeze(0)
        robot_table_rot = torch.eye(3).unsqueeze(0)
        self.coll.set_world_transform(robot_table_trans, robot_table_rot,
                                      robot_camera_trans, robot_camera_rot)

        self.coll.set_scene(camera_data['pc'], camera_data['pc_seg'])

        self.COLL_INIT = True

    def set_scene(self, camera_data):
        #if(not self.COLL_INIT):
        self.first_run(camera_data)
        self.camera_data = camera_data
        self.coll.set_scene(camera_data['pc'], camera_data['pc_seg'])
        self.SCENE_INIT = True


    def forward(self, link_pos_seq, link_rot_seq):
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]
        link_pos = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        #print(link_pos.shape, link_rot.shape)
        if(self.batch_size != batch_size):
            self.batch_size = batch_size
            self.coll.build_batch_features(self.batch_size*horizon, clone_pose=True, clone_points=True)
        
        res = self.coll.check_robot_sphere_collisions(link_pos, link_rot)
        self.res = res
        res = res.view(batch_size, horizon, n_links)
        # res = [batch,link]
        
        # negative res is outside mesh (not colliding)
        res += self.distance_threshold
        res[res <= 0.0] = 0.0

        res[res >= 0.5] = 0.5

        # rescale:
        res = res / 0.25

        # all values are positive now
        res = torch.sum(res, dim=-1)
        
        
        cost = res

        cost = self.weight * self.proj_gaussian(cost)
        
        return cost
