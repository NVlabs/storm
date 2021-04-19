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
import yaml
# import torch.nn.functional as F
from ...geom.geom_types import tensor_circle
from .gaussian_projection import GaussianProjection
from ...util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from ...geom.sdf.world import WorldImageCollision


class ImageCollisionCost(nn.Module):
    def __init__(self, weight=None, collision_file=None, bounds=[], dist_thresh=0.01, gaussian_params={}, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(ImageCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        
        # BUILD world and robot:
        world_image = join_path(get_assets_path(), collision_file)
        

        self.world_coll = WorldImageCollision(bounds=bounds, tensor_args=tensor_args)
        self.world_coll.update_world(world_image)
        self.dist_thresh = dist_thresh # meters
        
        self.t_mat = None
    def forward(self, pos_seq):
        
        inp_device = pos_seq.device
        batch_size = pos_seq.shape[0]
        horizon = pos_seq.shape[1]
        pos_batch = pos_seq.view(batch_size * horizon, 2)

        # query sdf for points:
        dist = self.world_coll.get_pt_value(pos_batch)
        
        
        dist = dist.view(batch_size, horizon, 1)
        # cost only when dist is less

        # values are signed distance: positive inside object, negative outside
        dist += self.dist_thresh
        dist[dist < 0.0] = 0.0
        dist[dist > 0.0] = 1.0


        res = self.weight * dist

        if(self.t_mat is None or self.t_mat.shape[0] != res.shape[1]):
            self.t_mat = torch.ones((res.shape[1], res.shape[1]), **self.tensor_args).tril()
            
        t_mat = self.t_mat

        res = res.squeeze(-1)
        
        return res.to(inp_device)



