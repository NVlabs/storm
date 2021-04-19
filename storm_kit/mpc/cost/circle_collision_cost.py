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



class CircleCollisionCost(nn.Module):
    def __init__(self, weight=None, collision_model=None,gaussian_params={}, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(CircleCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        
        # BUILD world and robot:
        world_yml = join_path(get_gym_configs_path(), collision_model)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)
        w_model = world_params['world_model']['coll_objs']
        self.world_spheres = torch.zeros((len(w_model.keys()),3), **tensor_args)
        for i,key in enumerate(w_model.keys()):
            d = w_model[key]

            self.world_spheres[i,:] = tensor_circle(pt=d['position'], radius=d['radius'],
                                                    tensor_args=self.tensor_args)

        self.dist = None
        self.t_mat = None
    def forward(self, pos_seq):
        
        inp_device = pos_seq.device
        batch_size = pos_seq.shape[0]
        horizon = pos_seq.shape[1]
        pos_batch = pos_seq.view(batch_size * horizon, 2)

        if(self.dist is None or self.dist.shape[0] != pos_batch.shape[0]):
            self.dist = torch.empty((pos_batch.shape[0],self.world_spheres.shape[0]), **self.tensor_args)
        for i in range(self.world_spheres.shape[0]):
            rel_position = torch.norm(pos_batch - self.world_spheres[i,:2], dim=-1)
            self.dist[:, i] = rel_position - self.world_spheres[i,2]
        
        dist = self.dist
        dist = dist.view(batch_size, horizon, self.world_spheres.shape[0])
        # cost only when dist is less

        dist[dist > 0.0] = 0.0
        dist *= -1.0

        cost = self.weight * dist.sum(dim=-1) 
        res = cost
        if(self.t_mat is None or self.t_mat.shape[0] != res.shape[1]):
            self.t_mat = torch.ones((res.shape[1], res.shape[1]), **self.tensor_args).tril()
            
        


        return res.to(inp_device)



