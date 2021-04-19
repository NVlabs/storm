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

from .gaussian_projection import GaussianProjection

class CapsuleCollisionCost(nn.Module):
    def __init__(self, weight=None, world_params=None, robot_params=None, gaussian_params={}, device=torch.device('cpu'), float_dtype=torch.float32):
        super(CapsuleCollisionCost, self).__init__()
        self.device = device
        self.float_dtype = float_dtype
        self.tensor_args = {'device':device, 'dtype':float_dtype}
        self.weight = torch.as_tensor(weight, device=device, dtype=float_dtype)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        # BUILD world spheres:
        self.radius = []
        self.position = []
        for obj in world_params['world_model']['coll_objs'].keys():
            self.radius.append(torch.tensor(world_params['world_model']['coll_objs'][obj]['radius'] + 0.1, **self.tensor_args))
            self.position.append(torch.tensor(world_params['world_model']['coll_objs'][obj]['position'], **self.tensor_args))

        
        
    def forward(self, position):
        inp_device = position.device
        position = position.to(self.device)
        i = 0
        cost = torch.norm(position - self.position[i],dim=-1) - self.radius[i]
        cost[cost > 0.0] = 0.0

        for i in range(1,len(self.position)):
            t_cost = torch.norm(position - self.position[i],dim=-1) - self.radius[i]
            t_cost[t_cost > 0.0] = 0.0
            cost += t_cost

        cost[cost < 0.0] = self.weight

        return cost.to(inp_device)



