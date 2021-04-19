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

from .gaussian_projection import GaussianProjection

class ZeroCost(nn.Module):
    def __init__(self, device=torch.device('cpu'), float_dtype=torch.float64,
                 hinge_val=100.0, weight=1.0, gaussian_params={}, max_vel=0.01):
        super(ZeroCost, self).__init__()
        self.device = device
        self.float_dtype = float_dtype
        self.Z = torch.zeros(1, device=self.device, dtype=self.float_dtype)
        self.weight = torch.as_tensor(weight, device=device, dtype=float_dtype)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        self.hinge_val = hinge_val
        self.max_vel = max_vel
    def forward(self, vels, goal_dist):
        inp_device = vels.device
        vel_err = torch.abs(vels.to(self.device))
        goal_dist = goal_dist.to(self.device)
        

        # max velocity threshold:
        vel_err[vel_err < self.max_vel] = 0.0

        if(self.hinge_val > 0.0):
            vel_err = torch.where(goal_dist <= self.hinge_val, vel_err, 0.0 * vel_err / goal_dist) #soft hinge

        cost = self.weight * self.proj_gaussian((torch.sum(torch.square(vel_err), dim=-1)))

        
        return cost.to(inp_device)
