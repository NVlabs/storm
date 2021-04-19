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

class PoseCost(nn.Module):
    """ Rotation cost 

    .. math::
     
    r  &=  \sum_{i=0}^{num_rows} (R^{i,:} - R_{g}^{i,:})^2 \\
    cost &= \sum w \dot r

    
    """
    def __init__(self, weight, vec_weight=[], position_gaussian_params={}, orientation_gaussian_params={}, tensor_args={'device':"cpu", 'dtype':torch.float32}, hinge_val=100.0,
                 convergence_val=[0.0,0.0]):

        super(PoseCost, self).__init__()
        self.tensor_args = tensor_args
        self.I = torch.eye(3,3, **tensor_args)
        self.weight = weight
        self.vec_weight = torch.as_tensor(vec_weight, **tensor_args)
        self.rot_weight = self.vec_weight[0:3]
        self.pos_weight = self.vec_weight[3:6]

        self.px = torch.tensor([1.0,0.0,0.0], **self.tensor_args).T
        self.py = torch.tensor([0.0,1.0,0.0], **self.tensor_args).T
        self.pz = torch.tensor([0.0,0.0,1.0], **self.tensor_args).T
        
        self.I = torch.eye(3,3,**self.tensor_args)
        self.Z = torch.zeros(1, **self.tensor_args)


        self.position_gaussian = GaussianProjection(gaussian_params=position_gaussian_params)
        self.orientation_gaussian = GaussianProjection(gaussian_params=orientation_gaussian_params)
        self.hinge_val = hinge_val
        self.convergence_val = convergence_val
        self.dtype = self.tensor_args['dtype']
        self.device = self.tensor_args['device']
    

    def forward(self, ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot):

        
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_rot_batch = ee_rot_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_goal_pos = ee_goal_pos.to(device=self.device,
                                     dtype=self.dtype)
        ee_goal_rot = ee_goal_rot.to(device=self.device,
                                     dtype=self.dtype)
        
        #Inverse of goal transform
        R_g_t = ee_goal_rot.transpose(-2,-1) # w_R_g -> g_R_w
        R_g_t_d = (-1.0 * R_g_t @ ee_goal_pos.t()).transpose(-2,-1) # -g_R_w * w_d_g -> g_d_g

        
        #Rotation part
        R_g_ee = R_g_t @ ee_rot_batch # g_R_w * w_R_ee -> g_R_ee
        
        
        #Translation part
        # transpose is done for matmul
        term1 = (R_g_t @ ee_pos_batch.transpose(-2,-1)).transpose(-2,-1) # g_R_w * w_d_ee -> g_d_ee
        d_g_ee = term1 + R_g_t_d # g_d_g + g_d_ee
        goal_dist = torch.norm(self.pos_weight * d_g_ee, p=2, dim=-1, keepdim=True)
        
        position_err = (torch.sum(torch.square(self.pos_weight * d_g_ee),dim=-1))
        #compute projection error
        rot_err = self.I - R_g_ee
        rot_err = torch.norm(rot_err, dim=-1)
        rot_err_norm = torch.norm(torch.sum(self.rot_weight * rot_err,dim=-1), p=2, dim=-1, keepdim=True)
        
        rot_err = torch.square(torch.sum(self.rot_weight * rot_err, dim=-1))


        if(self.hinge_val > 0.0):
            rot_err = torch.where(goal_dist.squeeze(-1) <= self.hinge_val, rot_err, self.Z) #hard hinge

        rot_err[rot_err < self.convergence_val[0]] = 0.0
        position_err[position_err < self.convergence_val[1]] = 0.0
        cost = self.weight[0] * self.orientation_gaussian(torch.sqrt(rot_err)) + self.weight[1] * self.position_gaussian(torch.sqrt(position_err))

        # dimension should be bacth * traj_length
        return cost.to(inp_device), rot_err_norm, goal_dist


