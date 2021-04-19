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
"""
Distance cost projected into the null-space of the Jacobian
"""

import torch
import torch.nn as nn

from ...differentiable_robot_model.coordinate_transform import matrix_to_euler_angles

class JacobianCost(nn.Module):
    def __init__(self, ndofs, device, float_dtype, retract_weight):
        self.ndofs = ndofs
        self.device = device
        self.float_dtype = float_dtype
        self.vel_idxs = torch.arange(self.ndofs,2*self.ndofs, dtype=torch.long, device=self.device)

        self.I = torch.eye(ndofs, device=device, dtype=self.float_dtype)

        self.retract_weight = torch.as_tensor(retract_weight, dtype=self.float_dtype, device=self.device)
        super(JacobianCost, self).__init__()
    
    def forward(self, state_batch, ee_pos_batch, ee_rot_batch, 
                goal_ee_pos, goal_ee_rot, jac_batch, dt,
                proj_type="transpose", dist_type="l2", weight=1.0, beta=1.0,
                retract_state=None):
        
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(self.device)
        ee_rot_batch = ee_rot_batch.to(self.device)
        goal_ee_pos = goal_ee_pos.to(self.device)
        goal_ee_rot = goal_ee_rot.to(self.device)
        jac_batch = jac_batch.to(self.device)
        
        #calculate desired x_dot (position+orientation)
        ee_pos_disp = (ee_pos_batch - goal_ee_pos) 

        # ee_euler_batch = matrix_to_euler_angles(ee_rot_batch, convention="XYZ")
        # goal_euler = matrix_to_euler_angles(goal_ee_rot, convention="XYZ")
        # ee_rot_disp = ee_euler_batch - goal_euler
        
        R_g_ee, _ = self.get_relative_transform(ee_pos_batch, ee_rot_batch,
                                                goal_ee_pos, goal_ee_rot)
        #print(R_g_ee.shape, state_batch.shape)
        ee_rot_disp = matrix_to_euler_angles(R_g_ee) * 0.0


        xdot_des = torch.cat((ee_pos_disp, ee_rot_disp), dim=-1) / dt
        
        # xdot_des = ee_pos_disp

        #use jacobian to get desired delta_q
        J_t = jac_batch.transpose(-2,-1)
        #print(xdot_des.unsqueeze(-1).shape, J_t.shape)
        qdot_des = torch.matmul(J_t, xdot_des.unsqueeze(-1)).squeeze(-1)
        #print(qdot_des.shape)
        # compute null space force and add:
        qdot = state_batch[:,:,self.ndofs:2*self.ndofs]
        # input('...')

        disp_vec = qdot - qdot_des# - qdot
        error = (0.5 * torch.sum(torch.square(disp_vec), dim=-1))
        cost = weight * error
        
        return cost.to(inp_device)


    def get_relative_transform(self, ee_pos_batch, ee_rot_batch,
                               goal_ee_pos, goal_ee_rot):

        #Inverse of goal transform
        R_g_t = goal_ee_rot.transpose(-2,-1)
        R_g_t_d = (-1.0* R_g_t @ goal_ee_pos.t()).transpose(-2,-1)

        #ee to goal transform
        #Rotation part
        R_g_ee = R_g_t @ ee_rot_batch
        #Translation part
        term1 = (R_g_t @ ee_pos_batch.transpose(-2,-1)).transpose(-2,-1)
        d_g_ee = term1 + R_g_t_d

        return R_g_ee, d_g_ee


