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

def get_inv_null_cost(J_full, goal_state, state_batch, device='cpu'):
    rho = 1e-2
    J_full_t = J_full.transpose(-2,-1)

    
    
    J_J_t = torch.matmul(J_full, J_full_t)
    J_J_t_inv = torch.inverse(J_J_t + (rho**2)*torch.eye(3, device=device).expand_as(J_J_t))
    J_pinv = torch.matmul(J_full_t, J_J_t_inv)

    J_pinv_J = torch.matmul(J_pinv, J_full)

    null_proj = torch.eye(6, device=device).expand_as(J_pinv_J) - J_pinv_J 
    
    null_disp = (state_batch[:,:, 0:6]-goal_state[:,0:6])
    null_disp_cost = torch.norm(torch.matmul(null_proj, null_disp.unsqueeze(-1)), dim=-2).squeeze(-1)
    return null_disp_cost

def get_transpose_null_cost(J_full, goal_state, state_batch, device='cpu'):
    rho = 1e-2
    J_full_t = J_full.transpose(-2,-1)
    J_t_J = torch.matmul(J_full_t, J_full)
    
    null_proj = torch.eye(6, device=device).expand_as(J_t_J) - J_t_J

    
    null_disp = (state_batch[:,:, 0:6]-goal_state[:,0:6])
    null_disp_cost = torch.norm(torch.matmul(null_proj, null_disp.unsqueeze(-1)), dim=-2).squeeze(-1)
    return null_disp_cost
