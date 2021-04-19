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

from .dist_cost import DistCost


eps = 0.01


class ProjectedDistCost(DistCost):
    def __init__(self, ndofs, weight=None, vec_weight=None, gaussian_params={}, device=torch.device('cpu'), float_dtype=torch.float32):
        super(ProjectedDistCost, self).__init__(weight, gaussian_params=gaussian_params, device=device, float_dtype=float_dtype)
        self.ndofs = ndofs
        self.float_dtype = float_dtype
        self.I = torch.eye(ndofs, device=device, dtype=self.float_dtype)
        self.task_I = torch.eye(6, device=device, dtype=self.float_dtype)
        self.vec_weight = torch.as_tensor(vec_weight, device=device, dtype=float_dtype)
    def forward(self, disp_vec, jac_batch, proj_type="transpose", dist_type="squared_l2", beta=1.0):
        inp_device = disp_vec.device
        disp_vec = self.vec_weight * disp_vec.to(self.device)

        if proj_type == "transpose":
            disp_vec_projected = self.get_transpose_null_disp(disp_vec, jac_batch)
        elif proj_type == "pseudo_inverse":
            disp_vec_projected = self.get_pinv_null_disp(disp_vec, jac_batch)
        elif proj_type == "identity":
            disp_vec_projected = disp_vec
        
        


        return super().forward(disp_vec_projected, dist_type, beta)


    def get_transpose_null_disp(self, disp_vec, jac_batch):
        J_t_J = torch.matmul(jac_batch.transpose(-2,-1), jac_batch)
        J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2,-1))
        score = 1.0 / (torch.sqrt(torch.det(J_J_t)) + 0.0001)
        return score



    def get_pinv_null_disp(self, disp_vec, jac_batch):
    
        jac_batch_t = jac_batch.transpose(-2,-1)

    
        J_J_t = torch.matmul(jac_batch, jac_batch_t)

        J_pinv = jac_batch_t @ torch.inverse(J_J_t + eps * self.task_I.expand_as(J_J_t))
    
        J_pinv_J = torch.matmul(J_pinv, jac_batch)
        

        null_proj = self.I.expand_as(J_pinv_J) - J_pinv_J
        

        null_disp = torch.matmul(null_proj, disp_vec.unsqueeze(-1)).squeeze(-1)
        return null_disp
