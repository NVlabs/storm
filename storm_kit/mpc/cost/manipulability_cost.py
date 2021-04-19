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

eps = 0.01



class ManipulabilityCost(nn.Module):
    def __init__(self, ndofs, weight=None, gaussian_params={}, device=torch.device('cpu'), float_dtype=torch.float32, thresh=0.1):
        super(ManipulabilityCost, self).__init__() 
        self.device = device
        self.float_dtype = float_dtype
        self.weight = torch.as_tensor(weight, device=device, dtype=float_dtype)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        self.ndofs = ndofs
        self.thresh = thresh
        self.i_mat = torch.ones((6,1), device=self.device, dtype=self.float_dtype)
    def forward(self, jac_batch):
        inp_device = jac_batch.device
        
        

        with torch.cuda.amp.autocast(enabled=False):
            
            J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2,-1))
            score = torch.sqrt(torch.det(J_J_t))
        score[score != score] = 0.0
        
        
        score[score > self.thresh] = self.thresh #1.0
        score = (self.thresh - score) / self.thresh

        cost = self.weight * score 
        
        return cost.to(inp_device)
    
