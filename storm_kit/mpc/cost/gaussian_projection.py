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
import math

class GaussianProjection(nn.Module):
    """
    Gaussian projection of weights following relaxedIK approach
    """
    def __init__(self, gaussian_params={'n':0,'c':0,'s':0,'r':0}):
        super(GaussianProjection, self).__init__()

        #self.tensor_args = tensor_args
        # model parameters: omega
        self.omega = gaussian_params
        self._ws = gaussian_params['s']
        self._wc = gaussian_params['c']
        self._wn = gaussian_params['n']
        self._wr = gaussian_params['r']
        
                            
        if(len(self.omega.keys()) > 0):
            self.n_pow = math.pow(-1.0, self.omega['n'])
    def forward(self, cost_value):
        if(self._wc == 0.0):
            return cost_value
        exp_term = torch.div(-1.0 * (cost_value - self._ws)**2, 2.0 * (self._wc**2))
        #print(self.omega['s'], cost_value)
        #print(torch.pow(-1.0, self.omega['n']))
        
        cost = 1.0 - self.n_pow * torch.exp(exp_term) + self._wr * torch.pow(cost_value - self._ws, 4)
        #cost = cost_value
        return cost
        
        
