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
# DEALINGS IN THE SOFTWARE.
#
# **********************************************************************
# The first version was licensed as "Original Source License"(see below).
# Several enhancements and bug fixes were done at NVIDIA CORPORATION
# since obtaining the first version. 
#
#
#
# Original Source License:
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.#

# Copyright (c) Facebook, Inc. and its affiliates.

import random
from contextlib import contextmanager

import numpy as np
import timeit
import torch
import operator
from functools import reduce


prod = lambda l: reduce(operator.mul, l, 1)
torch.set_default_tensor_type(torch.DoubleTensor)

def cross_product(vec3a, vec3b):
    vec3a = convert_into_at_least_2d_pytorch_tensor(vec3a)
    vec3b = convert_into_at_least_2d_pytorch_tensor(vec3b)
    skew_symm_mat_a = vector3_to_skew_symm_matrix(vec3a)
    return (skew_symm_mat_a @ vec3b.unsqueeze(2)).squeeze(2)


def bfill_lowertriangle(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.tril_indices(A.size(-2), k=-1, m=A.size(-1))
    A[..., ii, jj] = vec
    return A


def bfill_diagonal(A: torch.Tensor, vec: torch.Tensor):
    ii, jj = np.diag_indices(min(A.size(-2), A.size(-1)))
    A[..., ii, jj] = vec
    return A


def vector3_to_skew_symm_matrix(vec3):
    vec3 = convert_into_at_least_2d_pytorch_tensor(vec3)
    batch_size = vec3.shape[0]
    skew_symm_mat = vec3.new_zeros((batch_size, 3, 3))
    skew_symm_mat[:, 0, 1] = -vec3[:, 2]
    skew_symm_mat[:, 0, 2] = vec3[:, 1]
    skew_symm_mat[:, 1, 0] = vec3[:, 2]
    skew_symm_mat[:, 1, 2] = -vec3[:, 0]
    skew_symm_mat[:, 2, 0] = -vec3[:, 1]
    skew_symm_mat[:, 2, 1] = vec3[:, 0]
    return skew_symm_mat


def torch_square(x):
    return x * x


def exp_map_so3(omega, epsilon=1.0e-14):
    omegahat = vector3_to_skew_symm_matrix(omega).squeeze()

    norm_omega = torch.norm(omega, p=2)
    exp_omegahat = (torch.eye(3) +
                    ((torch.sin(norm_omega) / (norm_omega + epsilon)) * omegahat) +
                    (((1.0 - torch.cos(norm_omega)) / (torch_square(norm_omega + epsilon))) *
                     (omegahat @ omegahat))
                    )
    return exp_omegahat



def convert_into_pytorch_tensor(variable):
    if isinstance(variable, torch.Tensor):
        return variable
    elif isinstance(variable, np.ndarray):
        return torch.Tensor(variable)
    else:
        return torch.Tensor(variable)


def convert_into_at_least_2d_pytorch_tensor(variable):
    tensor_var = convert_into_pytorch_tensor(variable)
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var

