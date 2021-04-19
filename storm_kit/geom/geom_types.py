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

from ..differentiable_robot_model.coordinate_transform import CoordinateTransform

def tensor_circle(pt, radius, tensor=None, tensor_args={'device':"cpu", 'dtype':torch.float32}):
    if(tensor is None):
        tensor = torch.empty(3, **tensor_args)
    tensor[:2] = torch.as_tensor(pt, **tensor_args)
    tensor[2] = radius
    return tensor

def tensor_sphere(pt, radius, tensor=None, tensor_args={'device':"cpu", 'dtype':torch.float32}):
    if(tensor is None):
        tensor = torch.empty(4, **tensor_args)
    tensor[:3] = torch.as_tensor(pt, **tensor_args)
    tensor[3] = radius
    return tensor

def tensor_capsule(base, tip, radius, tensor=None, tensor_args={'device':"cpu", 'dtype':torch.float32}):
    if(tensor is None):
        tensor = torch.empty(7, **tensor_args)
    tensor[:3] = torch.as_tensor(base, **tensor_args)
    tensor[3:6] = torch.as_tensor(tip, **tensor_args)
    tensor[6] = radius
    return tensor


def tensor_cube(pose, dims, tensor_args={'device':"cpu", 'dtype':torch.float32}):
    w_T_b = CoordinateTransform(pose=pose, tensor_args=tensor_args)
    b_T_w = w_T_b.inverse()
    dims_t = torch.tensor([dims[0], dims[1], dims[2]], **tensor_args)
    cube = {'trans': w_T_b.translation(), 'rot': w_T_b.rotation(),
            'inv_trans': b_T_w.translation(), 'inv_rot': b_T_w.rotation(),
            'dims':dims_t}
    cube = [w_T_b.translation(), w_T_b.rotation(),
            b_T_w.translation(), b_T_w.rotation(),
            dims_t]
    return cube
