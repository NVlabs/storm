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
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ELU, ReLU6
from .network_macros import MLPRegression, scale_to_base, scale_to_net
from ...util_file import get_weights_path, join_path


class RobotSelfCollisionNet():
    """This class loads a network to predict the signed distance given a robot joint config."""
    
    def __init__(self, n_joints=0):
        """initialize class

        Args:
            n_joints (int, optional): Number of joints, same as number of channels for nn input. Defaults to 0.
        """        
        
        super().__init__()
        act_fn = ReLU6
        in_channels = n_joints
        
        out_channels = 1
        dropout_ratio = 0.1
        mlp_layers = [256, 64]
        self.model = MLPRegression(in_channels, out_channels, mlp_layers,
                                   dropout_ratio, batch_norm=False, act_fn=act_fn,
                                   layer_norm=False, nerf=True)

    def load_weights(self, f_name, tensor_args):
        """Loads pretrained network weights if available.

        Args:
            f_name (str): file name, this is relative to weights folder in this repo.
            tensor_args (Dict): device and dtype for pytorch tensors
        """        
        try:
            chk = torch.load(join_path(get_weights_path(), f_name))
            self.model.load_state_dict(chk["model_state_dict"])
            self.norm_dict = chk["norm"]
            for k in self.norm_dict.keys():
                self.norm_dict[k]['mean'] = self.norm_dict[k]['mean'].to(**tensor_args)
                self.norm_dict[k]['std'] = self.norm_dict[k]['std'].to(**tensor_args)
        except Exception:
            print('WARNING: Weights not loaded')
        self.model = self.model.to(**tensor_args)
        self.tensor_args = tensor_args
        self.model.eval()
        
            
    def compute_signed_distance(self, q):
        """Compute the signed distance given the joint config.

        Args:
            q (tensor): input batch of joint configs [b, n_joints]

        Returns:
            [tensor]: largest signed distance between any two non-consecutive links of the robot.
        """        
        with torch.no_grad():
            q_scale = scale_to_net(q, self.norm_dict,'x')
            dist = self.model.forward(q_scale)
            dist_scale = scale_to_base(dist, self.norm_dict, 'y')
        return dist_scale

    def check_collision(self, q):
        """Check collision given joint config. Requires classifier like training.

        Args:
            q (tensor): input batch of joint configs [b, n_joints]

        Returns:
            [tensor]: probability of collision of links, from sigmoid value.
        """        
        with torch.no_grad():
            q_scale = scale_to_net(q, self.norm_dict,'x')
            dist = torch.sigmoid(self.model.forward(q_scale))
        return dist
