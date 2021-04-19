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
import yaml
import numpy as np

from ...util_file import get_mpc_configs_path as mpc_configs_path
from ...mpc.rollout.simple_reacher import SimpleReacher
from ...mpc.control import MPPI
from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess
from ...util_file import get_assets_path, join_path, load_yaml, get_gym_configs_path
from .task_base import BaseTask


class SimpleTask(BaseTask):
    """
    .. inheritance-diagram:: SimpleTask
       :parts: 1

    """
    def __init__(self, robot_file='simple_reacher.yml', tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super().__init__(tensor_args=tensor_args)

        self.controller = self.init_mppi(robot_file)

        self.init_aux()
        
    def get_rollout_fn(self, **kwargs):
        rollout_fn = SimpleReacher(**kwargs)
        return rollout_fn

    def init_mppi(self, robot_file):
        mpc_yml_file = join_path(mpc_configs_path(), robot_file)

        with open(mpc_yml_file) as file:
            exp_params = yaml.load(file, Loader=yaml.FullLoader)

        rollout_fn = self.get_rollout_fn(exp_params=exp_params, tensor_args=self.tensor_args)



        mppi_params = exp_params['mppi']
        dynamics_model = rollout_fn.dynamics_model
        mppi_params['d_action'] = dynamics_model.d_action
        mppi_params['action_lows'] = -exp_params['model']['max_action'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        mppi_params['action_highs'] = exp_params['model']['max_action'] * torch.ones(dynamics_model.d_action, **self.tensor_args)
        init_action = torch.zeros((mppi_params['horizon'], dynamics_model.d_action), **self.tensor_args)
        mppi_params['init_mean'] = init_action
        mppi_params['rollout_fn'] = rollout_fn
        mppi_params['tensor_args'] = self.tensor_args
        controller = MPPI(**mppi_params)
        self.exp_params = exp_params
        return controller
