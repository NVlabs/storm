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
import numpy as np

from ...mpc.utils.state_filter import JointStateFilter
from ...mpc.utils.mpc_process_wrapper import ControlProcess

class BaseTask(): 
    def __init__(self, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        self.tensor_args = tensor_args
        self.prev_qdd_des = None
    def init_aux(self):
        self.state_filter = JointStateFilter(filter_coeff=self.exp_params['state_filter_coeff'], dt=self.exp_params['control_dt'])
        
        self.command_filter = JointStateFilter(filter_coeff=self.exp_params['cmd_filter_coeff'], dt=self.exp_params['control_dt'])
        self.control_process = ControlProcess(self.controller)
        self.n_dofs = self.controller.rollout_fn.dynamics_model.n_dofs
        self.zero_acc = np.zeros(self.n_dofs)
        
    def get_rollout_fn(self, **kwargs):
        raise NotImplementedError
    
    def init_mppi(self, **kwargs):
        raise NotImplementedError
    
    def update_params(self, **kwargs):
        self.controller.rollout_fn.update_params(**kwargs)
        self.control_process.update_params(**kwargs)
        return True


    def get_command(self, t_step, curr_state, control_dt, WAIT=False):

        # predict forward from previous action and previous state:
        #self.state_filter.predict_internal_state(self.prev_qdd_des)

        if(self.state_filter.cmd_joint_state is None):
            curr_state['velocity'] *= 0.0
        filt_state = self.state_filter.filter_joint_state(curr_state)
        state_tensor = self._state_to_tensor(filt_state)

        if(WAIT):
            next_command, val, info, best_action = self.control_process.get_command_debug(t_step, state_tensor.numpy(), control_dt=control_dt)
        else:
            next_command, val, info, best_action = self.control_process.get_command(t_step, state_tensor.numpy(), control_dt=control_dt)

        qdd_des = next_command
        self.prev_qdd_des = qdd_des
        cmd_des = self.state_filter.integrate_acc(qdd_des)

        return cmd_des



    def _state_to_tensor(self, state):
        state_tensor = np.concatenate((state['position'], state['velocity'], state['acceleration']))

        state_tensor = torch.tensor(state_tensor)
        return state_tensor
    def get_current_error(self, curr_state):
        state_tensor = self._state_to_tensor(curr_state).to(**self.controller.tensor_args).unsqueeze(0)

        
        ee_error,_ = self.controller.rollout_fn.current_cost(state_tensor)
        ee_error = [x.detach().cpu().item() for x in ee_error]
        return ee_error

    @property
    def mpc_dt(self):
        return self.control_process.mpc_dt
    @property
    def opt_dt(self):
        return self.control_process.opt_dt
    
    def close(self):
        self.control_process.close()
    @property
    def top_trajs(self):
        return self.control_process.top_trajs
    
