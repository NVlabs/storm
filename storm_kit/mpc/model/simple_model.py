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

#
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Dict, Optional, Any

from .model_base import DynamicsModelBase
from .integration_utils import tensor_step_vel, tensor_step_acc, build_int_matrix, build_fd_matrix, tensor_step_jerk, tensor_step_pos


class HolonomicModel(DynamicsModelBase):
    def __init__(self, dt, batch_size=1000, horizon=5, tensor_args={'device':'cpu','dtype':torch.float32}, dt_traj_params=None,
                 control_space='acc'):
        self.tensor_args = tensor_args
        self.dt = dt
        self.n_dofs = 2 # x,y position
        self.d_action = 2
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_traj_points = horizon #int(horizon / dt)
        # integration matrix:
        self._integrate_matrix = build_int_matrix(self.num_traj_points,device=self.tensor_args['device'],
                                                  dtype=self.tensor_args['dtype'])

        self._fd_matrix = build_fd_matrix(self.num_traj_points, device=self.tensor_args['device'],
                                          dtype=self.tensor_args['dtype'])
        
        if(dt_traj_params is None):
            dt_array = [self.dt] * int(1.0 * self.num_traj_points) #+ [self.dt * 5.0] * int(0.3 * self.num_traj_points)
        else:
            dt_array = [dt_traj_params['base_dt']] * int(dt_traj_params['base_ratio'] * self.num_traj_points) + [dt_traj_params['max_dt']] * int((1 - dt_traj_params['base_ratio']) * self.num_traj_points)
            self.dt = dt_traj_params['base_dt']
        if(len(dt_array) != self.num_traj_points):
            dt_array.insert(0,dt_array[0])
        self.dt_traj_params = dt_traj_params
        self._dt_h = torch.tensor(dt_array, **self.tensor_args)
        self.traj_dt = self._dt_h
        self._traj_tstep = torch.matmul(self._integrate_matrix, self._dt_h)
        self.d_state = 3 * self.n_dofs + 1
        self.state_seq = torch.zeros(self.batch_size, self.num_traj_points, self.d_state, **self.tensor_args)
        self.prev_state_buffer = None
        self.control_space = control_space
        if(control_space == 'acc'):
            self.step_fn = tensor_step_acc
        elif(control_space == 'vel'):
            self.step_fn = tensor_step_vel
        elif(control_space == 'jerk'):
            self.step_fn = tensor_step_jerk
        elif(control_space == 'pos'):
            self.step_fn = tensor_step_pos

        #self.order = 2
        #self.smooth_thresh = 10.1
        #self._nth_ing_mat = build_int_matrix(self.num_traj_points,device=self.tensor_args['device'],
        #                                     dtype=self.tensor_args['dtype'],
        #                                     order=self.order-1)
        #self._nth_fd_mat = build_fd_matrix(self.num_traj_points, device=self.tensor_args['device'],
        #                                   dtype=self.tensor_args['dtype'],
        #                                   order=self.order,
        #                                   PREV_STATE=True)

        self.device = self.tensor_args['device']
        self.float_dtype = self.tensor_args['dtype']
        self.prev_state_fd = build_fd_matrix(9, device=self.device, dtype=self.float_dtype, order=1, PREV_STATE=True)
        self.action_order = 0
        self._integrate_matrix_nth = build_int_matrix(self.num_traj_points, order=self.action_order, device=self.device, dtype=self.float_dtype, traj_dt=self.traj_dt)
        self._nth_traj_dt = torch.pow(self.traj_dt, self.action_order)
    
    def filter_actions(self, act_seq, state_seq, prev_state_buffer):
        prev_state = prev_state_buffer
        prev_state_tstep = prev_state_buffer[:,-1]

        order = self.order
        prev_dt = (prev_state_tstep)[-order:]
        n_mul = 1
        state = state_seq[:,:, self.n_dofs * n_mul:self.n_dofs * (n_mul + 1)]
        p_state = prev_state[-order:,self.n_dofs * n_mul: self.n_dofs * (n_mul + 1)].unsqueeze(0)
        p_state = p_state.expand(state.shape[0], -1, -1)
        state_buffer = torch.cat((p_state, state), dim=1)
       
        traj_dt = torch.cat((prev_dt, self.traj_dt))

        # do fd
        B, H, _ = state_buffer.shape
        H -= self.order
        '''
        if(self._nth_fd_mat.shape[0] != H):
            self._nth_fd_mat = build_fd_matrix(H,device=self.tensor_args['device'],
                                               dtype=self.tensor_args['dtype'], order=self.order,
                                               PREV_STATE=True)

        '''
        idx = 1
        # nth order value:
        # build nth order dt mat:
        #traj_dt = torch.pow(traj_dt, self.order)
        fd_dt = torch.pow(traj_dt, self.order-1)
        values = self._nth_fd_mat @ torch.diag(traj_dt) @ state_buffer
        #$print(self._nth_fd_mat)
        # find indices of values greater than thresh
        indices = torch.abs(values) > self.smooth_thresh
        # do a nth order integration to shift the orginal action to by value to make it smooth
        #axs[0].plot(values[idx,:,0].cpu().numpy())
        #plt.plot(values[0,:,0].cpu().numpy())
        clamp_values = torch.clamp(values, -self.smooth_thresh, self.smooth_thresh)
        #values[indices] = self.smooth_thresh
        
        
        #values[indices] = self.thresh 
        offset_act = clamp_values #self._nth_ing_mat @ torch.diag(fd_dt[self.order+1:]) @ clamp_values
        for i in range(self.order):
            offset_act = self._integrate_matrix @ torch.diag(self.traj_dt) @ offset_act
        #plt.plot(offset_act[0,:,0].cpu().numpy())
        #plt.plot(values[0,:,0].cpu().numpy())
        
        #act_seq[indices] = offset_act[indices]
        #act_seq[indices] = 1.0 * offset_act[indices]
        #act_seq = offset_act
        #act_seq = act_seq - offset_act
        
        #plt.show()
        if(False):
            fig, axs = plt.subplots(3)
            axs[0].plot(values[idx,:,0].cpu().numpy(),'r')
            axs[0].plot(clamp_values[idx,:,0].cpu().numpy(),'g')
            axs[1].plot(offset_act[idx,:,0].cpu().numpy(),'g')
            axs[2].plot(act_seq[idx,:,0].cpu().numpy(),'r')
            axs[2].plot(offset_act[idx,:,0].cpu().numpy(),'g')
            
            plt.show()
        raise NotImplementedError
        return offset_act
        #values[torch.abs(values) > self.smooth_thresh]

    def rollout_open_loop(self, start_state: torch.Tensor, act_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # get input device:
        inp_device = start_state.device
        start_state = start_state.to(**self.tensor_args)
        act_seq = act_seq.to(**self.tensor_args)

        if(self.prev_state_buffer is None):
            self.prev_state_buffer = torch.zeros((10, self.d_state), **self.tensor_args)
            self.prev_state_buffer[:,:] = start_state
        self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=0)
        self.prev_state_buffer[-1,:] = start_state

        # compute acceleration from prev buffer:
        
        
        curr_state = self.prev_state_buffer[-1:, :self.n_dofs * 3]

        # make actions be dynamically safe:
        # assuming sampled actions are snap:

        # do 3rd order integration to get acceleration:
        nth_act_seq = self.integrate_action(act_seq)
        #plt.plot(nth_act_seq[0,:,0].cpu().numpy())
        #nth_act_seq = act_seq
        #for i in range(self.action_order - 1):
        #    nth_act_seq = self._integrate_matrix @ torch.diag(self.traj_dt) @ nth_act_seq

        #plt.plot(nth_act_seq[0,:,0].cpu().numpy())
        #plt.show()

        
        
        # forward step with step matrix:
        state_seq = self.step_fn(curr_state, nth_act_seq, self.state_seq, self._dt_h, self.n_dofs, self._integrate_matrix, self._fd_matrix)
        
        state_seq[:,:, -1] = self._traj_tstep

        shape_tup = (self.batch_size * self.num_traj_points, self.n_dofs)
        
        
        state_dict = {'state_seq':state_seq.to(inp_device),
                      'prev_state_seq':self.prev_state_buffer.to(inp_device),
                      'nth_act_seq': nth_act_seq.to(inp_device)}
        return state_dict


        

    def integrate_action(self, act_seq):
        if(self.action_order == 0):
            return act_seq

        nth_act_seq = self._integrate_matrix_nth  @ act_seq
        return nth_act_seq

    def integrate_action_step(self, act, dt):
        for i in range(self.action_order):
            act = act * dt
        
        return act

    def get_next_state(self, curr_state, act, dt):
        """ Does a single step from the current state
        
        Args:
            curr_state: current state
            act: action
            dt: time to integrate

        Returns:
            next_state
        """
        
        if(self.control_space == 'jerk'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = curr_state[self.n_dofs:2*self.n_dofs] + act * dt
            curr_state[self.n_dofs:2*self.n_dofs] = curr_state[self.n_dofs:2*self.n_dofs] + curr_state[self.n_dofs*2:self.n_dofs*3] * dt
            
            curr_state[:self.n_dofs] = curr_state[:self.n_dofs] + curr_state[self.n_dofs:2*self.n_dofs] * dt
        elif(self.control_space == 'acc'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = act * dt
            curr_state[self.n_dofs:2*self.n_dofs] = curr_state[self.n_dofs:2*self.n_dofs] + curr_state[self.n_dofs*2:self.n_dofs*3] * dt
            
            curr_state[:self.n_dofs] = curr_state[:self.n_dofs] + curr_state[self.n_dofs:2*self.n_dofs] * dt
        elif(self.control_space == 'vel'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = 0.0
            curr_state[self.n_dofs:2*self.n_dofs] = act * dt
            
            curr_state[:self.n_dofs] = curr_state[:self.n_dofs] + curr_state[self.n_dofs:2*self.n_dofs] * dt
        elif(self.control_space == 'pos'):
            curr_state[2 * self.n_dofs:3 * self.n_dofs] = 0.0
            curr_state[1 * self.n_dofs:2 * self.n_dofs] = 0.0
            curr_state[:self.n_dofs] = act
        return curr_state

        
