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
from typing import List, Tuple, Dict, Optional, Any
import torch
from urdfpy import URDF

from ...differentiable_robot_model.differentiable_robot_model import DifferentiableRobotModel
from .model_base import DynamicsModelBase
from .integration_utils import build_int_matrix, build_fd_matrix, tensor_step_acc, tensor_step_vel, tensor_step_pos, tensor_step_jerk

class URDFKinematicModelBaseline(DynamicsModelBase):
    def __init__(self, urdf_path, dt, batch_size=1000, horizon=5,
                 tensor_args={'device':'cpu','dtype':torch.float32}, ee_link_name='ee_link', link_names=[], dt_traj_params=None, vel_scale=0.5, control_space='acc'):
        self.urdf_path = urdf_path
        self.device = tensor_args['device']

        self.float_dtype = tensor_args['dtype']
        self.tensor_args = tensor_args
        self.dt = dt
        self.ee_link_name = ee_link_name
        self.batch_size = batch_size
        self.horizon = horizon
        self.num_traj_points = int(round(horizon / dt))
        self.link_names = link_names

        self.robot_model = DifferentiableRobotModel(urdf_path, None, tensor_args=tensor_args)

        #self.robot_model.half()
        self.n_dofs = self.robot_model._n_dofs
        self.urdfpy_robot = URDF.load(urdf_path) #only for visualization
        
        self.d_state = 3 * self.n_dofs + 1
        self.d_action = self.n_dofs

        #Variables for enforcing joint limits
        self.joint_names = self.urdfpy_robot.actuated_joint_names
        self.joint_lim_dicts = self.robot_model.get_joint_limits()
        self.state_upper_bounds = torch.zeros(self.d_state, device=self.device, dtype=self.float_dtype)
        self.state_lower_bounds = torch.zeros(self.d_state, device=self.device, dtype=self.float_dtype)
        for i in range(self.n_dofs):
            self.state_upper_bounds[i] = self.joint_lim_dicts[i]['upper']
            self.state_lower_bounds[i] = self.joint_lim_dicts[i]['lower']
            self.state_upper_bounds[i+self.n_dofs] = self.joint_lim_dicts[i]['velocity'] * vel_scale
            self.state_lower_bounds[i+self.n_dofs] = -self.joint_lim_dicts[i]['velocity'] * vel_scale
            self.state_upper_bounds[i+2*self.n_dofs] = 10.0
            self.state_lower_bounds[i+2*self.n_dofs] = -10.0

        #print(self.state_upper_bounds, self.state_lower_bounds)
        # #pre-allocating memory for rollouts
        self.state_seq = torch.zeros(self.batch_size, self.num_traj_points, self.d_state, device=self.device, dtype=self.float_dtype)
        self.ee_pos_seq = torch.zeros(self.batch_size, self.num_traj_points, 3, device=self.device, dtype=self.float_dtype)
        self.ee_rot_seq = torch.zeros(self.batch_size, self.num_traj_points, 3, 3, device=self.device, dtype=self.float_dtype)
        self.Z = torch.tensor([0.], device=self.device, dtype=self.float_dtype) #torch.zeros(batch_size, self.n_dofs, device=self.device, dtype=self.float_dtype)

        self._integrate_matrix = build_int_matrix(self.num_traj_points, device=self.device, dtype=self.float_dtype)

        self.control_space = control_space
        if(control_space == 'acc'):
            self.step_fn = tensor_step_acc
        elif(control_space == 'vel'):
            self.step_fn = tensor_step_vel
        elif(control_space == 'jerk'):
            self.step_fn = tensor_step_jerk


        #self._integrate_matrix = torch.eye(self.num_traj_points, **self.tensor_args)
        #self._integrate_matrix_t0 = torch.tril(torch.ones((self.num_traj_points, self.num_traj_points), device=self.device, dtype=self.float_dtype), diagonal=-1)
        self._fd_matrix = build_fd_matrix(self.num_traj_points, device=self.device,
                                          dtype=self.float_dtype, order=1)
        if(dt_traj_params is None):
            dt_array = [self.dt] * int(1.0 * self.num_traj_points) #+ [self.dt * 5.0] * int(0.3 * self.num_traj_points)
        else:
            dt_array = [dt_traj_params['base_dt']] * int(dt_traj_params['base_ratio'] * self.num_traj_points)
            #smooth_blending = [dt_traj_params['max_dt']] * int((1 - dt_traj_params['base_ratio']) * self.num_traj_points)
            smooth_blending = torch.linspace(dt_traj_params['base_dt'],dt_traj_params['max_dt'], steps=int((1 - dt_traj_params['base_ratio']) * self.num_traj_points)).tolist()
            dt_array += smooth_blending
            
            
            self.dt = dt_traj_params['base_dt']
        if(len(dt_array) != self.num_traj_points):
            dt_array.insert(0,dt_array[0])
        self.dt_traj_params = dt_traj_params
        #self._dt_h = torch.tensor([self.dt] * self.num_traj_points, dtype=self.float_dtype, device=self.device)
        self._dt_h = torch.tensor(dt_array, dtype=self.float_dtype, device=self.device)
        self.dt_traj = self._dt_h
        self.traj_dt = self._dt_h
        #print(self.traj_dt)
        self._traj_tstep = torch.matmul(self._integrate_matrix, self._dt_h)
        
        self.link_pos_seq = torch.empty((self.batch_size, self.num_traj_points, len(self.link_names),3), **self.tensor_args)
        self.link_rot_seq = torch.empty((self.batch_size, self.num_traj_points, len(self.link_names),3,3), **self.tensor_args)

        self.lin_jac_seq = torch.empty((self.batch_size, self.num_traj_points,3,self.n_dofs), **self.tensor_args)
        self.ang_jac_seq = torch.empty((self.batch_size, self.num_traj_points,3,self.n_dofs), **self.tensor_args)

        self.prev_state_buffer = None # torch.zeros((self.num_traj_points, self.d_state), **self.tensor_args)
        self.prev_state_fd = build_fd_matrix(9, device=self.device, dtype=self.float_dtype, order=1, PREV_STATE=True)


        self.action_order = 0
        self._integrate_matrix_nth = build_int_matrix(self.num_traj_points, order=self.action_order, device=self.device, dtype=self.float_dtype, traj_dt=self.traj_dt)
        self._nth_traj_dt = torch.pow(self.traj_dt, self.action_order)
        
    def get_next_state(self, curr_state: torch.Tensor, act:torch.Tensor, dt):
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

    def step(self, state: torch.Tensor, act: torch.Tensor,dt=None,t=0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        curr_dt = self.dt if dt is None else dt
        batch_size = state.shape[0]
        # get input device:
        inp_device = state.device
        state = state.to(self.device)
        act = act.to(self.device)

        q_curr = state[:, :self.n_dofs]
        qd_curr = state[:, self.n_dofs:2*self.n_dofs]
        qdd_curr = state[:, -self.n_dofs:]
        
        qdd_new = act.reshape(batch_size,self.n_dofs)
        qd_new = qd_curr + qdd_curr * curr_dt
        q_new =  q_curr + qd_curr * curr_dt

        ee_pos, ee_rot, lin_jac, ang_jac = self.robot_model.compute_fk_and_jacobian(q_new, qd_new, link_name="ee_link")
        state[:, :self.n_dofs] = q_new
        state[:, self.n_dofs:2*self.n_dofs] = qd_new
        state[:, 2*self.n_dofs:3*self.n_dofs] = qdd_new
        #if(self.
        #print(lin_jac.shape)
        return state.to(inp_device), ee_pos.to(inp_device), ee_rot.to(inp_device), lin_jac, ang_jac

    def rollout_open_loop(self, start_state: torch.Tensor, act_seq: torch.Tensor, dt=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        curr_dt = self.dt if dt is None else dt
        curr_horizon = self.horizon
        batch_size, horizon, d_act = act_seq.shape
        # get input device:
        inp_device = start_state.device
        start_state = start_state.to(self.device)
        act_seq = act_seq.to(self.device)
        # batch_size, horizon, d_act = act_seq.shape
        curr_dt = self.dt if dt is None else dt
        curr_horizon = self.horizon
        # get input device:
        inp_device = start_state.device
        start_state = start_state.to(self.device, dtype=self.float_dtype)
        act_seq = act_seq.to(self.device, dtype=self.float_dtype)
        
        # add start state to prev state buffer:
        #print(start_state.shape, self.d_state)
        if(self.prev_state_buffer is None):
            self.prev_state_buffer = torch.zeros((10, self.d_state), **self.tensor_args)
            self.prev_state_buffer[:,:] = start_state
        self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=0)
        self.prev_state_buffer[-1,:] = start_state

        start_state = self.prev_state_buffer[-1:,:self.n_dofs * 3]
        
        curr_state = start_state.repeat(batch_size,1)
        
        state_seq = self.state_seq
        ee_pos_seq = self.ee_pos_seq
        ee_rot_seq = self.ee_rot_seq
        curr_horizon = self.horizon
        curr_batch_size = self.batch_size
        num_traj_points = self.num_traj_points
        link_pos_seq = self.link_pos_seq
        link_rot_seq = self.link_rot_seq
        lin_jac_seq = self.lin_jac_seq
        ang_jac_seq = self.ang_jac_seq
        for t in range(horizon):
            # state_seq[:, t] = curr_state.clone()
            # ee_pos_seq[:,t] = curr_ee_pos.clone()
            # ee_rot_seq[:,t] = curr_ee_rot.clone()
            curr_state, curr_ee_pos, curr_ee_rot, lin_jac, ang_jac = self.step(curr_state, act_seq[:, t],self.traj_dt[t],t)
            state_seq[:, t,:self.n_dofs*3] = curr_state
            state_seq[:, t,-1] = self._traj_tstep[t]
            ee_pos_seq[:,t,:] = curr_ee_pos
            ee_rot_seq[:,t,:,:] = curr_ee_rot
            lin_jac_seq[:,t,:,:] = lin_jac
            ang_jac_seq[:,t,:,:] = ang_jac
            
            # get link poses:
            for ki,k in enumerate(self.link_names):
                link_pos, link_rot = self.robot_model.get_link_pose(k)
                link_pos_seq[:,t,ki,:] = link_pos#.view((curr_batch_size, num_traj_points,3))
                link_rot_seq[:,t,ki,:,:] = link_rot#.view((curr_batch_size, num_traj_points,3,3))
            
        
        #ee_pos_seq = ee_pos_seq.view((curr_batch_size, num_traj_points, 3))
        #ee_rot_seq = ee_rot_seq.view((curr_batch_size, num_traj_points, 3, 3))
        #lin_jac_seq = lin_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        #ang_jac_seq = ang_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        #print(lin_jac_seq.shape)
        state_dict = {'state_seq':state_seq.to(inp_device),
                      'ee_pos_seq': ee_pos_seq.to(inp_device),
                      'ee_rot_seq': ee_rot_seq.to(inp_device),
                      'lin_jac_seq': lin_jac_seq.to(inp_device),
                      'ang_jac_seq': ang_jac_seq.to(inp_device),
                      'link_pos_seq':link_pos_seq.to(inp_device),
                      'link_rot_seq':link_rot_seq.to(inp_device),
                      'prev_state_seq':self.prev_state_buffer.to(inp_device)}
        return state_dict

    def old_tensor_step(self, state: torch.Tensor, act: torch.Tensor, state_seq: torch.Tensor, dt=None) -> torch.Tensor:
        """
        Args:
        state: [1,N]
        act: [H,N]
        todo:
        Integration  with variable dt along trajectory
        """
        inp_device = state.device
        state = state.to(self.device, dtype=self.float_dtype)
        act = act.to(self.device, dtype=self.float_dtype)
        nth_act_seq = self.integrate_action(act)
        
        
        #print(state.shape)
        state_seq = self.step_fn(state, nth_act_seq, state_seq, self._dt_h, self.n_dofs, self._integrate_matrix, self._fd_matrix)
        #state_seq = self.enforce_bounds(state_seq)
        # timestep array
        state_seq[:,:, -1] = self._traj_tstep

        
        return state_seq
        
        
    
    def _old_rollout_open_loop(self, start_state: torch.Tensor, act_seq: torch.Tensor,
                               dt=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # batch_size, horizon, d_act = act_seq.shape
        curr_dt = self.dt if dt is None else dt
        curr_horizon = self.horizon
        # get input device:
        inp_device = start_state.device
        start_state = start_state.to(self.device, dtype=self.float_dtype)
        act_seq = act_seq.to(self.device, dtype=self.float_dtype)
        
        # add start state to prev state buffer:
        #print(start_state.shape, self.d_state)
        if(self.prev_state_buffer is None):
            self.prev_state_buffer = torch.zeros((10, self.d_state), **self.tensor_args)
            self.prev_state_buffer[:,:] = start_state
        self.prev_state_buffer = self.prev_state_buffer.roll(-1, dims=0)
        self.prev_state_buffer[-1,:] = start_state

        
        #print(self.prev_state_buffer[:,-1])
        # compute dt w.r.t previous data?
        state_seq = self.state_seq
        ee_pos_seq = self.ee_pos_seq
        ee_rot_seq = self.ee_rot_seq
        curr_horizon = self.horizon
        curr_batch_size = self.batch_size
        num_traj_points = self.num_traj_points
        link_pos_seq = self.link_pos_seq
        link_rot_seq = self.link_rot_seq

        
        
        curr_state = self.prev_state_buffer[-1:,:self.n_dofs * 3]
 
        
        # forward step with step matrix:
        state_seq = self.tensor_step(curr_state, act_seq, state_seq, curr_dt)
        
        #print(start_state[:,self.n_dofs*2 : self.n_dofs*3])

        shape_tup = (curr_batch_size * num_traj_points, self.n_dofs)
        ee_pos_seq, ee_rot_seq, lin_jac_seq, ang_jac_seq = self.robot_model.compute_fk_and_jacobian(state_seq[:,:,:self.n_dofs].view(shape_tup),
                                                                                                    state_seq[:,:,self.n_dofs:2 * self.n_dofs].view(shape_tup),
                                                                                                    link_name=self.ee_link_name)

        # get link poses:
        for ki,k in enumerate(self.link_names):
            link_pos, link_rot = self.robot_model.get_link_pose(k)
            link_pos_seq[:,:,ki,:] = link_pos.view((curr_batch_size, num_traj_points,3))
            link_rot_seq[:,:,ki,:,:] = link_rot.view((curr_batch_size, num_traj_points,3,3))
            
        
        ee_pos_seq = ee_pos_seq.view((curr_batch_size, num_traj_points, 3))
        ee_rot_seq = ee_rot_seq.view((curr_batch_size, num_traj_points, 3, 3))
        lin_jac_seq = lin_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        ang_jac_seq = ang_jac_seq.view((curr_batch_size, num_traj_points, 3, self.n_dofs))
        
        state_dict = {'state_seq':state_seq.to(inp_device),
                      'ee_pos_seq': ee_pos_seq.to(inp_device),
                      'ee_rot_seq': ee_rot_seq.to(inp_device),
                      'lin_jac_seq': lin_jac_seq.to(inp_device),
                      'ang_jac_seq': ang_jac_seq.to(inp_device),
                      'link_pos_seq':link_pos_seq.to(inp_device),
                      'link_rot_seq':link_rot_seq.to(inp_device),
                      'prev_state_seq':self.prev_state_buffer.to(inp_device)}
        return state_dict


    

    def enforce_bounds(self, state_batch):
        """
            Project state into bounds
        """
        batch_size = state_batch.shape[0]
        bounded_state = torch.max(torch.min(state_batch, self.state_upper_bounds), self.state_lower_bounds)
        bounded_q = bounded_state[...,:,:self.n_dofs]
        bounded_qd = bounded_state[...,:,self.n_dofs:2*self.n_dofs]
        bounded_qdd = bounded_state[...,:,2*self.n_dofs:3*self.n_dofs]
        
        # #set velocity and acc to zero where position is at bound
        bounded_qd = torch.where(bounded_q < self.state_upper_bounds[:self.n_dofs], bounded_qd, self.Z)
        bounded_qd = torch.where(bounded_q > self.state_lower_bounds[:self.n_dofs], bounded_qd, self.Z)
        bounded_qdd = torch.where(bounded_q < self.state_upper_bounds[:self.n_dofs], bounded_qdd, -10.0*bounded_qdd)
        bounded_qdd = torch.where(bounded_q > self.state_lower_bounds[:self.n_dofs], bounded_qdd, -10.0*bounded_qdd)

        # #set acc to zero where vel is at bounds 
        bounded_qdd = torch.where(bounded_qd < self.state_upper_bounds[self.n_dofs:2*self.n_dofs], bounded_qdd, self.Z)
        bounded_qdd = torch.where(bounded_qd > self.state_lower_bounds[self.n_dofs:2*self.n_dofs], bounded_qdd, self.Z)
        state_batch[...,:,:self.n_dofs] = bounded_q
        #state_batch[...,:,self.n_dofs:self.n_dofs*2] = bounded_qd
        #state_batch[...,:,self.n_dofs*2:self.n_dofs*3] = bounded_qdd
        
        #bounded_state = torch.cat((bounded_q, bounded_qd, bounded_qdd), dim=-1) 
        return state_batch

    def integrate_action(self, act_seq):
        if(self.action_order == 0):
            return act_seq

        nth_act_seq = self._integrate_matrix_nth  @ act_seq
        return nth_act_seq

    def integrate_action_step(self, act, dt):
        for i in range(self.action_order):
            act = act * dt
        
        return act

    #Rendering
    def render(self, state):
        q = state[:, 0:self.n_dofs]
        state_dict = {}
        for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
            state_dict[joint.name] = q[:,i].item()
        self.urdfpy_robot.show(cfg=state_dict,use_collision=True) 


    def render_trajectory(self, state_list):
        state_dict = {}
        q = state_list[0][:, 0:self.n_dofs]
        for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
            state_dict[joint.name] = [q[:,i].item()]
        for state in state_list[1:]:
            q = state[:, 0:self.n_dofs]
            for (i,joint) in enumerate(self.urdfpy_robot.actuated_joints):
                state_dict[joint.name].append(q[:,i].item())
        self.urdfpy_robot.animate(cfg_trajectory=state_dict,use_collision=True) 

