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

from ...mpc.cost import DistCost, ZeroCost, FiniteDifferenceCost
from ...mpc.cost.stop_cost import StopCost
from ...mpc.model.simple_model import HolonomicModel
from ...mpc.cost.circle_collision_cost import CircleCollisionCost
from ...mpc.cost.image_collision_cost import ImageCollisionCost
from ...mpc.cost.bound_cost import BoundCost
from ...mpc.model.integration_utils import build_fd_matrix, tensor_linspace
from ...util_file import join_path, get_assets_path


class SimpleReacher(object):
    """
    This rollout function is for reaching a cartesian pose for a robot

    """

    def __init__(self, exp_params, tensor_args={'device':'cpu','dtype':torch.float32}):
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        mppi_params = exp_params['mppi']

        # initialize dynamics model:
        dynamics_horizon = mppi_params['horizon'] # model_params['dt']
        #Create the dynamical system used for rollouts

        self.dynamics_model = HolonomicModel(dt=exp_params['model']['dt'],
                                             dt_traj_params=exp_params['model']['dt_traj_params'],
                                             horizon=mppi_params['horizon'],
                                             batch_size=mppi_params['num_particles'],
                                             tensor_args=self.tensor_args,
                                             control_space=exp_params['control_space'])

        self.dt = self.dynamics_model.dt
        self.n_dofs = self.dynamics_model.n_dofs
        # rollout traj_dt starts from dt->dt*(horizon+1) as tstep 0 is the current state
        self.traj_dt = self.dynamics_model._dt_h #torch.arange(self.dt, (mppi_params['horizon'] + 1) * self.dt, self.dt,**self.tensor_args)

        self.goal_state = None
        

        self.goal_cost = DistCost(**exp_params['cost']['goal_state'],
                                  tensor_args=self.tensor_args)

        self.stop_cost = StopCost(**exp_params['cost']['stop_cost'],
                                  tensor_args=self.tensor_args,
                                  traj_dt=self.dynamics_model.traj_dt)
        self.stop_cost_acc = StopCost(**exp_params['cost']['stop_cost_acc'],
                                      tensor_args=self.tensor_args,
                                      traj_dt=self.dynamics_model.traj_dt)

        self.zero_vel_cost = ZeroCost(device=self.tensor_args['device'], float_dtype=self.tensor_args['dtype'], **exp_params['cost']['zero_vel'])

        self.fd_matrix = build_fd_matrix(10 - self.exp_params['cost']['smooth']['order'], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'], PREV_STATE=True, order=self.exp_params['cost']['smooth']['order'])

        self.smooth_cost = FiniteDifferenceCost(**self.exp_params['cost']['smooth'],
                                                tensor_args=self.tensor_args)

        
        self.image_collision_cost = ImageCollisionCost(
            **self.exp_params['cost']['image_collision'], bounds=exp_params['model']['position_bounds'],
            tensor_args=self.tensor_args)
        
        self.bound_cost = BoundCost(**exp_params['cost']['state_bound'],
                                    tensor_args=self.tensor_args,
                                    bounds=exp_params['model']['position_bounds'])

        self.terminal_cost = ImageCollisionCost(
            **self.exp_params['cost']['terminal'],
            bounds=exp_params['model']['position_bounds'],
            collision_file=self.exp_params['cost']['image_collision']['collision_file'],
            dist_thresh=self.exp_params['cost']['image_collision']['dist_thresh'],
            tensor_args=self.tensor_args)

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):
        

        state_batch = state_dict['state_seq']
        #print(action_batch)

        goal_state = self.goal_state.unsqueeze(0)
        
        cost, goal_dist = self.goal_cost.forward(goal_state - state_batch[:,:,:self.n_dofs], RETURN_GOAL_DIST=True)
        if self.exp_params['cost']['zero_vel']['weight'] > 0:
            vel_cost = self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=goal_dist.unsqueeze(-1))
            cost += vel_cost

        if(horizon_cost):
            vel_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2])
            cost += vel_cost
            acc_cost = self.stop_cost_acc.forward(state_batch[:, :, self.n_dofs*2:self.n_dofs * 3])
            cost += acc_cost


        if self.exp_params['cost']['smooth']['weight'] > 0 and horizon_cost:
            prev_state = state_dict['prev_state_seq']
            prev_state_tstep = state_dict['prev_state_seq'][:,-1]

            order = self.exp_params['cost']['smooth']['order']
            prev_dt = (self.fd_matrix @ prev_state_tstep)[-order:]
            #print(prev_state_tstep)
            #print(prev_dt.shape, self.traj_dt.shape)
            n_mul = 1
            state = state_batch[:,:, self.n_dofs * n_mul:self.n_dofs * (n_mul+1)]
            p_state = prev_state[-order:,self.n_dofs * n_mul: self.n_dofs * (n_mul+1)].unsqueeze(0)
            #print(p_state.shape, state.shape)
            p_state = p_state.expand(state.shape[0], -1, -1)
            state_buffer = torch.cat((p_state, state), dim=1)
            #print(self.traj_dt.shape, prev_dt.shape)
            traj_dt = torch.cat((prev_dt, self.traj_dt))
            #print(traj_dt)
            smooth_cost = self.smooth_cost.forward(state_buffer,
                                                   traj_dt)
            #print()
            #print(torch.min(smooth_cost),torch.max(smooth_cost))
            cost += smooth_cost

        if self.exp_params['cost']['image_collision']['weight'] > 0:
            # compute collision cost:
            coll_cost = self.image_collision_cost.forward(state_batch[:,:,:self.n_dofs])
            #print (coll_cost.shape)
            cost += coll_cost

        if self.exp_params['cost']['state_bound']['weight'] > 0:
            # compute collision cost:
            cost += self.bound_cost.forward(state_batch[:,:,:self.n_dofs])
        if self.exp_params['cost']['terminal']['weight'] > 0:
            # terminal cost:
            B, H, N = state_batch.shape
            # sample linearly from terminal position to goal:
            linear_pos_batch = torch.zeros_like(state_batch[:,:,:self.n_dofs])
            for i in range(self.n_dofs):
                data = tensor_linspace(state_batch[:,:,i], goal_state[0,0,i], H)
                linear_pos_batch[:,:,i] = data
            #print(linear_pos_batch.shape)
            term_cost = self.terminal_cost.forward(linear_pos_batch)
            #print(term_cost.shape, cost.shape)
            
            cost[:,-1] += torch.sum(term_cost, dim=-1)
        if(return_dist):
            return cost, goal_dist
        else:
            return cost
    
    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Args:
        
            action_seq: torch.Tensor [num_particles, horizon, d_act]
        """
        # rollout_start_time = time.time()
        #print("computing rollout")
        #print(act_seq)
        #print('step...')
        state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        #if('act_seq' in state_dict):
        #    act_seq = state_dict['act_seq']
            #print('action')
        #states = state_dict['state_seq']
        #acc = states[:,:, self.n_dofs*2: self.n_dofs*3]
        '''
        fig, axs = plt.subplots(4)
        acc = act_seq.cpu()
        for i in range(10):
            axs[3].plot(acc[i,:,0])


        states = state_dict['state_seq']
        acc = states[:,:, self.n_dofs*2: self.n_dofs*3]
        
        for i in range(10):
            axs[2].plot(acc[i,:,0])

        acc = states[:,:, self.n_dofs*1: self.n_dofs*2]

        for i in range(10):
            axs[1].plot(acc[i,:,0])
        acc = states[:,:, : self.n_dofs]
        for i in range(10):
            axs[0].plot(acc[i,:,0])

        plt.show()
        '''
        #link_pos_seq, link_rot_seq = self.dynamics_model.get_link_poses()
        
        cost_seq = self.cost_fn(state_dict,act_seq)
        
        sim_trajs = dict(
            actions=act_seq,#.clone(),
            costs=cost_seq,#clone(),
            rollout_time=0.0,
            state_seq=state_dict['state_seq']
        )
        
        return sim_trajs

    def update_params(self, goal_state=None):
        """
        Updates the goal targets for the cost functions.
        goal_state: n_dofs
        goal_ee_pos: 3
        goal_ee_rot: 3,3
        goal_ee_quat: 4

        """
        
        
        self.goal_state = torch.as_tensor(goal_state, **self.tensor_args).unsqueeze(0)
        
        return True
    def __call__(self, start_state, act_seq):
        return self.rollout_fn(start_state, act_seq)
    
    def current_cost(self, current_state):
        current_state = current_state.to(**self.tensor_args).unsqueeze(0)
        
        curr_batch_size = 1
        num_traj_points = 1
        state_dict = {'state_seq': current_state}

        cost = self.cost_fn(state_dict, None,no_coll=False, horizon_cost=False, return_dist=True)
        return cost, state_dict
