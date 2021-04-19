#!/usr/bin/env python
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

import copy

import numpy as np
import scipy.special
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.functional import normalize as f_norm

from .control_utils import cost_to_go, matrix_cholesky, batch_cholesky
from .olgaussian_mpc import OLGaussianMPC

class MPPI(OLGaussianMPC):
    """
    .. inheritance-diagram:: MPPI
       :parts: 1

    Class that implements Model Predictive Path Integral Controller
    
    Implementation is based on 
    Williams et. al, Information Theoretic MPC for Model-Based Reinforcement Learning
    with additional functions for updating the covariance matrix
    and calculating the soft-value function.

    """

    def __init__(self,
                 d_action,
                 horizon,
                 init_cov,
                 init_mean,
                 base_action,
                 beta,
                 num_particles,
                 step_size_mean,
                 step_size_cov,
                 alpha,
                 gamma,
                 kappa,
                 n_iters,
                 action_lows,
                 action_highs,
                 null_act_frac=0.,
                 rollout_fn=None,
                 sample_mode='mean',
                 hotstart=True,
                 squash_fn='clamp',
                 update_cov=False,
                 cov_type='sigma_I',
                 seed=0,
                 sample_params={'type': 'halton', 'fixed_samples': True, 'seed':0, 'filter_coeffs':None},
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32},
                 visual_traj='state_seq'):
        
        super(MPPI, self).__init__(d_action,
                                   action_lows, 
                                   action_highs,
                                   horizon,
                                   init_cov,
                                   init_mean,
                                   base_action,
                                   num_particles,
                                   gamma,
                                   n_iters,
                                   step_size_mean,
                                   step_size_cov, 
                                   null_act_frac,
                                   rollout_fn,
                                   sample_mode,
                                   hotstart,
                                   squash_fn,
                                   cov_type,
                                   seed,
                                   sample_params=sample_params,
                                   tensor_args=tensor_args)
        self.beta = beta
        self.alpha = alpha  # 0 means control cost is on, 1 means off
        self.update_cov = update_cov
        self.kappa = kappa
        self.visual_traj = visual_traj

    def _update_distribution(self, trajectories):
        """
           Update moments in the direction using sampled
           trajectories


        """
        costs = trajectories["costs"].to(**self.tensor_args)
        vis_seq = trajectories[self.visual_traj].to(**self.tensor_args)
        actions = trajectories["actions"].to(**self.tensor_args)
        w = self._exp_util(costs, actions)
        
        #Update best action
        best_idx = torch.argmax(w)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)

        top_values, top_idx = torch.topk(self.total_costs, 10)
        #print(ee_pos_seq.shape, top_idx)
        self.top_values = top_values
        self.top_idx = top_idx
        self.top_trajs = torch.index_select(vis_seq, 0, top_idx).squeeze(0)
        #print(self.top_traj.shape)
        #print(self.best_traj.shape, best_idx, w.shape)
        #self.best_trajs = torch.index_select(

        weighted_seq = w.T * actions.T

        sum_seq = torch.sum(weighted_seq.T, dim=0)

        new_mean = sum_seq
        #print(self.stomp_matrix.shape, self.full_scale_tril.shape)
        #cov = self.stomp_matrix #torch.matmul(self.full_scale_tril, self.stomp_matrix)
        #m_matrix = (1.0 / self.horizon) * cov # f_norm(cov,dim=0)
        #sum_seq = sum_seq.transpose(0,1)
        
        #new_mean = torch.matmul(m_matrix,sum_seq.reshape(self.horizon * self.d_action,1)).view(self.d_action, self.horizon).transpose(0,1)
        
        
        # plot mean:
        # = new_mean.cpu().numpy()
        #b = sum_seq.cpu().numpy()#.T
        #print(w, top_idx)
        #new_mean = sum_seq.T
        #matplotlib.use('tkagg')
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean
        #c = self.mean_action.cpu().numpy()
        #plt.plot(a[:,0])
        #plt.plot(b[:,0])
        #plt.plot(actions[top_idx[0],:,0].cpu().numpy())
        #plt.show()

        delta = actions - self.mean_action.unsqueeze(0)

        #Update Covariance
        if self.update_cov:
            if self.cov_type == 'sigma_I':
                #weighted_delta = w * (delta ** 2).T
                #cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0))
                #print(cov_update.shape, self.cov_action)
                raise NotImplementedError('Need to implement covariance update of form sigma*I')
            
            elif self.cov_type == 'diag_AxA':
                #Diagonal covariance of size AxA
                weighted_delta = w * (delta ** 2).T
                # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
                cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
            elif self.cov_type == 'diag_HxH':
                raise NotImplementedError
            elif self.cov_type == 'full_AxA':
                #Full Covariance of size AxA
                weighted_delta = torch.sqrt(w) * (delta).T
                weighted_delta = weighted_delta.T.reshape((self.horizon * self.num_particles, self.d_action))
                cov_update = torch.matmul(weighted_delta.T, weighted_delta) / self.horizon
            elif self.cov_type == 'full_HAxHA':# and self.sample_type != 'stomp':
                weighted_delta = torch.sqrt(w) * delta.view(delta.shape[0], delta.shape[1] * delta.shape[2]).T #.unsqueeze(-1)
                cov_update = torch.matmul(weighted_delta, weighted_delta.T)
                
                # weighted_cov = w * (torch.matmul(delta_new, delta_new.transpose(-2,-1))).T
                # weighted_cov = w * cov.T
                # cov_update = torch.sum(weighted_cov.T,dim=0)
                #
            #elif self.sample_type == 'stomp':
            #    weighted_delta = w * (delta ** 2).T
            #    cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
            #    self.cov_action = (1.0 - self.step_size_cov) * self.cov_action +\
            #        self.step_size_cov * cov_update
            #    #self.scale_tril = torch.sqrt(self.cov_action)
            #    return
            else:
                raise ValueError('Unidentified covariance type in update_distribution')
            
            self.cov_action = (1.0 - self.step_size_cov) * self.cov_action +\
                self.step_size_cov * cov_update
            #if(cov_update == 'diag_AxA'):
            #    self.scale_tril = torch.sqrt(self.cov_action)
            # self.scale_tril = torch.cholesky(self.cov_action)

        
    def _shift(self, shift_steps):
        """
            Predict good parameters for the next time step by
            shifting the mean forward one step and growing the covariance
        """
        if(shift_steps == 0):
            return
        super()._shift(shift_steps)

        if self.update_cov:
            if self.cov_type == 'sigma_I':
                self.cov_action += self.kappa #* self.init_cov_action
                self.scale_tril = torch.sqrt(self.cov_action)
                # self.inv_cov_action = 1.0 / self.cov_action

            elif self.cov_type == 'diag_AxA':
                self.cov_action += self.kappa #* self.init_cov_action
                #self.cov_action[self.cov_action < 0.0005] = 0.0005
                self.scale_tril = torch.sqrt(self.cov_action)
                # self.inv_cov_action = 1.0 / self.cov_action
                
            elif self.cov_type == 'full_AxA':
                self.cov_action += self.kappa*self.I
                self.scale_tril = matrix_cholesky(self.cov_action) # torch.cholesky(self.cov_action) #
                # self.scale_tril = torch.cholesky(self.cov_action)
                # self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)
            
            elif self.cov_type == 'full_HAxHA':
                self.cov_action += self.kappa * self.I

                #shift covariance up and to the left
                # self.cov_action = torch.roll(self.cov_action, shifts=(-self.d_action, -self.d_action), dims=(0,1))
                # self.cov_action = torch.roll(self.cov_action, shifts=(-self.d_action, -self.d_action), dims=(0,1))
                # #set bottom A rows and right A columns to zeros
                # self.cov_action[-self.d_action:,:].zero_()
                # self.cov_action[:,-self.d_action:].zero_()
                # #set bottom right AxA block to init_cov value
                # self.cov_action[-self.d_action:, -self.d_action:] = self.init_cov*self.I2 

                shift_dim = shift_steps * self.d_action
                I2 = torch.eye(shift_dim, **self.tensor_args)
                self.cov_action = torch.roll(self.cov_action, shifts=(-shift_dim, -shift_dim), dims=(0,1))
                #set bottom A rows and right A columns to zeros
                self.cov_action[-shift_dim:,:].zero_()
                self.cov_action[:,-shift_dim:].zero_()
                #set bottom right AxA block to init_cov value
                self.cov_action[-shift_dim:, -shift_dim:] = self.init_cov*I2 
                #update cholesky decomp
                self.scale_tril = torch.cholesky(self.cov_action)
                # self.inv_cov_action = torch.cholesky_inverse(self.scale_tril)


    def _exp_util(self, costs, actions):
        """
            Calculate weights using exponential utility
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)
        # if not self.time_based_weights: traj_costs = traj_costs[:,0]
        traj_costs = traj_costs[:,0]
        #control_costs = self._control_costs(actions)

        total_costs = traj_costs #+ self.beta * control_costs
        
        
        # #calculate soft-max
        w = torch.softmax((-1.0/self.beta) * total_costs, dim=0)
        self.total_costs = total_costs
        return w

    def _control_costs(self, actions):
        if self.alpha == 1:
            # if not self.time_based_weights:
            return torch.zeros(actions.shape[0], **self.tensor_args)
        else:
            # u_normalized = self.mean_action.dot(np.linalg.inv(self.cov_action))[np.newaxis,:,:]
            # control_costs = 0.5 * u_normalized * (self.mean_action[np.newaxis,:,:] + 2.0 * delta)
            # control_costs = np.sum(control_costs, axis=-1)
            # control_costs = cost_to_go(control_costs, self.gamma_seq)
            # # if not self.time_based_weights: control_costs = control_costs[:,0]
            # control_costs = control_costs[:,0]
            delta = actions - self.mean_action.unsqueeze(0)
            u_normalized = self.mean_action.matmul(self.full_inv_cov).unsqueeze(0)
            control_costs = 0.5 * u_normalized * (self.mean_action.unsqueeze(0) + 2.0 * delta)
            control_costs = torch.sum(control_costs, dim=-1)
            control_costs = cost_to_go(control_costs, self.gamma_seq)
            control_costs = control_costs[:,0]
        return control_costs
    
    def _calc_val(self, trajectories):
        costs = trajectories["costs"].to(**self.tensor_args)
        actions = trajectories["actions"].to(**self.tensor_args)
        delta = actions - self.mean_action.unsqueeze(0)
        
        traj_costs = cost_to_go(costs, self.gamma_seq)[:,0]
        control_costs = self._control_costs(delta)
        total_costs = traj_costs + self.beta * control_costs
        # calculate log-sum-exp
        # c = (-1.0/self.beta) * total_costs.copy()
        # cmax = np.max(c)
        # c -= cmax
        # c = np.exp(c)
        # val1 = cmax + np.log(np.sum(c)) - np.log(c.shape[0])
        # val1 = -self.beta * val1

        # val = -self.beta * scipy.special.logsumexp((-1.0/self.beta) * total_costs, b=(1.0/total_costs.shape[0]))
        val = -self.beta * torch.logsumexp((-1.0/self.beta) * total_costs)
        return val
        

