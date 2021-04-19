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
import math

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import ghalton


def scale_ctrl(ctrl, action_lows, action_highs, squash_fn='clamp'):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[np.newaxis, :, np.newaxis]
    act_half_range = (action_highs - action_lows) / 2.0
    act_mid_range = (action_highs + action_lows) / 2.0
    if squash_fn == 'clamp':
        # ctrl = torch.clamp(ctrl, action_lows[0], action_highs[0])
        ctrl = torch.max(torch.min(ctrl, action_highs), action_lows)
        return ctrl
    elif squash_fn == 'clamp_rescale':
        ctrl = torch.clamp(ctrl, -1.0, 1.0)
    elif squash_fn == 'tanh':
        ctrl = torch.tanh(ctrl)
    elif squash_fn == 'identity':
        return ctrl
    return act_mid_range.unsqueeze(0) + ctrl * act_half_range.unsqueeze(0)

#######################
## STOMP Covariance  ##
#######################

def get_stomp_cov(horizon, d_action,
                  tensor_args={'device':torch.device('cpu'),'dtype':torch.float32},
                  cov_mode='vel', RETURN_R=False):
    """ Computes the covariance matrix following STOMP motion planner

    Coefficients from here: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    More info here: https://github.com/ros-industrial/stomp_ros/blob/7fe40fbe6ad446459d8d4889916c64e276dbf882/stomp_core/src/utils.cpp#L36
    """
    acc_fd_array = [0,-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12, 0]
    #acc_fd_array = [1/90, -3/20, 3/2, -49/18, 3/2 , -3/20, 1/90 ]

    #jerk_fd_array = [0, 1 / 12.0, -17 / 12.0, 46 / 12.0, -46 / 12.0, 17 / 12.0, -1 / 12.0]
    jerk_fd_array = [1 / 8.0, -1, 13/8, 0 , -13/8, 1, -1/8]

    #snap_fd_array = [-1/6, 2.0, -13/2, 28/3, -13/2, 2, -1/6]
    snap_fd_array = [0, 1, -4, 6, -4, 1, 0]
    #vel_fd_array = [0, 1.0/12.0 , -2.0/3.0 , 0        , 2.0/3.0  , -1.0/12.0 , 0       ]
    vel_fd_array = [0, 0 , 1, -2       , 1,0, 0       ]
    
    fd_array = acc_fd_array
    A = torch.zeros((d_action * horizon, d_action * horizon), device=tensor_args['device'],dtype=torch.float64)


    if(cov_mode == 'vel'):
        for k in range(d_action):
            for i in range(0, horizon):
                for j in range(-3,4):
                    #print(j)
                    index = i + j
                    if(index < 0):
                        index = 0
                        continue
                    if(index >= horizon):
                        index = horizon - 1
                        continue
                    A[k * horizon + i,k * horizon + index] = fd_array[j + 3]
    elif(cov_mode == 'acc'):
        for k in range(d_action):
            for i in range(0, horizon):
                for j in range(-3,4):
                    #print(j)
                    index = i + j
                    if(index < 0):
                        index = 0
                        continue
                    if(index >= horizon):
                        index = horizon - 1
                        continue
                    if(index >= horizon/2):
                        #print(k * horizon + index - horizon//2)
                        A[k * horizon + i,k * horizon - index - horizon//2 -1] = fd_array[j + 3] #* float((horizon-index) / horizon)
                    else:
                        A[k * horizon + i,k * horizon + index] = fd_array[j + 3] #* float(index/horizon) 
    #plt.imshow(A)
    #plt.show()

    R = torch.matmul(A.transpose(-2,-1), A)
    #print(R[:horizon, :horizon])
    #plt.imshow(R)
    #plt.show()
    #print(R)
    #print(torch.det(R))
    
    cov = torch.inverse(R)
    cov = cov / torch.max(torch.abs(cov))
    #plt.imshow(cov)
    #plt.show()

    # also compute the cholesky decomposition:
    scale_tril = torch.zeros((d_action * horizon, d_action * horizon), **tensor_args)
    scale_tril = torch.cholesky(cov)
    '''
    k = 0
    act_cov_matrix = cov[k * horizon:k * horizon + horizon, k * horizon:k * horizon + horizon]
    print(act_cov_matrix.shape)
    print(torch.det(act_cov_matrix))
    local_cholesky = matrix_cholesky(act_cov_matrix)
    for k in range(d_action):
        
        scale_tril[k * horizon:k * horizon + horizon,k * horizon:k * horizon + horizon] = local_cholesky
    '''
    cov = cov.to(**tensor_args)
    scale_tril = scale_tril.to(**tensor_args) #* 0.1
    scale_tril = scale_tril / torch.max(scale_tril)
    if(RETURN_R):
        return cov, scale_tril, R
    return cov, scale_tril
    


#######################
## Gaussian Sampling ##
#######################


def generate_noise(cov, shape, base_seed, filter_coeffs=None, device=torch.device('cpu')):
    """
        Generate correlated Gaussian samples using autoregressive process
    """
    torch.manual_seed(base_seed)
    beta_0, beta_1, beta_2 = filter_coeffs
    N = cov.shape[0]
    m = MultivariateNormal(loc=torch.zeros(N).to(device), covariance_matrix=cov)
    eps = m.sample(sample_shape=shape)
    # eps = np.random.multivariate_normal(mean=np.zeros((N,)), cov = cov, size=shape)
    if filter_coeffs is not None:
        for i in range(2, eps.shape[1]):
            eps[:,i,:] = beta_0*eps[:,i,:] + beta_1*eps[:,i-1,:] + beta_2*eps[:,i-2,:]
    return eps 

def generate_noise_np(cov, shape, base_seed, filter_coeffs=None):
    """
        Generate correlated noisy samples using autoregressive process
    """
    np.random.seed(base_seed)
    beta_0, beta_1, beta_2 = filter_coeffs
    N = cov.shape[0]
    eps = np.random.multivariate_normal(mean=np.zeros((N,)), cov = cov, size=shape)
    if filter_coeffs is not None:
        for i in range(2, eps.shape[1]):
            eps[:,i,:] = beta_0*eps[:,i,:] + beta_1*eps[:,i-1,:] + beta_2*eps[:,i-2,:]
    return eps 

###########################
## Quasi-Random Sampling ##
###########################

def generate_prime_numbers(num):
    def is_prime(n):
        for j in range(2, ((n //2) + 1),1):
            if n % j == 0:
                return False
        return True

    primes = [0] * num #torch.zeros(num, device=device)
    primes[0] = 2
    curr_num = 1
    for i in range(1, num):
        while True:
            curr_num += 2
            if is_prime(curr_num):
                primes[i] = curr_num
                break
            
    return primes

def generate_van_der_corput_sample(idx, base):
    f, r = 1.0, 0
    while idx > 0:
        f /= base*1.0
        r += f * (idx % base)
        idx = idx // base
    return r

def generate_van_der_corput_samples_batch(idx_batch, base):
    inp_device = idx_batch.device
    batch_size = idx_batch.shape[0]
    f = 1.0 #torch.ones(batch_size, device=inp_device)
    r = torch.zeros(batch_size, device=inp_device)
    while torch.any(idx_batch > 0):
        f /= base*1.0
        r += f * (idx_batch % base) #* (idx_batch > 0)
        idx_batch = idx_batch // base
    return r


# def generate_van_der_corput_samples_batch_2(idx_batch, bases):
#     inp_device = idx_batch.device
#     batch_size = idx_batch.shape[0]
#     f = torch.ones(batch_size, device=inp_device)
#     r = torch.zeros(batch_size, device=inp_device)
    
#     while torch.any(idx_batch > 0):
#         f /= bases*1.0
#         r += f * (idx_batch % base) #* (idx_batch > 0)
#         idx_batch = idx_batch // base
    
#     return r

def generate_halton_samples(num_samples, ndims, bases=None, use_ghalton=True, seed_val=123, device=torch.device('cpu'), float_dtype=torch.float64):
    if not use_ghalton:
        samples = torch.zeros(num_samples, ndims, device=device, dtype=float_dtype)
        if not bases:
            bases = generate_prime_numbers(ndims)
        idx_batch = torch.arange(1,num_samples+1, device=device)
        for dim in range(ndims):
            samples[:, dim] = generate_van_der_corput_samples_batch(idx_batch, bases[dim])
    else:
        
        if ndims <= 100:
            perms = ghalton.EA_PERMS[:ndims]
            sequencer = ghalton.GeneralizedHalton(perms)
        else:
            sequencer = ghalton.GeneralizedHalton(ndims, seed_val)
        samples = torch.tensor(sequencer.get(num_samples), device=device, dtype=float_dtype)
    return samples


def generate_gaussian_halton_samples(num_samples, ndims, bases=None, use_ghalton=True, seed_val=123, device=torch.device('cpu'), float_dtype=torch.float64):
    uniform_halton_samples = generate_halton_samples(num_samples, ndims, bases, use_ghalton, seed_val, device, float_dtype)

    gaussian_halton_samples = torch.sqrt(torch.tensor([2.0],device=device,dtype=float_dtype)) * torch.erfinv(2 * uniform_halton_samples - 1)
    
    return gaussian_halton_samples


def generate_gaussian_sobol_samples(num_samples, ndims, seed_val, device=torch.device('cpu'), float_dtype=torch.float64):
    soboleng = torch.quasirandom.SobolEngine(dimension=ndims, scramble=True, seed=seed_val)
    uniform_sobol_samples = soboleng.draw(num_samples).to(device)

    gaussian_sobol_samples = torch.sqrt(torch.tensor([2.0],device=device,dtype=float_dtype)) * torch.erfinv(2 * uniform_sobol_samples - 1)
    return gaussian_sobol_samples
    
########################
## Gaussian Utilities ##
########################


def gaussian_logprob(mean, cov, x, cov_type="full"):
    """
    Calculate gaussian log prob for given input batch x
    Parameters
    ----------
    mean (np.ndarray): [N x num_samples] batch of means
    cov (np.ndarray): [N x N] covariance matrix
    x  (np.ndarray): [N x num_samples] batch of sample values

    Returns
    --------
    log_prob (np.ndarray): [num_sampls] log probability of each sample
    """
    N = cov.shape[0]
    if cov_type == "diagonal":
        cov_diag = cov.diagonal()
        cov_inv = np.diag(1.0 / cov_diag)
        cov_logdet = np.sum(np.log(cov_diag))
    else:
        cov_logdet = np.log(np.linalg.det(cov))
        cov_inv = np.linalg.inv(cov)
    diff = (x - mean).T
    mahalanobis_dist = -0.5 * np.sum((diff @ cov_inv) * diff, axis=1)
    const1 = -0.5 * N * np.log(2.0 * np.pi)    
    const2 = -0.5*cov_logdet
    log_prob = mahalanobis_dist + const1 + const2
    return log_prob

def gaussian_logprobgrad(mean, cov, x, cov_type="full"):
    if cov_type == "diagonal":
        cov_inv = np.diag(1.0/cov.diagonal())
    else:
        cov_inv = np.linalg.inv(cov)
    diff = (x - mean).T
    grad = diff @ cov_inv
    return grad

def gaussian_entropy(cov=None, L=None): #, cov_type="full"):
    """
    Entropy of multivariate gaussian given either covariance
    or cholesky decomposition of covariance
    
    """
    if cov is not None:
        inp_device = cov.device
        cov_logdet = torch.log(torch.det(cov))
        # print(np.linalg.det(cov.cpu().numpy()))
        # print(torch.det(cov))
        N = cov.shape[0]

    else:
        inp_device = L.device
        cov_logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        N = L.shape[0]
    # if cov_type == "diagonal":
        # cov_logdet =  np.sum(np.log(cov.diagonal())) 
    # else:
    # cov_logdet = np.log(np.linalg.det(cov))

    term1 = 0.5 * cov_logdet
    # pi = torch.tensor([math.pi], device=inp_device)
    # pre-calculate 1.0 + torch.log(2.0*pi) = 2.837877066
    term2 = 0.5 * N * 2.837877066

    ent = term1 + term2
    return ent.to(inp_device)

def gaussian_kl(mean0, cov0, mean1, cov1, cov_type="full"):
    """
    KL-divergence between Gaussians given mean and covariance
    KL(p||q) = E_{p}[log(p) - log(q)]

    """
    N = cov0.shape[0]
    if cov_type == "diagonal":
        cov1_diag = cov1.diagonal()
        cov1_inv = np.diag(1.0 / cov1_diag)
        cov0_logdet = np.sum(np.log(cov0.diagonal()))
        cov1_logdet = np.sum(np.log(cov1_diag))
    else:
        cov1_inv = np.linalg.inv(cov1)
        cov0_logdet = np.log(np.linalg.det(cov0))
        cov1_logdet = np.log(np.linalg.det(cov1))

    term1 = 0.5 * np.trace(cov1_inv @ cov0)
    diff = (mean1 - mean0).T
    mahalanobis_dist = 0.5 * np.sum((diff @ cov1_inv) * diff, axis=1)
    term3 = 0.5 * (-1.0*N + cov1_logdet - cov0_logdet)
    return term1 + mahalanobis_dist + term3



def cost_to_go(cost_seq, gamma_seq):
    """
        Calculate (discounted) cost to go for given cost sequence
    """
    # if torch.any(gamma_seq == 0):
    #     return cost_seq
    cost_seq = gamma_seq * cost_seq  # discounted cost sequence
    # cost_seq = torch.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq = torch.fliplr(torch.cumsum(torch.fliplr(cost_seq), axis=-1))  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq

def cost_to_go_np(cost_seq, gamma_seq):
    """
        Calculate (discounted) cost to go for given cost sequence
    """
    # if np.any(gamma_seq == 0):
    #     return cost_seq
    cost_seq = gamma_seq * cost_seq  # discounted reward sequence
    cost_seq = np.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
    cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq


############
##Cholesky##
############
def matrix_cholesky(A):
    L = torch.zeros_like(A)    
    for i in range(A.shape[-1]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s = s + L[i,k] * L[j,k]            
            
            L[i,j] = torch.sqrt(A[i,i] - s) if (i == j) else \
                      (1.0 / L[j,j] * (A[i,j] - s))
    return L

# Batched Cholesky decomp
def batch_cholesky(A):
    L = torch.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s = s + L[...,i,k] * L[...,j,k]

            L[...,i,j] = torch.sqrt(A[...,i,i] - s) if (i == j) else \
                      (1.0 / L[...,j,j] * (A[...,i,j] - s))
    return L
