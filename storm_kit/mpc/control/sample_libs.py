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


import numpy as np
from scipy.interpolate import BSpline
import scipy.interpolate as si
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from .control_utils import generate_noise, scale_ctrl, generate_gaussian_halton_samples, generate_gaussian_sobol_samples, gaussian_entropy, matrix_cholesky, batch_cholesky, get_stomp_cov

class SampleLib:
    def __init__(self, horizon=None, d_action=None, seed=0, mean=None, scale_tril=None,
                 covariance_matrix=None,
                 tensor_args={'device':"cpu", 'dtype':torch.float32}, fixed_samples=False,
                 filter_coeffs=None, **kwargs):
        self.tensor_args = tensor_args
        self.horizon = horizon
        self.d_action = d_action
        self.seed_val = seed
        self.Z = torch.zeros(self.horizon * self.d_action, **tensor_args)
        self.scale_tril = None
        if(scale_tril is None and covariance_matrix is not None):
            self.scale_tril = torch.cholesky(covariance_matrix)
        self.covariance_matrix = covariance_matrix
        self.fixed_samples = fixed_samples
        self.samples = None
        self.sample_shape = 0
        self.filter_coeffs = filter_coeffs
        self.ndims = horizon * d_action
        self.stomp_matrix, self.stomp_scale_tril = get_stomp_cov(self.horizon, self.d_action, tensor_args=self.tensor_args)
    def get_samples(self, sample_shape, base_seed, current_state=None, **kwargs):
        raise NotImplementedError
    
    def filter_samples(self, eps):
        #print(eps)
        if self.filter_coeffs is not None:
            beta_0, beta_1, beta_2 = self.filter_coeffs

            # This could be tensorized:
            for i in range(2, eps.shape[1]):
                eps[:,i,:] = beta_0 * eps[:,i,:] + beta_1 * eps[:,i-1,:] + beta_2 * eps[:,i-2,:]
        return eps
    def filter_smooth(self, samples):
        
        
        # scale by stomp matrix:
        if(samples.shape[0] == 0):
            return samples

        # fit bspline:
        
        filter_samples = (self.stomp_matrix[:self.horizon,:self.horizon] @ samples)
        #print(filter_samples.shape)
        filter_samples = filter_samples / torch.max(torch.abs(filter_samples))
        return filter_samples


class HaltonSampleLib(SampleLib):
    def __init__(self, horizon=0, d_action=0, seed=0, mean=None, scale_tril=None, covariance_matrix=None,
                 tensor_args={'device':"cpu", 'dtype':torch.float32}, fixed_samples=False, **kwargs):
        super(HaltonSampleLib, self).__init__(horizon=horizon, d_action=d_action,seed=seed, tensor_args=tensor_args,
                                              fixed_samples=fixed_samples)

       

    def get_samples(self, sample_shape, base_seed=None, filter_smooth=False, **kwargs):
        if(self.sample_shape != sample_shape or not self.fixed_samples):
            if(len(sample_shape) > 1):
                print('sample shape should be a single value')
                raise ValueError
            seed = self.seed_val if base_seed is None else base_seed
            self.sample_shape = sample_shape
            self.seed_val = seed
            
            self.samples = generate_gaussian_halton_samples(sample_shape[0],
                                                            self.ndims,
                                                            use_ghalton=True, seed_val=self.seed_val,
                                                            device=self.tensor_args['device'],
                                                            float_dtype=self.tensor_args['dtype'])
            self.samples = self.samples.view(self.samples.shape[0], self.horizon, self.d_action)

            if(filter_smooth):
                self.samples = self.filter_smooth(self.samples)
            else:
                self.samples = self.filter_samples(self.samples)


            
        return self.samples

def bspline(c_arr, t_arr=None, n=100, degree=3):
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()
    count = len(cv)

    if(t_arr is None):
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)

    #spl = BSpline(t, c, k, extrapolate=False)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    
    return samples


class KnotSampleLib(object):
    def __init__(self, horizon=0, d_action=0, n_knots=0, degree=3, seed=0, tensor_args={'device':"cpu", 'dtype':torch.float32}, sample_method='halton',
                 covariance_matrix = None, **kwargs):
        self.ndims = n_knots * d_action
        self.n_knots = n_knots
        self.horizon = horizon
        self.d_action = d_action
        self.tensor_args = tensor_args
        self.seed_val = seed
        self.degree = degree
        self.sample_method = sample_method
        self.Z = torch.zeros(self.ndims, **tensor_args)
        if(covariance_matrix is None):
            self.cov_matrix = torch.eye(self.ndims, **tensor_args)
        self.scale_tril = torch.cholesky(self.cov_matrix.to(dtype=torch.float32)).to(**tensor_args)
        self.mvn = MultivariateNormal(loc=self.Z, scale_tril=self.scale_tril, )
    def get_samples(self, sample_shape, **kwargs):
        # sample shape is the number of particles to sample
        if(self.sample_method=='halton'):
            self.knot_points = generate_gaussian_halton_samples(
                sample_shape[0],
                self.ndims,
                use_ghalton=True,
                seed_val=self.seed_val,
                device=self.tensor_args['device'],
                float_dtype=self.tensor_args['dtype'])
        elif(self.sample_method == 'random'):
            self.knot_points = self.mvn.sample(sample_shape=sample_shape)
            
    # Sample splines from knot points:
        # iteratre over action dimension:
        knot_samples = self.knot_points.view(sample_shape[0], self.d_action, self.n_knots)
        self.samples = torch.zeros((sample_shape[0], self.horizon, self.d_action), **self.tensor_args)
        for i in range(sample_shape[0]):
            for j in range(self.d_action):
                self.samples[i,:,j] = bspline(knot_samples[i,j,:], n=self.horizon, degree=self.degree)
        
        return self.samples
class RandomSampleLib(SampleLib):
    def __init__(self, horizon=0, d_action=0, seed=0, mean=None, scale_tril=None, covariance_matrix=None,
                 tensor_args={'device':"cpu", 'dtype':torch.float32}, fixed_samples=False, **kwargs):
        super(RandomSampleLib, self).__init__(horizon=horizon, d_action=d_action,seed=seed, tensor_args=tensor_args,
                                              fixed_samples=fixed_samples)
        
        if(self.scale_tril is None):
            self.scale_tril = torch.eye(self.ndims, **tensor_args)
        
        self.mvn = MultivariateNormal(loc=self.Z, scale_tril=self.scale_tril)

    def get_samples(self,sample_shape, base_seed=None, filter_smooth=False, **kwargs):

        if(base_seed is not None and base_seed != self.seed_val):
            self.seed_val = base_seed
            #print(self.seed_val)
            torch.manual_seed(self.seed_val)
        if(self.sample_shape != sample_shape or not self.fixed_samples):
            self.sample_shape = sample_shape
            self.samples = self.mvn.sample(sample_shape=self.sample_shape)
            self.samples = self.samples.view(self.samples.shape[0], self.horizon, self.d_action)
            if(filter_smooth):
                self.samples = self.filter_smooth(self.samples)
            else:
                self.samples = self.filter_samples(self.samples)
        return self.samples

class SineSampleLib(SampleLib):
    def __init__(self, horizon=None, d_action=None, seed=0, mean=None, scale_tril=None,
                 covariance_matrix=None,
                 tensor_args={'device':"cpu", 'dtype':torch.float32}, fixed_samples=False,
                 filter_coeffs=None, period=2, **kwargs):
        super(SineSampleLib, self).__init__(horizon=horizon, d_action=d_action, seed=seed,
                                            tensor_args=tensor_args,
                                            fixed_samples=fixed_samples)

        self.const_pi = torch.acos(torch.zeros(1)).item() 
        self.ndims = d_action
        self.period = period
        self.sine_wave = self.generate_sine_wave()
        self.diag_sine_wave = torch.diag(self.sine_wave)


    def get_samples(self, sample_shape, base_seed=None, **kwargs):
        if(self.sample_shape != sample_shape or not self.fixed_samples):
            if(len(sample_shape) > 1):
                print('sample shape should be a single value')
                raise ValueError
            seed = self.seed_val if base_seed is None else base_seed
            self.sample_shape = sample_shape
            self.seed_val = seed

            # sample only amplitudes from halton sequence:
            self.amplitude_samples = generate_gaussian_halton_samples(sample_shape[0],
                                                                      self.ndims,
                                                                      use_ghalton=True, seed_val=self.seed_val,
                                                                      device=self.tensor_args['device'],
                                                                      float_dtype=self.tensor_args['dtype'])

            
            self.amplitude_samples = self.filter_samples(self.amplitude_samples)
            self.amplitude_samples = self.amplitude_samples.unsqueeze(1).expand(-1, self.horizon, -1)
            
            # generate sine waves from samples for the full horizon:
            # amp_samples: [B, d_action], sine: [H] -> B, H, d_action 
            self.samples = self.diag_sine_wave @ self.amplitude_samples 

            #self.samples = self.samples.view(self.samples.shape[0], self.horizon, self.d_action)
            
        return self.samples
    def generate_sine_wave(self, horizon=None):
        horizon = self.horizon if horizon is None else horizon

        # generate a sine wave:
        x = torch.linspace(0, 4 * self.const_pi / self.period, horizon, **self.tensor_args) 
        sin_out = torch.sin(x)
        
        return sin_out
    
class StompSampleLib(SampleLib):
    def __init__(self, horizon=0, d_action=0, seed=0,
                 tensor_args={'device':"cpu", 'dtype':torch.float32}, fixed_samples=False, **kwargs):
        super(StompSampleLib, self).__init__(horizon=horizon, d_action=d_action,seed=seed, tensor_args=tensor_args,
                                             fixed_samples=fixed_samples)

        self.stomp_matrix, self.stomp_scale_tril = get_stomp_cov(self.horizon, self.d_action, tensor_args=self.tensor_args)
        self.Z = torch.zeros(self.horizon * self.d_action, **self.tensor_args)
        self._sample_cov = self.stomp_matrix
        self.mvn = MultivariateNormal(loc=self.Z, scale_tril=self.stomp_scale_tril)
        self.filter_coeffs = None
    def get_samples(self, sample_shape, base_seed=None, **kwargs):
        if(self.sample_shape != sample_shape or not self.fixed_samples):
            if(len(sample_shape) > 1):
                print('sample shape should be a single value')
                raise ValueError
            seed = self.seed_val if base_seed is None else base_seed
            self.sample_shape = sample_shape
            self.seed_val = seed
            torch.manual_seed(self.seed_val)
            
            self.samples = self.mvn.sample(sample_shape=self.sample_shape)
            self.samples = self.samples.view(self.samples.shape[0], self.d_action, self.horizon).transpose(-2,-1)
            self.samples = self.samples / torch.max(torch.abs(self.samples))
        return self.samples

class MultipleSampleLib(SampleLib):
    def __init__(self, horizon=0, d_action=0, seed=0, mean=None, covariance_matrix=None,
                 tensor_args={'device':"cpu", 'dtype':torch.float32}, fixed_samples=False,
                 sample_ratio={'halton':0.2, 'halton-knot':0.2, 'random':0.2, 'random-knot':0.2}, knot_scale=10, **kwargs):

        # sample from a mix of possibilities:

        # halton
        self.halton_sample_lib = HaltonSampleLib(horizon=horizon, d_action=d_action, seed=seed,
                                                 tensor_args=tensor_args,
                                                 fixed_samples=fixed_samples)

        self.knot_halton_sample_lib = KnotSampleLib(horizon=horizon, d_action=d_action, n_knots=horizon//knot_scale, degree=2, sample_method='halton', tensor_args=tensor_args)
        
        #random
        self.random_sample_lib = RandomSampleLib(horizon=horizon, d_action=d_action, seed=seed,
                                                 tensor_args=tensor_args,
                                                 fixed_samples=fixed_samples)
        self.knot_random_sample_lib = KnotSampleLib(horizon=horizon, d_action=d_action, n_knots=horizon//knot_scale, degree=2, sample_method='random', covariance_matrix=covariance_matrix, tensor_args=tensor_args)

        self.sample_ratio = sample_ratio
        self.sample_fns = []


        self.sample_fns = {'halton':self.halton_sample_lib.get_samples,
                           'halton-knot': self.knot_halton_sample_lib.get_samples,
                           'random': self.random_sample_lib.get_samples,
                           'random-knot': self.knot_random_sample_lib.get_samples}
        self.tensor_args = tensor_args
        self.fixed_samples = fixed_samples
        self.samples = None
    def get_samples(self, sample_shape, base_seed=None, **kwargs):
        if(self.fixed_samples and self.samples is None):
            cat_list = []
            sample_shape = list(sample_shape)
            for ki, k in enumerate(self.sample_ratio.keys()):
                if(self.sample_ratio[k] == 0.0):
                    continue
                n_samples = round(sample_shape[0] * self.sample_ratio[k])
                s_shape = torch.Size([n_samples])
                #if(k == 'halton' or k == 'random'):
                samples = self.sample_fns[k](sample_shape=s_shape)
                #else:
                #    samples = self.sample_fns[k](sample_shape=s_shape)
                cat_list.append(samples)
                samples = torch.cat(cat_list, dim=0)
                self.samples = samples
        return self.samples
        
class HaltonStompSampleLib(SampleLib):
    def __init__(self, horizon=0, d_action=0, seed=0, mean=None, scale_tril=None, covariance_matrix=None,
                 tensor_args={'device':"cpu", 'dtype':torch.float32}, fixed_samples=False,
                 sample_ratio=0.2,**kwargs):
        super(HaltonStompSampleLib, self).__init__(horizon=horizon, d_action=d_action,seed=seed,
                                                   tensor_args=tensor_args,
                                                   fixed_samples=fixed_samples)

        # sample ratio
        
        # generate stomp
        self.stomp_sample_lib = StompSampleLib(horizon=horizon, d_action=d_action, seed=seed,
                                               mean=mean, scale_tril=scale_tril,
                                               covariance_matrix=covariance_matrix,
                                               tensor_args=tensor_args,
                                               fixed_samples=fixed_samples)

        
        self.sine_sample_lib = SineSampleLib(horizon=horizon, d_action=d_action, seed=seed,
                                             mean=mean, scale_tril=scale_tril,
                                             covariance_matrix=covariance_matrix,
                                             tensor_args=tensor_args,
                                             fixed_samples=fixed_samples, period=4)
        # generate halton
        self.halton_sample_lib = HaltonSampleLib(horizon=horizon, d_action=d_action, seed=seed,
                                                 mean=mean, scale_tril=scale_tril,
                                                 covariance_matrix=covariance_matrix,
                                                 tensor_args=tensor_args,
                                                 fixed_samples=fixed_samples)

        self.knot_sample_lib = KnotSampleLib(horizon=horizon, d_action=d_action, n_knots=horizon//10, degree=2)
        # split ratio:
        self.halton_ratio = sample_ratio
        self.stomp_ratio = 1.0 - self.halton_ratio
        self.stomp_cov_matrix = self.stomp_sample_lib._sample_cov
        # some samples full halton, full random
        # some samples with zeros at the end
        self.zero_ratio = 0.1
    def get_samples(self, sample_shape, base_seed=None, **kwargs):
        #print(sample_shape, self.halton_ratio, self.stomp_ratio)
        halton_sample_size = list(sample_shape)
        halton_sample_size[0] = round(self.halton_ratio * halton_sample_size[0])

        halton_samples = self.knot_sample_lib.get_samples(sample_shape=torch.Size(halton_sample_size))
        #halton_samples = self.halton_sample_lib.get_samples(sample_shape=torch.Size(halton_sample_size,**self.tensor_args), filter_smooth=False)

        # filter halton samples:
        #print(halton_samples)
        if(self.halton_ratio == 1.0):
            return halton_samples
        
        stomp_sample_size = list(sample_shape)
        stomp_sample_size[0] = round(self.stomp_ratio * stomp_sample_size[0])
        stomp_samples = self.stomp_sample_lib.get_samples(sample_shape=torch.Size(stomp_sample_size))
        
        
        #sine_samples = self.sine_sample_lib.get_samples(sample_shape=torch.Size(stomp_sample_size,**self.tensor_args))

        samples = torch.cat((halton_samples, stomp_samples), dim=0)

        # zero out some samples after 0.1 tsteps:
        
        return samples
