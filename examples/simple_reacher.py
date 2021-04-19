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
torch.multiprocessing.set_start_method('spawn',force=True)
import copy
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from storm_kit.geom.geom_types import tensor_circle
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path
from storm_kit.mpc.rollout.simple_reacher import SimpleReacher
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.utils.state_filter import JointStateFilter, RobotStateFilter
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.mpc.task.simple_task import SimpleTask

traj_log = None

def holonomic_robot(args):
    # load
    tensor_args = {'device':'cpu','dtype':torch.float32}
    simple_task = SimpleTask(robot_file="simple_reacher.yml", tensor_args=tensor_args)
    

    goal_state = [0.4,0.3]
    
    simple_task.update_params(goal_state=goal_state)

    curr_state_tensor = torch.zeros((1,4), **tensor_args)
    filter_coeff = {'position':1.0, 'velocity':1.0, 'acceleration':1.0}
    current_state = {'position':np.array([0.05, 0.2]), 'velocity':np.zeros(2) + 0.0}
    
    i = 0
    exp_params = simple_task.exp_params
    controller = simple_task.controller
    sim_dt = exp_params['control_dt']
    
    
    global traj_log
    image = controller.rollout_fn.image_collision_cost.world_coll.im
    extents = np.ravel(exp_params['model']['position_bounds'])

    traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],
                'acc':[], 'world':image, 'bounds':extents}

    zero_acc = np.zeros(2)
    t_step = 0.0
    full_act = None
    curr_state = np.hstack((current_state['position'], current_state['velocity'], zero_acc, t_step))
    curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)

    update_goal = False

    filtered_state = copy.deepcopy(current_state)
    plan_length = 200

    traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],
                'acc':[], 'world':image, 'bounds':extents}
    

    while(i < plan_length):
        
        current_state = {'position':current_state['position'],
                         'velocity':current_state['velocity'],
                         'acceleration': current_state['position']*0.0}
        filtered_state = current_state
        curr_state = np.hstack((filtered_state['position'], filtered_state['velocity'], filtered_state['acceleration'], t_step))
            

        curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
        error, _ = simple_task.get_current_error(filtered_state)
        
        command = simple_task.get_command(t_step, filtered_state, sim_dt, WAIT=True)
        
        if(i == 0):
            top_trajs = simple_task.top_trajs
            traj_log['top_traj'] = top_trajs.cpu().numpy()
                
        current_state = command
            
        print(i, command['position'])
        traj_log['position'].append(filtered_state['position'])
        traj_log['error'].append(error)
        traj_log['velocity'].append(filtered_state['velocity'])
        traj_log['command'].append(command['acceleration'])
        traj_log['acc'].append(command['acceleration'])
        traj_log['des'].append(copy.deepcopy(goal_state))
        t_step += sim_dt
        i += 1
        
    matplotlib.use('tkagg')
    plot_traj(traj_log)


def plot_traj(traj_log):
    position = np.matrix(traj_log['position'])
    vel = np.matrix(traj_log['velocity'])
    err = np.matrix(traj_log['error'])
    acc = np.matrix(traj_log['acc'])
    act = np.matrix(traj_log['command'])
    des = np.matrix(traj_log['des'])

    c_map = [x / position.shape[0] for x in range(position.shape[0])]
    #print(c_map)
    #fig, axs = plt.subplots(5)

    axs = [plt.subplot(4,1,i+1) for i in range(4)]
    #axs = [plt.subplot(1,1,i+1) for i in range(1)]

    
    if(len(axs) > 3):
        axs[0].set_title('Position')
        axs[1].set_title('Velocity')
        axs[2].set_title('Acceleration')

        axs[3].set_title('Trajectory Position')
        axs[0].plot(position[:,0], 'r', label='x')
        axs[0].plot(position[:,1], 'g',label='y')

        axs[0].plot(des[:,0], 'r-.', label='x_des')
        axs[0].plot(des[:,1],'g-.', label='y_des')
        axs[0].legend()

        axs[1].plot(vel[:,0], 'r',label='x')
        axs[1].plot(vel[:,1], 'g', label='y')
        axs[2].plot(acc[:,0], 'r', label='acc')
        axs[2].plot(acc[:,1], 'g', label='acc')
    extents = (traj_log['bounds'][0], traj_log['bounds'][1],
               traj_log['bounds'][2], traj_log['bounds'][3])
    axs[-1].imshow(traj_log['world'], extent=extents, cmap='gray', alpha=0.4)
    axs[-1].plot(np.ravel(position[0,0]), np.ravel(position[0,1]), 'rX', linewidth=3.0, markersize=15)
    axs[-1].plot(des[:,0], des[:,1],'gX', linewidth=3.0, markersize=15)
    axs[-1].plot(np.ravel(position[:,0]), np.ravel(position[:,1]), 'k-.', linewidth=3.0)
    
    


    for k in range(traj_log['top_traj'].shape[0]):
        d = traj_log['top_traj'][k,:,:2]

    axs[-1].axis('square')
    axs[-1].set_xlim(traj_log['bounds'][0], traj_log['bounds'][1])
    axs[-1].set_ylim(traj_log['bounds'][2], traj_log['bounds'][3])
    plt.show()
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    
    
    holonomic_robot(args)
