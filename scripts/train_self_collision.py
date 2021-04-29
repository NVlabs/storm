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
import torch.nn.functional as F
import numpy as np
from storm_kit.mpc.rollout.arm_base import ArmBase
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path, get_mpc_configs_path, get_weights_path
import yaml
from storm_kit.mpc.control.control_utils import generate_halton_samples
from storm_kit.geom.nn_model.robot_self_collision import RobotSelfCollisionNet
import os
import matplotlib.pyplot as plt
class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, x,y,y_gt):
        self.x = x
        self.y = y
        self.y_gt = y_gt
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, idx):
        sample = {'x': self.x[idx,:], 'y': self.y[idx,:], 'y_gt': self.y_gt[idx,:]}
        return sample
def create_dataset(robot_name):
    checkpoints_dir = get_weights_path()+'/robot_self'
    num_particles = 5000
    task_file = robot_name+'_reacher.yml'
    # load robot model:
    device = torch.device('cuda', 0) 
    tensor_args = {'device':device, 'dtype':torch.float32}
    mpc_yml_file = join_path(get_mpc_configs_path(), task_file)

    with open(mpc_yml_file) as file:
        exp_params = yaml.load(file, Loader=yaml.FullLoader)
    exp_params['robot_params'] = exp_params['model'] #robot_params
    exp_params['cost']['primitive_collision']['weight'] = 0.0
    exp_params['control_space'] = 'pos'
    exp_params['mppi']['horizon'] = 2
    exp_params['mppi']['num_particles'] = num_particles
    rollout_fn = ArmBase(exp_params, tensor_args, world_params=None)
    
    # sample joint angles
    dof = rollout_fn.dynamics_model.d_action
    q_samples = generate_halton_samples(num_particles*2, dof, use_ghalton=True,
                                        device=tensor_args['device'],
                                        float_dtype=tensor_args['dtype'])

    # scale samples by joint range:
    up_bounds = rollout_fn.dynamics_model.state_upper_bounds[:dof]
    low_bounds = rollout_fn.dynamics_model.state_lower_bounds[:dof]

    range_b = up_bounds - low_bounds
    q_samples = q_samples * range_b + low_bounds
    q_samples = q_samples.view(num_particles,2,dof)

    
    start_state = torch.zeros((rollout_fn.dynamics_model.d_state), **tensor_args)

    state_dict = rollout_fn.dynamics_model.rollout_open_loop(start_state, q_samples)
    
    link_pos_seq = state_dict['link_pos_seq']
    link_rot_seq = state_dict['link_rot_seq']
    # compute link poses
    cost = rollout_fn.robot_self_collision_cost.distance
    dist = cost(link_pos_seq, link_rot_seq)


    # dataset:
    x = q_samples.view(num_particles*2, dof)

    x_data  = x.cpu().numpy()
    y = dist.view(num_particles*2,1) #* 100.0

    #torch.save(x, 'x_data.p')
    #torch.save(y, 'y_data.p')
    #plt.scatter(x_data[:,1], x_data[:,3], c=y.cpu().numpy(), vmin=-0.1, vmax=0.1,cmap='coolwarm')
    #plt.show()
    #print(torch.min(y), torch.max(y))
    #x = x[y > -0.02]
    #y[y < -0.02] = -0.02
    #y[y < -0.1] = 0.1
    #y[y >= -0.02] = 1.0
    #y[y < -0.02] = 0.0
    
    print(torch.min(y), torch.max(y))
    
    n_size = x.shape[0]
    #print(n_size)
    # 
    nn_model = RobotSelfCollisionNet(n_joints=dof)
    nn_model.model.to(**tensor_args)
    model = nn_model.model

    # load training set:
    x_train = x[:int((n_size)*0.7),:]
    y_train = y[:int((n_size)*0.7)]
    x_coll = x_train[y_train[:,0]>-0.02]#.cpu().numpy()
    y_coll = y_train[y_train[:,0]>-0.02]#.cpu().numpy()

    #x_data = x_train[y_train[:,0]>-0.01].cpu().numpy()
    #y_data = y_train[y_train[:,0]>-0.01].cpu().numpy()
    #x_train = x_train.cpu().numpy()
    #plt.scatter(x_data[:,1], x_data[:,3],c=y_data,vmin=-0.1, vmax=0.1, cmap='coolwarm')
    #plt.show()

    # scale dataset:
    mean_x = torch.mean(x, dim=0)#* 0.0 #+ 1.0
    std_x = torch.mean(x, dim=0)* 0.0 + 1.0
    mean_y = torch.mean(y, dim=0)#* 0.0 #+ 1.0
    std_y = torch.mean(y, dim=0)#* 0.0 + 1.0
    
    x_train = torch.div((x_train - mean_x),std_x)
    #x_train[x_train!=x_train] = 0.0
    x_coll = torch.div(x_coll - mean_x, std_x).detach()
    y_coll = torch.div(y_coll - mean_y, std_y).detach()
    y_train_true = y_train.clone()
    y_train = torch.div((y_train - mean_y),std_y)
    #y_train[y_train!=y_train] = 0.0
    #d = y[int((n_size*2)*0.9):]
    #print(d[d>0.0].shape)
    #exit()
    x = torch.div(x - mean_x,std_x)
    x[x!=x] = 0.0
    y = torch.div(y - mean_y,std_y)
    y[y!=y] = 0.0
    x_val = x[int((n_size)*0.7):int((n_size)*0.9),:]
    y_val = y[int((n_size)*0.7):int((n_size)*0.9)]
    x_test = x[int((n_size)*0.9):,:]
    y_test = y[int((n_size)*0.9):]

    train_dataset = RobotDataset(x_train.detach(), y_train.detach(), y_train_true.detach())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    coll_dataset = RobotDataset(x_coll.detach(), y_coll.detach(), y_coll.detach())
    collloader = torch.utils.data.DataLoader(coll_dataset, batch_size=32, shuffle=True)


    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)#,momentum=0.97)

    #print(model)
    epochs = 100
    min_loss = 100.0
    # training:
    for e in range(epochs):
        model.train()
        loss = []
        i = 0
        x_train = x_train[torch.randperm(x_train.size()[0])]
        for i, data in enumerate(trainloader):
            
            optimizer.zero_grad()
            
            y = data['y'].to(device)
            y_gt = data['y_gt'].to(device)
            x = data['x'].to(device)

            coll_data = next(iter(collloader))

            x_coll_batch = coll_data['x'].to(device)
            y_coll_batch = coll_data['y'].to(device)
            
            y_pred = (model.forward(x))
            y_coll_pred = (model.forward(x_coll_batch))
            #print(y_coll_pred)#, y_coll_batch, x_coll_batch)
            alpha = 1.0 #torch.where(y_gt > -0.1, 100.0, 1.0)
            #train_loss = F.binary_cross_entropy_with_logits(y_pred,y) + F.binary_cross_entropy_with_logits(y_coll_pred, y_coll_batch)
            train_loss = F.mse_loss(y_pred, y, reduction='mean') + 1.0*F.mse_loss(y_coll_pred, y_coll_batch, reduction='mean')
            #train_loss = torch.nn.BCEWithLogitsLoss(
            train_loss.backward()
            optimizer.step()
            loss.append(train_loss.item())
            #print(train_loss.item())
            #i += batch_size

        model.eval()
        
        y_pred = model.forward(x_val)
        #y_pred = torch.sigmoid(model.forward(x_val))
        #print(x_val)
        #print(y_pred[0,0])
        val_loss = F.mse_loss(y_pred, y_val, reduction='mean')
        #val_loss = F.binary_cross_entropy_with_logits(y_pred,y_val)
        train_loss = np.mean(loss)
        if(val_loss < min_loss and e>20):
            print('saving model', val_loss.item())
            torch.save(
                {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'norm':{'x':{'mean':mean_x, 'std':std_x},
                            'y':{'mean':mean_y, 'std':std_y}}
                },
                join_path(checkpoints_dir,
                          robot_name+'_self_sdf.pt'))
            min_loss = val_loss
        print(e, train_loss, val_loss.item())

    with torch.no_grad():
        x = x_test#[y_test[:,0] > 0.0]
        y = y_test#[y_test[:,0] > 0.0]
        
        #print(x.shape)
        y_pred = model.forward(x)
        y_pred = torch.mul(y_pred, std_y) + mean_y
        y_test = torch.mul(y, std_y) + mean_y
        print(y_test[y_test>0.0])
        print(y_pred[y_test>0.0])
        #print(y_pred.shape, y_test.shape)
        loss = F.l1_loss(y_pred, y_test, reduction='mean')
        print(torch.median(y_pred), torch.mean(y_pred))
        print(loss.item())
            
if __name__=='__main__':
    create_dataset('franka')
    # load robot model
    
    # load dataset


    # 
