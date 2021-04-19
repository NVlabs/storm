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
# DEALINGS IN THE SOFTWARE.
#
# **********************************************************************
# The first version was licensed as "Original Source License"(see below).
# Several enhancements and bug fixes were done at NVIDIA CORPORATION
# since obtaining the first version. 
#
#
#
# Original Source License:
#
# MIT License
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.#

# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from . import utils
from .coordinate_transform import (
    CoordinateTransform,
    z_rot,
    y_rot,
    x_rot,
)

import hydra


class DifferentiableRigidBody(torch.nn.Module):
    """
    Differentiable Representation of a link
    """

    def __init__(self, rigid_body_params, tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super().__init__()

        self.tensor_args = tensor_args
        self.device = tensor_args['device']
        self.joint_id = rigid_body_params["joint_id"]
        self.name = rigid_body_params["link_name"]

        # dynamics parameters
        self.mass = rigid_body_params["mass"]
        self.com = rigid_body_params["com"]
        self.inertia_mat = rigid_body_params["inertia_mat"]
        self.joint_damping = rigid_body_params["joint_damping"]

        # kinematics parameters
        self.trans = rigid_body_params["trans"]
        self.rot_angles = rigid_body_params["rot_angles"].unsqueeze(0)

        
        roll = self.rot_angles[:,0]
        pitch = self.rot_angles[:,1]
        yaw = self.rot_angles[:,2]
        self.fixed_rotation = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)
        
        self.joint_limits = rigid_body_params["joint_limits"]

        # local joint axis (w.r.t. joint coordinate frame):
        self.joint_axis = rigid_body_params["joint_axis"]
        self.axis_idx = torch.nonzero(self.joint_axis.squeeze(0))
        if self.axis_idx.nelement() > 0:
            self.axis_idx = self.axis_idx[0]
        
        if self.joint_axis[0, 0] == 1:
            self.axis_rot_fn = x_rot
        elif self.joint_axis[0, 1] == 1:
            self.axis_rot_fn = y_rot
        else:
            self.axis_rot_fn = z_rot

        self.joint_pose = CoordinateTransform(tensor_args=tensor_args) #.to(device)
        self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))
        self._batch_size = -1
        self._batch_trans = None
        self._batch_rot = None
        
        # local velocities and accelerations (w.r.t. joint coordinate frame):
        # in spatial vector terminology: linear velocity v
        self.joint_lin_vel = torch.zeros((1, 3), **self.tensor_args)
        # in spatial vector terminology: angular velocity w
        self.joint_ang_vel = torch.zeros((1, 3), **self.tensor_args)
        # in spatial vector terminology: linear acceleration vd
        self.joint_lin_acc = torch.zeros((1, 3), **self.tensor_args)
        # in spatial vector terminology: angular acceleration wd
        self.joint_ang_acc = torch.zeros((1, 3), **self.tensor_args)

        self.update_joint_state(torch.zeros((1, 1), **self.tensor_args), torch.zeros((1, 1), **self.tensor_args))


        # self.update_joint_acc(torch.zeros(1, 1).to(self.device, dtype=self.float_dtype))
        self.update_joint_acc(torch.zeros((1, 1), **self.tensor_args))


        self.pose = CoordinateTransform(tensor_args=self.tensor_args)
        # I have different vectors for angular/linear motion/force, but they usually always appear as a pair
        # meaning we usually always compute both angular/linear components.
        # Maybe worthwile thinking of a structure for this - in math notation we would use the notion of spatial vectors
        # drake uses some form of spatial vector implementation
        self.lin_vel = torch.zeros((1, 3), **self.tensor_args)
        self.ang_vel = torch.zeros((1, 3), **self.tensor_args)
        self.lin_acc = torch.zeros((1, 3), **self.tensor_args)
        self.ang_acc = torch.zeros((1, 3), **self.tensor_args)

        # in spatial vector terminology this is the "linear force f"
        self.lin_force = torch.zeros((1, 3), **self.tensor_args)
        # in spatial vector terminology this is the "couple n"
        self.ang_force = torch.zeros((1, 3), **self.tensor_args)

        return

    def update_joint_state(self, q, qd):
        batch_size = q.shape[0]
            
        self.joint_ang_vel = qd @ self.joint_axis


        if(batch_size != self._batch_size):
            #print('calling once', self._batch_size, batch_size)
            self._batch_size = batch_size
            self._batch_trans = self.trans.unsqueeze(0).repeat(self._batch_size,1)
            self._batch_rot = self.fixed_rotation.repeat(self._batch_size, 1, 1)
            
        # when we update the joint angle, we also need to update the transformation
        
        self.joint_pose.set_translation(self._batch_trans)#
        #self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))
        #print(q.shape)
        rot = self.axis_rot_fn(q.squeeze(1))
        
        self.joint_pose.set_rotation(self._batch_rot @ rot)
        return

    def update_joint_acc(self, qdd):
        # local z axis (w.r.t. joint coordinate frame):
        self.joint_ang_acc = qdd @ self.joint_axis
        return

    def multiply_inertia_with_motion_vec(self, lin, ang):

        mass, com, inertia_mat = self._get_dynamics_parameters_values()

        mcom = com * mass
        com_skew_symm_mat = utils.vector3_to_skew_symm_matrix(com)
        inertia = inertia_mat + mass * (
            com_skew_symm_mat @ com_skew_symm_mat.transpose(-2, -1)
        )

        batch_size = lin.shape[0]

        new_lin_force = mass * lin - utils.cross_product(
            mcom.repeat(batch_size, 1), ang
        )
        new_ang_force = (inertia.repeat(batch_size, 1, 1) @ ang.unsqueeze(2)).squeeze(
            2
        ) + utils.cross_product(mcom.repeat(batch_size, 1), lin)

        return new_lin_force, new_ang_force

    def _get_dynamics_parameters_values(self):
        return self.mass, self.com, self.inertia_mat

    def get_joint_limits(self):
        return self.joint_limits

    def get_joint_damping_const(self):
        return self.joint_damping


class LearnableRigidBody(DifferentiableRigidBody):
    r"""

    Learnable Representation of a link

    """

    def __init__(self, learnable_rigid_body_config, gt_rigid_body_params, device="cpu", float_dtype=torch.float32):

        super().__init__(rigid_body_params=gt_rigid_body_params, device=device)

        # we overwrite dynamics parameters
        if "mass" in learnable_rigid_body_config.learnable_dynamics_params:
            self.mass_fn = hydra.utils.instantiate(
                learnable_rigid_body_config.mass_parametrization, device=device
            )
        else:
            self.mass_fn = lambda: self.mass

        if "com" in learnable_rigid_body_config.learnable_dynamics_params:
            self.com_fn = hydra.utils.instantiate(
                learnable_rigid_body_config.com_parametrization, device=device
            )
        else:
            self.com_fn = lambda: self.com

        if "inertia_mat" in learnable_rigid_body_config.learnable_dynamics_params:
            self.inertia_mat_fn = hydra.utils.instantiate(learnable_rigid_body_config.inertia_parametrization)
        else:
            self.inertia_mat_fn = lambda: self.inertia_mat

        self.joint_damping = gt_rigid_body_params["joint_damping"]

        # kinematics parameters
        if "trans" in learnable_rigid_body_config.learnable_kinematics_params:
            self.trans = torch.nn.Parameter(
                torch.rand_like(gt_rigid_body_params["trans"])
            )
            self.joint_pose.set_translation(torch.reshape(self.trans, (1, 3)))

        if "rot_angles" in learnable_rigid_body_config.learnable_kinematics_params:
            self.rot_angles = torch.nn.Parameter(gt_rigid_body_params["rot_angles"])

        return

    def _get_dynamics_parameters_values(self):
        return self.mass_fn(), self.com_fn(), self.inertia_mat_fn()
