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
from typing import List, Tuple, Dict, Optional
import os

import torch


from .utils import cross_product
from .differentiable_rigid_body import (
    DifferentiableRigidBody,
    LearnableRigidBody,
)
from .urdf_utils import URDFRobotModel
import torch.autograd.profiler as profiler
#import diff_robot_data
#robot_description_folder = diff_robot_data.__path__[0]


class DifferentiableRobotModel(torch.nn.Module):
    """
    Differentiable Robot Model
    """

    def __init__(
            self, urdf_path: str, learnable_rigid_body_config=None, name="",
            tensor_args={'device':"cpu", 'dtype':torch.float32}):

        super().__init__()

        self.name = name

        self.device = tensor_args['device']
        self.float_dtype = tensor_args['dtype']
        self.tensor_args = tensor_args
        self.urdf_path = urdf_path
        
        self._urdf_model = URDFRobotModel(
            urdf_path=urdf_path, tensor_args=self.tensor_args
        )
        self._bodies = torch.nn.ModuleList()
        self._n_dofs = 0
        self._controlled_joints = []

        # initialize zero tensors:
        self._batch_size = 1
        self._base_lin_vel = torch.zeros((self._batch_size, 3), **self.tensor_args)
        self._base_ang_vel = torch.zeros((self._batch_size, 3), **self.tensor_args)
        self._base_pose_trans = torch.zeros(self._batch_size,3, **self.tensor_args)
        self._base_pose_rot = torch.eye(3, **self.tensor_args).expand(self._batch_size,3,3)
        
        # here we're making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # joint is at the beginning of a link
        self._name_to_idx_map = dict()

        for (i, link) in enumerate(self._urdf_model.robot.links):

            rigid_body_params = self._urdf_model.get_body_parameters_from_urdf(i, link)

            if (learnable_rigid_body_config is not None) and (link.name in learnable_rigid_body_config.learnable_links):
                body = LearnableRigidBody(
                    learnable_rigid_body_config=learnable_rigid_body_config,
                    gt_rigid_body_params=rigid_body_params,
                    tensor_args=self.tensor_args
                )
            else:
                body = DifferentiableRigidBody(
                    rigid_body_params=rigid_body_params, tensor_args=self.tensor_args
                )

            if rigid_body_params["joint_type"] != "fixed":
                self._n_dofs += 1
                self._controlled_joints.append(i)

            self._bodies.append(body)
            self._name_to_idx_map[body.name] = i
    def delete_lxml_objects(self):
        self._urdf_model = None
    def load_lxml_objects(self):
        self._urdf_model = URDFRobotModel(
            urdf_path=self.urdf_path, tensor_args=self.tensor_args
        )
    def update_kinematic_state(self, q: torch.Tensor, qd: torch.Tensor) -> None:
        r"""

        Updates the kinematic state of the robot
        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]

        Returns:

        """
        '''
        assert q.ndim == 2
        assert qd.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs
        '''
        q = q.to(**self.tensor_args)
        qd = qd.to(**self.tensor_args)
        
        batch_size = q.shape[0]

        if(batch_size != self._batch_size):
            self._batch_size = batch_size
            self._base_lin_vel = torch.zeros((self._batch_size, 3), **self.tensor_args)
            self._base_ang_vel = torch.zeros((self._batch_size, 3), **self.tensor_args)
            self._base_pose_trans = torch.zeros(self._batch_size,3, **self.tensor_args)
            self._base_pose_rot = torch.eye(3, **self.tensor_args).expand(self._batch_size,3,3)
            

        # we assume a non-moving base
        parent_body = self._bodies[0]
        parent_body.lin_vel = self._base_lin_vel

        parent_body.ang_vel = self._base_ang_vel


        # Below two lines are not in the source repo, this is done to initialize?
        parent_body.pose.set_translation(self._base_pose_trans)
        parent_body.pose.set_rotation(self._base_pose_rot)
        
        # propagate the new joint state through the kinematic chain to update bodies position/velocities
        with profiler.record_function("robot_model/fk/for_loop"):
            for i in range(1, len(self._bodies)):
                if(i in self._controlled_joints):
                    idx = self._controlled_joints.index(i)
                    self._bodies[i].update_joint_state(q[:,idx].unsqueeze(1), qd[:,idx].unsqueeze(1))
                body = self._bodies[i]

                parent_name = self._urdf_model.get_name_of_parent_body(body.name)
                # find the joint that has this link as child
                parent_body = self._bodies[self._name_to_idx_map[parent_name]]

                # transformation operator from child link to parent link
                childToParentT = body.joint_pose

                # the position and orientation of the body in world coordinates, with origin at the joint
                body.pose = parent_body.pose.multiply_transform(childToParentT)
                
                '''
                parentToChildT = childToParentT.inverse()
                # we rotate the angular velocity of the parent's link into the child frame
                new_ang_vel = (
                    parentToChildT.rotation() @ parent_body.ang_vel.unsqueeze(2)
                ).squeeze(2)

                # this body's angular velocity is combination of the velocity experienced at it's parent's link
                # + the velocity created by this body's joint
                body.ang_vel = body.joint_ang_vel + new_ang_vel

                # transform linear velocity of parent link frame to this body's link fram
                new_lin_vel = (
                    parentToChildT.trans_cross_rot() @ parent_body.ang_vel.unsqueeze(2)
                ).squeeze(2) + (
                    parentToChildT.rotation() @ parent_body.lin_vel.unsqueeze(2)
                ).squeeze(
                    2
                )

                # combining linear velocity of parent link with linear velocity induced by this links joint
                body.lin_vel = body.joint_lin_vel + new_lin_vel
                '''
        return

    def compute_forward_kinematics(
            self, q: torch.Tensor, qd: torch.Tensor,link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            link_name: name of link

        Returns: translation and rotation of the link frame

        """
        #assert q.ndim == 2
        inp_device = q.device
        q = q.to(**self.tensor_args)
        qd = qd.to(**self.tensor_args)
        self.update_kinematic_state(q, qd)

        pose = self._bodies[self._name_to_idx_map[link_name]].pose
        pos = pose.translation().to(inp_device)
        rot = pose.rotation().to(inp_device)#get_quaternion()
        return pos, rot


    def get_link_pose(self, link_name: str):
        
        pose = self._bodies[self._name_to_idx_map[link_name]].pose
        pos = pose.translation()#.to(inp_device)
        rot = pose.rotation()#.to(inp_device)#get_quaternion()
        return pos, rot
    def iterative_newton_euler(
        self, base_lin_acc: torch.Tensor, base_ang_acc: torch.Tensor
    ) -> None:
        r"""

        Args:
            base_lin_acc: linear acceleration of base (for fixed manipulators this is zero)
            base_ang_acc: angular acceleration of base (for fixed manipulators this is zero)

        """

        body = self._bodies[0]
        body.lin_acc = base_lin_acc
        body.ang_acc = base_ang_acc

        for i in range(1, len(self._bodies)):
            body = self._bodies[i]
            parent_name = self._urdf_model.get_name_of_parent_body(body.name)

            parent_body = self._bodies[self._name_to_idx_map[parent_name]]

            # get the inverse of the current joint pose
            inv_pose = body.joint_pose.inverse()

            # new wd
            new_ang_acc = (
                inv_pose.rotation() @ parent_body.ang_acc.unsqueeze(2)
            ).squeeze(2) + body.joint_ang_acc
            # new vd
            new_lin_acc = (
                (inv_pose.trans_cross_rot() @ parent_body.ang_acc.unsqueeze(2)).squeeze(
                    2
                )
                + (inv_pose.rotation() @ parent_body.lin_acc.unsqueeze(2)).squeeze(2)
                + body.joint_lin_acc
            )

            # body velocity cross joint vel
            new_w = cross_product(body.ang_vel, body.joint_ang_vel)
            new_v = cross_product(
                body.ang_vel, body.joint_lin_vel
            ) + cross_product(body.lin_vel, body.joint_ang_vel)

            body.lin_acc = new_lin_acc + new_v
            body.ang_acc = new_ang_acc + new_w

        child_body = self._bodies[-1]

        # after recursion is done, we propagate forces back up (from endeffector link to base)
        for i in range(len(self._bodies) - 2, 0, -1):
            body = self._bodies[i]
            joint_pose = child_body.joint_pose

            # pose x children_force
            child_ang_force = (
                joint_pose.trans_cross_rot() @ child_body.lin_force.unsqueeze(2)
            ).squeeze(2) + (
                joint_pose.rotation() @ child_body.ang_force.unsqueeze(2)
            ).squeeze(
                2
            )
            child_lin_force = (
                joint_pose.rotation() @ child_body.lin_force.unsqueeze(2)
            ).squeeze(2)

            [IcAcc_lin, IcAcc_ang] = body.multiply_inertia_with_motion_vec(
                body.lin_acc, body.ang_acc
            )
            [IcVel_lin, IcVel_ang] = body.multiply_inertia_with_motion_vec(
                body.lin_vel, body.ang_vel
            )

            # body vel x IcVel
            tmp_ang_force = cross_product(
                body.ang_vel, IcVel_ang
            ) + cross_product(body.lin_vel, IcVel_lin)
            tmp_lin_force = cross_product(body.ang_vel, IcVel_lin)

            body.lin_force = IcAcc_lin + tmp_lin_force + child_lin_force
            body.ang_force = IcAcc_ang + tmp_ang_force + child_ang_force
            child_body = body
        return

    def compute_inverse_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd_des: torch.Tensor,
        include_gravity: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            qdd_des: desired joint accelerations [batch_size x n_dofs]
            include_gravity: when False, we assume gravity compensation is already taken care off

        Returns: forces to achieve desired accelerations

        """
        assert q.ndim == 2
        assert qd.ndim == 2
        assert qdd_des.ndim == 2
        assert q.shape[1] == self._n_dofs
        assert qd.shape[1] == self._n_dofs
        assert qdd_des.shape[1] == self._n_dofs
        
        inp_device = q.device
        q = q.to(**self.tensor_args)
        qd = qd.to(**self.tensor_args)
        qdd_des = qdd_des.to(**self.tensor_args)

        batch_size = qdd_des.shape[0]
        force = torch.zeros_like(qdd_des)

        # we set the current state of the robot
        self.update_kinematic_state(q, qd)

        # we set the acceleration of all controlled joints to the desired accelerations
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            self._bodies[idx].update_joint_acc(qdd_des[:, i].unsqueeze(1))

        # forces at the base are either 0, or gravity
        base_ang_acc = q.new_zeros((batch_size, 3))
        base_lin_acc = q.new_zeros((batch_size, 3))
        if include_gravity:
            base_lin_acc[:, 2] = 9.81 * torch.ones(batch_size, **self.tensor_args)

        # we propagate the base forces
        self.iterative_newton_euler(base_lin_acc, base_ang_acc)

        # we extract the relevant forces for all controlled joints
        for i in range(qdd_des.shape[1]):
            idx = self._controlled_joints[i]
            rot_axis = torch.zeros((batch_size, 3)).to(**self.tensor_args)
            rot_axis[:, 2] = torch.ones(batch_size).to(**self.tensor_args)
            force[:, i] += (
                self._bodies[idx].ang_force.unsqueeze(1) @ rot_axis.unsqueeze(2)
            ).squeeze()

        # we add forces to counteract damping
        damping_const = torch.zeros(1, self._n_dofs, **self.tensor_args)
        for i in range(self._n_dofs):
            idx = self._controlled_joints[i]
            damping_const[:, i] = self._bodies[idx].get_joint_damping_const()
        force += damping_const.repeat(batch_size, 1) * qd

        return force

    def compute_non_linear_effects(
        self, q: torch.Tensor, qd: torch.Tensor, include_gravity: Optional[bool] = True
    ) -> torch.Tensor:
        r"""

        Compute the non-linear effects (Coriolis, centrifugal, gravitational, and damping effects).

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        zero_qdd = q.new_zeros(q.shape)
        return self.compute_inverse_dynamics(q, qd, zero_qdd, include_gravity)

    def compute_lagrangian_inertia_matrix(
        self, q: torch.Tensor, include_gravity: Optional[bool] = True
    ) -> torch.Tensor:
        r"""

        Args:
            q: joint angles [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns:

        """
        assert q.shape[1] == self._n_dofs
        batch_size = q.shape[0]
        identity_tensor = torch.eye(q.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
        zero_qd = q.new_zeros(q.shape)
        zero_qdd = q.new_zeros(q.shape)
        if include_gravity:
            gravity_term = self.compute_inverse_dynamics(
                q, zero_qd, zero_qdd, include_gravity
            )
        else:
            gravity_term = q.new_zeros(q.shape)

        H = torch.stack(
            [
                (
                    self.compute_inverse_dynamics(
                        q, zero_qd, identity_tensor[:, :, j], include_gravity
                    )
                    - gravity_term
                )
                for j in range(self._n_dofs)
            ],
            dim=2,
        )
        return H

    def compute_forward_dynamics(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        f: torch.Tensor,
        include_gravity: Optional[bool] = True,
    ) -> torch.Tensor:
        r"""
        Computes next qdd by solving the Euler-Lagrange equation
        qdd = H^{-1} (F - Cv - G - damping_term)

        Args:
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]
            f: forces to be applied [batch_size x n_dofs]
            include_gravity: set to False if your robot has gravity compensation

        Returns: accelerations that are the result of applying forces f in state q, qd

        """

        nle = self.compute_non_linear_effects(
            q=q, qd=qd, include_gravity=include_gravity
        )
        inertia_mat = self.compute_lagrangian_inertia_matrix(
            q=q, include_gravity=include_gravity
        )

        # Solve H qdd = F - Cv - G - damping_term
        qdd = torch.solve(f.unsqueeze(2) - nle.unsqueeze(2), inertia_mat)[0].squeeze(2)

        return qdd

    def compute_endeffector_jacobian(
            self, q: torch.Tensor, qd:torch.Tensor, link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_link_jacobian(q, qd, link_name)
    
    def compute_link_jacobian(
            self, q: torch.Tensor, qd: torch.Tensor, link_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            link_name: name of link name for the jacobian
            q: joint angles [batch_size x n_dofs]

        Returns: linear and angular jacobian

        """
        inp_device = q.device
        q = q.to(**self.tensor_args)
        qd = qd.to(**self.tensor_args)
        
        self.compute_forward_kinematics(q, qd, link_name)

        e_pose = self._bodies[self._name_to_idx_map[link_name]].pose
        p_e = e_pose.translation()[0]

        lin_jac, ang_jac = torch.zeros([3, self._n_dofs],**self.tensor_args), torch.zeros(
            [3, self._n_dofs], **self.tensor_args
        )

        # any joints larger than this joint, will have 0 in the jacobian
        parent_name = self._urdf_model.get_name_of_parent_body(link_name)
        parent_joint_id = self._urdf_model.find_joint_of_body(parent_name)

        for i, idx in enumerate(self._controlled_joints):
            if (idx -1) > parent_joint_id:
                continue
            pose = self._bodies[idx].pose
            axis = self._bodies[idx].joint_axis
            axis_idx = int(torch.where(axis[0])[0])
            p_i, z_i = pose.translation()[0], pose.rotation()[0, :, axis_idx]
            lin_jac[:, i] = torch.cross(z_i, p_e - p_i)
            ang_jac[:, i] = z_i

        return lin_jac.to(inp_device), ang_jac.to(inp_device)


    def compute_fk_and_jacobian(
                self, q: torch.Tensor, qd:torch.Tensor, link_name: str
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            link_name: name of link name for the jacobian
            q: joint angles [batch_size x n_dofs]
            qd: joint velocities [batch_size x n_dofs]

        Returns: ee_pos, ee_rot and linear and angular jacobian

        """
        # st=time.time()
        inp_device = q.device
        q = q.to(**self.tensor_args)
        qd = qd.to(**self.tensor_args)

        batch_size = q.shape[0]
        # print("8", time.time()-st)
        # st = time.time()
        with profiler.record_function("robot_model/fk"):
            ee_pos, ee_rot = self.compute_forward_kinematics(q, qd, link_name)
        # print("9", time.time()-st)
        # st=time.time()
        # e_pose = self._bodies[self._name_to_idx_map[link_name]].pose
        # p_e = ee_pos[:, 0] #e_pose.translation()[0]
        lin_jac = torch.zeros([batch_size, 3, self._n_dofs], **self.tensor_args)
        ang_jac = torch.zeros([batch_size, 3, self._n_dofs], **self.tensor_args)
        # print("10", time.time()-st)
        # st=time.time()
        # any joints larger than this joint, will have 0 in the jacobian
        #parent_name = self._urdf_model.get_name_of_parent_body(link_name)
        parent_joint_id = self._urdf_model.find_joint_of_body(link_name)

        # do this as a tensor cross product:
        with profiler.record_function("robot_model/jac"):
            for i, idx in enumerate(self._controlled_joints):
                if (idx - 1) > parent_joint_id:
                    continue
                pose = self._bodies[idx].pose
                axis_idx = self._bodies[idx].axis_idx
                p_i = pose.translation()
                z_i = torch.index_select(pose.rotation(),-1, axis_idx).squeeze(-1)
                lin_jac[:, :, i] = torch.cross(z_i, ee_pos - p_i)
                ang_jac[:, :, i] = z_i
        # print("12", time.time()-st)
        # st=time.time()
        return ee_pos.to(inp_device), ee_rot.to(inp_device), lin_jac.to(inp_device), ang_jac.to(inp_device)

    def get_joint_limits(self) -> List[Dict[str, torch.Tensor]]:
        r"""

        Returns: list of joint limit dict, containing joint position, velocity and effort limits

        """
        limits = []
        for idx in self._controlled_joints:
            limits.append(self._bodies[idx].get_joint_limits())
        return limits

    def get_link_names(self) -> List[str]:
        r"""

        Returns: a list containing names for all links

        """

        link_names = []
        for i in range(len(self._bodies)):
            link_names.append(self._bodies[i].name)
        return link_names
    
    def print_link_names(self) -> None:
        r"""

        print the names of all links

        """
        for i in range(len(self._bodies)):
            print(self._bodies[i].name)

    def print_learnable_params(self) -> None:
        r"""

        print the name and value of all learnable parameters

        """
        for name, param in self.named_parameters():
            print(f"{name}: {param}")


class LearnableRigidBodyConfig:
    def __init__(self, learnable_links=[], learnable_kinematics_params=[], learnable_dynamics_params=[]):
        self.learnable_links = learnable_links
        self.learnable_kinematics_params = learnable_kinematics_params
        self.learnable_dynamics_params = learnable_dynamics_params


