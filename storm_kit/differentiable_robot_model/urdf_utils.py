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
import os
import torch
from urdf_parser_py.urdf import URDF


class URDFRobotModel(object):
    def __init__(self, urdf_path, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        self.robot = URDF.from_xml_file(urdf_path)
        self.urdf_path = urdf_path
        self._device = tensor_args['device']
        self.tensor_args = tensor_args

    def find_joint_of_body(self, body_name):
        for (i, joint) in enumerate(self.robot.joints):
            if joint.child == body_name:
                return i
        return -1
    def find_link_idx(self, link_name):
        for (i,link) in enumerate(self.robot.links):
            if(link.name == link_name):
                return i
        return -1

    def get_name_of_parent_body(self, link_name):
        jid = self.find_joint_of_body(link_name)
        joint = self.robot.joints[jid]
        return joint.parent

    def get_link_collision_mesh(self, link_name):
        idx = self.find_link_idx(link_name)
        link = self.robot.links[idx]
        mesh_fname = link.collision.geometry.filename
        mesh_origin = link.collision.origin
        origin_pose = torch.zeros(6).to(**self.tensor_args)
        if(mesh_origin is not None):
            origin_pose[:3] = mesh_origin.position
            origin_pose[3:6] = mesh_origin.rotation
            
        # join to urdf path
        mesh_fname = os.path.join(os.path.dirname(self.urdf_path), mesh_fname)
        return mesh_fname, origin_pose
    def get_body_parameters_from_urdf(self, i, link):
        body_params = {}
        body_params['joint_id'] = i
        body_params['link_name'] = link.name

        if i == 0:
            rot_angles = torch.zeros(3).to(**self.tensor_args)
            trans = torch.zeros(3).to(**self.tensor_args)
            joint_name = "base_joint"
            joint_type = "fixed"
            joint_limits = None
            joint_damping = None
            joint_axis = torch.zeros((1, 3), **self.tensor_args)
        else:
            link_name = link.name
            jid = self.find_joint_of_body(link_name)
            joint = self.robot.joints[jid]
            joint_name = joint.name
            # find joint that is the "child" of this body according to urdf

            rpy = torch.tensor(joint.origin.rotation, **self.tensor_args)
            rot_angles = torch.tensor([rpy[0], rpy[1], rpy[2]], **self.tensor_args)
            trans = torch.tensor(joint.origin.position, **self.tensor_args)
            joint_type = joint.type
            joint_limits = None
            joint_damping = torch.zeros(1, **self.tensor_args)
            joint_axis = torch.zeros((1, 3), **self.tensor_args)
            if joint_type != 'fixed':
                joint_limits = {'effort': joint.limit.effort,
                                'lower': joint.limit.lower,
                                'upper': joint.limit.upper,
                                'velocity': joint.limit.velocity}
                try:
                    joint_damping = torch.tensor(joint.dynamics.damping, **self.tensor_args)
                except AttributeError:
                    joint_damping = torch.tensor(0.0, **self.tensor_args)
                joint_axis = torch.tensor(joint.axis, **self.tensor_args).reshape(1, 3)

        body_params['rot_angles'] = rot_angles
        body_params['trans'] = trans
        body_params['joint_name'] = joint_name
        body_params['joint_type'] = joint_type
        body_params['joint_limits'] = joint_limits
        body_params['joint_damping'] = joint_damping
        body_params['joint_axis'] = joint_axis
        #body_params['collision_mesh'] = link.collision.geometry.mesh.filename
        if link.inertial is not None:
            mass = torch.tensor(link.inertial.mass, **self.tensor_args)
            com = torch.tensor(link.inertial.origin.position, **self.tensor_args).reshape((1, 3))

            inert_mat = torch.zeros((3, 3), **self.tensor_args)
            inert_mat[0, 0] = link.inertial.inertia.ixx
            inert_mat[0, 1] = link.inertial.inertia.ixy
            inert_mat[0, 2] = link.inertial.inertia.ixz
            inert_mat[1, 0] = link.inertial.inertia.ixy
            inert_mat[1, 1] = link.inertial.inertia.iyy
            inert_mat[1, 2] = link.inertial.inertia.iyz
            inert_mat[2, 0] = link.inertial.inertia.ixz
            inert_mat[2, 1] = link.inertial.inertia.iyz
            inert_mat[2, 2] = link.inertial.inertia.izz

            inert_mat = inert_mat.unsqueeze(0)
            body_params['mass'] = mass
            body_params['com'] = com
            body_params['inertia_mat'] = inert_mat
        else:
            body_params['mass'] = None
            body_params['com'] = None
            body_params['inertia_mat'] = None
            print("no dynamics information for link: {}".format(link.name))

        return body_params

