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
import yaml

import numpy as np
import torch
import trimesh

from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, rpy_angles_to_matrix, multiply_transform, transform_point
from ...differentiable_robot_model.urdf_utils import URDFRobotModel
from ...geom.geom_types import tensor_capsule, tensor_sphere
from ...util_file import join_path, get_mpc_configs_path
from ...geom.nn_model.robot_self_collision import RobotSelfCollisionNet
from typing import List

class RobotCapsuleCollision:
    """ This class holds a batched collision model where the robot is represented as capsules [one per link]
    """
    def __init__(self, robot_collision_params, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        # read capsules
        self.batch_size = batch_size
        self.tensor_args = tensor_args
        # keep track of their pose in world frame
        self._link_capsules = None
        self.link_capsules = None
        self.l_T_c = CoordinateTransform(tensor_args=self.tensor_args)
        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)
    
    def load_robot_collision_model(self, robot_collision_params):
        
        robot_links = robot_collision_params['link_objs']

        # we store as [Batch, n_link, 7]
        self._link_capsules = torch.empty((self.batch_size, len(robot_links), 7), **self.tensor_args)
        for j_idx, j in enumerate(robot_links):
            pose = robot_links[j]['pose_offset']
            # create a transform from pose offset:
            trans = torch.tensor(pose[0:3], **self.tensor_args).unsqueeze(0)
            rpy = torch.tensor(pose[3:], **self.tensor_args).unsqueeze(0)
            # rotation matrix from euler:
            rot = rpy_angles_to_matrix(rpy)
            
            
            l_T_c = CoordinateTransform(trans=trans, rot=rot, tensor_args=self.tensor_args)
            
            r = robot_links[j]['radius']

            # transform base, tip by pose_offset:
            
            base = torch.tensor(robot_links[j]['base'], **self.tensor_args).unsqueeze(0)
            
            tip = torch.tensor(robot_links[j]['tip'], **self.tensor_args).unsqueeze(0)
            base = l_T_c.transform_point(base)
            tip = l_T_c.transform_point(tip)
            self._link_capsules[:, j_idx,:] = tensor_capsule(base, tip, r, tensor_args=self.tensor_args).unsqueeze(0).repeat(self.batch_size, 1)
        #print(self.link_capsules)
        self.link_capsules = self._link_capsules.clone()
    
    def update_robot_link_poses(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [batch, n_links , 3]
           link_rot: [batch, n_links , 3 , 3]
        """
        if(links_pos.shape[0] != self.batch_size):
            self.batch_size = links_pos.shape[0]
            self.load_robot_collision_model(self.robot_collision_params)
        
        # This contains coordinate tranforms as [batch_size * n_links ]
        self.l_T_c.set_translation(links_pos)
        self.l_T_c.set_rotation(links_rot)
        
        # Update tranform of link points:
        self.link_capsules[:,:,:3] = self.l_T_c.transform_point(self._link_capsules[:,:,:3])
        self.link_capsules[:,:,3:6] = self.l_T_c.transform_point(self._link_capsules[:,:,3:6])
        
       
    def get_robot_link_objs(self):
        # return capsule spheres in world frame
        
        return self.link_capsules
    
    def get_robot_link_points(self):
        
        raise NotImplementedError


class RobotMeshCollision: 
    """ This class holds a batched collision model with meshes loaded using trimesh. 
    Points are sampled from the mesh which can be used for collision checking.
    """
    def __init__(self, robot_collision_params, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        # read capsules
        self.batch_size = batch_size
        self.tensor_args = tensor_args
        # keep track of their pose in world frame
        
        #self.link_points = None
        self._batch_link_points = None
        self._link_points = None
        self._link_collision_trans = None
        self._link_collision_rot = None
        self._batch_link_collision_trans = None
        self._batch_link_collision_rot = None

        self._robot_collision_trans = None
        self._robot_collision_rot = None

        self._batch_robot_collision_trans = None
        self._batch_robot_collision_rot = None

        self.w_link_points = None
        self.w_batch_link_points = None
        
        self.l_T_c = CoordinateTransform(tensor_args=self.tensor_args)
        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)
        
        
    def load_robot_collision_model(self, robot_collision_params):
        
        robot_links = robot_collision_params['link_objs']
        robot_urdf = robot_collision_params['urdf']
        n_pts = robot_collision_params['sample_points']

        # read robot urdf
        robot_urdf = URDFRobotModel(robot_urdf, self.tensor_args)
        
        # read meshes, sample points and store
        
        
        # we store as [n_link, 7]
        self._link_points = torch.empty((len(robot_links), n_pts, 3), **self.tensor_args)
        self._link_collision_trans = torch.empty((len(robot_links), 3), **self.tensor_args)
        self._link_collision_rot = torch.empty((len(robot_links), 3, 3), **self.tensor_args)
        
        for j_idx, j in enumerate(robot_links):
            # read mesh
            mesh_fname, mesh_origin = robot_urdf.get_link_collision_mesh(j)
            
            # sample points
            mesh = trimesh.load_mesh(mesh_fname)
            mesh_centroid = mesh.centroid 
            mesh.vertices = mesh.vertices - mesh_centroid #* 0.0
            points = torch.tensor(trimesh.sample.sample_surface(mesh, n_pts)[0], **self.tensor_args)
            #points = torch.tensor(trimesh.sample.volume_mesh(mesh, n_pts), **self.tensor_args)

            
            # transform points from mesh frame to link frame:
            pose = mesh_origin
            # create a transform from pose offset:
            trans = torch.tensor(pose[0:3], **self.tensor_args).unsqueeze(0)
            rpy = torch.tensor(pose[3:], **self.tensor_args).unsqueeze(0)
            
            # rotation matrix from euler:
            rot = rpy_angles_to_matrix(rpy)
            mesh_cent = torch.as_tensor(mesh_centroid, **self.tensor_args).unsqueeze(0)#.unsqueeze(0)


            trans = trans + (mesh_cent @ rot.transpose(-1,-2))

            l_T_c = CoordinateTransform(trans=trans, rot=rot, tensor_args=self.tensor_args)

            
                        
            #points = l_T_c.transform_point(points)

            # store points

            self._link_points[j_idx, :,:] = points

            # store tranform:
            self._link_collision_rot[j_idx,:,:] = l_T_c.rotation().squeeze(0)
            self._link_collision_trans[j_idx,:] = l_T_c.translation().squeeze(0)
        
    def build_batch_features(self, clone_points=False, clone_pose=True, batch_size=None):
        if(batch_size is not None):
            self.batch_size = batch_size
        if(clone_points):
            
            self._batch_link_points = self._link_points.unsqueeze(0).repeat(self.batch_size, 1, 1,1).clone()
        if(clone_pose):
            self._batch_link_collision_trans = self._link_collision_trans.unsqueeze(0).repeat(self.batch_size, 1, 1).clone()
            self._batch_link_collision_rot = self._link_collision_rot.unsqueeze(0).repeat(self.batch_size, 1, 1, 1).clone()
    def update_batch_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [batch, n_links , 3] 
           link_rot: [batch, n_links , 3 , 3] 

        """
        
        (self._batch_robot_collision_rot,
         self._batch_robot_collision_trans) = multiply_transform(links_rot, links_pos,
                                                                 self._batch_link_collision_rot,
                                                                 self._batch_link_collision_trans)
        return True
        
    def update_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [n_links, 3]
           link_rot: [n_links, 3, 3]

        """

        self._robot_collision_rot, self._robot_collision_trans = (
            multiply_transform(links_rot, links_pos,
                               self._link_collision_rot,
                               self._link_collision_trans,
                               ))
        
        
        return True

    def update_robot_collision_points(self, links_pos, links_rot):

        self.update_robot_collision_pose(links_pos, links_rot)

        self.w_link_points = transform_point(self._link_points, self._robot_collision_rot, self._robot_collision_trans)
        

    def update_batch_robot_collision_points(self, links_pos, links_rot):
        self.update_batch_robot_collision_pose(links_pos, links_rot)
        self.w_batch_link_points = transform_point(self._batch_link_points,
                                                   self._batch_robot_collision_rot,
                                                   self._batch_robot_collision_trans.unsqueeze(-2))
    def get_robot_link_objs(self):
        raise NotImplementedError
    def get_batch_robot_link_points(self):
        return self.w_batch_link_points
    def get_robot_link_points(self):
        return self.w_link_points
    def get_link_points(self):
        return self._link_points

class RobotSphereCollision:
    """ This class holds a batched collision model where the robot is represented as spheres.
        All points are stored in the world reference frame, obtained by using update_pose calls.
    """
    
    def __init__(self, robot_collision_params, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        """ Initialize with robot collision parameters, look at franka_reacher.py for an example.

        Args:
            robot_collision_params (Dict): collision model parameters
            batch_size (int, optional): Batch size of parallel sdf computation. Defaults to 1.
            tensor_args (dict, optional): compute device and data type. Defaults to {'device':"cpu", 'dtype':torch.float32}.
        """        
        # read capsules
        self.batch_size = batch_size
        self.tensor_args = tensor_args

        
        # keep track of their pose in world frame
        
        #self.link_points = None
        self._link_spheres = None
        self._batch_link_spheres = None

        self._link_points = None
        self._link_collision_trans = None
        self._link_collision_rot = None
        self._batch_link_collision_trans = None
        self._batch_link_collision_rot = None

        self._robot_collision_trans = None
        self._robot_collision_rot = None

        self._batch_robot_collision_trans = None
        self._batch_robot_collision_rot = None

        self.w_link_points = None
        self.w_batch_link_spheres = None
        
        self.l_T_c = CoordinateTransform(tensor_args=self.tensor_args)
        self.robot_collision_params = robot_collision_params
        self.load_robot_collision_model(robot_collision_params)
        
        self.dist = None

        # load nn collision model:
        dof = robot_collision_params['dof']
        
        self.robot_nn = RobotSelfCollisionNet(n_joints=dof)
        self.robot_nn.load_weights(robot_collision_params['self_collision_weights'], tensor_args)
    
    def load_robot_collision_model(self, robot_collision_params):
        """Load robot collision model, called from constructor

        Args:
            robot_collision_params (Dict): loaded from yml file
        """        
        robot_links = robot_collision_params['link_objs']

        # load collision file:
        print(robot_collision_params)
        coll_yml = join_path(get_mpc_configs_path(), robot_collision_params['collision_spheres'])
        with open(coll_yml) as file:
            coll_params = yaml.load(file, Loader=yaml.FullLoader)

        coll_params = coll_params['collision_spheres']

        self._link_spheres = []

        
        # we store as [n_link, 7]
        self._link_collision_trans = torch.empty((len(robot_links), 3), **self.tensor_args)
        self._link_collision_rot = torch.empty((len(robot_links), 3, 3), **self.tensor_args)

        for j_idx, j in enumerate(robot_links):
            
            n_spheres = len(coll_params[j])
            link_spheres = torch.zeros((n_spheres, 4), **self.tensor_args)

            for i in range(n_spheres):
                
                link_spheres[i,:] = tensor_sphere(coll_params[j][i]['center'], coll_params[j][i]['radius'], tensor_args=self.tensor_args, tensor=link_spheres[i,:])
            self._link_spheres.append(link_spheres)
            
        self._w_link_spheres = self._link_spheres
    def build_batch_features(self, clone_objs=False, clone_pose=True, batch_size=None):
        """clones poses/object instances for computing across batch. Use this once per batch size change to avoid re-initialization over repeated calls.

        Args:
            clone_objs (bool, optional): clones objects. Defaults to False.
            clone_pose (bool, optional): clones pose. Defaults to True.
            batch_size ([type], optional): batch_size to clone. Defaults to None.
        """        
        if(batch_size is not None):
            self.batch_size = batch_size
        if(clone_objs):
            self._batch_link_spheres = []
            for i in range(len(self._link_spheres)):
                self._batch_link_spheres.append(self._link_spheres[i].unsqueeze(0).repeat(self.batch_size, 1, 1).clone())
        self.w_batch_link_spheres = copy.deepcopy(self._batch_link_spheres)
        
    def update_batch_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [batch, n_links , 3] 
           link_rot: [batch, n_links , 3 , 3] 

        """
        '''
        (self._batch_robot_collision_rot,
         self._batch_robot_collision_trans) = multiply_transform(links_rot, links_pos,
                                                                 self._batch_link_collision_rot,
                                                                 self._batch_link_collision_trans)
        '''
        return True
        
    def update_robot_collision_pose(self, links_pos, links_rot):
        """
        Update link collision poses
        Args:
           link_pos: [n_links, 3]
           link_rot: [n_links, 3, 3]

        """
        '''
        self._robot_collision_rot, self._robot_collision_trans = (
            multiply_transform(links_rot, links_pos,
                               self._link_collision_rot,
                               self._link_collision_trans,
                               ))
        
        '''
        return True

    def update_robot_collision_objs(self, links_pos, links_rot):
        '''update pose of link spheres

        Args:
        links_pos: nx3
        links_rot: nx3x3
        '''
        
        # transform link points:
        for i in range(len(self._link_spheres)):
            self._w_link_spheres[i][:,:3] = transform_point(self._link_spheres[:,:3], links_rot[i,:,:], links_pos[i,:,:])
        

    def update_batch_robot_collision_objs(self, links_pos, links_rot):
        '''update pose of link spheres

        Args:
        links_pos: bxnx3
        links_rot: bxnx3x3
        '''
        
        b, n, _ = links_pos.shape
        
        for i in range(n):
            link_pts = self._batch_link_spheres[i][:,:,:3]
            self.w_batch_link_spheres[i][:,:,:3] = transform_point(self._batch_link_spheres[i][:,:,:3], links_rot[:,i,:,:], links_pos[:,i,:].unsqueeze(-2))

    def check_self_collisions_nn(self, q):
        """compute signed distance using NN, uses an instance of :class:`.nn_model.robot_self_collision.RobotSelfCollisionNet`

        Args:
            q ([type]): [description]

        Returns:
            [type]: [description]
        """        
        dist = self.robot_nn.compute_signed_distance(q)
        return dist


    def check_self_collisions(self, link_trans, link_rot):
        """Analytic method to compute signed distance between links. This is used to train the NN method :func:`check_self_collisions_nn` amd is not used directly as it is slower.

        Args:
            link_trans ([tensor]): link translation as batch [b,3]
            link_rot ([type]): link rotation as batch [b,3,3]

        Returns:
            [tensor]: signed distance [b,1]
        """        
        n_links = len(self.w_batch_link_spheres)
        b, _, _ = link_trans.shape
        if self.dist is None or b != self.dist.shape[0]:
            self.update_batch_robot_collision_objs(link_trans, link_rot)
            self.dist = torch.zeros((b,n_links,n_links), **self.tensor_args) - 100.0
        dist = self.dist
        dist = find_link_distance(self.w_batch_link_spheres, dist)
        
        return dist
    def get_robot_link_objs(self):
        raise NotImplementedError

    def get_batch_robot_link_spheres(self):
        return self.w_batch_link_spheres

    def get_robot_link_points(self):
        return self.w_link_points

    def get_link_points(self):
        return self._link_points


@torch.jit.script
def compute_spheres_distance(spheres_1, spheres_2):
    
    b, n, _ = spheres_1.shape
    b_l, n_l, _ = spheres_2.shape
    
    #dist = torch.zeros((b, n), device=spheres_1.device,
    #                   dtype=spheres_2.dtype)
    
    


    j = 0
    link_sphere_pts = spheres_1[:,j,:]
    link_sphere_pts = link_sphere_pts.unsqueeze(1)
    # find closest distance to other link spheres:
    
    
    
    #print(l_spheres.shape, link_sphere_pts.shape)
    s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
    s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist
    max_dist = torch.max(s_dist, dim=-1)[0]
    
    
    for j in range(1,n):
        link_sphere_pts = spheres_1[:,j,:]
        link_sphere_pts = link_sphere_pts.unsqueeze(1)
        # find closest distance to other link spheres:
        s_dist = torch.norm(spheres_2[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
        s_dist = spheres_2[:,:,3] + link_sphere_pts[:,:,3] - s_dist
        s_dist = torch.max(s_dist, dim=-1)[0]
        max_dist = torch.maximum(max_dist, s_dist)
        
    dist = max_dist #torch.max(dist,dim=-1)[0]
    return dist

@torch.jit.script
def find_closest_distance(link_idx, links_sphere_list):
    # type: (int, List[Tensor]) -> Tensor
    """closet distance computed via iteration between sphere sets.

    Args:
        link_idx ([type]): [description]
        links_sphere_list ([type]): [description]

    Returns:
        [type]: [description]
    """

    spheres = links_sphere_list[link_idx]
    b, n, _ = spheres.shape
    #spheres = spheres.view(b * n, 4)
    #link_pts = spheres[:,:,:3]
    #link_dist = torch.zeros((b,len(links_sphere_list)), **self.tensor_args)
    dist = torch.zeros((b,len(links_sphere_list), n), device=spheres.device,
                       dtype=spheres.dtype)
    for j in range(n):
        # for every sphere in current link
        link_sphere_pts = spheres[:,j,:]
        link_sphere_pts = link_sphere_pts.unsqueeze(1)
        # find closest distance to other link spheres:
        
        for i in range(len(links_sphere_list)):
            if(i == link_idx or i==link_idx-1 or i==link_idx+1):
                dist[:,i,j] = -100.0
                continue
            # transform link_idx spheres to current link frame:
            # given a link and another link, find closest distance between them:
            l_spheres = links_sphere_list[i]
        
            b_l, n_l, _ = l_spheres.shape
            
            #print(l_spheres.shape, link_sphere_pts.shape)
            s_dist = torch.norm(l_spheres[:,:,:3] - link_sphere_pts[:,:,:3], dim=-1)
            s_dist = l_spheres[:,:,3] + link_sphere_pts[:,:,3] - s_dist 

            # dist: b, n_l -> b
            dist[:,i,j] = torch.max(s_dist, dim=-1)[0]
    link_dist = torch.max(dist,dim=-1)[0]
    return link_dist

@torch.jit.script
def find_link_distance(links_sphere_list, dist):
    # type: (List[Tensor], Tensor) -> Tensor
    futures : List[torch.jit.Future[torch.Tensor]] = []

    b, n, _ = links_sphere_list[0].shape
    spheres = links_sphere_list[0]
    n_links = len(links_sphere_list)
    dist *= 0.0
    dist -= 100.0
    #dist = torch.zeros((b,n_links,n_links), device=spheres.device,
    #                   dtype=spheres.dtype) - 100.0

    for i in range(n_links):
        # for every link, compute the distance to the other links:
        current_spheres = links_sphere_list[i]
        for j in range(i + 2, n_links):
            compute_spheres = links_sphere_list[j]

            # find the distance between the two links:
            d = torch.jit.fork(compute_spheres_distance, current_spheres, compute_spheres)
            futures.append(d)


    k = 0
    for i in range(n_links):
        # for every link, compute the distance to the other links:
        for j in range(i + 2, n_links):
            d = torch.jit.wait(futures[k])
            dist[:,i,j] = d
            dist[:,j,i] = d
            k += 1
    link_dist = torch.max(dist,dim=-1)[0]
    return link_dist
