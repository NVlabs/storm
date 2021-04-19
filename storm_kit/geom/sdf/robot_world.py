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
import torch

from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, rpy_angles_to_matrix, multiply_transform, transform_point
from ...geom.sdf.primitives import sdf_capsule_to_sphere
from .robot import RobotCapsuleCollision, RobotMeshCollision, RobotSphereCollision
from .world import WorldPointCloudCollision, WorldPrimitiveCollision


class RobotWorldCollision:
    def __init__(self, robot_collision, world_collision):
        self.tensor_args = robot_collision.tensor_args
        self.robot_coll = robot_collision
        self.world_coll = world_collision
    def update_robot_link_poses(self, links_pos, links_rot):
        self.robot_coll.update_robot_link_poses(links_pos, links_rot)
    def update_world_robot_pose(self, w_pos, w_rot):
        self.world_coll.update_reference_frame(w_pos, w_rot)
    
        
class RobotWorldCollisionCapsule(RobotWorldCollision):
    """Collision checking between capsule robot and sphere world"""    
    def __init__(self, robot_collision_params, world_collision_params, robot_batch_size=1,
                 world_batch_size=1,tensor_args={'device':"cpu", 'dtype':torch.float32}):
        robot_collision = RobotCapsuleCollision(robot_collision_params, tensor_args=tensor_args, batch_size=robot_batch_size)
        world_collision = WorldPrimitiveCollision(world_collision_params, tensor_args=tensor_args, batch_size=world_batch_size)
        super().__init__(robot_collision, world_collision)
        self.dist = None
    
    def get_signed_distance(self):

        link_capsules = self.robot_coll.get_robot_link_objs()
        world_spheres = self.world_coll.get_objs()
        if(self.dist is None or self.dist.shape[0] != link_capsules.shape[0]):
            self.dist = torch.empty((link_capsules.shape[0],link_capsules.shape[1],world_spheres.shape[1]), **self.tensor_args)
        dist = self.dist
        for i in range(world_spheres.shape[1]):
        
            sphere_pt = world_spheres[:,i,:3]
            sphere_radius = world_spheres[:,i,3]
            dist[:,:,i] = sdf_capsule_to_sphere(link_capsules[:,:,:3],
                                                link_capsules[:,:,3:6], link_capsules[:,:,6],
                                                sphere_pt, sphere_radius)
        return dist

class RobotWorldCollisionPrimitive(RobotWorldCollision):
    def __init__(self, robot_collision_params, world_collision_params, robot_batch_size=1,
                 world_batch_size=1,tensor_args={'device':"cpu", 'dtype':torch.float32},
                 bounds=None, grid_resolution=None):
        robot_collision = RobotSphereCollision(robot_collision_params, robot_batch_size, tensor_args)

        
        world_collision = WorldPrimitiveCollision(world_collision_params, tensor_args=tensor_args, batch_size=world_batch_size, bounds=bounds, grid_resolution=grid_resolution)
        self.robot_batch_size = robot_batch_size

        super().__init__(robot_collision, world_collision)
        self.dist = None

    def build_batch_features(self, batch_size, clone_pose=True, clone_points=True):
        self.batch_size = batch_size
        self.robot_coll.build_batch_features(clone_objs=clone_points, batch_size=batch_size)

    def check_robot_sphere_collisions(self, link_trans, link_rot):
        """get signed distance from stored grid [very fast]

        Args:
            link_trans (tensor): [b,3]
            link_rot (tensor): [b,3,3]

        Returns:
            tensor: signed distance [b,1]
        """        
        batch_size = link_trans.shape[0]
        # update link pose:
        if(self.robot_batch_size != batch_size):
            self.robot_batch_size = batch_size
            self.build_batch_features(self.robot_batch_size, clone_pose=True, clone_points=True)

        self.robot_coll.update_batch_robot_collision_objs(link_trans, link_rot)

        w_link_spheres = self.robot_coll.get_batch_robot_link_spheres()
        
                
        
        n_links = len(w_link_spheres)

        if(self.dist is None or self.dist.shape[0] != n_links):
            self.dist = torch.zeros((batch_size, n_links), **self.tensor_args)
        dist = self.dist

        for i in range(n_links):
            spheres = w_link_spheres[i]
            b, n, _ = spheres.shape
            spheres = spheres.view(b * n, 4)

            # compute distance between world objs and link spheres
            sdf = self.world_coll.check_pts_sdf(spheres[:,:3]) + spheres[:,3]
            sdf = sdf.view(b,n)
            dist[:,i] = torch.max(sdf, dim=-1)[0]
 
        return dist



        
    def get_robot_env_sdf(self, link_trans, link_rot):
        """Compute signed distance via analytic functino

        Args:
            link_trans (tensor): [b,3]
            link_rot (tensor): [b,3,3]

        Returns:
            tensor : signed distance [b,1]
        """        
        batch_size = link_trans.shape[0]
        # update link pose:
        if(self.robot_batch_size != batch_size):
            self.robot_batch_size = batch_size
            self.build_batch_features(self.robot_batch_size, clone_pose=True, clone_points=True)

        self.robot_coll.update_batch_robot_collision_objs(link_trans, link_rot)

        w_link_spheres = self.robot_coll.get_batch_robot_link_spheres()
        
                
        
        n_links = len(w_link_spheres)

        if(self.dist is None or self.dist.shape[0] != n_links):
            self.dist = torch.empty((batch_size, n_links), **self.tensor_args)
        dist = self.dist

        

        
        for i in range(n_links):
            spheres = w_link_spheres[i]
            #b, n, _ = spheres.shape
            #spheres = spheres.view(b * n, 4)

            # compute distance between world objs and link spheres
            d = self.world_coll.get_sphere_distance(spheres)
            
            dist[:,i] = torch.max(torch.max(d, dim=-1)[0], dim=-1)[0]

        return dist




class RobotWorldCollisionVoxel():
    '''
    This class can check collision between robot and sdf grid of camera pointcloud.
    '''
    def __init__(self, robot_collision_params, batch_size, label_map, bounds=None, grid_resolution=0.02,
                 tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):


        self.robot = RobotMeshCollision(robot_collision_params, batch_size, tensor_args)
        self.robot_sphere_model = RobotSphereCollision(robot_collision_params, batch_size, tensor_args)
        self._batch_table_link_rot = None
        self.batch_size = batch_size
        self.res = None
        self.tensor_args = tensor_args
        self.device = self.tensor_args['device']
        # bounds
        self.bounds = torch.tensor(bounds, **tensor_args)
        # label_map
        self.world = WorldPointCloudCollision(label_map, grid_resolution=grid_resolution, bounds=self.bounds,
                                              tensor_args=tensor_args)
        

    
    def set_world_transform(self, robot_table_trans, robot_R_table, robot_c_trans, robot_R_c):

        # For scene collisionnet, all points need to be in table frame:

        # camera -> robot frame, robot frame -> table frame
        self.robot_camera_transform = CoordinateTransform(trans=robot_c_trans,
                                                          rot=robot_R_c,
                                                          tensor_args=self.tensor_args)#.inverse()

        self.robot_table_transform = CoordinateTransform(trans=robot_table_trans,
                                                         rot=robot_R_table,
                                                         tensor_args=self.tensor_args)#.inverse()
        
        self.table_robot_transform = self.robot_table_transform.inverse()
        self.table_camera_transform = self.table_robot_transform.multiply_transform(self.robot_camera_transform)

        self.world.update_camera_transform(self.table_camera_transform.translation(),
                                           self.table_camera_transform.rotation())



    def set_scene(self, camera_pointcloud, scene_labels):
        """Loads scene pointcloud of the environment

        Args:
            camera_pointcloud: pointcloud of scene from camera [nx3]
            scene_labels: labels of pointcloud ["robot", "ground"]
        """
        
        self.world.update_world_pc(camera_pointcloud, scene_labels)
        self.world.update_world_sdf(self.world.scene_pc)
        


    def build_batch_features(self, batch_size, clone_pose=True, clone_points=True):
        self.batch_size = batch_size
        self.robot.build_batch_features(clone_pose=clone_pose, clone_points=clone_points, batch_size=batch_size)
        self.robot_sphere_model.build_batch_features(clone_objs=clone_points, batch_size=batch_size)

    def transform_to_table(self, link_trans, link_rot):
        table_link_rot, table_link_trans = multiply_transform(self.table_robot_transform.rotation(),
                                                              self.table_robot_transform.translation(),
                                                              link_rot,
                                                              link_trans)
        return table_link_trans, table_link_rot

    def check_robot_sphere_collisions(self, link_trans, link_rot):
        """Checks collision between robot spheres and pointcloud sdf grid

        Args:
            link_trans ([type]): [description]
            link_rot ([type]): [description]

        Returns:
            [type]: [description]
        """        
        table_link_trans, table_link_rot = self.transform_to_table(link_trans, link_rot)
        
        self.robot_sphere_model.update_batch_robot_collision_objs(table_link_trans, table_link_rot)

        if(self.res is None or self.res.shape[0] != link_trans.shape[0]):
            self.res = torch.zeros((link_trans.shape[0], link_trans.shape[1]), **self.tensor_args)
            
        res = self.res

        # get points:
        # pts: batch, n_links, n_pts, 3
        w_link_spheres = self.robot_sphere_model.get_batch_robot_link_spheres()

        
        
        n_links = len(w_link_spheres)
        
        for i in range(n_links):
            spheres = w_link_spheres[i]
            b, n, _ = spheres.shape
            spheres_arr = spheres.view(b * n, 4)
            sdf = self.world.check_pts_sdf(spheres_arr[:,:3])
            sdf = sdf + spheres_arr[:,3]
            sdf = sdf.view(b,n)
            # find largest sdf:
            res[:,i] = torch.max(sdf, dim=-1)[0]
            
        return res

    def check_robot_mesh_collisions(self, link_trans, link_rot, threshold=0.45,return_point_values=False):
        """Checks collision between robot represented by sampling points on mesh surface and pointcloud sdf.

        Args:
            link_trans ([type]): [description]
            link_rot ([type]): [description]
            threshold (float, optional): [description]. Defaults to 0.45.
            return_point_values (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """        
        # transform these pos, rot to table frame:
        table_link_trans, table_link_rot = self.transform_to_table(link_trans, link_rot)

        self.robot.update_batch_robot_collision_points(table_link_trans, table_link_rot)

        # link_trans: [batch, n_links]
        # result is [batch, n_links,n_pts]
        if(self.res is None or self.res.shape[0] != link_trans.shape[0]):
            self.res = torch.zeros((link_trans.shape[0], link_trans.shape[1]), **self.tensor_args)
            
        res = self.res

        # get points:
        # pts: batch, n_links, n_pts, 3
        pts = self.robot.get_batch_robot_link_points()
        batch_size = pts.shape[0]
        n_links = pts.shape[1]
        n_pts = pts.shape[2]
        
        
        pts = pts.view(batch_size * n_links * n_pts,3)
        
        
        # convert sdf values to be one per link:
        # reshape to get points per link:
        # find largest distance and make this as the sdf value

        
        
        sdf = self.world.check_pts_sdf(pts)
        if(return_point_values):
            res = sdf.view(batch_size, n_links, n_pts)
        else:
            res = torch.max(sdf.view(batch_size, n_links, n_pts), dim=-1)[0]
        
        # make out of bounds collision values as false
        
        # if binary:
        #res = self.world.scene_voxels[pt_idx]
        #res = res.view(batch_size, n_links, n_pts).sum(axis=-1)
        # batch, n_links:
        #res = res / float(n_pts)
        
        #res[res <= 5.0] = 0.0
        #res[res > 5.0] = 1.0
        

        return res



    
