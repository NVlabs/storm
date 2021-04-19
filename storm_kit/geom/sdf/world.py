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

import cv2
import numpy as np
import trimesh
from trimesh.voxel.creation import voxelize
import torch
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt


from ...differentiable_robot_model.coordinate_transform import CoordinateTransform, rpy_angles_to_matrix, transform_point
from ...geom.geom_types import tensor_capsule, tensor_sphere, tensor_cube
from ...geom.sdf.primitives import get_pt_primitive_distance, get_sphere_primitive_distance

class WorldCollision:
    def __init__(self, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        # read capsules
        self.batch_size = batch_size
        self.tensor_args = tensor_args
        
    def load_collision_model(self):
        raise NotImplementedError


class WorldGridCollision(WorldCollision):
    """This template class can be used to build a sdf grid using a signed distance function for fast lookup.
    """    
    def __init__(self, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32},bounds=None, grid_resolution=0.05):
        super().__init__(batch_size, tensor_args)
        self.bounds = torch.as_tensor(bounds, **tensor_args)
        self.grid_resolution = grid_resolution
        self.pitch = self.grid_resolution
        self.scene_sdf = None
        self.scene_sdf_matrix = None

    def update_world_sdf(self):
        sdf_grid = self._compute_sdfgrid()
        self.scene_sdf_matrix = sdf_grid
        self.scene_sdf = sdf_grid.flatten()

    def get_signed_distance(self, pts):
        """This needs to be implemented

        Args:
            pts (tensor): [b,3]

        Raises:
            NotImplementedError: Raises error as this function needs to be implemented in a child class

        Returns:
            tensor: distance [b,1]
        """        
        raise NotImplementedError
        dist = None
        return dist
    def view_sdf_grid(self, sdf_grid):
        ax = plt.axes(projection='3d')
        ind_matrix = [[x,y,z] for x in range(sdf_grid.shape[0]) for y in range(sdf_grid.shape[1]) for z in range(sdf_grid.shape[2])]
        ind_matrix = np.matrix(ind_matrix)
        xdata = ind_matrix[:,0]
        ydata = ind_matrix[:, 1]
        zdata = ind_matrix[:,2]
        
        c_data = torch.flatten(sdf_grid).cpu().numpy()
        ax.scatter3D(xdata, ydata, zdata, c=c_data, cmap='coolwarm')#, vmin=-0.1, vmax=0.1)
        plt.show()

    def build_transform_matrices(self, bounds, pitch):
        '''
        Args:
        bounds: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        pitch: float
        '''
        origin = bounds[0]
        # build pt to idx
        # given a point (x,y,z), convert to an index assuming pt is within bounds
        rot = torch.eye(3) * (1.0 / pitch)
        trans = torch.as_tensor(origin)
        self.proj_pt_idx = CoordinateTransform(trans=-1.0 * trans * (1.0 / pitch), rot=rot, tensor_args=self.tensor_args)
        
        
        # build idx to pt
        rot = torch.eye(3) * (pitch)
        #trans = torch
        self.proj_idx_pt = CoordinateTransform(trans=1.0 * trans, rot=rot, tensor_args=self.tensor_args)

    def _compute_sdfgrid(self):
        # voxel grid has different bounds

        # create a sdf grid for scene bounds and pitch:
        sdf_grid_dims = torch.Size(((self.bounds[1] - self.bounds[0]) / self.grid_resolution).int())
        self.build_transform_matrices(self.bounds, self.grid_resolution)

        sdf_grid = torch.zeros(sdf_grid_dims, **self.tensor_args)
        self.num_voxels = torch.tensor([sdf_grid.shape[0], sdf_grid.shape[1],
                                        sdf_grid.shape[2]],
                                       **self.tensor_args)

        # get indices

        ind_matrix = [[x,y,z] for x in range(sdf_grid.shape[0]) for y in range(sdf_grid.shape[1]) for z in range(sdf_grid.shape[2])]
        
        ind_matrix = torch.tensor(ind_matrix, **self.tensor_args)
        self.ind_matrix = ind_matrix
        pt_matrix = self.proj_idx_pt.transform_point(ind_matrix)

        dist_matrix = torch.flatten(self.get_signed_distance(pt_matrix))
        self.dist_matrix = dist_matrix
        
        # get corresponding points
        # Get closest distance from indice points to points in pointcloud:
        for i in range(sdf_grid.shape[0]):
            for j in range(sdf_grid.shape[1]):
                for k in range(sdf_grid.shape[2]):
                    sdf_grid[i,j,k] = dist_matrix[i * (sdf_grid.shape[1] * sdf_grid.shape[2])+ j * (sdf_grid.shape[2]) + k]
                    
                    
        return sdf_grid
    
    def check_pts_sdf(self, pts):
        '''
        finds the signed distance for the points from the stored grid
        Args:
        pts: [n,3]
        '''
        #print(self.bounds, self.pitch)
        in_bounds = (pts > self.bounds[0] + self.pitch).all(dim=-1)
        in_bounds &= (pts < self.bounds[1] - self.pitch).all(dim=-1)

        #pts[~in_bounds] = self.bounds[0]
        pt_idx = self.voxel_inds(pts)
        
        pt_idx[~in_bounds] = 0
        
        # remove points outside voxel region:
        # check collisions:
        # batch * n_links, n_pts
        # get sdf from scene voxels:
        # negative distance is outside mesh:
        sdf = self.scene_sdf[pt_idx]
        sdf[~in_bounds] = -10.0
        return sdf

    def voxel_inds(self, pt, scale=1):

        pt = self.proj_pt_idx.transform_point(pt)

        pt = (pt).to(dtype=torch.int64)

        
        ind_pt = (pt[...,0]) * (self.num_voxels[1] * self.num_voxels[2]) + pt[...,1] * self.num_voxels[2] + pt[...,2]
        
        ind_pt = ind_pt.to(dtype=torch.int64)
        
        self.ind_pt = ind_pt
        
        return self.ind_pt

class WorldPrimitiveCollision(WorldGridCollision):
    """ This class holds a batched collision model
    """
    def __init__(self, world_collision_params, batch_size=1, tensor_args={'device':"cpu", 'dtype':torch.float32}, bounds=None, grid_resolution=0.05):
        super().__init__(batch_size, tensor_args, bounds, grid_resolution)
        self._world_spheres = None
        self._world_cubes = None
        
        self.n_objs = 0

        self.l_T_c = CoordinateTransform(tensor_args=self.tensor_args)
        self.load_collision_model(world_collision_params)
        self.dist = torch.zeros((1,1,1), **self.tensor_args)

        if(bounds is not None):
            self.update_world_sdf()

    def load_collision_model(self, world_collision_params):
        
        world_objs = world_collision_params['coll_objs']
        sphere_objs = world_objs['sphere']
        if('cube' in world_objs):
            cube_objs = world_objs['cube']
        else:
            cube_objs = []
            

        # we store as [Batch, n_link, 7]
        self._world_spheres = torch.empty((self.batch_size, len(sphere_objs), 4), **self.tensor_args)
        self._world_cubes = []

        for j_idx, j in enumerate(sphere_objs):
            position = sphere_objs[j]['position']
            
            r = sphere_objs[j]['radius']
            
            self._world_spheres[:, j_idx,:] = tensor_sphere(position, r, tensor_args=self.tensor_args).unsqueeze(0).repeat(self.batch_size, 1)
        
        for j_idx, j in enumerate(cube_objs):
            pose = cube_objs[j]['pose']
            pose_fixed = [pose[0], pose[1], pose[2], pose[6], pose[3], pose[4], pose[5]]
            dims = cube_objs[j]['dims']
            cube = tensor_cube(pose_fixed, dims, tensor_args=self.tensor_args)
            self._world_cubes.append(cube)

            
            
            
        self.n_objs = self._world_spheres.shape[1] + len(self._world_cubes)
        
    
    def update_obj_poses(self, objs_pos, objs_rot):
        """
        Update collision object poses
        Args:
           link_pos: [batch, n_links , 3]
           link_rot: [batch, n_links , 3 , 3]
        """
        
        # This contains coordinate tranforms as [batch_size * n_links ]
        
        self.l_T_c.set_translation(objs_pos)
        self.l_T_c.set_rotation(objs_rot)
        
        # Update tranform of link points:
        self._world_spheres[:,:,:3] = self.l_T_c.transform_point(self._world_spheres[:,:,:3])

        # TODO for cube:
        

    def update_reference_frame(self, r_pos, r_rot):
        """
        Update world collision poses
        Args:
           link_pos: [batch, n_links , 3]
           link_rot: [batch, n_links , 3 , 3]
        """
        
        # This contains coordinate tranforms as [batch_size * n_links ]
        
        self.l_T_c.set_translation(r_pos)
        self.l_T_c.set_rotation(r_rot)

        for i in range(self._world_spheres.shape[1]):
            self._world_spheres[:,i,:3] = self.l_T_c.transform_point(self._world_spheres[:,i,:3])
            
    def get_sphere_objs(self):
        # return capsule spheres in world frame
        return self._world_spheres

    def get_cube_objs(self):
        # return capsule spheres in world frame
        return self._world_cubes

    def get_sphere_distance(self, w_sphere):
        """
        Computes the signed distance via analytic function
        Args:
        tensor_sphere: b, n, 4
        """
        dist = torch.zeros((w_sphere.shape[0], self.n_objs, w_sphere.shape[1]), **self.tensor_args)
        dist = get_sphere_primitive_distance(w_sphere, self._world_spheres, self._world_cubes)
        return dist

    def get_pt_distance(self, w_pts):
        """
        Args:
        w_pts: b, n, 3
        """
        if(len(w_pts.shape) == 2):
            w_pts = w_pts.view(w_pts.shape[0], 1, 3)
        if(self.dist.shape[0] != w_pts.shape[0] or self.dist.shape[1] != self.n_objs or self.dist_shape[2] != w_pts.shape[1]):
            self.dist = torch.zeros((w_pts.shape[0], self.n_objs, w_pts.shape[1]), **self.tensor_args)
        dist = self.dist
        dist = get_pt_primitive_distance(w_pts, self._world_spheres, self._world_cubes, dist)
        return dist

    def get_signed_distance(self, w_pts):
        dist = torch.max(self.get_pt_distance(w_pts), dim=1)[0]
        return dist
    

class WorldPointCloudCollision(WorldGridCollision):
    def __init__(self, label_map, bounds, grid_resolution=0.02, tensor_args={'device':"cpu", 'dtype':torch.float32}, batch_size=1):
        super().__init__(batch_size, tensor_args, bounds, grid_resolution)

        self.label_map = label_map
        self.camera_transform = None
        self.scene_pc = None
        self.scale = 1
        self._flat_tensor = None
        self.proj_pt_idx = None
        self.trimesh_scene_voxel = None
        self.ind_pt = None

    def update_camera_transform(self, w_c_trans, w_R_c):
        self.camera_transform = CoordinateTransform(trans=w_c_trans,
                                                    rot=w_R_c,
                                                    tensor_args=self.tensor_args)#.inverse()
        
    def update_world_pc(self, pointcloud, seg_labels):
        # Fill pointcloud in tensor:
        orig_scene_pc = torch.as_tensor(pointcloud, **self.tensor_args)
        # filter seg labels:
        scene_labels = torch.as_tensor(seg_labels.astype(int), device=self.tensor_args['device'])

        scene_pc_mask = torch.logical_and(scene_labels != self.label_map["robot"],
                                          scene_labels != self.label_map["ground"])

        vis_mask = scene_pc_mask.flatten()
        scene_pc = orig_scene_pc[vis_mask]

        scene_pc = self.camera_transform.transform_point(scene_pc)
        min_bound = self.bounds[0]
        max_bound = self.bounds[1]
        mask_bound = torch.logical_and(torch.all(scene_pc > min_bound, dim=-1), torch.all(scene_pc < max_bound, dim=-1))

        
        scene_pc = scene_pc[mask_bound]
        self.scene_pc = scene_pc
        
        
    def update_world_voxel(self, scene_pc):
        
        # Fill pointcloud in tensor:

        # marching cubes:


        scene_pc = trimesh.PointCloud(scene_pc.cpu().numpy())

        pitch = self.grid_resolution #scene_pc.extents.max() / self.voxel_scale

        self.pitch = pitch
        scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(scene_pc.vertices, pitch=pitch)
        self.trimesh_scene_mesh = scene_mesh
        scene_voxel = voxelize(scene_mesh, pitch=pitch,method='subdivide')
        self.trimesh_scene_voxel = scene_voxel
        scene_voxel_tensor = torch.tensor(scene_voxel.matrix, **self.tensor_args)
        self.scene_voxel_matrix = scene_voxel_tensor
        # this pushes the sdf to be only available close to the voxel grid
        self.trimesh_bounds = torch.as_tensor(self.trimesh_scene_voxel.bounds, **self.tensor_args)

        
    def update_world_sdf(self, scene_pc):
        self.update_world_voxel(scene_pc)
        
        sdf_grid = self._compute_sdfgrid()


        self.scene_sdf_matrix = sdf_grid
        self.scene_sdf = sdf_grid.flatten()
        
    def _update_trimesh_projection(self):
        scene_voxel = self.trimesh_scene_voxel
        scene_voxel_tensor = self.scene_voxel_matrix
        ind_matrix = torch.tensor(scene_voxel._transform.inverse_matrix, **self.tensor_args).unsqueeze(0)
        pt_matrix = torch.tensor(scene_voxel._transform.matrix, **self.tensor_args).unsqueeze(0)

        self.proj_pt_idx = CoordinateTransform(trans=ind_matrix[:,:3,3], rot=ind_matrix[:,:3,:3], tensor_args=self.tensor_args)

        self.proj_idx_pt = CoordinateTransform(trans=pt_matrix[:,:3,3], rot=pt_matrix[:,:3,:3], tensor_args=self.tensor_args)

        num_voxels = torch.tensor([scene_voxel_tensor.shape[0], scene_voxel_tensor.shape[1],
                                   scene_voxel_tensor.shape[2]],
                                  **self.tensor_args)

        
        flat_tensor = torch.tensor([num_voxels[1:].prod(),# // (self.scale ** 2),
                                    num_voxels[2],# // self.scale,
                                    1], device=self.tensor_args['device'], dtype=torch.int64)
        self.scene_voxels = torch.flatten(scene_voxel_tensor)
        
        
        self.num_voxels = num_voxels
        self._flat_tensor = flat_tensor
        
    def get_signed_distance(self, pts):
        dist = trimesh.proximity.signed_distance(self.trimesh_scene_mesh, pts.cpu().numpy())
        return dist
    
    def get_scene_pts_from_voxelgrid(self):

        pts = self.trimesh_scene_voxel.points
        return pts

    def get_scene_mesh_from_voxelgrid(self):
        
        mesh = self.trimesh_scene_voxel.as_boxes() # marching_cubes
        return mesh
        
class WorldImageCollision(WorldCollision):
    def __init__(self, bounds, tensor_args={'device':"cpu", 'dtype':torch.float32}):
        super().__init__(1, tensor_args)
        self.bounds = torch.as_tensor(bounds, **tensor_args)
        self.scene_im = None

        self._flat_tensor = None
        self.proj_pt_idx = None
        

        self.ind_pt = None
    
    def update_world(self, image_path):
        im = cv2.imread(image_path,0)
        _,im = cv2.threshold(im,10,255,cv2.THRESH_BINARY)

        self.im = im
        
        #load image
        im_obstacle = cv2.bitwise_not(im)
        dist_obstacle = cv2.distanceTransform(im_obstacle, cv2.DIST_L2,3)
        dist_outside = cv2.distanceTransform(im, cv2.DIST_L2,3)
        
        dist_map = dist_obstacle - dist_outside

        self.dist_map = dist_map
        
        # transpose and flip to get the map to normal x, y axis
        a = torch.as_tensor(dist_map, **self.tensor_args).T
        scene_im = torch.flip(a, [1])

        
        self.scene_im = scene_im

        # get pixel range and rescale to meter bounds
        x_range,y_range = scene_im.shape
        self.im_dims = torch.tensor([x_range, y_range], **self.tensor_args)
        pitch = (self.im_dims) / ((self.bounds[:,1] - self.bounds[:,0]))
        self.pitch = pitch

        
        num_voxels = self.im_dims
        
        flat_tensor = torch.tensor([num_voxels[1], 1], device=self.tensor_args['device'], dtype=torch.int64)
        self.scene_voxels = torch.flatten(self.scene_im) * (1 / self.pitch[0])

        
        self.num_voxels = num_voxels
        self._flat_tensor = flat_tensor
        self.im_bounds = self.bounds 
        self.num_voxels = num_voxels
    def voxel_inds(self, pt):
        pt = (self.pitch * pt).to(dtype=torch.int64)
        
        ind_pt = (pt[...,0]) * (self.num_voxels[0]) + pt[...,1]
        ind_pt = ind_pt.to(dtype=torch.int64)
        return ind_pt
    def get_pt_value(self, pt):
        bound_mask = torch.logical_and(torch.all(pt < self.im_bounds[:,1] - (1.0/self.pitch),dim=-1),
                                       torch.all(pt > self.im_bounds[:,0] + (1.0/self.pitch),dim=-1))
        flat_mask = bound_mask.flatten()
        ind = self.voxel_inds(pt)
        ind[~flat_mask] = 0
        pt_coll = self.scene_voxels[ind]
        
        # values are signed distance: positive inside object, negative outside
        pt_coll[~flat_mask] = -10.0
        
        return pt_coll

