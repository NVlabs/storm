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

# This file contains a generic robot class that can load a robot asset into sim and gives access to robot's state and control.


import copy

import numpy as np
from quaternion import from_rotation_matrix, as_float_array, as_rotation_matrix, as_quat_array
try:
    from  isaacgym import gymapi
    from isaacgym import gymutil
except Exception:
    print("ERROR: gym not loaded, this is okay when generating doc")

import torch

from .helpers import load_struct_from_dict
from ..util_file import join_path

def inv_transform(gym_transform):
    mat = np.eye(4)
    mat[0:3, 3] = np.ravel([gym_transform.p.x,gym_transform.p.y, gym_transform.p.z])
    # get rotation matrix from quat:
    q = gym_transform.r

    rot = as_rotation_matrix(as_quat_array([q.w, q.x, q.y, q.z]))
    mat[0:3, 0:3] = rot
    inv_mat = np.linalg.inv(mat)

    quat = as_float_array(from_rotation_matrix(inv_mat[0:3,0:3]))
    new_transform = gymapi.Transform(p=gymapi.Vec3(inv_mat[0,3], inv_mat[1,3],
                                                   inv_mat[2,3]),
                                     r=gymapi.Quat(quat[1],
                                                   quat[2],
                                                   quat[3],
                                                   quat[0]))
    
    return new_transform
# Write some helper functions:
def pose_from_gym(gym_pose):
    pose = np.array([gym_pose.p.x, gym_pose.p.y, gym_pose.p.z,
                     gym_pose.r.x, gym_pose.r.y, gym_pose.r.z, gym_pose.r.w])
    return pose

class RobotSim():
    def __init__(self, device='cpu', gym_instance=None, sim_instance=None,
                 asset_root='', sim_urdf='', asset_options='', init_state=None, collision_model=None, **kwargs):
        self.gym = gym_instance
        self.sim = sim_instance
        self.device = device
        self.dof = None
        self.init_state = init_state
        self.joint_names = []
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options = load_struct_from_dict(robot_asset_options, asset_options)

        self.camera_handle = None
        self.collision_model_params = collision_model
        self.DEPTH_CLIP_RANGE = 6.0
        self.ENV_SEG_LABEL = 1
        self.ROBOT_SEG_LABEL = 2
        
        self.robot_asset = self.load_robot_asset(sim_urdf,
                                                 robot_asset_options,
                                                 asset_root)

        
    def init_sim(self, gym_instance, sim_instance):
        self.gym = gym_instance
        self.sim = sim_instance
        
    def load_robot_asset(self, sim_urdf, asset_options, asset_root):

        if ((self.gym is None) or (self.sim is None)):
            raise AssertionError
        robot_asset = self.gym.load_asset(self.sim, asset_root,
                                          sim_urdf, asset_options)
        #print(asset_options.disable_gravity)
        return robot_asset

    def spawn_robot(self, env_handle, robot_pose, robot_asset=None, coll_id=-1, init_state=None):
        p = gymapi.Vec3(robot_pose[0], robot_pose[1], robot_pose[2])
        robot_pose = gymapi.Transform(p=p, r=gymapi.Quat(robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6]))
        self.spawn_robot_pose = robot_pose
        # also store inverse:
        #self.inv_robot_pose = self.spawn_robot_pose.inverse()
        
        if(robot_asset is None):
            robot_asset = self.robot_asset
        robot_handle = self.gym.create_actor(env_handle, robot_asset,
                                             robot_pose, 'robot', coll_id, coll_id, self.ROBOT_SEG_LABEL) # coll_id, mask_filter, mask_vision

        # set friction prop:
        shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, robot_handle)
        for i in range(len(shape_props)):
            shape_props[i].friction = 1.5
        self.gym.set_actor_rigid_shape_properties(env_handle,robot_handle,shape_props)
        
        # set initial position:
        robot_joint_names = self.gym.get_actor_dof_names(env_handle, robot_handle)
        self.joint_names = robot_joint_names
            
        # todo - move to shared data
        # get joint limits and ranges for robot
        robot_dof_props = self.gym.get_actor_dof_properties(env_handle, robot_handle)

        self.dof = len(robot_dof_props['effort'])

        robot_lower_limits = robot_dof_props['lower']
        robot_upper_limits = robot_dof_props['upper']
        
        if(init_state is None):
            if(self.init_state is None):
                init_state = (robot_lower_limits + robot_upper_limits) / 2 
            else:
                init_state = self.init_state
        self.init_state = init_state

        # for torque control:
        #robot_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
        #robot_dof_props['stiffness'].fill(0.0) # = self.joint_stiffnness[:self.num_dofs]
        #robot_dof_props['damping'].fill(0.0) # To avoidxb oscilaatuions?

        # for position control:
        robot_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
        robot_dof_props['stiffness'].fill(400.0) # = self.joint_stiffnness[:self.num_dofs]
        robot_dof_props['damping'].fill(40.0) # To avoidxb oscilaatuions?
        robot_dof_props['stiffness'][-2:] = 100.0
        robot_dof_props['damping'][-2:] = 5.0
        

        self.gym.set_actor_dof_properties(env_handle, robot_handle, robot_dof_props)            
        
        robot_dof_states = copy.deepcopy(self.gym.get_actor_dof_states(env_handle, robot_handle,
                                                                       gymapi.STATE_ALL))

        for i in range(len(robot_dof_states['pos'])):
            robot_dof_states['pos'][i] = self.init_state[i]
            robot_dof_states['vel'][i] = 0.0
        self.init_robot_state = robot_dof_states
        self.gym.set_actor_dof_states(env_handle, robot_handle, robot_dof_states, gymapi.STATE_ALL)

        if(self.collision_model_params is not None):
            self.init_collision_model(self.collision_model_params, env_handle, robot_handle)

        return robot_handle
    def get_state(self, env_handle, robot_handle):
        robot_state = self.gym.get_actor_dof_states(env_handle, robot_handle, gymapi.STATE_ALL)
        
        # reformat state to be similar ros jointstate:
        joint_state = {'name':self.joint_names, 'position':[], 'velocity':[], 'acceleration':[]}

        for i in range(len(robot_state)):
            joint_state['position'].append(robot_state[i][0])
            joint_state['velocity'].append(robot_state[i][1])
        joint_state['position'] = np.ravel(joint_state['position'])
        joint_state['velocity'] = np.ravel(joint_state['velocity'])
        joint_state['acceleration'] = np.ravel(joint_state['velocity'])*0.0
        
        return joint_state
    

    def command_robot(self, tau, env_handle, robot_handle):
        self.gym.apply_actor_dof_efforts(env_handle, robot_handle, np.float32(tau))
        
    def command_robot_position(self, q_des, env_handle, robot_handle):
        self.gym.set_actor_dof_position_targets(env_handle, robot_handle, np.float32(q_des))


    def set_robot_state(self, q_des, qd_des, env_handle, robot_handle):
        robot_dof_states = copy.deepcopy(self.gym.get_actor_dof_states(env_handle, robot_handle,
                                                                       gymapi.STATE_ALL))

        for i in range(len(robot_dof_states['pos'])):
            robot_dof_states['pos'][i] = q_des[i]
            robot_dof_states['vel'][i] = qd_des[i]
        self.init_robot_state = robot_dof_states
        self.gym.set_actor_dof_states(env_handle, robot_handle, robot_dof_states, gymapi.STATE_ALL)

    def update_collision_model(self, link_poses, env_ptr, robot_handle):
        w_T_r = self.spawn_robot_pose
        for i in range(len(link_poses)):
            #print(i)
            link = self.link_colls[i]
            link_pose = gymapi.Transform()
            link_pose.p = gymapi.Vec3(link_poses[i][0], link_poses[i][1], link_poses[i][2])
            link_pose.r = gymapi.Quat(link_poses[i][4], link_poses[i][5], link_poses[i][6],link_poses[i][3])
            w_p1 = w_T_r * link_pose * link['pose_offset'] * link['base']
            self.gym.set_rigid_transform(env_ptr, link['p1_body_handle'], w_p1)
            w_p2 = w_T_r * link_pose * link['pose_offset'] * link['tip']
            self.gym.set_rigid_transform(env_ptr, link['p2_body_handle'], w_p2)

    def init_collision_model(self, robot_collision_params, env_ptr, robot_handle):
        
        # get robot w_T_r
        w_T_r = self.spawn_robot_pose
        # transform all points based on this:
        
        robot_links = robot_collision_params['link_objs']
        #x,y,z = 0.1, 0.1, 0.2
        obs_color = gymapi.Vec3(0.0, 0.5, 1.0)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002

        # link pose is in robot base frame:
        link_pose = gymapi.Transform()
        link_pose.p = gymapi.Vec3(0, 0, 0)
        link_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.link_colls = []
        
        for j in robot_links:
            #print(j)
            pose = robot_links[j]['pose_offset']
            pose_offset = gymapi.Transform()
            pose_offset.p = gymapi.Vec3(pose[0], pose[1], pose[2])
            #pose_offset.r = gymapi.Quat.from_rpy(pose[3], pose[4], pose[5])
            r = robot_links[j]['radius']
            base = np.ravel(robot_links[j]['base'])
            tip = np.ravel(robot_links[j]['tip'])
            width = np.linalg.norm(base - tip)
            pt1_pose = gymapi.Transform()
            pt1_pose.p = gymapi.Vec3(base[0], base[1], base[2])
            link_p1_asset = self.gym.create_sphere(self.sim, r, asset_options)
            link_p1_handle = self.gym.create_actor(env_ptr, link_p1_asset,w_T_r * pose_offset * pt1_pose, j, 2, 2)
            self.gym.set_rigid_body_color(env_ptr, link_p1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          obs_color)
            link_p1_body = self.gym.get_actor_rigid_body_handle(env_ptr, link_p1_handle, 0)
            
            pt2_pose = gymapi.Transform()
            pt2_pose.p = gymapi.Vec3(tip[0], tip[1], tip[2])
            link_p2_asset = self.gym.create_sphere(self.sim, r, asset_options)
            link_p2_handle = self.gym.create_actor(env_ptr, link_p2_asset, w_T_r * pose_offset * pt2_pose, j, 2, 2)
            link_p2_body = self.gym.get_actor_rigid_body_handle(env_ptr, link_p2_handle, 0)
            self.gym.set_rigid_body_color(env_ptr, link_p2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          obs_color)
        
    
            link_coll = {'base':pt1_pose, 'tip':pt2_pose, 'pose_offset':pose_offset, 'radius':r,
                         'p1_body_handle':link_p1_body, 'p2_body_handle': link_p2_body}
            self.link_colls.append(link_coll)

    def spawn_camera(self, env_ptr, fov, width, height, robot_camera_pose):
        """
        Spawn a camera in the environment
        Args:
        env_ptr: environment pointer
        fov, width, height: camera params
        robot_camera_pose: Camera pose w.r.t robot_body_handle [x, y, z, qx, qy, qz, qw]
        """
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = fov
        camera_props.height = height
        camera_props.width = width
        camera_props.use_collision_geometry = False

        self.num_cameras = 1
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        robot_camera_pose = gymapi.Transform(
            gymapi.Vec3(robot_camera_pose[0], robot_camera_pose[1], robot_camera_pose[2]),
            gymapi.Quat(robot_camera_pose[3], robot_camera_pose[4], robot_camera_pose[5], robot_camera_pose[6]))

        # quat (q.x, q.y, q.z, q.w)
        # as_float_array(q.w, q.x, q.y, q.z)
        world_camera_pose = self.spawn_robot_pose * robot_camera_pose
        
        #print('Spawn camera pose:',world_camera_pose.p)
        self.gym.set_camera_transform(
            camera_handle,
            env_ptr,
            world_camera_pose)

        self.camera_handle = camera_handle
        
        return camera_handle

        
        
    def observe_camera(self, env_ptr):
        self.gym.render_all_camera_sensors(self.sim)
        self.current_env_observations = []
        
        camera_handle = self.camera_handle

        w_c_mat = self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle).T
        #print('View matrix',w_c_mat)
        #p = gymapi.Vec3(w_c_mat[3,0], w_c_mat[3,1], w_c_mat[3,2])
        #p = gymapi.Vec3(w_c_mat[0,3], w_c_mat[1,3], w_c_mat[2,3])
        #quat = as_float_array(from_rotation_matrix(w_c_mat[0:3, 0:3]))
        #r = gymapi.Quat(quat[1], quat[2], quat[3], quat[0])
        camera_pose = self.spawn_robot_pose.inverse()

        proj_matrix = self.gym.get_camera_proj_matrix(
            self.sim, env_ptr, camera_handle
        )
        view_matrix = self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)#.T
        #view_matrix = view_matrix_t
        #view_matrix[0:3,3] = view_matrix_t[3,0:3]
        #view_matrix[3,0:3] = 0.0
        q = camera_pose.r
        p = camera_pose.p
        camera_pose = [p.x,p.y, p.z, q.x, q.y, q.z, q.w]
        
        
        color_image = self.gym.get_camera_image(
            self.sim,
            env_ptr,
            camera_handle,
            gymapi.IMAGE_COLOR)
        color_image = np.reshape(color_image, [480, 640, 4])[:, :, :3]

        depth_image = self.gym.get_camera_image(
            self.sim,
            env_ptr,
            camera_handle,
            gymapi.IMAGE_DEPTH,
        )
        depth_image[depth_image == np.inf] = 0
        #depth_image[depth_image > self.DEPTH_CLIP_RANGE] = 0
        segmentation = self.gym.get_camera_image(
            self.sim,
            env_ptr,
            camera_handle,
            gymapi.IMAGE_SEGMENTATION,
        )
        
        camera_data = {'color':color_image, 'depth':depth_image,
                       'segmentation':segmentation, 'robot_camera_pose':camera_pose,
                       'proj_matrix':proj_matrix, 'label_map':{'robot': self.ROBOT_SEG_LABEL,
                                                               'ground': 0},
                       'view_matrix':view_matrix,
                       'world_robot_pose': self.spawn_robot_pose}
        return camera_data
