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
try:
    from  isaacgym import gymapi
    from isaacgym import gymutil
except Exception:
    print("ERROR: gym not loaded, this is okay when generating docs")

from quaternion import from_rotation_matrix

from .helpers import load_struct_from_dict

class Gym(object):
    def __init__(self,sim_params={}, physics_engine='physx', compute_device_id=0, graphics_device_id=1, num_envs=1, headless=False, **kwargs):

        if(physics_engine=='physx'):
            physics_engine = gymapi.SIM_PHYSX
        elif(physics_engine == 'flex'):
            physics_engine = gymapi.SIM_FLEX
        # create physics engine struct
        sim_engine_params = gymapi.SimParams()
        
        # find params in kwargs and fill up here:
        sim_engine_params = load_struct_from_dict(sim_engine_params, sim_params)
        self.headless = headless

        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device_id,
                                       graphics_device_id,
                                       physics_engine,
                                       sim_engine_params)

        self.env_list = []#None
        self.viewer = None
        self._create_envs(num_envs, num_per_row=int(np.sqrt(num_envs)))
        if(not headless):
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            cam_pos = gymapi.Vec3(-1.5, 1.8, 1.2)
            cam_target = gymapi.Vec3(6, 0.0, -6)
            #cam_pos = gymapi.Vec3(2, 2.0, -2)
            #cam_target = gymapi.Vec3(-6, 0.0,6)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        #self.gym.add_ground(self.sim, gymapi.PlaneParams())

        self.dt = sim_engine_params.dt
    def step(self):
        
        ## step through the physics regardless, only apply torque when sim time matches the real time
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        #if self.mode == 'human':
        # update the viewer
        if(not self.headless):
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

        self.gym.sync_frame_time(self.sim)
        return True
    
    def _create_envs(self, num_envs, spacing=1.0, num_per_row=1):
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        for _ in range(num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.env_list.append(env_ptr)
    def get_sim_time(self):
        return self.gym.get_sim_time(self.sim)
    def clear_lines(self):
        if(self.viewer is not None):
            self.gym.clear_lines(self.viewer)
    def draw_lines(self, pts, color=[0.5,0.0,0.0], env_idx=0, w_T_l=None):
        if(self.viewer is None):
            return
        verts = np.empty((pts.shape[0] - 1, 2), dtype=gymapi.Vec3.dtype)
        colors = np.empty(pts.shape[0] - 1, dtype=gymapi.Vec3.dtype)
        for i in range(pts.shape[0] - 1):
            p1 = pts[i]
            p2 = pts[i + 1]
            verts[i][0] = (p1[0], p1[1], p1[2])
            verts[i][1] = (p2[0], p2[1], p2[2])

            if(w_T_l is not None):
                verts[i][0] = w_T_l * verts[i][0]
                verts[i][1] = w_T_l * verts[i][1]
            colors[i] = (color[0], color[1], color[2])

        

        self.gym.add_lines(self.viewer,self.env_list[env_idx],pts.shape[0] - 1,verts, colors)
        #self.gym.add_lines(self.viewer,self.env_list[env_idx],pts.shape[0] - 1,verts, colors)

class World(object):
    def __init__(self, gym_instance, sim_instance, env_ptr, world_params=None, w_T_r=None):
        self.gym = gym_instance
        self.sim = sim_instance
        self.env_ptr = env_ptr
        
        self.radius = []
        self.position = []
        color = [0.6, 0.6, 0.6]
        obj_color = gymapi.Vec3(color[0], color[1], color[2])
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        self.ENV_SEG_LABEL = 1
        self.BG_SEG_LABEL = 0
        self.robot_pose = w_T_r
        self.table_handles = []

        if(world_params is None):
            return
        spheres = world_params['world_model']['coll_objs']['sphere']
        for obj in spheres.keys():
            radius = spheres[obj]['radius']
            position = spheres[obj]['position']

            
            
            # get pose
            
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(position[0], position[1], position[2])
            object_pose.r = gymapi.Quat(0, 0, 0,1)
            object_pose = w_T_r * object_pose

            #

            obj_asset = gym_instance.create_sphere(sim_instance,radius, asset_options)
            obj_handle = gym_instance.create_actor(env_ptr, obj_asset, object_pose, obj, 2, 2, self.ENV_SEG_LABEL)
            gym_instance.set_rigid_body_color(env_ptr, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_color)

        if('cube' in world_params['world_model']['coll_objs']):
            cube = world_params['world_model']['coll_objs']['cube']
            for obj in cube.keys():
                dims = cube[obj]['dims']
                pose = cube[obj]['pose']
                self.add_table(dims, pose, color=color)
            

    
    def add_table(self, table_dims, table_pose, color=[1.0,0.0,0.0]):

        table_dims = gymapi.Vec3(table_dims[0], table_dims[1], table_dims[2])

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        obj_color = gymapi.Vec3(color[0], color[1], color[2])
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(table_pose[0], table_pose[1], table_pose[2])
        pose.r = gymapi.Quat(table_pose[3], table_pose[4], table_pose[5], table_pose[6])
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z,
                                          asset_options)

        table_pose = self.robot_pose * pose
        table_handle = self.gym.create_actor(self.env_ptr, table_asset, table_pose,'table',
                                             2,2,self.ENV_SEG_LABEL)
        self.gym.set_rigid_body_color(self.env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_color)
        self.table_handles.append(table_handle)

    def spawn_object(self, asset_file, asset_root, pose, color=[], name='object'):
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        #pose = gymapi.Transform()
        #pose.p = gymapi.Vec3(pose[0], pose[1], pose[2])
        #pose.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
        
        obj_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        obj_handle = self.gym.create_actor(self.env_ptr, obj_asset, pose,name,
                                           2,2,self.BG_SEG_LABEL)
        return obj_handle

    def get_pose(self, body_handle):
        pose = self.gym.get_rigid_transform(self.env_ptr, body_handle)
        #pose = pose.p

        return pose
