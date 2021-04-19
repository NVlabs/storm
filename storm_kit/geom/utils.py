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
import open3d as o3d

def get_open3d_pointcloud(points, color=[1,0,0], translation=np.zeros(3),rot=None):
    pcd = o3d.geometry.PointCloud()

    if(rot is not None):
        # project points:
    
        points = np.dot(points,rot.T)

    data = o3d.utility.Vector3dVector(points + translation)
    
    pcd.points = data
    color_array = np.array([color for x in range(len(points))])
    color_data = o3d.utility.Vector3dVector(color_array)

    pcd.colors = color_data
    return pcd


def get_pointcloud_from_depth(camera_data={'proj_matrix':None, 'segmentation':None,
                                           'depth':None}):
    proj_matrix = camera_data['proj_matrix']
    
    fu = 2 / proj_matrix[0, 0]
    fv = 2 / proj_matrix[1, 1]
    seg_buffer = camera_data['segmentation']
    depth_buffer = camera_data['depth']
    cam_width = camera_data['depth'].shape[1]
    cam_height = camera_data['depth'].shape[0]
    points = []
    # Ignore any points which originate from ground plane or empty space
    depth_buffer[seg_buffer == 0] = -10001
    #print(cam_width)
    vinv = np.linalg.inv(np.matrix(camera_data['view_matrix']))
    
    #print(vinv)
    centerU = cam_width / 2
    centerV = cam_height / 2
    pc_seg = []
    for i in range(cam_width):
        for j in range(cam_height):
            if depth_buffer[j, i] < -10000:
                continue
            u = -(i-centerU)/(cam_width)  # image-space coordinate
            v = (j-centerV)/(cam_height)  # image-space coordinate
            d = depth_buffer[j, i]  # depth buffer value
            X2 = np.matrix([d*fu*u, d*fv*v, d, 1])#.T # deprojection vector
            #p2 = X2
            
            #print(vinv.shape, X2.shape)
            p2 = X2 * vinv #(vinv * X2).T   # Inverse camera view to get world coordinates
            #print(p2)
            points.append([p2[0,0], p2[0,1],p2[0,2]])
            #points.append([p2[0,2], p2[0,0], p2[0,1]])
            pc_seg.append(seg_buffer[j,i])
    camera_data['pc'] = points#np.matrix(points)
    camera_data['pc_seg'] = np.ravel(pc_seg)
    return camera_data
