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
from ...differentiable_robot_model.coordinate_transform import transform_point

def sdf_capsule_to_pt(capsule_base, capsule_tip, capsule_radius, pt):
    """Computes distance between a capsule and a point

    Args:
        capsule_base (tensor): x,y,z in batch [b,3]
        capsule_tip (tensor): x,y,z in batch [b,3]
        capsule_radius (tensor): radius of capsule in batch [b,1]
        pt (tensor): query point x,y,z in batch [b,3]

    Returns:
        (tensor): signed distance (negative outside, positive inside) [b,1]
    """
    pt_base = pt - capsule_base
    tip_base = capsule_tip - capsule_base

    h = torch.clamp(torch.dot(pt_base, tip_base) / torch.dot(tip_base,tip_base), 0.0, 1.0)
    
    dist = torch.norm(pt_base - tip_base * h) - capsule_radius
    return dist

def sdf_capsule_to_sphere(capsule_base, capsule_tip, capsule_radius, sphere_pt, sphere_radius):
    """Compute signed distance between capsule and sphere.

    Args:
        capsule_base (tensor): x,y,z in batch [b,3]
        capsule_tip (tensor): x,y,z in batch [b,3]
        capsule_radius (tensor): radius of capsule in batch [b,1]
        sphere_pt (tensor): query sphere origin x,y,z in batch [b,3]
        sphere_radius (tensor): radius of sphere [b,1]

    Returns:
        (tensor): signed distance (negative outside, positive inside) [b,1]
    """    
    pt_base = sphere_pt - capsule_base
    tip_base = capsule_tip - capsule_base
    
    pt_dot = (pt_base * tip_base).sum(-1)
    cap_dot = (tip_base * tip_base).sum(-1)

    h = torch.clamp(pt_dot / cap_dot, 0.0, 1.0)
    norm = torch.norm(pt_base - tip_base * h.unsqueeze(-1),dim=-1)
    dist = (norm - capsule_radius) - sphere_radius
    return dist



def sdf_pt_to_sphere(sphere_pt, sphere_radius, query_pt):
    """signed distance between sphere and point. Also works for 2d case.

    Args:
        sphere_pt (tensor): origin of sphere [b,3]
        sphere_radius (tensor): radius of sphere [b,1]
        query_pt (tensor): query point [b,1]

    Returns:
        (tensor): signed distance (negative outside, positive inside) [b,1]
    """    
    return jit_sdf_pt_to_sphere(sphere_pt,sphere_radius,query_pt)

def sdf_pt_to_box(box_dims, box_trans, box_rot, query_pts):
    """signed distance between box and point. Points are assumed to be in world frame.

    Args:
        box_dims (tensor): dx,dy,dz of box [b,3], this is around origin (-dx/2,dx/2...).
        box_trans (tensor): origin of box in the world frame [b,3].
        box_rot (tensor): rotation of box as a rotation matrix in the world frame. [b,3,3]
        query_pts (tensor): pts in world frame to query sdf. [b,3]

    Returns:
        (tensor): signed distance (negative outside, positive inside) [b,1]
    """    

    return jit_sdf_pt_to_box(box_dims, box_trans, box_rot, query_pts)

@torch.jit.script
def jit_sdf_pt_to_box(box_dims, box_trans, box_rot, query_pts):
    # transform points to pose:
    l_pts = transform_point(query_pts, box_rot, box_trans)
    

    dmin = l_pts - (-box_dims / 2.0)
    dmin[dmin > 0.0] = 0.0
    
    dmax = l_pts - (box_dims / 2.0)
    dmax[dmax < 0.0] = 0.0
    
    dist = torch.norm(dmin + dmax , dim=-1)

    
    in_bounds = torch.logical_and(torch.all(l_pts < box_dims/2.0, dim=-1),
                                  torch.all(l_pts > -1.0 * box_dims/2.0, dim=-1))
    dist[~in_bounds] *= -1.0

    return dist

@torch.jit.script
def jit_sdf_pt_to_sphere(sphere_pt, sphere_radius, query_pt):
    
    dist = sphere_radius - torch.norm(query_pt - sphere_pt,dim=-1)
    
    return dist

@torch.jit.script
def get_pt_primitive_distance(w_pts, world_spheres, world_cubes, dist):
    # type: (Tensor, Tensor, List[List[Tensor]], Tensor) -> Tensor
    
    for i in range(world_spheres.shape[1]):
        # compute distance between w_pts and sphere:
        # world_spheres: b, 0, 3
        d = sdf_pt_to_sphere(world_spheres[:,i,:3],
                             world_spheres[:,i,3],
                             w_pts)
        dist[:,i,:] = d
        
    # cube signed distance:
    for i in range(len(world_cubes)):
        
        cube = world_cubes[i]
        #print(cube['inv_trans'], cube['trans'])
        d = sdf_pt_to_box(cube[-1], cube[2], cube[3], w_pts)
        dist[:,i + world_spheres.shape[1],:] = d
    return dist

@torch.jit.script
def get_sphere_primitive_distance(w_sphere, world_spheres, world_cubes):
    # type: (Tensor, Tensor, List[List[Tensor]]) -> Tensor
    dist = torch.zeros((w_sphere.shape[0], world_spheres.shape[1]+len(world_cubes), w_sphere.shape[1]), device=w_sphere.device, dtype=w_sphere.dtype)

    for i in range(world_spheres.shape[1]):
        # compute distance between w_pts and sphere:
        # world_spheres: b, 0, 3
        d = sdf_pt_to_sphere(world_spheres[:,i,:3],
                             world_spheres[:,i,3],
                             w_sphere[...,:3]) + w_sphere[...,3]
        
        dist[:,i,:] = d
        
    
    # cube signed distance:
    for i in range(len(world_cubes)):
        cube = world_cubes[i]
        d = sdf_pt_to_box(cube[-1], cube[2], cube[3], w_sphere[...,:3])
        dist[:,i + world_spheres.shape[1],:] = d + w_sphere[...,3]
        
    return dist


