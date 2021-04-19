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
import math

#import utils
from .utils import vector3_to_skew_symm_matrix

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])

def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


# def matrix_to_euler_angles(matrix, convention: str):
#     """
#     Convert rotations given as rotation matrices to Euler angles in radians.

#     Args:
#         matrix: Rotation matrices as tensor of shape (..., 3, 3).
#         convention: Convention string of three uppercase letters.

#     Returns:
#         Euler angles in radians as tensor of shape (..., 3).
#     """
#     if len(convention) != 3:
#         raise ValueError("Convention must have 3 letters.")
#     if convention[1] in (convention[0], convention[2]):
#         raise ValueError(f"Invalid convention {convention}.")
#     for letter in convention:
#         if letter not in ("X", "Y", "Z"):
#             raise ValueError(f"Invalid letter {letter} in convention string.")
#     if matrix.size(-1) != 3 or matrix.size(-2) != 3:
#         raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
#     i0 = _index_from_letter(convention[0])
#     i2 = _index_from_letter(convention[2])
#     tait_bryan = i0 != i2
#     if tait_bryan:
#         central_angle = torch.asin(
#             matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
#         )
#     else:
#         central_angle = torch.acos(matrix[..., i0, i0])

#     o = (
#         _angle_from_tan(
#             convention[0], convention[1], matrix[..., i2], False, tait_bryan
#         ),
#         central_angle,
#         _angle_from_tan(
#             convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
#         ),
#     )
#     return torch.stack(o, -1)

def matrix_to_euler_angles(R, cy_thresh=1e-6):
    # if cy_thresh is None:
    #     try:
    #         cy_thresh = np.finfo(M.dtype).eps * 4
    #     except ValueError:
    #         cy_thresh = _FLOAT_EPS_4
    # r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    inp_device = R.device
    #if(len(R.shape) == 4):
    Z = torch.zeros(R.shape[:-2], device=inp_device, dtype=R.dtype)
    #print(Z.shape)
    #else:
    #    Z = torch.zeros(R.shape[0], device=inp_device, dtype=R.dtype)
    r11 = R[...,0,0]
    r12 = R[...,0,1]
    r13 = R[...,0,2]
    r21 = R[...,1,0]
    r22 = R[...,1,1]
    r23 = R[...,1,2]
    r31 = R[...,2,0]
    r32 = R[...,2,1]
    r33 = R[...,2,2]


    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = torch.sqrt(r33*r33 + r23*r23)

    cond = cy > cy_thresh

    z = torch.where(cond, torch.atan2(-r12,  r11), torch.atan2(r21,  r22)).unsqueeze(-1)
    y = torch.atan2(r13,  cy).unsqueeze(-1)
    x = torch.where(cond, torch.atan2(-r23, r33), Z).unsqueeze(-1) 

    # if cy > cy_thresh: # cos(y) not close to zero, standard form
    #     z = torch.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
    #     y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
    #     x = torch.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    # else: # cos(y) (close to) zero, so x -> 0.0 (see above)
    #     # so r21 -> sin(z), r22 -> cos(z) and
    #     z = torch.atan2(r21,  r22)
    #     y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
    #     x = 0.0
    
    # return z, y, x
    return torch.cat([x, y, z], dim=-1)

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4). [qw, qx,qy,qz]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    zero = matrix.new_zeros((1,))
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 + m11 + m22))
    x = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 - m11 - m22))
    y = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 + m11 - m22))
    z = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 - m11 + m22))
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
    mask_c2 = (1 - mask_d2) * mask_d0_nd1
    mask_c3 = (1 - mask_d2) * (1 - mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def x_rot(angle):
    #  if len(angle.shape) == 0:
    # angle = angle.unsqueeze(0)
    # print("x_rot", angle.shape)
    # angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device, dtype=angle.dtype)
    R[:, 0, 0] = torch.ones(batch_size, device=angle.device, dtype=angle.dtype)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 1, 2] = -torch.sin(angle)
    R[:, 2, 1] = torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def y_rot(angle):
    #  if len(angle.shape) == 0:
    # angle = angle.unsqueeze(0)

    # print("y_rot", angle.shape)
    # print(angle.shape)
    # angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device, dtype=angle.dtype)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 2] = torch.sin(angle)
    R[:, 1, 1] = torch.ones(batch_size, device=angle.device, dtype=angle.dtype)
    R[:, 2, 0] = -torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def z_rot(angle):
    #  if len(angle.shape) == 0:
    # angle = angle.unsqueeze(0)
    # print("z_rot", angle.shape)
    # angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    # print("z_rot2", angle.shape)
    # angle = angle.squeeze(0)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3), device=angle.device, dtype=angle.dtype)
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 1] = -torch.sin(angle)
    R[:, 1, 0] = torch.sin(angle)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 2, 2] = torch.ones(batch_size, device=angle.device, dtype=angle.dtype)
    return R

def rpy_angles_to_matrix(euler_angles):
    """
    Convert rotations given as RPY euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    roll = euler_angles[:,0]
    pitch = euler_angles[:,1]
    yaw = euler_angles[:,2]
    matrices = (z_rot(yaw) @ y_rot(pitch)) @ x_rot(roll)

    return matrices


class CoordinateTransform(object):
    def __init__(self, rot=None, trans=None, tensor_args={'device':"cpu", 'dtype':torch.float32}, pose=None):

        
        self.tensor_args = tensor_args
        
        if rot is None:
            self._rot = torch.eye(3, **tensor_args).unsqueeze(0) #.to(device)
        else:
            self._rot = rot.to(**tensor_args)
        #if len(self._rot.shape) == 2:
        #    self._rot = self._rot.unsqueeze(0)

        if trans is None:
            self._trans = torch.zeros(1, 3, **tensor_args) #.to(device)
        else:
            self._trans = trans.to(**tensor_args)
        if len(self._trans.shape) == 1:
            self._trans = self._trans.unsqueeze(0)

        if(pose is not None):
            self.set_pose(pose)
    def set_pose(self, pose):
        """
        Args:
        pose: x, y, z, qw, qx, qy, qz
        """
        
        self._trans[0,:] = torch.as_tensor(pose[:3], **self.tensor_args)
        self._rot = quaternion_to_matrix(torch.as_tensor(pose[3:]).unsqueeze(0)).to(**self.tensor_args)
        
    def set_translation(self, t):
        self._trans = t.to(**self.tensor_args)
        #if len(self._trans.shape) == 1:
        #    self._trans = self._trans.unsqueeze(0)
        return

    def set_rotation(self, rot):
        self._rot = rot.to(**self.tensor_args)
        #if len(self._rot.shape) == 2:
        #    self._rot = self._rot.unsqueeze(0)
        return

    def rotation(self):
        return self._rot

    def translation(self):
        return self._trans

    def inverse(self):
        rot_transpose = self._rot.transpose(-2, -1)
        return CoordinateTransform(rot_transpose, -(rot_transpose @ self._trans.unsqueeze(2)).squeeze(2), tensor_args=self.tensor_args)

    def multiply_transform(self, coordinate_transform):
        new_rot, new_trans = multiply_transform(self._rot, self._trans, coordinate_transform.rotation(), coordinate_transform.translation())
        #new_rot = self._rot @ coordinate_transform.rotation()
        #new_trans = (self._rot @ coordinate_transform.translation().unsqueeze(-1)).squeeze(-1) + self._trans
        return CoordinateTransform(new_rot, new_trans, tensor_args=self.tensor_args)

    def multiply_inv_transform(self, coordinate_transform):
        new_rot, new_trans = multiply_inv_transform(coordinate_transform.rotation(), coordinate_transform.translation(), self._rot, self._trans)
        #new_rot = self._rot @ coordinate_transform.rotation()
        #new_trans = (self._rot @ coordinate_transform.translation().unsqueeze(-1)).squeeze(-1) + self._trans
        return CoordinateTransform(new_rot, new_trans, tensor_args=self.tensor_args)
    
    def trans_cross_rot(self):
        return vector3_to_skew_symm_matrix(self._trans) @ self._rot

    def get_transform_matrix(self):
        mat = torch.eye(4, **self.tensor_args)
        mat[:3,:3] = self._rot
        mat[:3,3] = self._trans
        return mat
    def get_quaternion(self):
        batch_size = self._rot.shape[0]
        M = torch.zeros((batch_size, 4, 4)).to(self._rot.device)
        M[:, :3, :3] = self._rot
        M[:, :3, 3] = self._trans
        M[:, 3, 3] = 1
        q = torch.empty((batch_size, 4)).to(self._rot.device)
        t = torch.einsum('bii->b', M) #torch.trace(M)
        for n in range(batch_size):
            tn = t[n]
            if tn > M[n, 3, 3]:
                q[n, 3] = tn
                q[n, 2] = M[n, 1, 0] - M[n, 0, 1]
                q[n, 1] = M[n, 0, 2] - M[n, 2, 0]
                q[n, 0] = M[n, 2, 1] - M[n, 1, 2]
            else:
                i, j, k = 0, 1, 2
                if M[n, 1, 1] > M[n, 0, 0]:
                    i, j, k = 1, 2, 0
                if M[n, 2, 2] > M[n, i, i]:
                    i, j, k = 2, 0, 1
                tn = M[n, i, i] - (M[n, j, j] + M[n, k, k]) + M[n, 3, 3]
                q[n, i] = tn
                q[n, j] = M[n, i, j] + M[n, j, i]
                q[n, k] = M[n, k, i] + M[n, i, k]
                q[n, 3] = M[n, k, j] - M[n, j, k]
                #q = q[[3, 0, 1, 2]]
            q[n, :] *= 0.5 / math.sqrt(tn * M[n, 3, 3])
        return q

    def transform_point(self, point):
        #if(len(point.shape) == 1):
        #    point = point.unsqueeze(-1)
        #new_point = transform_point(point, self._rot, self._trans)
        #new_point = point @ self._rot.transpose(-1,-2) + self._trans
        new_point = (self._rot @ (point).unsqueeze(-1)).squeeze(-1) + self._trans
        return new_point



@torch.jit.script   
def multiply_transform(w_rot_l, w_trans_l, l_rot_c, l_trans_c):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    #print(l_rot_c.shape, w_rot_l.shape)
    w_rot_c = w_rot_l @ l_rot_c
    
    w_trans_c = (w_rot_l @ l_trans_c.unsqueeze(-1)).squeeze(-1) + w_trans_l
    
    #w_trans_c = (l_trans_c @ w_rot_l.transpose(-1,-2)) + w_trans_l
    #print(w_trans_c - w_trans_l, l_trans_c)
    return w_rot_c, w_trans_c

@torch.jit.script   
def multiply_inv_transform(l_rot_w, l_trans_w, l_rot_c, l_trans_c):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    w_rot_l = l_rot_w.transpose(-1,-2)
    w_rot_c = w_rot_l @ l_rot_c


    w_trans_l = -(w_rot_l @ l_trans_w.unsqueeze(2)).squeeze(2)
    w_trans_c = (w_rot_l @ l_trans_c.unsqueeze(-1)).squeeze(-1) + w_trans_l
    
    #w_trans_c = (l_trans_c @ w_rot_l.transpose(-1,-2)) + w_trans_l
    #print(w_trans_c - w_trans_l, l_trans_c)
    return w_rot_c, w_trans_c


@torch.jit.script
def transform_point(point, rot, trans):
    # type: (Tensor, Tensor, Tensor) -> Tensor

    #new_point = (rot @ (point).unsqueeze(-1)).squeeze(-1) + trans
    new_point = (point @ rot.transpose(-1,-2)) + trans
    return new_point
