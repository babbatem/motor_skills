import math
import numpy as np

import open3d as o3d

"""
Estimates normals at all points in a cloud using a local neighborhood of points

@param cloud:      point cloud in an arbitrary frame
@param camera_pos: 3-tuple with camera's position in that frame:
    [0,0,0] if cloud is in the camera frame or
    camera position in world frame if already transformed to world frame

@return cloud normals: np array containing normal vectors.
    one per point, so if input cloud has size [n, 3],
    normals should also be [n, 3].
"""
def estimateNormals(cloud, camera_pos):
    if cloud.shape[-1] != 3:
        print("Could not estimate cloud normals, expected point cloud points to have three elements.")
        raise ValueError
    return cloud

"""
Generates transformation matrix from position and quaternion

@param pos:  x-y-z position tuple
@param quat: w-x-y-z quaternion rotation tuple

@return trans_mat: 4x4 transformation matrix
"""
def posQuat2Mat(pos, quat):
    if len(pos) != 3 or len(quat) != 4:
        print("Position and quaternion", pos, quat, "are invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    '''
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    '''

    #https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    trans_mat_row_0 = [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), pos[0]]
    trans_mat_row_1 = [2 * (xy + zw), - x2 + y2 - z2 + w2, 2 * (yz - xw), pos[1]]
    trans_mat_row_2 = [2 * (xz - yw), 2 * (yz + xw), - x2 - y2 + z2 + w2, pos[2]]
    trans_mat_row_3 = [0., 0., 0., 1.]

    return np.array([trans_mat_row_0, trans_mat_row_1, trans_mat_row_2, trans_mat_row_3])

"""
Transforms point cloud using given transformation matrix.
Note that if a transformation representing the camera pose in the world frame,
    it should be inverted before passing it to this function to transform the
    cloud in the camera frame to the world frame

@param pos:  x-y-z position tuple
@param quat: w-x-y-z quaternion rotation tuple

@return trans_mat: 4x4 transformation matrix
"""
def transformCloud(cloud, transform):
    if cloud.shape[-1] != 3:
        print("Could not transform cloud, expected point cloud points to have three elements.")
        raise ValueError
    # Input cloud is nx3
    # 3xn
    transpose_cloud = np.transpose(cloud)
    # 4xn cloud matrix by adding extra row of 1s
    transpose_cloud_and_ones = np.concatenate((transpose_cloud, np.ones((1, transpose_cloud.shape[1]))), axis=0)
    # Transform
    transformed_transpose_cloud_and_ones = np.matmul(transform, transpose_cloud_and_ones)
    # 3xn - remove the ones
    transformed_transpose_cloud = np.delete(transformed_transpose_cloud_and_ones, (3), axis=0)
    # nx3 transpose back
    transformed_cloud = np.transpose(transformed_transpose_cloud)
    return transformed_cloud

def transformNormals(normals, transform):
    if normals.shape[-1] != 3:
        print("Could not transform normals, expected normal vectors to have three elements.")
        raise ValueError
    # Input normals is nx3
    # 3xn
    transpose_normals = np.transpose(normals)
    # 3x3
    rot_mat = transform[:3, :3]
    # 3xn
    transformed_transpose_normals = np.matmul(rot_mat, transpose_normals)
    # nx3
    transformed_normals = np.transpose(transformed_transpose_normals)
    return transformed_normals

"""
Removes points outside a rectangular bounding box.
If any element of any point is below the corresponding axis's min or above its
    max value, it is removed from the cloud. Limit values can be set to
    None to not crop a min and/or max along a certain axis.

@param cloud:    point cloud in the same frame as the bounding box (often world)
@param x_limits: list or tuple with two elements: min and max points to keep
    along x axis.
@param y_limits: list or tuple with two elements: min and max points to keep
    along y axis.
@param z_limits: list or tuple with two elements: min and max points to keep
    along z axis.

@return cropped_cloud: cloud with points outside the specified box removed.
"""
def cropPointsOutside(cloud, x_limits, y_limits, z_limits):    
    # TODO(mcorsaro): does using True do what we excpect?
    cropped_cloud = cloud[np.logical_and(\
        np.logical_and(\
            np.logical_and((True if x_limits[0] == None else cloud[:,0] > x_limits[0]), (True if x_limits[1] == None else cloud[:,0] < x_limits[1])),\
            np.logical_and((True if y_limits[0] == None else cloud[:,1] > y_limits[0]), (True if y_limits[1] == None else cloud[:,1] < y_limits[1]))),\
        np.logical_and((True if z_limits[0] == None else cloud[:,2] > z_limits[0]), (True if z_limits[1] == None else cloud[:,2] < z_limits[1])))]
    return cropped_cloud
    '''
    cloud = cloud[np.logical_and(\
        np.logical_and(\
        np.logical_and(\
            np.abs(cloud[:,0]) < 0.09, \
            np.abs(cloud[:,1]) < 0.076), \
            cloud[:,2] < 0.155),\
        cloud[:,2] > -0.085)]'''
