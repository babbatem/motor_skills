import copy
import math
import numpy as np

import open3d as o3d

"""
Rotates a given matrix about z-axis (approach) by specified angle

@param rotation_matrix: rotation matrix to be rotated about its z-axis
@param angle_in_rad:    angle to rotate frame about z by (radians)

@return rotated_rotation_matrix: original matrix rotated about its z-axis
"""
def rotate_about_z(rotation_matrix, angle_in_rad):
    rotation_about_z = np.array([math.cos(angle_in_rad), -1*math.sin(angle_in_rad), 0, \
        math.sin(angle_in_rad), math.cos(angle_in_rad), 0, \
        0, 0, 1]).reshape(3,3)
    return np.matmul(rotation_matrix, rotation_about_z)

"""
Returns pose some distance away from position along negative z-axis (approach)

@param grasp_pose: (4x4) numpy array representing grasp pose, with z-axis
    being the approach direction.
@param dist:       distance to translate this frame along negative z-axis

@returns translated_pose: given pose translated along negative z-axis
"""
def translateFrameNegativeZ(grasp_pose, dist):
    translated_pose = copy.deepcopy(grasp_pose)
    dist_to_move_in_cloud_frame = -1*np.matmul(grasp_pose, np.array([0, 0, dist, 0]))
    translated_pose[:3, 3] += dist_to_move_in_cloud_frame[:3]
    return translated_pose

"""
Class that generates grasp candidate poses given a point cloud with
    ten Pas et al.'s heuristic used in Grasp Pose Detection in Point Clouds
"""
class GraspPoseGenerator(object):
    """
    initialization function

    @param point_cloud_with_normals: point cloud to generate grasps on. We
        assume this cloud has been computed prior to constructing an object
        of this class, and a new object must be created for a new cloud. The
        frame that this cloud is in doesn't matter, though I provide a
        combination of several clouds in the world frame.
    @param rotation_values_about_approach: this heuristic generates a grasp
        frame from a sampled point's covariance matrix, but also rotates this
        frame about its z-axis (approach direction) to expand the space of
        possible grasp poses. This parameter is a list of angles in radians
        to rotate the default grasp frame about. I've used [0, pi/2], but the
        default is 6 angles from -pi/3 to pi/2 in increments of pi/6.
    """
    def __init__(self, point_cloud_with_normals, rotation_values_about_approach=[(i-2)*math.pi/6 for i in range(6)]):
        super(GraspPoseGenerator, self).__init__()

        # We assume a cloud with its pre-computed normals are passed in here.
        # We don't comput normals since we assume this is a composite cloud,
        #   and that normals were estimated before combining
        self.o3d_cloud = point_cloud_with_normals

        # points and normals as numpy arrays
        self.cloud_points = np.asarray(self.o3d_cloud.points)
        self.cloud_norms = np.asarray(self.o3d_cloud.normals)
        if self.cloud_points.shape != self.cloud_norms.shape:
            print("Error: Point cloud provided to GraspPoseGenerator has points of shape", \
                self.cloud_points.shape[0], "but normals of shape", normals_shape)
            raise ValueError

        # http://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html
        # KDTree used for nearest neighbor search
        self.o3d_kdtree = o3d.geometry.KDTreeFlann(self.o3d_cloud)

        # same values as mj_point_clouds.py uses to estimate_normals
        self.search_radius = 0.03
        self.max_nns = 250

        # angles to rotate grasp frame about z-axis by
        self.rotation_values = rotation_values_about_approach

    """
    Generates a set of grasp poses given an index corresponding to a point
        in the cloud to use as a centroid

    @param index: int < number of points in provided cloud

    @return grasp poses: list of 4x4 numpy arrays representing grasp poses
        in the point cloud's frame.
    """
    def proposeGraspPosesAtCloudIndex(self, index):
        if index >= self.cloud_points.shape[0]:
            print("Error: Requested grasp pose corresponding to point with index", index, \
                "but cloud contains", self.cloud_points.shape[0], "points.")
            raise ValueError
        # sample point to use as grasp centroid
        sample_point = self.cloud_points[index]
        # vector anti-parallel to normal provided with input cloud
        approach_vector = -1*self.cloud_norms[index]

        # k indices corresponding to points within a self.search_radius radius
        #   of the sampled point.
        [k, local_point_ids, _] = self.o3d_kdtree.search_radius_vector_3d(self.o3d_cloud.points[index], self.search_radius)
        # self.max_nns closest points to sampled point
        nn_ids = local_point_ids[:self.max_nns]
        # create a new point cloud with sampled point and these local points
        local_cloud = o3d.geometry.PointCloud()
        local_cloud.points = o3d.utility.Vector3dVector(self.cloud_points[nn_ids, :])
        local_cloud.normals = o3d.utility.Vector3dVector(self.cloud_norms[nn_ids, :])

        # covariance matrix of this local cloud
        [mean, covar_mat] = local_cloud.compute_mean_and_covariance()

        # compute the eigenvalues and eigenvectors of this local cloud's
        #   covariance matrix to estimate normals and curvature at sampled point
        eigenvalues, eigenvectors = np.linalg.eig(covar_mat)
        min_eig_id = eigenvalues.argmin()
        max_eig_id = eigenvalues.argmax()
        mid_eig_id = 3 - max_eig_id - min_eig_id
        # approach direction is eigenvector with smallest eigenvalue (normal)
        hand_rot_approach = eigenvectors[:, min_eig_id]
        # closing direction is eigenvector with middle eigenvalue
        hand_rot_closing = eigenvectors[:, mid_eig_id]
        # ensure approach estimated from local cloud is in the same direction
        #   as given approach vector (anti-parallel to provided normal), which
        #   has presumably been oriented based on camera position. Note
        #   that these vectors may be slightly different if original provided
        #   normals were estimated with single cloud and composite cloud is
        #   provided.
        if np.dot(approach_vector, hand_rot_approach) < 0:
            hand_rot_approach = -1*hand_rot_approach

        # We choose x-y-z order for these vectors based on MuJoCo Jaco gripper
        #   frame: approach is z, fingers close along x.
        hand_rot_mat = np.zeros((3,3))
        hand_rot_mat[:,0] = hand_rot_closing
        # y = z cross x
        hand_rot_mat[:,1] = np.cross(hand_rot_approach, hand_rot_closing)
        hand_rot_mat[:,2] = hand_rot_approach

        # Create several orientations at this sample point by rotating about
        #   approach axis.
        grasp_poses = []
        for rotation_value in self.rotation_values:
            grasp_pose = np.eye(4)
            hand_rot_mat_rotated_about_approach = rotate_about_z(hand_rot_mat, rotation_value)
            grasp_pose[:3, :3] = hand_rot_mat_rotated_about_approach
            # Note that sampled point is returned as position here - use
            #   translateFrameNegativeZ to generate a grasp and pre-grasp pose
            #   depending on how far you want the gripper to be from the target
            #   object. Note that the MuJoCo Jaco's EE frame is in between
            #   the fingers, so the sample point actually works well as
            #   the grasp centroid in the EE frame
            grasp_pose[:3, 3] = sample_point
            grasp_poses.append(grasp_pose)
        return grasp_poses
        