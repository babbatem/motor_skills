import math
import numpy as np

import open3d as o3d

import motor_skills.planner.mj_point_clouds as mjpc

def rotate_about_z(rotation_matrix, angle_in_rad):
    rotation_about_z = np.array([math.cos(angle_in_rad), -1*math.sin(angle_in_rad), 0, \
        math.sin(angle_in_rad), math.cos(angle_in_rad), 0, \
        0, 0, 1]).reshape(3,3)
    return np.matmul(rotation_matrix, rotation_about_z)

"""
Class that generates grasp candidate poses given a point cloud with
    ten Pas et al.'s heuristic used in Grasp Pose Detection in Point Clouds
"""
class GraspPoseGenerator(object):
    """
    initialization function

    # from -pi/3 to pi/2 in increments of pi/6    
    """
    def __init__(self, point_cloud_with_normals, rotation_values_about_approach=[(i-2)*math.pi/6 for i in range(6)]):
        super(GraspPoseGenerator, self).__init__()

        self.o3d_cloud = point_cloud_with_normals

        self.cloud_points = np.asarray(self.o3d_cloud.points)
        self.cloud_norms = np.asarray(self.o3d_cloud.normals)
        if self.cloud_points.shape != self.cloud_norms.shape:
            print("Error: Point cloud provided to GraspPoseGenerator has points of shape", \
                self.cloud_points.shape[0], "but normals of shape", normals_shape)
            raise ValueError

        # http://www.open3d.org/docs/latest/tutorial/Basic/kdtree.html
        self.o3d_kdtree = o3d.geometry.KDTreeFlann(self.o3d_cloud)

        # same values as mj_point_clouds.py estimate_normals
        self.search_radius = 0.03
        self.max_nns = 250

        self.rotation_values = rotation_values_about_approach

    def proposeGraspPosesAtCloudIndex(self, index):
        if index >= self.cloud_points.shape[0]:
            print("Error: Requested grasp pose corresponding to point with index", index, \
                "but cloud contains", self.cloud_points.shape[0], "points.")
            raise ValueError
        sample_point = self.cloud_points[index]
        # vector anti-parallel to normal
        approach_vector = -1*self.cloud_norms[index]

        [k, local_point_ids, _] = self.o3d_kdtree.search_radius_vector_3d(self.o3d_cloud.points[index], self.search_radius)
        nn_ids = local_point_ids[:self.max_nns]
        local_cloud = o3d.geometry.PointCloud()
        local_cloud.points = o3d.utility.Vector3dVector(self.cloud_points[nn_ids, :])
        local_cloud.normals = o3d.utility.Vector3dVector(self.cloud_norms[nn_ids, :])

        [mean, covar_mat] = local_cloud.compute_mean_and_covariance()

        eigenvalues, eigenvectors = np.linalg.eig(covar_mat)
        print("values\n", eigenvalues,"\nvectors\n", eigenvectors, "\nnormal\n", self.cloud_norms[index])
        min_eig_id = eigenvalues.argmin()
        max_eig_id = eigenvalues.argmax()
        mid_eig_id = 3 - max_eig_id - min_eig_id
        print("min", eigenvalues[min_eig_id], "max", eigenvalues[max_eig_id], "mid", eigenvalues[mid_eig_id])
        hand_rot_approach = eigenvectors[:, min_eig_id]
        hand_rot_closing = eigenvectors[:, mid_eig_id]
        print("Approach\n", hand_rot_approach, "\nclosing\n", hand_rot_closing)
        if np.dot(approach_vector, hand_rot_approach) < 0:
            hand_rot_approach = -1*hand_rot_approach
        # y = z cross x
        hand_rot_third_axis = np.cross(hand_rot_approach, hand_rot_closing)
        hand_rot_mat = np.zeros((3,3))
        hand_rot_mat[:,0] = hand_rot_closing
        hand_rot_mat[:,1] = hand_rot_third_axis
        hand_rot_mat[:,2] = hand_rot_approach
        print("Hand rot mat\n", hand_rot_mat)

        grasp_poses = []
        for rotation_value in self.rotation_values:
            grasp_pose = np.eye(4)
            hand_rot_mat_rotated_about_approach = rotate_about_z(hand_rot_mat, rotation_value)
            grasp_pose[:3, :3] = hand_rot_mat_rotated_about_approach
            grasp_pose[:3, 3] = sample_point
            grasp_poses.append(grasp_pose)
        return grasp_poses
        # TODO(mcorsaro): rotate matrix about approach axis by pi/2, pi/3, pi/6, -pi/3, -pi/6
        
        