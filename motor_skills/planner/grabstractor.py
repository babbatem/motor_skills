import copy
import datetime
import math
import os
import time

import itertools

from colorsys import hls_to_rgb

import numpy as np
import open3d as o3d

from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

import motor_skills.planner.mj_point_clouds as mjpc
import motor_skills.planner.grasp_pose_generator as gpg

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

class Grabstractor(object):
    def __init__(self, cloud_with_normals, grasp_poses, obj="door", use_obj_frame=True):
        self.cloud_with_normals = copy.deepcopy(cloud_with_normals)
        self.grasp_poses = copy.deepcopy(grasp_poses)
        self.vis_font = font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 33)
        self.obj = obj
        if self.obj == "door":
            self.obj_frame = mjpc.posRotMat2Mat([0.185, 0.348, 0.415], mjpc.quat2Mat([0.7071, 0, 0.7071, 0]))
            self.visualization_view_param_file = "/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/DoorOpen3DCamPose.json"
        elif self.obj == "cylinder":
            self.obj_frame = mjpc.posRotMat2Mat([0, 0.5, 0], mjpc.quat2Mat([1, 0, 0, 0]))
            self.visualization_view_param_file = "/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/CylinderOpen3DCamPose.json"
        if use_obj_frame:
            self.visualization_view_param_file = self.visualization_view_param_file[:-5] + '_obj_frame.json'
            self.cloud_with_normals.transform(np.linalg.inv(self.obj_frame))
            self.grasp_poses = [np.matmul(np.linalg.inv(self.obj_frame), grasp_pose) for grasp_pose in self.grasp_poses]

    def loadGripperMesh(self):
        gripper_model_path = '/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/assets/kinova_j2s6s300/hand_3finger.STL'
        finger_proximal_model_path = '/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/assets/kinova_j2s6s300/finger_proximal.STL'
        finger_distal_model_path = '/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/assets/kinova_j2s6s300/finger_distal.STL'
        gripper_mesh = o3d.io.read_triangle_mesh(gripper_model_path)
        finger_1_proximal_mesh = o3d.io.read_triangle_mesh(finger_proximal_model_path)
        finger_1_distal_mesh = o3d.io.read_triangle_mesh(finger_distal_model_path)
        finger_2_proximal_mesh = o3d.io.read_triangle_mesh(finger_proximal_model_path)
        finger_2_distal_mesh = o3d.io.read_triangle_mesh(finger_distal_model_path)
        finger_3_proximal_mesh = o3d.io.read_triangle_mesh(finger_proximal_model_path)
        finger_3_distal_mesh = o3d.io.read_triangle_mesh(finger_distal_model_path)

        proximal_color = [0, 0, 0]
        distal_color = [0.5, 0.5, 0.5]
        finger_1_proximal_mesh.paint_uniform_color(proximal_color)
        finger_1_distal_mesh.paint_uniform_color(distal_color)
        finger_2_proximal_mesh.paint_uniform_color(proximal_color)
        finger_2_distal_mesh.paint_uniform_color(distal_color)
        finger_3_proximal_mesh.paint_uniform_color(proximal_color)
        finger_3_distal_mesh.paint_uniform_color(distal_color)

        # Transforms pulled from URDF
        proximal2distal = mjpc.posEuler2Mat([0.044, -0.003, 0], [0, 0, 0])
        hand2proximal1 = mjpc.posEuler2Mat([0.00279, 0.03126, -0.11467], [-1.570796327, .649262481663582, 1.35961148639407])
        hand2proximal2 = mjpc.posEuler2Mat([0.02226, -0.02707, -0.11482], [-1.570796327, .649262481663582, -1.38614049188413])
        hand2proximal3 = mjpc.posEuler2Mat([-0.02226, -0.02707, -0.11482], [-1.570796327, .649262481663582, -1.75545216211587])

        finger_1_proximal_mesh.transform(hand2proximal1)
        finger_1_distal_mesh.transform(np.matmul(hand2proximal1, proximal2distal))
        finger_2_proximal_mesh.transform(hand2proximal2)
        finger_2_distal_mesh.transform(np.matmul(hand2proximal2, proximal2distal))
        finger_3_proximal_mesh.transform(hand2proximal3)
        finger_3_distal_mesh.transform(np.matmul(hand2proximal3, proximal2distal))

        self.gripper_meshes = [gripper_mesh, finger_1_proximal_mesh, finger_1_distal_mesh, finger_2_proximal_mesh, finger_2_distal_mesh, finger_3_proximal_mesh, finger_3_distal_mesh]
        self.gripper_transform = None

    def transformGripperMesh(self, t_mat):
        rotation_about_x = np.eye(4)
        rotation_about_x[:3, :3] = mjpc.quat2Mat([0, 1, 0, 0])
        # by default, hand points into negative z; flip about x by pi
        hand_transform = np.matmul(t_mat, rotation_about_x)
        # origin isn't end effector frame, so pull it back by distance in URDF's j2s6s300_joint_end_effector
        hand_transform = gpg.translateFrameNegativeZ(hand_transform, -0.16)
        for mesh in self.gripper_meshes:
            # since mesh.transform transforms from the current position, set it back to the original pose
            #   first by transforming with inverse of last transform
            if self.gripper_transform is not None:
                mesh.transform(np.linalg.inv(self.gripper_transform))
            mesh.transform(hand_transform)
        self.gripper_transform = hand_transform

    def saveO3DScreenshot(self, filepath, filename, point_cloud, grasp_pose=None, labeled_grabstraction=None):
        vis_list = [point_cloud]
        if grasp_pose is not None:
            grasp_pose_to_visualize = mjpc.o3dTFAtPose(grasp_pose)
            self.transformGripperMesh(grasp_pose)
            vis_list.append(grasp_pose_to_visualize)
            vis_list += self.gripper_meshes

        full_filename_with_path = filepath + '/' + filename

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        view_ctrl = vis.get_view_control()
        view_parameters = o3d.io.read_pinhole_camera_parameters(self.visualization_view_param_file)
        original_intrinsics = view_ctrl.convert_to_pinhole_camera_parameters()
        # https://github.com/intel-isl/Open3D/issues/1164
        view_parameters.intrinsic = original_intrinsics.intrinsic
        for mesh in vis_list:
            vis.add_geometry(mesh)
            vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        view_ctrl.convert_from_pinhole_camera_parameters(view_parameters)
        screenshot = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        np_screenshot = np.asarray(screenshot)
        pil_screenshot = Image.fromarray(np.uint8(np_screenshot*255)).convert('RGB')
        draw = ImageDraw.Draw(pil_screenshot)
        if grasp_pose is not None and labeled_grabstraction is not None:
            pos, quat = mjpc.mat2PosQuat(grasp_pose)
            text_to_draw = "pos:\n    {:.4f}\n    {:.4f}\n    {:.4f}\nquat:\n    {:.4f}\n    {:.4f}\n    {:.4f}\n    {:.4f}\nGrabstraction:\n".format(pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]) + \
                ''.join(["    {:.4f}\n".format(v) for v in labeled_grabstraction])
            draw.text((0,0), text_to_draw, (0,0,0), font=self.vis_font)
        pil_screenshot.save(full_filename_with_path)

    def visualizationVideoSample(self, num_grasps_to_display=50, num_values_per_range=4, filepath="/home/mcorsaro/grabstraction_results/"):

        file_dir= filepath + '/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        os.mkdir(file_dir)

        for grasp_family_i, grabstracted_input in enumerate(self.grabstracted_inputs):
            grabstraction_dir = file_dir + '/' + str(grasp_family_i) + '/'
            os.mkdir(grabstraction_dir)
            grabstraction_ranges = np.array((grabstracted_input.min(0), grabstracted_input.max(0)))

            for dim_to_vary in range(self.embedding_dim):
                vals_to_vary = [grabstraction_ranges[0, dim_to_vary]+i*(grabstraction_ranges[1, dim_to_vary]-grabstraction_ranges[0, dim_to_vary])/(num_grasps_to_display-1) for i in range(num_grasps_to_display)]
                vals_in_other_dims = []
                for other_dim in range(self.embedding_dim):
                    if other_dim != dim_to_vary:
                        vals_in_other_dims.append([grabstraction_ranges[0, other_dim]+i*(grabstraction_ranges[1, other_dim]-grabstraction_ranges[0, other_dim])/(num_values_per_range-1) for i in range(num_values_per_range)])
                combinations_of_other_dim_vals = list(itertools.product(*vals_in_other_dims))
                for other_dim_val_combination in combinations_of_other_dim_vals:
                    for varied_val in vals_to_vary:
                        abstract_grasp_np = np.array(other_dim_val_combination[:dim_to_vary] + tuple([varied_val]) + other_dim_val_combination[dim_to_vary:])
                        np_grasp_pose = self.embeddings[grasp_family_i].inverse_transform(abstract_grasp_np)
                        grasp_pose = mjpc.npGraspArr2Mat(np_grasp_pose)
                        filename = str(dim_to_vary) + ''.join(["_{:.9f}".format(v) for v in abstract_grasp_np]) + ".jpg"
                        self.saveO3DScreenshot(grabstraction_dir, filename, self.cloud_with_normals, grasp_pose, abstract_grasp_np)

    def visualizationProjectManifold(self, filepath="/home/mcorsaro/grabstraction_results/"):
        file_dir= filepath + '/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        os.mkdir(file_dir)

        self.saveFamilyCloudScreenshot(file_dir)

        cloud_points = np.asarray(self.cloud_with_normals.points)
        cloud_color = np.empty((cloud_points.shape))
        min_z, max_z = cloud_points.min(0)[2], cloud_points.max(0)[2]
        cloud_angle = np.arctan2(cloud_points[:,1], cloud_points[:,0])
        cloud_angle = cloud_angle - (math.pi*2 *(cloud_angle > math.pi/2))
        min_a, max_a = cloud_angle.min(), cloud_angle.max()
        for point_i in range(cloud_points.shape[0]):
            z = cloud_points[point_i, 2]
            a = cloud_angle[point_i]
            # multiply by 0.8 to avoid end of spectrum where purple turns to pink
            z_rel = (z-min_z)/(max_z-min_z)*0.8
            a_rel = (a-min_a)/(max_a-min_a)
            cloud_color[point_i] = hls_to_rgb(z_rel, a_rel, 0.5)
        self.cloud_with_normals.colors = o3d.utility.Vector3dVector(cloud_color)
        #obj_axes = mjpc.o3dTFAtPose(self.obj_frame)

        cloud_filename = 'cloud_screenshot.jpg'
        self.saveO3DScreenshot(file_dir, cloud_filename, self.cloud_with_normals)

        for grasp_family_i in range(len(self.grabstracted_inputs)):

            grabstraction_filename = 'grabstraction_' + str(grasp_family_i) + '.jpg'
            grabstracted_input = self.grabstracted_inputs[grasp_family_i]

            if grabstracted_input.shape[1] == 2:
                plt.scatter(grabstracted_input[:, 0], grabstracted_input[:, 1], c=cloud_color[self.grasp_family_indices[grasp_family_i]])
                plt.savefig(file_dir + '/' + grabstraction_filename)
                plt.clf()
            elif grabstracted_input.shape[1] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(grabstracted_input[:, 0], grabstracted_input[:, 1], grabstracted_input[:, 2], c=cloud_color[self.grasp_family_indices[grasp_family_i]])
                plt.savefig(file_dir + '/' + grabstraction_filename)
                plt.clf()

    def saveFamilyCloudScreenshot(self, file_dir):
        family_cloud = copy.deepcopy(self.cloud_with_normals)
        family_cloud_points = np.asarray(family_cloud.points)
        cloud_color = np.empty((family_cloud_points.shape))
        unique_colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (255, 128, 128), (128, 128, 128), (128, 0, 0), (255, 128, 0)]
        for fam_i, family_indices in enumerate(self.grasp_family_indices):
            for point_index in family_indices:
                cloud_color[point_index] = unique_colors[fam_i]
        family_cloud.colors = o3d.utility.Vector3dVector(cloud_color)
        self.saveO3DScreenshot(file_dir, 'grasp_families.jpg', family_cloud)

    def clusterGraspsIntoFamilyIndices(self, original_space):
        # TODO(mcorsaro): algorithm that works on any object
        if self.obj == 'door':
            grasp_family_space_indices = range(original_space.shape[0])
            return [grasp_family_space_indices]
        if self.obj == 'cylinder':
            indices_on_top = []
            indices_on_side = []
            max_z = original_space.max(0)[2]
            for i in range(original_space.shape[0]):
                if original_space[i, 2] >= max_z-0.005:
                    indices_on_top.append(i)
                else:
                    indices_on_side.append(i)
            grasp_family_space_indices = [indices_on_top, indices_on_side]
            return grasp_family_space_indices

    def generateGrabstraction(self, compression_alg="pca", embedding_dim=3):

        self.loadGripperMesh()
        # TODO(mcorsaro): choose manually per grasp family
        self.embedding_dim=embedding_dim

        #TODO(mcorsaro): grasp_poses, cloud_with_normals, and original_full_space should be condensed into one variable
        # right now, we assume same size, I think

        # use quaternion as placeholder for rotation, but they're not continuous.. see https://arxiv.org/pdf/1812.07035.pdf
        grasp_pose_space = np.empty((len(self.grasp_poses), 7))
        normals = np.asarray(self.cloud_with_normals.normals)
        point_normal_space = np.empty((len(self.grasp_poses), 6))
        for i, grasp_pose in enumerate(self.grasp_poses):
            grasp_position, grasp_orientation = mjpc.mat2PosQuat(grasp_pose)
            normal = (normals[i]).tolist()
            point_normal_space[i] = grasp_position + normal
            grasp_pose_space[i] = grasp_position + grasp_orientation
        #self.original_space = self.original_space[:, :3]

        original_full_space = point_normal_space#grasp_pose_space
        # after clustering
        self.grasp_family_indices = self.clusterGraspsIntoFamilyIndices(original_full_space)
        grasp_family_spaces = [original_full_space[ind_list] for ind_list in self.grasp_family_indices]

        if compression_alg=="isomap":
            # Isomap isn't invertible.. https://openreview.net/forum?id=iox4AjpZ15
            self.embeddings = [Isomap(n_neighbors=250, n_components=self.embedding_dim) for grasp_families in grasp_family_spaces]
            self.grabstracted_inputs = [self.embeddings[i].fit_transform(grasp_family_space) for i, grasp_family_space in enumerate(grasp_family_spaces)]

        elif compression_alg=="pca":
            self.embeddings = [PCA(n_components=self.embedding_dim) for grasp_families in grasp_family_spaces]
            self.grabstracted_inputs = [self.embeddings[i].fit_transform(grasp_family_space) for i, grasp_family_space in enumerate(grasp_family_spaces)]

        '''
        if embedding_dim<=3:
            #https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
            grabstracted_inputs_normed = (self.grabstracted_inputs - self.grabstracted_inputs.min(0)) / self.grabstracted_inputs.ptp(0)
            # add a column of zeros if less than 3 dims
            if embedding_dim<3:
                grabstracted_inputs_normed = np.append(grabstracted_inputs_normed, np.zeros((grabstracted_inputs_normed.shape[0], 1)), axis=1)
            # add an additional column of zeros if originally 1-dim
            if embedding_dim==1:
                grabstracted_inputs_normed = np.append(grabstracted_inputs_normed, np.zeros((grabstracted_inputs_normed.shape[0], 1)), axis=1)
            self.cloud_with_normals.colors = o3d.utility.Vector3dVector(grabstracted_inputs_normed)
        '''

        '''
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        min_np_grasp = self.embedding.inverse_transform(self.grabstracted_inputs.min(0))
        min_grasp_pose = mjpc.npGraspArr2Mat(min_np_grasp)
        grasp_pose_to_visualize = mjpc.o3dTFAtPose(min_grasp_pose)
        self.transformGripperMesh(min_grasp_pose)
        o3d.visualization.draw_geometries([world_axes, self.cloud_with_normals, grasp_pose_to_visualize] + self.gripper_meshes)

        max_np_grasp = self.embedding.inverse_transform(self.grabstracted_inputs.max(0))
        max_grasp_pose = mjpc.npGraspArr2Mat(max_np_grasp)
        grasp_pose_to_visualize = mjpc.o3dTFAtPose(max_grasp_pose)
        self.transformGripperMesh(max_grasp_pose)
        o3d.visualization.draw_geometries([world_axes, self.cloud_with_normals, grasp_pose_to_visualize] + self.gripper_meshes)
        '''
