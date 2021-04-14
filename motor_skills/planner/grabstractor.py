import datetime
import os
import time

import itertools

import numpy as np
import open3d as o3d

from PIL import Image, ImageFont, ImageDraw

import motor_skills.planner.mj_point_clouds as mjpc
import motor_skills.planner.grasp_pose_generator as gpg

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

class Grabstractor(object):
    def __init__(self, cloud_with_normals, grasp_poses):
        self.cloud_with_normals = cloud_with_normals
        self.grasp_poses = grasp_poses
        self.vis_font = font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 33)

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

    def saveO3DScreenshot(self, grasp_pose, labeled_grabstraction, filepath, filename, view_param_file="/home/mcorsaro/.mujoco/motor_skills/motor_skills/planner/DoorOpen3DCamPose.json"):
        grasp_pose_to_visualize = mjpc.o3dTFAtPose(grasp_pose)
        self.transformGripperMesh(grasp_pose)
        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        vis_list = [world_axes, self.cloud_with_normals, grasp_pose_to_visualize] + self.gripper_meshes

        full_filename_with_path = filepath + '/' + filename

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        view_ctrl = vis.get_view_control()
        view_parameters = o3d.io.read_pinhole_camera_parameters(view_param_file)
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
        pos, quat = mjpc.mat2PosQuat(grasp_pose)
        text_to_draw = "pos:\n    {:.4f}\n    {:.4f}\n    {:.4f}\nquat:\n    {:.4f}\n    {:.4f}\n    {:.4f}\n    {:.4f}\nGrabstraction:\n".format(pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]) + \
            ''.join(["    {:.4f}\n".format(v) for v in labeled_grabstraction])
        draw.text((0,0), text_to_draw, (0,0,0), font=self.vis_font)
        pil_screenshot.save(full_filename_with_path)

    def visualizationVideoSample(self, num_grasps_to_display=50, num_values_per_range=4, filepath="/home/mcorsaro/grabstraction_results/"):

        grabstraction_ranges = np.array((self.grabstracted_inputs.min(0), self.grabstracted_inputs.max(0)))

        file_dir= filepath + '/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
        os.mkdir(file_dir)

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
                    np_grasp_pose = self.embedding.inverse_transform(abstract_grasp_np)
                    grasp_pose = mjpc.npGraspArr2Mat(np_grasp_pose)
                    filename = str(dim_to_vary) + ''.join(["_{:.4f}".format(v) for v in abstract_grasp_np]) + ".jpg"
                    self.saveO3DScreenshot(grasp_pose, abstract_grasp_np, file_dir, filename)

    def generateGrabstraction(self, compression_alg="pca", embedding_dim=2):

        self.loadGripperMesh()
        self.embedding_dim=embedding_dim

        # use quaternion as placeholder for rotation, but they're not continuous.. see https://arxiv.org/pdf/1812.07035.pdf
        self.grasp_pose_space = np.empty((len(self.grasp_poses), 7))
        for i, grasp_pose in enumerate(self.grasp_poses):
            grasp_position, grasp_orientation = mjpc.mat2PosQuat(grasp_pose)
            self.grasp_pose_space[i] = grasp_position + grasp_orientation
        
        if compression_alg=="isomap":
            # Isomap isn't invertible.. https://openreview.net/forum?id=iox4AjpZ15
            self.embedding = Isomap(n_neighbors=250, n_components=self.embedding_dim)
            self.grabstracted_inputs = self.embedding.fit_transform(grasp_pose_space)

        elif compression_alg=="pca":
            self.embedding = PCA(n_components=self.embedding_dim)
            self.grabstracted_inputs = self.embedding.fit_transform(self.grasp_pose_space)

        if embedding_dim==3:
            #https://stackoverflow.com/questions/29661574/normalize-numpy-array-columns-in-python
            grabstracted_inputs_normed = (self.grabstracted_inputs - self.grabstracted_inputs.min(0)) / self.grabstracted_inputs.ptp(0)
            self.cloud_with_normals.colors = o3d.utility.Vector3dVector(grabstracted_inputs_normed)

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
