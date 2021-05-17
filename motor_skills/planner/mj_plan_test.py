import numpy as np
import datetime
import copy
import math
import time
import os

from PIL import Image
import matplotlib.pyplot as plt

import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.planner.pbplanner import PbPlanner

import motor_skills.planner.mj_point_clouds as mjpc
import motor_skills.planner.grasp_pose_generator as gpg
import motor_skills.planner.grabstractor as grb
import motor_skills.planner.learnTaskGraspClassifier as tgc

from mujoco_py import cymj
from mujoco_py import load_model_from_path, MjSim, MjViewer

import open3d as o3d

# mujoco to pybullet
def wxyz2xyzw(wxyz):
    return wxyz[1:] + [wxyz[0]]
# pybullet to mujoco
def xyzw2wxyz(xyzw):
    return [xyzw[-1]] + list(xyzw[:-1])

class MujocoPlanExecutor(object):
    def __init__(self, obj, visualize=True):
        super(MujocoPlanExecutor, self).__init__()
        self.obj = obj

        motor_skills_dir = "/home/mcorsaro/.mujoco/motor_skills/"
        model_file = None
        if self.obj == 'door':
            model_file = 'motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
        elif self.obj == 'cylinder':
            model_file = 'motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_cylinder.xml'
        elif self.obj == 'box':
            model_file = 'motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_box.xml'
        model = load_model_from_path(motor_skills_dir + '/' + model_file)
                #load_model_from_path(motor_skills_dir + '/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_nodoor.xml')
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim) if visualize else None

        ##### Fingers #####
        # compute indices and per timestep delta q
        # Arm DoF
        self.aDOF=6
        # Gripper DoF
        self.gDOF = 6
        # Total robot dof
        self.tDOF = self.aDOF + self.gDOF
        self.max_finger_delta = 1.3
        self.grasp_steps = 500
        self.pregrasp_steps = 500
        self.pregrasp_goal = [0., 0., -0.04]
        self.num_links_per_finger = 2

        self.finger_joint_idxs = []
        self.finger_base_idxs = []
        self.finger_tip_idxs = []
        for i in range(1,int(self.gDOF/self.num_links_per_finger) + 1):
            base_idx = self.sim.model.joint_name2id("j2s6s300_joint_finger_" + str(i))
            tip_idx = self.sim.model.joint_name2id("j2s6s300_joint_finger_tip_" + str(i))
            self.finger_joint_idxs.append(base_idx)
            self.finger_joint_idxs.append(tip_idx)
            self.finger_base_idxs.append(base_idx)
            self.finger_tip_idxs.append(tip_idx)

        door_bounds = [(-2., -2., 0.05), (2., 2., 2.)]
        handle_bounds = [(0.189, -2., 0.05), (2., 0.4, 2.)]
        cylinder_bounds = [(-1, 0.1, 0.01), (1., 1., 1.)]
        box_bounds = [(-1, 0.1, 0.01), (1., 1., 1.)]
        crop_bounds = None
        if self.obj == 'door':
            crop_bounds = handle_bounds
        elif self.obj == 'cylinder':
            crop_bounds = cylinder_bounds
        elif self.obj == 'box':
            crop_bounds = box_bounds
        self.pc_gen = mjpc.PointCloudGenerator(self.sim, min_bound=crop_bounds[0], max_bound=crop_bounds[1])

        self.door_dofs = None if self.obj != 'door' else [self.sim.model.joint_name2id('door_hinge'), self.sim.model.joint_name2id('latch')]
        self.finger_joint_range = self.sim.model.jnt_range[:self.tDOF, ]

    def mj_render(self):
        if self.viewer is not None:
            self.viewer.render()

    def executePlan(self, plan):
        try:
            plan.interpolate(200)
            H = plan.getStateCount()
            for t in range(H):
                state_t = plan.getState(t)
                target_q = []
                target_qd = []
                for i in range(self.aDOF):
                    target_q.append(state_t[i])
                    target_qd.append(0.0)

                torques=mjc.pd(None, target_qd, target_q, self.sim, ndof=self.aDOF, kp=np.eye(self.aDOF)*300)
                self.sim.data.ctrl[:self.aDOF]=torques
                self.sim.step()
                self.mj_render()
                time.sleep(0.01)

            # Make sure it reaches the goal
            for t in range(200):
                torques=mjc.pd(None, target_qd, target_q, self.sim, ndof=self.aDOF, kp=np.eye(self.aDOF)*100)
                self.sim.data.ctrl[:self.aDOF]=torques
                self.sim.step()
                self.mj_render()
            return True
        except:
            return False

    # https://github.com/babbatem/motor_skills/blob/impedance/motor_skills/cip/MjGraspHead.py
    def closeFingers(self):
        # % close fingers
        new_pos = copy.deepcopy(self.sim.data.qpos[:self.tDOF])
        for t in range(self.grasp_steps):
            # % see which sensors are reporting force
            # TODO(mcorsaro): confirm that these values are reported correctly - why the first 6
            touched = np.where(self.sim.data.sensordata[:6] != 0.0)[0]

            # check each finger to determine if links are in contact

            fingers_stopped = [False]*int(self.gDOF/self.num_links_per_finger)
            touched_finger_idxs = [touched_link + self.aDOF for touched_link in touched]
            for finger_i in range(int(self.gDOF/self.num_links_per_finger)):
                base_idx = self.finger_base_idxs[finger_i]
                tip_idx = self.finger_tip_idxs[finger_i]
                if tip_idx in touched_finger_idxs:
                    fingers_stopped[finger_i] = True
                elif base_idx in touched_finger_idxs:
                    new_pos[tip_idx] += self.max_finger_delta/self.grasp_steps
                else:
                    new_pos[base_idx] += self.max_finger_delta/self.grasp_steps
                    new_pos[tip_idx] += self.max_finger_delta/self.grasp_steps

            if len(fingers_stopped) == sum(fingers_stopped):
                break

            # Make sure goals are within limits
            for f_idx in self.finger_joint_idxs:
                new_pos[f_idx] = max(self.finger_joint_range[f_idx, 0], new_pos[f_idx])
                new_pos[f_idx] = min(self.finger_joint_range[f_idx, 1], new_pos[f_idx])

            # % compute torque and step
            self.sim.data.ctrl[:] = mjc.pd([0] * self.tDOF, [0] * self.tDOF, new_pos, self.sim, ndof=self.tDOF, kp=np.eye(self.tDOF)*300)
            self.sim.forward()
            self.sim.step()

            self.mj_render()

        # Make sure goals are within limits
        for f_idx in self.finger_joint_idxs:
            new_pos[f_idx] = max(self.finger_joint_range[f_idx, 0], new_pos[f_idx])
            new_pos[f_idx] = min(self.finger_joint_range[f_idx, 1], new_pos[f_idx])

        # Make sure it reaches the goal
        for t in range(200):
            self.sim.data.ctrl[:] = mjc.pd([0] * self.tDOF, [0] * self.tDOF, new_pos, self.sim, ndof=self.tDOF, kp=np.eye(self.tDOF)*300)
            self.sim.step()
            self.mj_render()

    def resetDoor(self):
        for mj_id in self.door_dofs:
            self.sim.data.qpos[mj_id]=0.0

    def setUpSimAndGenClouds(self, prm_file="/home/mcorsaro/Desktop/TESTPRM"):
        self.planner = PbPlanner(self.obj, prm_file)
        #start = planner.validityChecker.sample_state()
        self.start_joints = [0, np.pi, np.pi, 0, np.pi, 0]
        self.sim.data.qpos[:self.aDOF] = self.start_joints
        self.sim.step()
        self.mj_render()

        self.planner.validityChecker.open_finger_state = copy.deepcopy(self.sim.data.qpos[self.aDOF:self.tDOF])
        self.planner.validityChecker.current_finger_state = copy.deepcopy(self.sim.data.qpos[self.aDOF:self.tDOF])

        self.cloud_with_normals = self.pc_gen.generateCroppedPointCloud()

    def setUpSimAndGenCloudsAndGenCandidates(self, rotation_values_about_approach=[0, math.pi/2], prm_file="/home/mcorsaro/Desktop/TESTPRM"):
        self.setUpSimAndGenClouds(prm_file)
        num_points = np.asarray(self.cloud_with_normals.points).shape[0]
        pose_gen = gpg.GraspPoseGenerator(self.cloud_with_normals, rotation_values_about_approach=rotation_values_about_approach)

        self.grasp_poses = []
        for i in range(num_points):
            self.grasp_poses += pose_gen.proposeGraspPosesAtCloudIndex(i)

        print("Generated", len(self.grasp_poses), "grasp poses.")

    def executeGrasp(self, sampled_pose):

        start_time = time.time()

        error_code = None

        grasp_pose = copy.deepcopy(sampled_pose)
        pregrasp_pose = gpg.translateFrameNegativeZ(grasp_pose, self.dist_from_point_to_ee_link + self.pregrasp_dist)
        grasp_pose = gpg.translateFrameNegativeZ(grasp_pose, self.dist_from_point_to_ee_link)
        # Pregrasp and grasp orientation are the same, so just use grasp_or
        pregrasp_position, _ = mjpc.mat2PosQuat(pregrasp_pose)
        grasp_position, grasp_orientation = mjpc.mat2PosQuat(grasp_pose)

        open_grasp_goal_pos, open_grasp_goal_quat = mjpc.mat2PosQuat(np.matmul(self.handle_transform, grasp_pose))

        # TODO(mcorsaro): reset pybullet
        self.sim.reset()
        self.sim.data.qpos[:self.aDOF] = self.start_joints
        self.sim.data.qpos[self.aDOF:self.tDOF] = self.planner.validityChecker.open_finger_state
        self.sim.step()
        self.planner.validityChecker.checking_other_ids = True

        # compute pre-grasp joint state, check for collision
        self.planner.validityChecker.updateFingerState(self.planner.validityChecker.open_finger_state)
        pregrasp_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, pregrasp_position, wxyz2xyzw(grasp_orientation), starting_state=self.start_joints)
        pregrasp_invalid_code = self.planner.validityChecker.isInvalid(pregrasp_goal)
        if pregrasp_invalid_code:
            error_code = pregrasp_invalid_code
        else:
            # compute grasp pose, check for collision
            grasp_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, grasp_position, wxyz2xyzw(grasp_orientation), starting_state=pregrasp_goal)
            grasp_invalid_code = self.planner.validityChecker.isInvalid(grasp_goal)
            if grasp_invalid_code:
                error_code = grasp_invalid_code+3
            else:
                # Stop checking for collisions with door
                self.planner.validityChecker.checking_other_ids = False
                open_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, open_grasp_goal_pos, wxyz2xyzw(open_grasp_goal_quat), starting_state=grasp_goal)
                open_invalid_code = self.planner.validityChecker.isInvalid(open_goal)
                # Checking for collisions with door again
                self.planner.validityChecker.checking_other_ids = True
                if open_invalid_code:
                    error_code = open_invalid_code+6
                #elif True:
                #    error_code = 0
                else:
                    current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
                    pregrasp_path=self.planner.plan(current_joint_vals, pregrasp_goal, check_validity=False)
                    ps = time.time()
                    if not self.executePlan(pregrasp_path):
                        error_code = 9
                    else:
                        print("Executed pregrasp in", time.time()-ps)
                        current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
                        current_pos, current_quat_xyzw = self.planner.calculateForwardKinematics(0, self.aDOF, current_joint_vals.tolist())
                        current_quat = xyzw2wxyz(current_quat_xyzw)
                        #print("Error after execution:", self.planner.distBetweenPoses(current_pos, pregrasp_position, current_quat, grasp_orientation))

                        current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
                        grasp_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, grasp_position, wxyz2xyzw(grasp_orientation), starting_state=current_joint_vals.tolist())
                        grasp_invalid_code = self.planner.validityChecker.isInvalid(grasp_goal)
                        if grasp_invalid_code:
                            error_code = grasp_invalid_code+10
                        else:
                            current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
                            grasp_path=self.planner.plan(current_joint_vals, grasp_goal, check_validity=False)
                            gs = time.time()
                            if not self.executePlan(grasp_path):
                                error_code = 13
                            else:
                                print("Executed grasp in", time.time()-gs)
                                #current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
                                #print("Grasp pose", self.planner.calculateForwardKinematics(0, self.aDOF, current_joint_vals.tolist()))
                                #print("Grasp desired pose", grasp_position, grasp_orientation)
                                fs = time.time()
                                self.closeFingers()
                                print("Closed fingers in", time.time()-fs)
                                self.planner.validityChecker.updateFingerState(copy.deepcopy(self.sim.data.qpos[self.aDOF:self.tDOF]))
                                #TODO(mcorsaro): don't look for collision between fingers and door

                                # Stop checking for collisions with door
                                self.planner.validityChecker.checking_other_ids = False
                                open_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, open_grasp_goal_pos, wxyz2xyzw(open_grasp_goal_quat), starting_state=current_joint_vals.tolist())
                                open_invalid_code = self.planner.validityChecker.isInvalid(open_goal)
                                if open_invalid_code:
                                    error_code = open_invalid_code+14
                                    self.planner.validityChecker.checking_other_ids = True
                                else:
                                    current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
                                    open_path=self.planner.plan(current_joint_vals, open_goal, check_validity=False)
                                    os = time.time()
                                    if not self.executePlan(open_path):
                                        error_code = 17
                                        self.planner.validityChecker.checking_other_ids = True
                                    else:
                                        print("Executed opening in", time.time()-os)
                                        # Checking for collisions with door again
                                        self.planner.validityChecker.checking_other_ids = True
                                        error_code = 0
        return (copy.deepcopy(self.sim.data.qpos[self.tDOF:]), error_code, time.time()-start_time)

    def generateData(self):

        self.setUpSimAndGenCloudsAndGenCandidates(rotation_values_about_approach=[0])
        fam_gen = grb.Grabstractor(self.cloud_with_normals, self.grasp_poses, obj=self.obj)

        self.world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        result_error_codes = []
        result_door_states = []
        result_time = []
        c = 0
        batch_start_time = time.time()
        print("Now attempting", len(self.grasp_poses), "grasp poses.")
        self.pregrasp_dist = 0.15
        self.dist_from_point_to_ee_link = 0.01

        handle_translation = mjpc.posRotMat2Mat([-0.14, -0.348, -0.415], mjpc.quat2Mat([1, 0, 0, 0]))
        # point centered at handle origin, orientation is 90 degrees past y axis
        handle_rotation = mjpc.posRotMat2Mat([0, 0, 0], mjpc.quat2Mat([0.7071068, 0, 0.7071068, 0]))
        self.handle_transform = np.matmul(np.matmul(np.linalg.inv(handle_translation), handle_rotation), handle_translation)

        # The arm should be straight up
        '''current_joint_vals = copy.deepcopy(self.sim.data.qpos[:self.aDOF])
        initial_position, initial_orientation = self.planner.calculateForwardKinematics(0, self.aDOF, current_joint_vals.tolist())
        initial_position, initial_orientation = list(initial_position), list(initial_orientation)
        initial_axes = mjpc.o3dTFAtPose(mjpc.posRotMat2Mat(initial_position, mjpc.quat2Mat(initial_orientation)))
        o3d.visualization.draw_geometries([self.cloud_with_normals, world_axes, initial_axes])
        print("Initial pose", initial_position, initial_orientation)'''
        # y is approach, z is closing (2 to 1), x is negative x (left, looking towards door)
        for grasp_pose in self.grasp_poses:

            door_state, error_code, runtime = self.executeGrasp(grasp_pose)
            result_door_states.append(door_state)
            result_error_codes.append(error_code)
            result_time.append(runtime)
            c+=1
            if c%100==0:
                print(c, "in", time.time()-batch_start_time)
                batch_start_time=time.time()

        label_types = list(np.unique(result_error_codes))
        label_counts = [(l, result_error_codes.count(l)) for l in label_types]
        print("Times:", result_time)
        print("Completed attempts for all generated grasp poses:", label_counts)
        for label_type in label_types:
            indices = np.where(np.array(result_error_codes) == label_type)[0]
            all_times_this_label = [result_time[ind] for ind in indices]
            print("Average, min, max time for label", label_type, sum(all_times_this_label)/len(all_times_this_label), min(all_times_this_label), max(all_times_this_label))

        result_file = "/home/mcorsaro/grabstraction_results/trial_results_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S') + '.txt'
        rf = open(result_file, "w")
        for i in range(len(self.grasp_poses)):
            gp, go = mjpc.mat2PosQuat(self.grasp_poses[i])
            rf.write(str(result_error_codes[i]) + ' ' + str(result_door_states[i].tolist()) + ' ' + str(gp) + ' ' + str(go) + '\n')
        fam_gen.visualizeGraspLabels(result_error_codes)
        rf.close()

def loadGraspFile(filename, filepath="/home/mcorsaro/grabstraction_results/"):
    grasp_label_file = filepath + '/' + filename
    f=open(grasp_label_file, 'r')
    error_codes, door_states, grasp_poses = [], [], []
    lines = f.readlines()
    for line in lines:
        line = line.replace('[','').replace(']','').replace(',','')
        split_line = line.split()
        error_code = int(split_line[0])
        door_state = [float(val) for val in split_line[1:3]]
        grasp_pos = [float(val) for val in split_line[3:6]]
        grasp_quat = [float(val) for val in split_line[6:]]
        grasp_pose = mjpc.posRotMat2Mat(grasp_pos, mjpc.quat2Mat(grasp_quat))
        error_codes.append(error_code)
        door_states.append(door_state)
        grasp_poses.append(grasp_pose)
    return error_codes, door_states, grasp_poses

def errorCodesAndDoorStatesToLabels(error_codes, door_states, grasp_poses, handle=True, thresh=0.2):
    doorstate_array = np.array(door_states)
    ec0_indices = []
    for i in range(len(error_codes)):
        if error_codes[i] == 0:
            ec0_indices.append(i)
    doorstate_array_ec0 = doorstate_array[ec0_indices]
    labels = None
    if handle:
        labels = doorstate_array_ec0[:,1] > thresh
    else:
        print("Only handle opening verification implemented so far.")
        sys.exit()
    #print(np.sum(labels), labels.shape)
    return labels, ec0_indices

def averageOverSeeds(fam_gen, labels, indices, train_percent, input_feature_size='full', num_seeds_to_avg_over=3):
    this_run_test_accs = []
    this_run_best_params = []
    train_sizes = None
    for j in range(num_seeds_to_avg_over):
        clf = tgc.TaskGraspClassifier(fam_gen, labels, indices, input_feature_size=input_feature_size, percent_train_set_to_use=train_percent)
        # Doesn't work if there's not at least one positive and one negative label, so try 5 times until it's balanced
        single_run_test_accs, single_run_best_params, train_set_size = None, None, None
        for attempt in range(5):
            try:
                single_run_test_accs, single_run_best_params, train_set_size = clf.gridSearch()
                break
            except:
                print("Attempt failed, trying again.")
                continue
        if single_run_test_accs is None:
            print("Failed 5 times in a row with parameters", train_percent)
            sys.exit()
        train_sizes = train_set_size
        this_run_test_accs.append(single_run_test_accs)
        this_run_best_params.append(single_run_best_params)
        print("Achieved", single_run_test_accs, "test accuracy with parameters", single_run_best_params)
    return this_run_test_accs, this_run_best_params, train_sizes

if __name__ == '__main__':
    obj = 'door'
    mjp = MujocoPlanExecutor(obj=obj)

    task = 'train_classifier'

    if task == 'generate_data':
        # Generate data
        mjp.generateData()

    elif task == 'learn_manifold':
        # Generate low-D manifold
        mjp.setUpSimAndGenCloudsAndGenCandidates(rotation_values_about_approach=[0], prm_file=None)
        fam_gen = grb.Grabstractor(mjp.cloud_with_normals, mjp.grasp_poses, obj=obj)
        fam_gen.generateGrabstraction(compression_alg="pca", embedding_dim=2)
        fam_gen.visualizationVideoSample()
        #fam_gen.visualizeGraspPoses(vis_every_n=200)
        fam_gen.visualizationProjectManifold()

    elif task == 'train_classifier':
        # Learn labels
        mjp.setUpSimAndGenClouds(prm_file=None)
        loaded_grasp_error_codes, loaded_grasp_door_states, loaded_grasp_poses = loadGraspFile("door_labels_turn.txt")
        fam_gen = grb.Grabstractor(mjp.cloud_with_normals, loaded_grasp_poses, obj=obj)
        # convert error codes (kinematics) and door states to binary labels
        labels, indices = errorCodesAndDoorStatesToLabels(loaded_grasp_error_codes, loaded_grasp_door_states, loaded_grasp_poses)

        train_sizes = []
        test_accs = []
        test_std_devs = []
        best_params = []

        ae_train_sizes = []
        ae_test_accs = []
        ae_test_std_devs = []
        ae_best_params = []

        train_set_size_percentages = [0.01, 0.1]

        for train_percent in train_set_size_percentages:
            this_run_test_accs, this_run_best_params, this_run_train_sizes = averageOverSeeds(fam_gen, labels, indices, train_percent, input_feature_size='full')
            test_accs.append(sum(this_run_test_accs)/len(this_run_test_accs))
            test_std_devs.append(np.std(this_run_test_accs))
            train_sizes.append(this_run_train_sizes)
            best_params.append(this_run_best_params)

            ae_this_run_test_accs, ae_this_run_best_params, ae_this_run_train_sizes = averageOverSeeds(fam_gen, labels, indices, train_percent, input_feature_size='compressed')
            ae_test_accs.append(sum(ae_this_run_test_accs)/len(ae_this_run_test_accs))
            ae_test_std_devs.append(np.std(ae_this_run_test_accs))
            ae_train_sizes.append(ae_this_run_train_sizes)
            ae_best_params.append(ae_this_run_best_params)

        plot_dir = "/home/mcorsaro/grabstraction_results/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S') + '/'
        os.mkdir(plot_dir)

        print(train_sizes, test_accs)
        plt.plot(train_sizes, test_accs, label="Pose")
        plt.plot(ae_train_sizes, ae_test_accs, label="3D Autoencoder")
        plt.legend(loc='lower right')
        plt.savefig(plot_dir + "test_accs.jpg")
        plt.close()

        plt.errorbar(train_sizes, test_accs, yerr=test_std_devs, label="Pose")
        plt.errorbar(ae_train_sizes, ae_test_accs, yerr=ae_test_std_devs, label="3D Autoencoder")
        plt.legend(loc='lower right')
        plt.savefig(plot_dir + "test_accs_err.jpg")

        '''fam_gen.visualizeGraspLabelsWithErrorCodes(labels, loaded_grasp_error_codes, indices)
        color_labeled_cloud = fam_gen.visualizeGraspLabels(loaded_grasp_error_codes)
        time.sleep(1)
        fam_gen.visualizeGraspPoses(vis_every_n=1, error_codes=loaded_grasp_error_codes, given_cloud=color_labeled_cloud)'''
