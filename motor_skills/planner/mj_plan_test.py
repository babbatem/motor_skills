import numpy as np
import time
import copy
import math
import time

from PIL import Image

import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.planner.pbplanner import PbPlanner

import motor_skills.planner.mj_point_clouds as mjpc
import motor_skills.planner.grasp_pose_generator as gpg
import motor_skills.planner.grabstractor as grb

from mujoco_py import cymj
from mujoco_py import load_model_from_path, MjSim, MjViewer

import open3d as o3d

class MujocoPlanExecutor(object):
    def __init__(self, load_door=True):
        super(MujocoPlanExecutor, self).__init__()

        self.load_door = load_door

        motor_skills_dir = "/home/mcorsaro/.mujoco/motor_skills/"

        model = load_model_from_path(motor_skills_dir + '/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_door.xml') if load_door else \
                load_model_from_path(motor_skills_dir + '/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_nodoor.xml')
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)

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
        self.pc_gen = mjpc.PointCloudGenerator(self.sim, min_bound=handle_bounds[0], max_bound=handle_bounds[1])

        self.door_dofs = [self.sim.model.joint_name2id('door_hinge'), self.sim.model.joint_name2id('latch')]
        self.finger_joint_range = self.sim.model.jnt_range[:self.tDOF, ]

    def executePlan(self, plan):
        plan.interpolate(1000)
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
            self.viewer.render()
            time.sleep(0.01)

        # Make sure it reaches the goal
        for t in range(200):
            torques=mjc.pd(None, target_qd, target_q, self.sim, ndof=self.aDOF, kp=np.eye(self.aDOF)*100)
            self.sim.data.ctrl[:self.aDOF]=torques
            self.sim.step()
            self.viewer.render()

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

            if self.viewer is not None:
                self.viewer.render()

        # Make sure goals are within limits
        for f_idx in self.finger_joint_idxs:
            new_pos[f_idx] = max(self.finger_joint_range[f_idx, 0], new_pos[f_idx])
            new_pos[f_idx] = min(self.finger_joint_range[f_idx, 1], new_pos[f_idx])

        # Make sure it reaches the goal
        for t in range(200):
            self.sim.data.ctrl[:] = mjc.pd([0] * self.tDOF, [0] * self.tDOF, new_pos, self.sim, ndof=self.tDOF, kp=np.eye(self.tDOF)*300)
            self.sim.step()
            if self.viewer != None:
                self.viewer.render()

    def resetDoor(self):
        for mj_id in self.door_dofs:
            self.sim.data.qpos[mj_id]=0.0

    def setUpSimAndGenCloudsAndGenCandidates(self, rotation_values_about_approach=[0, math.pi/2]):
        self.planner = PbPlanner(self.load_door)
        #start = planner.validityChecker.sample_state()
        self.start_joints = [0, np.pi, np.pi, 0, np.pi, 0]
        self.sim.data.qpos[:self.aDOF] = self.start_joints
        self.sim.step()
        self.viewer.render()

        self.cloud_with_normals = self.pc_gen.generateCroppedPointCloud()
        num_points = np.asarray(self.cloud_with_normals.points).shape[0]
        pose_gen = gpg.GraspPoseGenerator(self.cloud_with_normals, rotation_values_about_approach=rotation_values_about_approach)

        self.grasp_poses = []
        for i in range(num_points):
            self.grasp_poses += pose_gen.proposeGraspPosesAtCloudIndex(i)

        print("Generated", len(self.grasp_poses), "grasp poses.")

    def generateData(self):

        self.setUpSimAndGenCloudsAndGenCandidates()

        world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
        '''
        0 - attempt failed
        1 - attempt succeeded
        2 - pregrasp ik failed
        3 - grasp ik failed
        '''
        result_labels = []
        result_time = []
        c = 0
        batch_start_time = time.time()
        for grasp_pose in self.grasp_poses:
            start_time = time.time()
            pregrasp_pose = gpg.translateFrameNegativeZ(grasp_pose, 0.15)

            # Pregrasp and grasp orientation are the same, so just use grasp_or
            pregrasp_position, _ = mjpc.mat2PosQuat(pregrasp_pose)
            grasp_position, grasp_orientation = mjpc.mat2PosQuat(grasp_pose)

            self.sim.data.qpos[:self.aDOF] = self.start_joints
            self.sim.step()

            ##############################################################
            '''
            grasp_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, grasp_position, grasp_orientation)#, starting_state=pregrasp_goal)
            self.planner.validityChecker.isValid(grasp_goal)
            self.sim.data.qpos[:6] = grasp_goal[:6]
            self.sim.step()
            o3d.visualization.draw_geometries([world_axes, self.cloud_with_normals, mjpc.o3dTFAtPose(grasp_pose), mjpc.o3dTFAtPose(pregrasp_pose)])
            while True:
                self.viewer.render()
            print(grasp_goal)
            '''
            ################################################################

            pregrasp_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, pregrasp_position, grasp_orientation)
            if not self.planner.validityChecker.isValid(pregrasp_goal):
                result_labels.append(2)
            else:
                grasp_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, grasp_position, grasp_orientation, starting_state=pregrasp_goal)
                if not self.planner.validityChecker.isValid(grasp_goal):
                    result_labels.append(3)
                else:
                    result_labels.append(1)
            result_time.append(time.time()-start_time)
            c+=1
            if c%100==0:
                print(c, "in", time.time()-batch_start_time)
                batch_start_time=time.time()

            #o3d.visualization.draw_geometries([world_axes, self.cloud_with_normals, mjpc.o3dTFAtPose(grasp_pose), mjpc.o3dTFAtPose(pregrasp_pose)])
        label_types = list(np.unique(result_labels))
        label_counts = [(l, result_labels.count(l)) for l in label_types]
        print("Times:", result_time)
        print("Completed attempts for all generated grasp poses:", label_counts)
        for label_type in label_types:
            indices = np.where(np.array(result_labels) == label_type)[0]
            all_times_this_label = [result_time[ind] for ind in indices]
            print("Average, min, max time for label", label_type, sum(all_times_this_label)/len(all_times_this_label), min(all_times_this_label), max(all_times_this_label))

        while True:
            self.viewer.render()

        print("Now planning pre-grasp motion.")
        pregrasp_path=self.planner.plan(start, pregrasp_goal)
        _=input('enter to start execution')

        self.executePlan(pregrasp_path)

        grasp_goal = self.planner.accurateCalculateInverseKinematics(0, self.aDOF, grasp_position, grasp_orientation)
        print("Now planning grasp motion.")
        current_joint_vals = self.sim.data.qpos[:self.aDOF]
        grasp_path=self.planner.plan(current_joint_vals, grasp_goal)
        _=input('enter to start execution')
        self.executePlan(grasp_path)

        # TODO(mcorsaro): Update finger position in planner after executing grasp
        self.closeFingers()

        current_joint_vals = self.sim.data.qpos[:self.aDOF]
        back_result = self.planner.plan(current_joint_vals, start)

        self.viewer.render()
        _=input('enter to start execution')
        self.executePlan(back_result)

        while True:
            self.viewer.render()

if __name__ == '__main__':
    mjp = MujocoPlanExecutor()
    #mjp.generateData()
    mjp.setUpSimAndGenCloudsAndGenCandidates(rotation_values_about_approach=[0])
    fam_gen = grb.Grabstractor(mjp.cloud_with_normals, mjp.grasp_poses)
    fam_gen.generateGrabstraction(compression_alg="pca")
    fam_gen.visualizationVideoSample()