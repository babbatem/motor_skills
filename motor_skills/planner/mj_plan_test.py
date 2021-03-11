import numpy as np
import time
import copy

import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.planner.pbplanner import PbPlanner
from motor_skills.planner.mj_point_clouds import PointCloudGenerator

from mujoco_py import cymj
from mujoco_py import load_model_from_path, MjSim, MjViewer

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

        # List of body names
        # self.sim.model.body_names

        '''
        print("cam_bodyid", self.sim.model.cam_bodyid)
        print("cam_fovy", self.sim.model.cam_fovy)
        print("cam_ipd", self.sim.model.cam_ipd)
        print("cam_mat0", self.sim.model.cam_mat0)
        print("cam_mode", self.sim.model.cam_mode)
        print("cam_pos", self.sim.model.cam_pos)
        print("cam_pos0", self.sim.model.cam_pos0)
        print("cam_poscom0", self.sim.model.cam_poscom0)
        print("cam_quat", self.sim.model.cam_quat)
        print("cam_targetbodyid", self.sim.model.cam_targetbodyid)
        print("cam_user", self.sim.model.cam_user)
        print("camera_id2name", self.sim.model.camera_id2name)
        print("camera_name2id", self.sim.model.camera_name2id)
        print("camera_names", self.sim.model.camera_names)
        for cam_id in self.sim.model.cam_bodyid:
            print("body_pos", self.sim.model.body_pos[cam_id])
            print("body_quat", self.sim.model.body_quat[cam_id])
        '''
        

        self.pc_gen = PointCloudGenerator(self.sim)

        self.door_dofs = [self.sim.model.joint_name2id('door_hinge'), self.sim.model.joint_name2id('latch')]

        self.finger_joint_range = self.sim.model.jnt_range[:self.tDOF, ]

        cloud_with_normals = self.pc_gen.generateCroppedPointCloud()
        print("Received cloud and normals of shape", cloud_with_normals.shape, cloud_with_normals.min(), cloud_with_normals.max())
        sys.exit()

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

    def runDemo(self):

        # make some plans
        planner = PbPlanner(self.load_door)
        #start = planner.validityChecker.sample_state()
        start = [0, np.pi, np.pi, 0, np.pi, 0]
        self.sim.data.qpos[:self.aDOF] = start
        self.sim.step()
        self.viewer.render()
        
        pregrasp_position = [0.3, 0.3, 0.4]
        grasp_orientation = [0.5, -0.5, -0.5, -0.5]
        pregrasp_goal = planner.accurateCalculateInverseKinematics(0, self.aDOF, pregrasp_position, grasp_orientation)
        print("Now planning pre-grasp motion.")
        pregrasp_path=planner.plan(start, pregrasp_goal)
        _=input('enter to start execution')

        self.executePlan(pregrasp_path)

        grasp_position = [0.3, 0.33, 0.4]
        grasp_goal = planner.accurateCalculateInverseKinematics(0, self.aDOF, grasp_position, grasp_orientation)
        print("Now planning grasp motion.")
        current_joint_vals = self.sim.data.qpos[:self.aDOF]
        grasp_path=planner.plan(current_joint_vals, grasp_goal)
        _=input('enter to start execution')
        self.executePlan(grasp_path)

        # TODO(mcorsaro): Update finger position in planner after executing grasp
        self.closeFingers()

        current_joint_vals = self.sim.data.qpos[:self.aDOF]
        back_result = planner.plan(current_joint_vals, start)

        self.viewer.render()
        _=input('enter to start execution')
        self.executePlan(back_result)

        while True:
            self.viewer.render()

if __name__ == '__main__':
    mjp = MujocoPlanExecutor()
    mjp.runDemo()