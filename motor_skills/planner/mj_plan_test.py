import numpy as np
import time
import copy

import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.planner.pbplanner import PbPlanner

from mujoco_py import cymj
from mujoco_py import load_model_from_path, MjSim, MjViewer

class MujocoPlanExecutor(object):
    def __init__(self, load_door=True):
        super(MujocoPlanExecutor, self).__init__()

        self.NDOF=6
        self.load_door = load_door

        model = load_model_from_path('/home/mcorsaro/.mujoco/motor_skills/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_door.xml') if load_door else \
            load_model_from_path('/home/mcorsaro/.mujoco/motor_skills/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_nodoor.xml')
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)

        ##### Fingers #####
        # compute indices and per timestep delta q
        self.gDOF = 6
        self.tDOF = self.NDOF + self.gDOF
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

        self.door_dofs = [self.sim.model.joint_name2id('door_hinge'), self.sim.model.joint_name2id('latch')]

        self.finger_joint_range = self.sim.model.jnt_range[:self.tDOF, ]

    def executePlan(self, plan):
        plan.interpolate(1000)
        H = plan.getStateCount()
        for t in range(H):
            state_t = plan.getState(t)
            target_q = []
            target_qd = []
            for i in range(self.NDOF):
                target_q.append(state_t[i])
                target_qd.append(0.0)

            torques=mjc.pd(None, target_qd, target_q, self.sim, ndof=self.NDOF, kp=np.eye(self.NDOF)*300)
            self.sim.data.ctrl[:self.NDOF]=torques
            self.sim.step()
            self.viewer.render()
            time.sleep(0.01)

        # Make sure it reaches the goal
        for t in range(200):
            torques=mjc.pd(None, target_qd, target_q, self.sim, ndof=self.NDOF, kp=np.eye(self.NDOF)*100)
            self.sim.data.ctrl[:self.NDOF]=torques
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
            touched_finger_idxs = [touched_link + self.NDOF for touched_link in touched]
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
        self.sim.data.qpos[:self.NDOF] = start
        self.sim.step()
        self.viewer.render()
        
        pregrasp_position = [0.3, 0.3, 0.4]
        grasp_orientation = [0.5, -0.5, -0.5, -0.5]
        pregrasp_goal = planner.accurateCalculateInverseKinematics(0, self.NDOF, pregrasp_position, grasp_orientation)
        print("Now planning pre-grasp motion.")
        pregrasp_path=planner.plan(start, pregrasp_goal)
        _=input('enter to start execution')

        self.executePlan(pregrasp_path)

        grasp_position = [0.3, 0.33, 0.4]
        grasp_goal = planner.accurateCalculateInverseKinematics(0, self.NDOF, grasp_position, grasp_orientation)
        print("Now planning grasp motion.")
        current_joint_vals = self.sim.data.qpos[:self.NDOF]
        grasp_path=planner.plan(current_joint_vals, grasp_goal)
        _=input('enter to start execution')
        self.executePlan(grasp_path)

        # TODO(mcorsaro): Update finger position in planner after executing grasp
        self.closeFingers()

        current_joint_vals = self.sim.data.qpos[:self.NDOF]
        back_result = planner.plan(current_joint_vals, start)

        self.viewer.render()
        _=input('enter to start execution')
        self.executePlan(back_result)

        while True:
            self.viewer.render()

if __name__ == '__main__':
    mjp = MujocoPlanExecutor()
    mjp.runDemo()