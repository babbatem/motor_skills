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

        self.delta = np.zeros(self.tDOF)
        self.finger_joint_idxs = []
        for i in range(1,4):
            base_idx = self.sim.model.joint_name2id("j2s6s300_joint_finger_" + str(i))
            tip_idx = self.sim.model.joint_name2id("j2s6s300_joint_finger_tip_" + str(i))
            self.finger_joint_idxs.append(base_idx)
            self.finger_joint_idxs.append(tip_idx)
            self.delta[base_idx] = self.max_finger_delta/self.grasp_steps
            self.delta[tip_idx] = self.max_finger_delta/self.grasp_steps

        # List of body names
        # self.sim.model.body_names

        self.door_dofs = [self.sim.model.joint_name2id('door_hinge'), self.sim.model.joint_name2id('latch')]

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
            new_pos += self.delta

            # % see which sensors are reporting force
            touched = np.where(self.sim.data.sensordata[:6] != 0.0)[0]

            # % if they are all in contact, we're done
            if len(touched) == 6:
                break

            # % otherwise, compute new setpoints for those which are not in contact
            current_pos = self.sim.data.qpos
            for touch_point in touched:
                new_pos[self.finger_joint_idxs[touch_point]] = current_pos[self.finger_joint_idxs[touch_point]]

            # % compute torque and step
            # TODO(mcorsaro): stop at joint limits
            self.sim.data.ctrl[:] = mjc.pd([0] * self.tDOF, [0] * self.tDOF, new_pos, self.sim, ndof=self.tDOF, kp=np.eye(self.tDOF)*300)
            self.sim.forward()
            self.sim.step()

            if self.viewer is not None:
                self.viewer.render()

    def resetDoor(self):
        for mj_id in self.door_dofs:
            self.sim.data.qpos[mj_id]=0.0

    def runDemo(self):

        # make some plans
        planner = PbPlanner(self.load_door)
        start = planner.validityChecker.sample_state()
        goal_position = [0.3, 0.3, 0.4]
        goal_orientation = [0.5, -0.5, -0.5, -0.5]
        goal = planner.accurateCalculateInverseKinematics(0, self.NDOF, goal_position, goal_orientation)
        if planner.validityChecker.isValid(goal):
            #g = planner.validityChecker.sample_state()
            result=planner.plan(start, goal)

            self.sim.data.qpos[:self.NDOF] = start
            self.sim.step()
            self.viewer.render()
            _=input('enter to start execution')

            self.executePlan(result)

            self.closeFingers()

            current_joint_vals = self.sim.data.qpos[:self.NDOF]
            if planner.validityChecker.isValid(current_joint_vals):
                if planner.validityChecker.isValid(start):
                    back_result = planner.plan(current_joint_vals, start)

                    self.viewer.render()
                    _=input('enter to start execution')
                    self.executePlan(back_result)
                else:
                    print("Starting state is not valid, but it was before???", start)
            else:
                print("Grasp values")

            '''_=input('enter to see goal')
            self.sim.data.qpos[:self.NDOF] = goal
            self.sim.step()'''
        else:
            print("Didn't work, goal was invalid.")

        while True:
            self.viewer.render()

if __name__ == '__main__':
    mjp = MujocoPlanExecutor()
    mjp.runDemo()