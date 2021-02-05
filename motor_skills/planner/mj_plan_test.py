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

    def run_demo(self):

        # make some plans
        planner = PbPlanner(self.load_door)
        start = planner.validityChecker.sample_state()
        goal = [-0.1589180907085973, 1.1245753846210882, 0.8936496967854405, -1.4150251357344403, 1.5338966490406984, 1.3421998853827806]
        if planner.validityChecker.isValid(goal):
            #g = planner.validityChecker.sample_state()
            result=planner.plan(start, goal)

            self.sim.data.qpos[:self.NDOF] = start
            self.sim.step()
            self.viewer.render()
            _=input('enter to start execution')

            self.executePlan(result)

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
    mjp.run_demo()