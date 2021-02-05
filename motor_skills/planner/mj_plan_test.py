import pickle
import numpy as np
import time
import copy

import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.planner.pbplanner import PbPlanner

from mujoco_py import cymj
from mujoco_py import load_model_from_path, MjSim, MjViewer

NDOF=6

load_door = True

model = load_model_from_path('/home/mcorsaro/.mujoco/motor_skills/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_door.xml') if load_door else \
    load_model_from_path('/home/mcorsaro/.mujoco/motor_skills/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_nodoor.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

# make some plans
planner = PbPlanner(load_door)
s = planner.validityChecker.sample_state()
goal = [-0.1589180907085973, 1.1245753846210882, 0.8936496967854405, -1.4150251357344403, 1.5338966490406984, 1.3421998853827806]
if planner.validityChecker.isValid(goal):
    #g = planner.validityChecker.sample_state()
    result=planner.plan(s, goal)

    sim.data.qpos[:NDOF] = s
    sim.step()
    viewer.render()
    _=input('enter to start execution')

    result.interpolate(1000)
    H = result.getStateCount()
    for t in range(H):
        state_t = result.getState(t)
        target_q = []
        target_qd = []
        for i in range(NDOF):
            target_q.append(state_t[i])
            target_qd.append(0.0)

        torques=mjc.pd(None, target_qd, target_q, sim, ndof=NDOF, kp=np.eye(NDOF)*300)
        sim.data.ctrl[:NDOF]=torques
        sim.step()
        viewer.render()
        time.sleep(0.01)

    for t in range(200):
        torques=mjc.pd(None, target_qd, target_q, sim, ndof=NDOF, kp=np.eye(NDOF)*100)
        sim.data.ctrl[:NDOF]=torques
        sim.step()
        viewer.render()

    _=input('enter to see goal')
    sim.data.qpos[:NDOF] = goal
    sim.step()
else:
    print("Didn't work, goal was invalid.")

while True:
    viewer.render()
