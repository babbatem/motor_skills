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

model = load_model_from_path('/home/abba/msu_ws/src/motor_skills/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_door.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

# make some plans
planner = PbPlanner()
s = planner.validityChecker.sample_state()
g = planner.validityChecker.sample_state()
result=planner.plan(s, g)

sim.data.qpos[:6] = s
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

    torques=mjc.pd(None, target_qd, target_q, sim, ndof=6, kp=np.eye(6)*300)
    sim.data.ctrl[:6]=torques
    sim.step()
    viewer.render()
    time.sleep(0.01)

for t in range(200):
    torques=mjc.pd(None, target_qd, target_q, sim, ndof=6, kp=np.eye(6)*100)
    sim.data.ctrl[:6]=torques
    sim.step()
    viewer.render()



_=input('enter to see goal')
sim.data.qpos[:6] = g
sim.step()

while True:
    viewer.render()
