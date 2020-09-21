import pickle
import numpy as np

import copy
import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv, MjJacoDoor

from mujoco_py import cymj
from mujoco_py import load_model_from_path, MjSim, MjViewer

model = load_model_from_path('/home/abba/msu_ws/src/motor_skills/motor_skills/envs/mj_jaco/assets/kinova_j2s6s300/mj-j2s6s300_door.xml')
sim = MjSim(model)
viewer = MjViewer(sim)

parent_dir_path ='/home/abba/msu_ws/src/motor_skills/motor_skills/envs/mj_jaco/'
start_pose_file = open(parent_dir_path + "/assets/MjJacoDoorGrasps", 'rb')
start_poses = pickle.load(start_pose_file)
sim.data.qpos[:6] = start_poses[8]
sim.step()

for t in range(10000):
    sim.data.ctrl[:] = mjc.pd(None, None, None, sim, ndof=12)
    sim.forward()
    sim.step()
    viewer.render()

    force = sim.data.sensordata[-6:-3]
    torque = sim.data.sensordata[-3:]
    print('---')
    print('force: ', force)
    print('torque: ', torque)
    print(sim.data.sensordata)
