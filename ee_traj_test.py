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
viewer.render()

y_star = copy.deepcopy(sim.data.body_xpos[6])
y_star[0]+=0.4
y_star[2]+=0.4
ys = mjc.ee_traj(y_star, 1000, sim, 6)

current_ee_quat = copy.deepcopy(sim.data.body_xquat[6])
for y in ys:
    torques = mjc.ee_reg2(y, current_ee_quat, sim, 6)
    sim.data.ctrl[:]=torques
    sim.step()
    viewer.render()

print('done')
while True:
    torques = mjc.ee_reg2(y, current_ee_quat, sim, 6)
    sim.data.ctrl[:]=torques
    sim.step()
    viewer.render()
