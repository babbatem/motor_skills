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

ee_index = 6
ee_xpos_goal = copy.deepcopy(sim.data.body_xpos[ee_index])
ee_xpos_goal[2] += 0.1
ee_quat_goal = [1,0,0,0]

while True:
    sim.data.ctrl[:] = mjc.ee_reg2(ee_xpos_goal, ee_quat_goal,
                                   sim, ee_index,
                                   kp=None, kv=None, ndof=12)
    # sim.data.ctrl[:] = sim.data.qfrc_bias[:12]
    sim.step()
    sim.forward()
    viewer.render()

    print('-----')
    print(ee_xpos_goal - sim.data.body_xpos[ee_index])
    print(ee_quat_goal - sim.data.body_xquat[ee_index])
