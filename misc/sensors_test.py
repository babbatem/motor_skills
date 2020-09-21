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
ee_xpos_goal[1] += 0.2
ee_xpos_goal[2] -= 0.005

for t in range(1000):
    sim.data.ctrl[:] = mjc.ee_regulation(ee_xpos_goal, sim, ee_index, kp=None, kv=None, ndof=12)
    sim.step()
    sim.forward()
    viewer.render()

print(sim.data.qpos)
obj_type = 3 # 3 for joint, 1 for body
joint_idxs = np.array([])
offset = np.zeros(12)
for i in range(1,4):
    base_idx = cymj._mj_name2id(sim.model, obj_type,"j2s6s300_joint_finger_" + str(i))
    tip_idx = cymj._mj_name2id(sim.model, obj_type,"j2s6s300_joint_finger_tip_" + str(i))
    offset[base_idx] = 2
    offset[tip_idx] = 2
    new_pos = sim.data.qpos[:12] + offset

for t in range(10000):
    sim.data.ctrl[:] = mjc.pd([0] * 12, [0] * 12, new_pos, sim, ndof=12)
    sim.forward()
    sim.step()
    viewer.render()
    print(sim.data.sensordata)
