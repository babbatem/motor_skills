import numpy as np
import copy
import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoDoorImpedance
from mujoco_py import cymj

# %%
if __name__ == '__main__':

    env = MjJacoDoorImpedance(vis=True)
    env.reset()

    ee_index = 6
    ee_current_pos = env.sim.data.body_xpos[ee_index]
    ee_current_quat = env.sim.data.body_xquat[ee_index]

    obj_type = 1 # 3 for joint, 1 for body
    body_idx = cymj._mj_name2id(env.sim.model, obj_type,"latch")
    ee_xpos_goal = env.sim.data.body_xpos[body_idx]
    ee_quat_goal = env.sim.data.body_xquat[body_idx]



    for t in range(10000):
        delta_pos = [0, 0, 0.3]
        delta_ori = [0,0,0]
        delta_kp = [0,0,0]
        delta_kv = [0,0,0]
        delta_gripper_pos = [0, 0, 0, 0, 0, 0]
        action=np.concatenate((delta_pos, delta_ori,
                               delta_kp, delta_kp,
                               delta_kv, delta_kv,
                               delta_gripper_pos))
        # torques = mjc.ee_reg2(ee_current_pos, ee_current_quat,
        #                                env.sim, ee_index,
        #                                kp=None, kv=None, ndof=12)

        env.step(action)
