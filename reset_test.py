import time
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

    delta_pos = [0, 0, 0.0]
    delta_ori = [0,0,0]
    delta_kp = [0,0,0]
    delta_kv = [0,0,0]
    action=np.concatenate((delta_pos, delta_ori,
                           delta_kp, delta_kp, delta_kv, delta_kv))
    for i in range(100):
        env.reset()
        env.step(action)
        time.sleep(0.1)
