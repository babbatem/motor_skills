import numpy as np
import copy
import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoDoorImpedance

# %%
if __name__ == '__main__':

    env = MjJacoDoorImpedance(vis=True)
    env.reset()

    for t in range(10000):
        # delta_pos = [0, 0, 0.1]
        # delta_ori = [0,0,0]
        # delta_kp = [0,0,0]
        # delta_kv = [0,0,0]
        # action=np.concatenate((delta_pos, delta_ori,
        #                        delta_kp, delta_kp, delta_kv, delta_kv))
        action=np.zeros(6)
        env.step(action)
