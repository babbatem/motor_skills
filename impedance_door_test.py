import copy
import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoDoorImpedance

# %%
if __name__ == '__main__':

    env = MjJacoDoorImpedance(vis=True)
    env.reset()

    for t in range(10000):
        env.step([0]*9)
