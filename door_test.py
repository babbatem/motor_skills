import copy
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv, MjJacoDoor

# %%
if __name__ == '__main__':

    env = MjJacoDoor(vis=True)

    for t in range(10000):
        action = mjc.gravity_comp(env.sim)

        # q = [0]*9
        # qd = [0]*9
        # qdd = [0]*9
        # action = pid(qdd,qd,q,env.sim)

        # q=None
        # qd[0] += 1.0
        # action = pid(qdd,qd,q,env.sim)

        # print('---')
        # print('goal: ', test_x)
        # print('current: ', env.sim.data.body_xpos[9])
        # print('---')
        # action = ee_regulation(test_x, env.sim, 9)
        env.step(action)
