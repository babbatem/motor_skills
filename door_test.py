import copy
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv, MjJacoDoor

# %%
last_time = 0.0
if __name__ == '__main__':

    env = MjJacoDoor(vis=True)

    for t in range(10000):

        # action = mjc.gravity_comp(env.sim)
        # env.step(action)

        env.step_test()

        # q = [0]*9
        # qd = [0]*9
        # qdd = [0]*9
        # action = mjc.pd(qdd,qd,q,env.sim)

        # q=None
        # qd[0] += 1.0
        # action = mjc.pd(qdd,qd,q,env.sim)

        # print('---')
        # print('goal: ', test_x)
        # print('current: ', env.sim.data.body_xpos[9])
        # print('---')
        # action = ee_regulation(test_x, env.sim, 9)
        # env.step(action)
        # env.sim.data.qfrc_bias[-1] = env.sim.data.qfrc_applied[-1]
        # env.step_test()
        # print('--')
        # print(env.sim.data.qfrc_bias[-1])
        # print(env.sim.data.qfrc_applied[-1])
        # print(env.model.jnt_range[-1])
        # print(1.0 / (last_time - env.sim.data.time))
        # print(env.sim.data.qpos)
        # last_time = copy.deepcopy(env.sim.data.time)
        # print(env.sim.data.site_xpos)
