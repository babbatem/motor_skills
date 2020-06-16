import copy
import motor_skills
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv, MjJacoDoor

# %%
if __name__ == '__main__':

    env = MjJacoDoor(vis=True)
    env.reset()

    ee_index = 6
    ee_xpos_goal = copy.deepcopy(env.sim.data.body_xpos[ee_index])
    ee_xpos_goal[1] += 0.5

    for t in range(10000):
        env.sim.data.ctrl[:]=mjc.ee_regulation(ee_xpos_goal, env.sim, ee_index, kp=None, kv=None, ndof=12)
        env.sim.step()
        env.sim.forward()

        env.viewer.render()
