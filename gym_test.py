import numpy as np
import motor_skills
import motor_skills.core.mj_control as mjc
import gym

env=gym.make("motor_skills:mj_jaco_door-v0", vis=True)
while True:

    # env.step(np.zeros(6))
    action = mjc.gravity_comp(env.sim)
    env.step(action)
