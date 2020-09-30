import time
import numpy as np
from motor_skills.envs.mj_jaco.MjJacoDoorImpedanceCIP import MjJacoDoorImpedanceCIP
from motor_skills.envs.mj_jaco.MjJacoDoorImpedanceNaive import MjJacoDoorImpedanceNaive

def seed_properly(seed_value=123):

    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    import random
    random.seed(seed_value)

    import numpy as np
    np.random.seed(seed_value)


seed=int(time.time())
seed_properly(seed)
env = MjJacoDoorImpedanceCIP(vis=True, start_idx=2)
# env = MjJacoDoorImpedanceNaive(vis=True)
env.reset()

while True:
    action = np.zeros(env.action_dim)

    action[4] = 0.5
    env.step(action)



    # env.sim.step()
    # env.render()
