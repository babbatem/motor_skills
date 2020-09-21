import time
import numpy as np
from motor_skills.envs.mj_jaco.MjJacoDoorImpedanceCIP import MjJacoDoorImpedanceCIP

def seed_properly(seed_value=123):

    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    import random
    random.seed(seed_value)

    import numpy as np
    np.random.seed(seed_value)


seed=int(time.time())
seed_properly(seed)
env = MjJacoDoorImpedanceCIP(vis=True)
env.reset()

while True:
    env.sim.step()
    env.render()
