import copy
from scipy.spatial.transform import Rotation as R
import numpy as np
import motor_skills.core.mj_control as mjc
from motor_skills.envs.mj_jaco import MjJacoEnv
from motor_skills.rmp.jaco_rmp import JacoFlatRMP

# %%
env = MjJacoEnv(vis=True)
rmp = JacoFlatRMP()
qdd_cap = 1000
while True:

    # evaluate RMP for goal
    q = env.sim.data.qpos[:6]
    qd = env.sim.data.qvel[:6]
    qdd = rmp.eval(q, qd)
    action = mjc.pd(qdd, qd, q, env.sim, ndof=6)

    action_norm = np.linalg.norm(action)
    if action_norm > qdd_cap:
        action = action / action_norm * qdd_cap

    try:
        env.step(action)
    except:
        print("bad qdd: " + str(qdd))
        break

    #print('qpos: ' + str(env.sim.data.qpos[:6]))
    print('xpos: ' + str(env.sim.data.body_xpos[6]))
    quat = mjc.quat_to_scipy(env.sim.data.body_xquat[6])
    r = R.from_quat(quat)
    #print('rot: ' + str(r.as_euler('xyz', degrees=False)))
    #print('qdd: ' + str(qdd))
