from motor_skills.envs.mj_jaco.MjJacoEnv import MjJacoEnv
from motor_skills.envs.mj_jaco.MjJacoDoor import MjJacoDoor
from gym.envs.registration import register

register(
    id='KukaDrawer-v0',
    entry_point='kuka_gym.envs:KukaDrawerGymEnv'
)

register(
    id='KukaCabinet-v0',
    entry_point='kuka_gym.envs:KukaCabinetGymEnv'
)

register(
    id='KukaDynamic-v0',
    entry_point='kuka_gym.envs:KukaDynamicGymEnv'
)
