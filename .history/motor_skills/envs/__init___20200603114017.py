from gym.envs.registration import register

register(
    id='MjJacoDoor',
    entry_point='motor_skills.envs.mj_jaco:MjJacoDoor'
)