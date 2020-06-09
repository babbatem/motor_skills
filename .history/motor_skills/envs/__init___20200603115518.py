from gym.envs.registration import register

register(
    id='mj_jaco_env-v0',
    entry_point='motor_skills.envs.mj_jaco:MjJacoEnv'
)