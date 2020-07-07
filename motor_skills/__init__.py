from gym.envs.registration import register

register(
    id='mj_jaco_door-v0',
    entry_point='motor_skills.envs.mj_jaco:MjJacoDoor'
)

register(
    id='mj_jaco_door_impedance-v0',
    entry_point='motor_skills.envs.mj_jaco:MjJacoDoorImpedance'
)
