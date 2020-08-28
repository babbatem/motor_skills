from gym.envs.registration import register

register(
    id='mj_jaco_door-v0',
    entry_point='motor_skills.envs.mj_jaco:MjJacoDoor'
)

# register(
#     id='mj_jaco_door_impedance-v0',
#     entry_point='motor_skills.envs.mj_jaco:MjJacoDoorImpedance'
# )
#

register(
    id='mj_jaco_door_naive-v0',
    entry_point='motor_skills.envs.mj_jaco:MjJacoDoorImpedanceNaive'
)

register(
    id='mj_jaco_door_cip-v0',
    entry_point='motor_skills.envs.mj_jaco:MjJacoDoorImpedanceCIP'
)
