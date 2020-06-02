from distutils.core import setup
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

setup(
    name='motor_skills',
    version='0.1dev',
    packages=['motor_skills',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)
