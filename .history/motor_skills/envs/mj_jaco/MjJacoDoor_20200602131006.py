import time
import copy
import pathlib
import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv

THRESHOLD = np.pi / 2

class MjJacoDoor(gym.Env):
    """docstring for MjJacoDoor."""

    def __init__(self, vis=False, n_steps=int(1000)):
        parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        self.fname = parent_dir_path + '/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
        self.model = load_model_from_path(self.fname)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.vis=vis

        a_low = np.full(7, -float('inf'))
        a_high = np.full(7, float('inf'))
        self.action_space = gym.spaces.Box(a_low,a_high)

        o_low = np.full(8, -float('inf'))
        o_high = np.full(8, float('inf')) 
        self.observation_space=gym.spaces.Box(o_low,o_high)
        self.env=self
        self.n_steps = n_steps

    def step(self, action):
        for i in range(len(action)):
            self.sim.data.ctrl[i]=action[i]

        self.sim.forward()
        self.sim.step()
        self.viewer.render() if self.vis else None

        reward = self.sim.data.qpos[-2] > THRESHOLD

        info={'goal_achieved': reward}

        done = self.sim.data.time == self.n_steps - 1

        return self.sim.data.qpos, reward, done, info
