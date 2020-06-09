import time
import copy
import pathlib
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv

class MjJacoDoor:
    """docstring for MjJacoDoor."""

    def __init__(self, vis=False):
        parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        self.fname = parent_dir_path + '/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
        self.model = load_model_from_path(self.fname)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.vis=vis

    def step(self, action):
        for i in range(len(action)):
            self.sim.data.ctrl[i]=action[i]

        self.sim.forward()
        self.sim.step()
        self.viewer.render() if self.vis else None
        return self.sim.data.qpos
