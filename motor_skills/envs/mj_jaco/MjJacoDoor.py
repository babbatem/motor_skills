import time
import copy
import pathlib
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv

class MjJacoDoor(MjJacoEnv):
    """docstring for MjJacoDoor."""

    def __init__(self, vis=False):
        super(MjJacoDoor, self).__init__()
        parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        self.fname = parent_dir_path + '/assets/mjdoor.xml'
        self.model = load_model_from_path(self.fname)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.vis=vis

    def move_door(self):
        print(len(self.sim.data.body_xpos))
