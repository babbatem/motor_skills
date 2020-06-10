import time
import copy
import pathlib
import pickle
import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv

THRESHOLD = np.pi / 2

class MjJacoDoor(gym.Env):
    """docstring for MjJacoDoor."""

    def __init__(self, vis=False, n_steps=int(1000)):
        self.parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
        self.vis=vis
        self.fname = self.parent_dir_path + '/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
        self.model = load_model_from_path(self.fname)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim) if self.vis else None

        a_low = np.full(12, -float('inf'))
        a_high = np.full(12, float('inf'))
        self.action_space = gym.spaces.Box(a_low,a_high)

        obs_space = self.model.nq + self.model.nsensordata
        o_low = np.full(obs_space, -float('inf'))
        o_high = np.full(obs_space, float('inf'))
        self.observation_space=gym.spaces.Box(o_low,o_high)
        self.env=self
        self.n_steps = n_steps


    def reset(self):

        # pick a random start arm pose (DoFs 1-6)
        start_pose_file = open(self.parent_dir_path + "/assets/MjJacoDoorGrasps", 'rb')
        self.start_poses = pickle.load(start_pose_file)
        idx = np.random.randint(len(self.start_poses))
        self.sim.data.qpos[:6]=self.start_poses[idx]
        self.sim.step()

        # TODO: close the gripper here

        # reset the object
        self.sim.data.qpos[-1]=0.0
        self.sim.data.qpos[-2]=0.0

        self.elapsed_steps=0
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])
        return obs

    def step(self, action):

        # TODO: failure predicate here

        for i in range(len(action)):
            self.sim.data.ctrl[i]=action[i]+self.sim.data.qfrc_bias[i]

        self.sim.forward()
        self.sim.step()
        self.viewer.render() if self.vis else None

        reward = self.sim.data.qpos[-2] > THRESHOLD

        info={'goal_achieved': reward}

        done = self.elapsed_steps == (self.n_steps - 1)

        obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])

        self.elapsed_steps+=1
        return obs, reward, done, info
