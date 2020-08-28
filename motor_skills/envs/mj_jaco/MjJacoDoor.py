import time

import copy
import pathlib
import pickle
import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv
import motor_skills.core.mj_control as mjc
from mujoco_py import cymj
from scipy.spatial.transform import Rotation as R

from motor_skills.envs.mj_jaco.mj_cip_utils import sample_random_pose, door_open_success

class MjJacoDoor(gym.Env):
	"""
		provides a standalone gym environment for the JacoDoor problem (opening)
		also functions as a *base class* for CIP experiment environments.
	"""

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

	def model_reset(self):

		# %% close object
		self.sim.data.qpos[-1]=0.0
		self.sim.data.qpos[-2]=0.0

	def reset(self):
		self.model_reset()

		# %% sample random (valid) 6DoF pose, set it, step.
		sample_random_pose(self.sim, self.model)

		obs = np.concatenate( [self.sim.data.qpos, self.sim.data.sensordata] )
		self.elapsed_steps=0
		return obs

	def step(self, action):
		"""
			interprets action as torque input; performs gravity comp.
			returns binary reward based on threshold.
		"""

		for i in range(len(action)):
			self.sim.data.ctrl[i]=action[i]+self.sim.data.qfrc_bias[i]

		self.sim.forward()
		self.sim.step()
		self.render() if self.vis else None

		reward = door_opening_success(self.sim)

		info={'goal_achieved': reward}

		done = self.elapsed_steps == (self.n_steps - 1)

		obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])

		self.elapsed_steps+=1
		return obs, reward, done, info

	def render(self):
		try:
			self.viewer.render()
		except Exception as e:
			self.viewer = MjViewer(self.sim)
			self.viewer.render()
