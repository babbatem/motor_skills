import time
import copy
import pathlib
import pickle
import hjson
import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv

from mujoco_py import cymj
from scipy.spatial.transform import Rotation as R

from motor_skills.cip.ImpedanceCIP import ImpedanceCIP
from motor_skills.envs.mj_jaco.MjJacoDoor import MjJacoDoor

class MjJacoDoorCIPBase(MjJacoDoor):
	"""
		base class for MjJacoDoor controlled by an ImpedanceCIP.
		init_cip method left not implemented such that the appropriate CIP may be loaded.
	"""

	def __init__(self, vis=False, n_steps=int(2000)):

		# % call super to load model
		super(MjJacoDoorCIPBase, self).__init__(vis=vis,n_steps=n_steps)

		# % call init_cip, which should be overridden.
		self.init_cip()
		self.set_cip_data()

		# %% override inherited action and obs spaces
		action_dim = self.cip.controller.action_dim + 6
		a_low = np.full(action_dim, -float('inf'))
		a_high = np.full(action_dim, float('inf'))
		self.action_space = gym.spaces.Box(a_low,a_high)

		obs_space = self.model.nq + self.model.nsensordata
		o_low = np.full(obs_space, -float('inf'))
		o_high = np.full(obs_space, float('inf'))
		self.observation_space=gym.spaces.Box(o_low,o_high)
		self.env=self
		self.n_steps = n_steps

	def init_cip(self):
		raise NotImplementedError

	def set_cip_data(self):
		self.control_timestep = 1.0 / self.cip.controller.control_freq
		self.model_timestep = self.sim.model.opt.timestep
		self.arm_dof = self.cip.controller.control_dim
		self.gripper_indices = np.arange(self.arm_dof, len(self.sim.data.ctrl))

	def reset(self):
		"""
		NOTE: qpos reset behavior largely deferred to self.cip.learning_reset()
		"""

		# %% first, reset the object's qpos
		self.model_reset()

		# %% reset cip controller
		self.cip.controller.reset()

		# % reset cip for start of learning
		self.cip.learning_reset()

		obs = np.concatenate( [copy.deepcopy(self.sim.data.qpos),
							   copy.deepcopy(self.sim.data.sensordata] )
		self.elapsed_steps=0
		return obs


	def step(self, action):
		"""
		uses CIP to interpret action.
		"""

		policy_step = True
		for i in range(int(self.control_timestep / self.model_timestep)):

			# %% interpret as torques here (gravity comp done in the CIP)
			torques = self.cip.get_action(action, policy_step)
			self.sim.data.ctrl[:] = torques
			self.sim.step()
			policy_step = False
			self.elapsed_steps+=1

		self.viewer.render() if self.vis else None
		reward = -1*self.cip.learning_cost(self.sim)
		done = self.elapsed_steps >= (self.n_steps - 1)
		obs = np.concatenate( [copy.deepcopy(self.sim.data.qpos),
							   copy.deepcopy(self.sim.data.sensordata] )
		return obs, reward, done, info
