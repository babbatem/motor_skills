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

def binary_touch(sensordata):
	return np.array([s_i > 0 for s_i in sensordata])

class MjJacoDoorCIPBase(MjJacoDoor):
	"""
		base class for MjJacoDoor controlled by an ImpedanceCIP.
		init_cip method left to be implemented such that the appropriate CIP may be loaded.

		sensor_type variable controls how tactile sensors behave:
			'perfect' - full 3D force vector of all contact points inside site
						(not yet implemented)
			'normal'  - real valued normal force from sim.sensordata
			'binary'  - normal force binary flag
			None 	  - touch sensing disabled

		wrist_sensor boolean variable controls presence of wrist FT sensor
			True -> 6DoF readings included in observations
			False -> omitted
	"""

	def __init__(self, vis=False, vis_head=False, n_steps=int(2000), sensor_type='normal', wrist_sensor=True):

		# % call super to load model
		super(MjJacoDoorCIPBase, self).__init__(vis=vis,n_steps=n_steps)

		# % call init_cip, which should be overridden.
		self.init_cip()
		self.set_cip_data()

		# %% override inherited action and obs spaces
		# action_dim = self.cip.controller.action_dim + 6
		action_dim = self.cip.controller.action_dim      # NOTE: gripper control disabled
		a_low = np.full(action_dim, -float('inf'))
		a_high = np.full(action_dim, float('inf'))
		self.action_space = gym.spaces.Box(a_low,a_high)
		self.action_dim = action_dim

		self.env=self
		self.n_steps = n_steps
		self.sensor_type=sensor_type
		self.wrist_sensor=wrist_sensor

		if sensor_type == 'perfect':
			raise NotImplementedError('sensor_type "perfect" is yet to be implemented')

		# % compute dimensionality of haptic sensors
		self.n_touch_sensors=0
		self.touch_sensor_dim=0
		for i in range(len(self.model.sensor_type)):

			# % check if it is a touch sensor
			if self.model.sensor_type[i] == 0:

				# % if so increment count and dim count
				self.n_touch_sensors+=1
				self.touch_sensor_dim+=self.model.sensor_dim[i]

		if sensor_type is None:
			n_touch_vars = 0
		else:
			n_touch_vars = self.touch_sensor_dim

		if wrist_sensor:
			n_wrist_vars = 6
		else:
			n_wrist_vars = 0

		obs_space = self.model.nq + n_touch_vars + n_wrist_vars
		o_low = np.full(obs_space, -float('inf'))
		o_high = np.full(obs_space, float('inf'))
		self.observation_space=gym.spaces.Box(o_low,o_high)


	def init_cip(self):
		raise NotImplementedError

	def set_cip_data(self):
		self.control_timestep = 1.0 / self.cip.controller.control_freq
		self.model_timestep = self.sim.model.opt.timestep
		self.arm_dof = self.cip.controller.control_dim
		self.gripper_indices = np.arange(self.arm_dof, len(self.sim.data.ctrl))

	def get_obs(self):
		# % get appropriate sensor readings
		tmp = self.sim.data.sensordata
		raw_touch = copy.deepcopy(tmp[:self.touch_sensor_dim])
		raw_FT = copy.deepcopy(tmp[self.touch_sensor_dim:])
		if self.sensor_type is None:
			touch_output = np.array([])
		elif self.sensor_type == 'normal':
			touch_output = raw_touch
		elif self.sensor_type == 'binary':
			touch_output = binary_touch(raw_touch)

		if self.wrist_sensor:
			wrist_output = raw_FT
		else:
			wrist_output=np.array([])

		# % concatenate qpos and tactile sensor output
		obs = np.concatenate( [copy.deepcopy(self.sim.data.qpos),
							   touch_output, wrist_output] )

		return obs

	def reset(self):
		"""
		NOTE: qpos reset behavior largely deferred to self.cip.learning_reset()
		"""

		# %% first, reset the object's qpos
		self.model_reset()

		# %% reset cip controller
		self.cip.controller.reset()

		# % reset cip for start of learning
		self.start_idx  = self.cip.learning_reset()

		self.grp_target = copy.deepcopy(self.sim.data.qpos[:len(self.sim.data.ctrl)])
		self.cip.set_gripper_target(self.grp_target)

		obs=self.get_obs()
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

		info={'success': self.sim.data.qpos[-2] > np.pi / 4.0 }
		self.viewer.render() if self.vis else None
		reward = -1*self.cip.learning_cost(self.sim)
		done = self.elapsed_steps >= (self.n_steps - 1)
		obs=self.get_obs()
		return obs, reward, done, info
