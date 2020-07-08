import time
import copy
import pathlib
import pickle
import hjson
import gym
import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
from motor_skills.envs.mj_jaco import MjJacoEnv
import motor_skills.core.mj_control as mjc
from mujoco_py import cymj
from scipy.spatial.transform import Rotation as R

from motor_skills.cip.ImpedanceCIP import ImpedanceCIP

DOOR_GOAL = np.pi / 2
DOOR_WEIGHT = 100
HANDLE_GOAL = np.pi / 2
HANDLE_WEIGHT = 10
GRASP_WEIGHT = 10

class MjJacoDoorImpedance(gym.Env):
	"""docstring for MjJacoDoor."""

	def __init__(self, vis=True, n_steps=int(1000)):

		# %% setup MuJoCo
		self.parent_dir_path = str(pathlib.Path(__file__).parent.absolute())
		self.vis=vis
		self.fname = self.parent_dir_path + '/assets/kinova_j2s6s300/mj-j2s6s300_door.xml'
		self.model = load_model_from_path(self.fname)
		self.sim = MjSim(self.model)
		self.viewer = MjViewer(self.sim) if self.vis else None

		# %% load the CIP
		controller_file = self.parent_dir_path + '/assets/controller_config.hjson'
		self.cip = ImpedanceCIP(controller_file, self.sim)
		self.control_timestep = 1.0 / self.cip.controller.control_freq
		self.model_timestep = self.sim.model.opt.timestep
		self.arm_dof = self.cip.controller.control_dim

		# %% configure action space (+6 for the gripper)
		action_dim = self.cip.controller.action_dim
		a_low = np.full(action_dim, -float('inf'))
		a_high = np.full(action_dim, float('inf'))
		self.action_space = gym.spaces.Box(a_low,a_high)

		obs_space = self.model.nq + self.model.nsensordata
		o_low = np.full(obs_space, -float('inf'))
		o_high = np.full(obs_space, float('inf'))
		self.observation_space=gym.spaces.Box(o_low,o_high)
		self.env=self
		self.n_steps = n_steps

	def sample_random_pose(self):
		# %% TODO: seeding elsewhere please.
		np.random.seed(0)

		# %% sample random start in [-pi,pi], make sure it is valid.
		random_q = 2 * np.pi * np.random.rand(6) - np.pi
		for i in range(6):
			if not self.model.jnt_limited[i]:
				self.sim.data.qpos[i]=random_q[i]
			else:
				if (random_q[i] >= self.model.jnt_range[i][0]):
					if (random_q[i] <= self.model.jnt_range[i][1]):
						self.sim.data.qpos[i]=random_q[i]
					else:
						self.sim.data.qpos[i]=self.model.jnt_range[i][1]
				else:
					self.sim.data.qpos[i]=self.model.jnt_range[i][0]

		self.sim.step()

		# %% TODO: finish this, call recursively
		# %% make sure we aren't in penetration
		my_max_contacts=10
		for i in range(my_max_contacts):

			# %% break if we are into un-filled contacts
			if (self.sim.data.contact[i].geom1 == 0 \
				and self.sim.data.contact[i].geom2 == 0):
				break

		return


	def reset(self):

		# %% close gripper
		for i in range(6):
			self.sim.data.qpos[i+6]=0.0

		# %% close object
		self.sim.data.qpos[-1]=0.0
		self.sim.data.qpos[-2]=0.0

		# %% gravity comp
		for i in range(len(self.sim.data.ctrl)):
			self.sim.data.qfrc_applied[i]=self.sim.data.qfrc_bias[i]

		# %% sample random (valid) 6DoF pose
		self.sample_random_pose()

		# %% reset controller
		self.cip.controller.reset()
		obs = np.concatenate( [self.sim.data.qpos, self.sim.data.sensordata] )
		self.elapsed_steps=0
		return obs

	def cost(self):
		# %% TODO: smoothness cost?
		# %% TODO: handle distance?
		gripper_idx = cymj._mj_name2id(self.sim.model, 1, "j2s6s300_link_6")
		handle_idx  = cymj._mj_name2id(self.sim.model, 1, "latch")
		gripper_handle_displacement = self.sim.data.body_xpos[gripper_idx] - \
									  self.sim.data.body_xpos[handle_idx]

		print(gripper_handle_displacement)
		gripper_handle_distance = np.linalg.norm(gripper_handle_displacement)
		cost = DOOR_WEIGHT*(DOOR_GOAL - self.sim.data.qpos[-2])**2 + \
			   HANDLE_WEIGHT*(HANDLE_GOAL - self.sim.data.qpos[-1])**2 + \
			   GRASP_WEIGHT*(gripper_handle_distance)

		print('--')
		print(self.elapsed_steps)
		print(DOOR_WEIGHT*(DOOR_GOAL - self.sim.data.qpos[-2])**2)
		print(HANDLE_WEIGHT*(HANDLE_GOAL - self.sim.data.qpos[-1])**2)
		print(GRASP_WEIGHT*(gripper_handle_distance))
		return cost


	def step(self, action):

		# %% interpret action as target [pos, ori] of gripper
		policy_step = True
		for i in range(int(self.control_timestep / self.model_timestep)):
			torques=action
			# torques = self.cip.get_action(action, policy_step)
			torques += self.sim.data.qfrc_bias[:self.arm_dof]
			self.sim.data.ctrl[:self.arm_dof] = torques
			#
			# # %% TODO: gripper action
			# self.sim.step()
			# policy_step = False
			# self.elapsed_steps+=1

		self.viewer.render() if self.vis else None

		reward = -1*self.cost()

		info={'goal_achieved': reward > -1e-1 }
		done = self.elapsed_steps >= (self.n_steps - 1)
		obs = np.concatenate([self.sim.data.qpos, self.sim.data.sensordata])
		return obs, reward, done, info
